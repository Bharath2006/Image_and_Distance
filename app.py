import cv2
import numpy as np
import pyttsx3
from flask import Flask, render_template, Response, jsonify, request
import traceback
import threading
from datetime import datetime
import time

app = Flask(__name__)

detection_enabled = False
voice_enabled = False
state_lock = threading.Lock()
detection_history = []
MAX_HISTORY = 200

def load_yolo():
    try:
        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        layer_names = net.getLayerNames()
        unconnected_out_layers = net.getUnconnectedOutLayers()
        if hasattr(unconnected_out_layers, "flatten"):
            out_idxs = unconnected_out_layers.flatten()
        else:
            out_idxs = [int(x) for x in unconnected_out_layers]
        output_layers = [layer_names[i - 1] for i in out_idxs]
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return net, classes, output_layers
    except Exception as e:
        print("❌ Error loading YOLO:", e)
        traceback.print_exc()
        raise

net, classes, output_layers = load_yolo()

engine = pyttsx3.init()
tts_lock = threading.Lock()

focal_length = 615
known_width = 0.5

def four_word_description(label):

    words = f"{label} ahead take care"
    parts = words.split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    else:
        return (words + " now now now").split()[:4]
    
def speak_async(text):
    def _speak(txt):
        with tts_lock:
            try:
                engine.say(txt)
                engine.runAndWait()
            except Exception:
                pass
    thr = threading.Thread(target=_speak, args=(text,), daemon=True)
    thr.start()

def detect_objects(frame):
    try:
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                     (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                if len(scores) == 0:
                    continue
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > 0.5:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = []
        if boxes:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if hasattr(indexes, "flatten"):
                indexes = indexes.flatten().tolist()
            else:
                try:
                    indexes = [int(i) for i in indexes]
                except:
                    indexes = []

        return boxes, confidences, class_ids, indexes
    except Exception as e:
        print("❌ Detection error:", e)
        traceback.print_exc()
        return [], [], [], []

def record_detection(label, confidence, distance):
    entry = {
        "label": str(label),
        "confidence": round(float(confidence), 3),
        "distance_m": round(float(distance), 2) if distance is not None else None,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": four_word_description(label),
        "commands": ["stop", "pause", "describe"]      }
    with state_lock:
        detection_history.insert(0, entry)
        if len(detection_history) > MAX_HISTORY:
            detection_history.pop()

def draw_labels(boxes, confidences, class_ids, indexes, classes, frame):
    try:
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]] if class_ids[i] < len(classes) else str(class_ids[i])
                confidence = confidences[i]
                color = (0, 255, 0) 
                dangerous_objects = ["knife", "gun", "rifle", "pistol", "firearm"]
                is_danger = label.lower() in dangerous_objects
                if is_danger:
                    color = (0, 0, 255)
                    alert_text = f"WARNING! {label.upper()} detected nearby!"
                    with tts_lock:
                        try:
                            engine.setProperty('volume', 1.0)
                            engine.say(alert_text)
                            engine.runAndWait()
                        except:
                            pass
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                distance = None
                if w != 0:
                    distance = (known_width * focal_length) / w
                    cv2.putText(frame, f"Distance: {distance:.2f}m",
                                (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                entry = {
                    "label": label,
                    "confidence": round(confidence, 3),
                    "distance_m": round(distance, 2) if distance else None,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "description": four_word_description(label),
                    "commands": ["stop", "pause", "describe"],
                    "danger": is_danger
                }
                with state_lock:
                    detection_history.insert(0, entry)
                    if len(detection_history) > MAX_HISTORY:
                        detection_history.pop()

                with state_lock:
                    v_on = voice_enabled
                if v_on and not is_danger:
                    speak_async(f"{label} detected at {distance:.2f} meters" if distance else f"{label} detected")
    except Exception as e:
        print("❌ Label drawing error:", e)

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not found.")
        return

    last_frame_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break
        with state_lock:
            enabled = detection_enabled

        if enabled:
            boxes, confidences, class_ids, indexes = detect_objects(frame)
            draw_labels(boxes, confidences, class_ids, indexes, classes, frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_enabled
    with state_lock:
        detection_enabled = True
    return jsonify({"status": "started"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_enabled
    with state_lock:
        detection_enabled = False
    return jsonify({"status": "stopped"})

@app.route('/voice_on', methods=['POST'])
def voice_on():
    global voice_enabled
    with state_lock:
        voice_enabled = True
    return jsonify({"voice": "on"})

@app.route('/voice_off', methods=['POST'])
def voice_off():
    global voice_enabled
    with state_lock:
        voice_enabled = False
    return jsonify({"voice": "off"})

@app.route('/get_history')
def get_history():
    with state_lock:
        return jsonify(detection_history[:50])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    with state_lock:
        detection_history.clear()
    return jsonify({"cleared": True})


@app.route('/describe_last', methods=['POST'])
def describe_last():
    with state_lock:
        if not detection_history:
            return jsonify({"status":"no_detection"})
        last = detection_history[0]
        label = last.get("label", "object")
        distance = last.get("distance_m", None)
        desc = last.get("description", "object detected")
    
    if distance is not None:
        speak_text = f"{label} detected at {distance:.2f} meters. {desc}."
    else:
        speak_text = f"{label} detected. {desc}."
    
    speak_async(speak_text)
    return jsonify({"status":"spoken", "description": speak_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
