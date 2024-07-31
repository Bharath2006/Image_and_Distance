import cv2
import numpy as np
import pyttsx3


def load_yolo():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    print("Unconnected Out Layers:", unconnected_out_layers)  # Debug statement
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes, output_layers

def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

def draw_labels(boxes, confidences, class_ids, indexes, classes, img, engine, focal_length, known_width):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            distance = (known_width * focal_length) / w
            cv2.putText(img, f"Distance: {distance:.2f}m", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Speak the label and distance
            engine.say(f"{label} detected at a distance of {distance:.2f} meters")
            engine.runAndWait()

def start_video():
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    engine = pyttsx3.init()
    focal_length = 615  # Example focal length in pixels (calibrate your camera to get the exact value)
    known_width = 0.5  # Example known width of the object in meters (use a known object for accuracy)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, confidences, class_ids, indexes = detect_objects(frame, net, output_layers)
        draw_labels(boxes, confidences, class_ids, indexes, classes, frame, engine, focal_length, known_width)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video()
