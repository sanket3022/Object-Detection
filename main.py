import cv2
import numpy as np

net = cv2.dnn.readNet("C:\\Users\\Sanket Sharma\\Desktop\\SANKET MAJOR PROJECT\\objec_dec main\\objec_dec\\yolov3.weights", "C:\\Users\\Sanket Sharma\\Desktop\\SANKET MAJOR PROJECT\\objec_dec main\\objec_dec\\yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

with open("C:\\Users\\Sanket Sharma\\Desktop\\SANKET MAJOR PROJECT\\objec_dec main\\objec_dec\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)

display_size = (800, 600)
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", display_size[0], display_size[1])

# Object detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3: 
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
