import cv2
import numpy as np
from tkinter import * 
from pydub import AudioSegment
from pydub.playback import play
import time
import os

# Load YOLO model
net = cv2.dnn.readNet("C:\\Users\\Sanket Sharma\\Desktop\\SANKET MAJOR PROJECT\\objec_dec main\\objec_dec\\yolov3.weights", "C:\\Users\\Sanket Sharma\\Desktop\\SANKET MAJOR PROJECT\\objec_dec main\\objec_dec\\yolov3.cfg")

# Load class names
with open("C:\\Users\\Sanket Sharma\\Desktop\\SANKET MAJOR PROJECT\\objec_dec main\\objec_dec\\coco.names", "r") as f:
    classes = f.read().strip().split("\n")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Function to calculate distance from object based on its size in the image
def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

def main():
    
    known_distance = 50 
    known_width = 15 

    
    cap = cv2.VideoCapture(0)

    focal_length = 800

    objects_distances = {}

    while True:
        
        ret, frame = cap.read()

        
        height, width, channels = frame.shape

       
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    label = str(classes[class_id])

                    # Calculate the distance to the object
                    distance = distance_to_camera(known_width, focal_length, w)

                    # Store object name and distance
                    objects_distances[label] = distance

                    # Print the distance and object name
                    print(f"Object: {label}, Distance: {distance:.2f} cm")
                    if f"{distance:.2f}"<"30":
                        #song = AudioSegment.from_mp3("C:\\Users\\Sanket Sharma\\Desktop\\SANKET MAJOR PROJECT\\objec_dec main\\objec_dec\\beep.mp3")
                        #play(song)
                        file="beep.mp3"
                        os.system(file)

                    # Display the distance on the frame
                    cv2.putText(frame, f"{label}: {distance:.2f} cm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw rectangle around the object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            time.sleep(2)
        
        cv2.imshow('Object Detection', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        


    cap.release()
    cv2.destroyAllWindows()

    
    print("Objects and distances:")
    for obj, dist in objects_distances.items():
        print(f"{obj}: {dist:.2f} cm")
        # if dist<20:
        #   print("hello")
        

if __name__ == "__main__":
    main()
