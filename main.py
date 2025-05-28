import cv2
import numpy as np
import time
from pygame import mixer
import os
import sys

# Check if video file path is provided as command line argument
video_path = None
if len(sys.argv) > 1 and sys.argv[1] == '--video' and len(sys.argv) > 2:
    video_path = sys.argv[2]

print("Starting object detection application...")

# Initialize sound system
try:
    mixer.init()
    sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beep.mp3")
    print(f"Loading sound from: {sound_path}")
    beep_sound = mixer.Sound(sound_path)
    print("Sound loaded successfully")
except Exception as e:
    print(f"Error initializing sound: {e}")
    exit(1)

# Initialize YOLO model with absolute paths
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "yolov3.weights")
    config_path = os.path.join(base_dir, "yolov3.cfg")
    names_path = os.path.join(base_dir, "coco.names")
    
    print(f"Loading YOLO model from: {weights_path} and {config_path}")
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getUnconnectedOutLayersNames()
    print("YOLO model loaded successfully")

    print(f"Loading class names from: {names_path}")
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(classes)} classes")
except Exception as e:
    print(f"Error initializing model: {e}")
    exit(1)

# Function to calculate distance based on object width
def distance_to_camera(known_width, focal_length, per_width):
    # Prevent division by zero
    if per_width == 0:
        return float('inf')
    return (known_width * focal_length) / per_width

# Function to calculate IOU (Intersection Over Union) between two boxes
def calculate_iou(box1, box2):
    # box format: [x, y, w, h]
    # Convert to [x1, y1, x2, y2] format
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

# Dictionary of known object widths in cm (Expanded with more objects)
KNOWN_WIDTHS = {
    # People and animals
    "person": 40,       # Average width of a person
    "dog": 30,          # Average width of a dog
    "cat": 20,          # Average width of a cat
    "bird": 10,         # Average width of a bird
    
    # Vehicles
    "car": 180,         # Average width of a car
    "bicycle": 60,      # Average width of a bicycle
    "motorcycle": 80,   # Average width of a motorcycle
    "bus": 250,         # Average width of a bus
    "truck": 240,       # Average width of a truck
    
    # Electronics
    "laptop": 35,       # Average width of a laptop
    "cell phone": 7,    # Average width of a cell phone
    "tv": 100,          # Average width of a TV
    "mouse": 6,         # Average width of a mouse
    "keyboard": 35,     # Average width of a keyboard
    "remote": 5,        # Average width of a remote
    
    # Furniture
    "chair": 50,        # Average width of a chair
    "couch": 200,       # Average width of a couch
    "bed": 150,         # Average width of a bed
    "dining table": 120,# Average width of a dining table
    "desk": 120,        # Average width of a desk
    
    # Kitchen items
    "bottle": 8,        # Average width of a bottle
    "cup": 8,           # Average width of a cup
    "bowl": 15,         # Average width of a bowl
    "plate": 25,        # Average width of a plate
    "fork": 2,          # Average width of a fork
    "knife": 2,         # Average width of a knife
    "spoon": 2,         # Average width of a spoon
    
    # Common objects
    "book": 15,         # Average width of a book
    "clock": 20,        # Average width of a clock
    "vase": 15,         # Average width of a vase
    "scissors": 8,      # Average width of scissors
    "teddy bear": 25,   # Average width of a teddy bear
    "hair drier": 20,   # Average width of a hair drier
    "toothbrush": 2,    # Average width of a toothbrush
    
    # Sports equipment
    "tennis racket": 30,# Average width of a tennis racket
    "baseball bat": 8,  # Average width of a baseball bat
    "baseball glove": 25,# Average width of a baseball glove
    "skateboard": 20,   # Average width of a skateboard
    "sports ball": 15,  # Average width of a sports ball
}

# Adjust default width and focal length
DEFAULT_WIDTH = 20  # Increased from 15
FOCAL_LENGTH = 800

# Set up video capture
try:
    if video_path:
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        exit(1)
    print("Video source opened successfully")
except Exception as e:
    print(f"Error with video source: {e}")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer if processing a video file
if video_path:
    output_path = os.path.splitext(video_path)[0] + '_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Output video will be saved as: {output_path}")

display_size = (800, 600)
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", display_size[0], display_size[1])

# Variables for sound control
last_beep_time = 0
beep_cooldown = 2  # seconds between beeps
close_object_detected = False
proximity_threshold = 30  # cm

# Create a dictionary to track object IDs and their distances
tracked_objects = {}
# Dictionary to store previous frame's boxes for each class
prev_boxes_by_class = {}

print("Starting main detection loop...")

# Object detection
frame_count = 0
try:
    while True:
        start_time = time.time()
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Processing frame {frame_count}")
            
        ret, frame = cap.read()
        if not ret:
            if video_path:
                print("End of video file reached")
            else:
                print("Error: Can't receive frame from camera")
            break

        height, width, _ = frame.shape

        # Prepare image for YOLO with multi-scale detection
        scales = [0.8, 1.0, 1.2]  # Multiple scales for better detection
        all_boxes = []
        all_confidences = []
        all_class_ids = []
        
        for scale in scales:
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image for this scale
            if scale != 1.0:
                scaled_frame = cv2.resize(frame, (new_width, new_height))
            else:
                scaled_frame = frame
            
            # Prepare blob for this scale
            blob = cv2.dnn.blobFromImage(scaled_frame, 1/255.0, (416, 416), 
                                       swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward(layer_names)

            # Process detections for this scale
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Lower confidence threshold for more detections
                    if confidence > 0.4:  # Reduced from 0.6
                        center_x = int(obj[0] * new_width)
                        center_y = int(obj[1] * new_height)
                        w = int(obj[2] * new_width)
                        h = int(obj[3] * new_height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Scale back coordinates if needed
                        if scale != 1.0:
                            x = int(x / scale)
                            y = int(y / scale)
                            w = int(w / scale)
                            h = int(h / scale)

                        all_boxes.append([x, y, w, h])
                        all_confidences.append(float(confidence))
                        all_class_ids.append(class_id)

        # Use all collected detections
        boxes = all_boxes
        confidences = all_confidences
        class_ids = all_class_ids

        # Group boxes by class
        boxes_by_class = {}
        for i, class_id in enumerate(class_ids):
            if class_id not in boxes_by_class:
                boxes_by_class[class_id] = []
            boxes_by_class[class_id].append((i, boxes[i], confidences[i]))

        # Create a new list that will store the indices of boxes to keep after class-wise filtering
        filtered_indices = []
        
        # Process each class separately with adjusted NMS
        for class_id, class_boxes in boxes_by_class.items():
            # Sort boxes by confidence (highest first)
            class_boxes.sort(key=lambda x: x[2], reverse=True)
            
            # Get indices of boxes for this class
            indices_for_class = [box[0] for box in class_boxes]
            boxes_for_class = [box[1] for box in class_boxes]
            
            # Apply class-specific NMS with adjusted threshold
            nms_results = cv2.dnn.NMSBoxes(boxes_for_class, 
                                          [confidences[i] for i in indices_for_class], 
                                          0.4,    # Lower confidence threshold (was 0.6)
                                          0.3)    # Higher NMS threshold (was 0.2) to keep more overlapping boxes
            
            if len(nms_results) > 0:
                if isinstance(nms_results[0], list):  # Handle older OpenCV versions
                    nms_results = [item[0] for item in nms_results]
                
                # Add the filtered indices for this class
                for idx in nms_results:
                    filtered_indices.append(indices_for_class[idx])
        
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Detected {len(boxes)} objects, {len(filtered_indices)} after filtering")
        
        # Reset close object detection flag for this frame
        close_object_detected = False
        
        # Current tracked objects in this frame
        current_objects = {}
        
        # Final processed boxes for display and distance calculation
        final_boxes = []
        
        # Process the filtered boxes
        for i in filtered_indices:
            box = boxes[i]
            x, y, w, h = box
            class_id = class_ids[i]
            label = classes[class_id]
            confidence = confidences[i]
            
            # Skip very small detections (reduced minimum size)
            min_size = 15  # Reduced from 20 pixels
            if w < min_size or h < min_size:
                continue
                
            # Get appropriate width for distance calculation based on the object type
            known_width = KNOWN_WIDTHS.get(label, DEFAULT_WIDTH)
            
            # Calculate distance from camera
            distance = distance_to_camera(known_width, FOCAL_LENGTH, w)
            
            # Smooth distance with previous measurements if available
            obj_id = f"{label}_{i}"  # Create a simple object ID
            if obj_id in tracked_objects:
                # Apply simple moving average for smoother distance estimates
                prev_distance = tracked_objects[obj_id]
                distance = 0.7 * prev_distance + 0.3 * distance  # 70% previous, 30% new
            
            # Store the current distance
            tracked_objects[obj_id] = distance
            current_objects[obj_id] = True
            
            # Add to final boxes for rendering
            final_boxes.append((box, class_id, label, confidence, distance))
        
        # Draw the final boxes
        for box, class_id, label, confidence, distance in final_boxes:
            x, y, w, h = box
            
            # Color based on distance (green if far, yellow if medium, red if close)
            color = (0, 255, 0)  # Default green
            if distance < 50:
                color = (0, 255, 255)  # Yellow
            if distance < proximity_threshold:
                color = (0, 0, 255)  # Red
                close_object_detected = True
                if frame_count % 10 == 0:  # Print less frequently
                    print(f"Close object detected: {label} at {distance:.1f} cm")
                
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_text = f"{label}: {distance:.1f} cm ({confidence:.2f})"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Clean up tracked objects that are no longer visible
        for obj_id in list(tracked_objects.keys()):
            if obj_id not in current_objects:
                del tracked_objects[obj_id]
        
        # Play sound if object is closer than proximity_threshold and cooldown period has elapsed
        current_time = time.time()
        if close_object_detected and (current_time - last_beep_time > beep_cooldown):
            print(f"Playing sound alert - object detected within {proximity_threshold}cm")
            beep_sound.play()
            last_beep_time = current_time

        # Display the frame
        cv2.imshow("Object Detection", frame)
        
        # Save frame to output video if processing a video file
        if video_path:
            out.write(frame)
        
        # Calculate and display FPS
        end_time = time.time()
        fps = 1/(end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit key pressed, exiting...")
            break

except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    # Clean up
    print("Cleaning up resources...")
    cap.release()
    if video_path:
        out.release()
    cv2.destroyAllWindows()
    print("Application closed")
