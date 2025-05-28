import cv2
import numpy as np
import time
import os
from pygame import mixer
import csv
from datetime import datetime
import json
import sys

print("Starting object detection application (obj1.py)...")

# Check if video file path is provided as command line argument
video_path = None
if len(sys.argv) > 1 and sys.argv[1] == '--video' and len(sys.argv) > 2:
    video_path = sys.argv[2]

try:
    mixer.init()
    sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beep.mp3")
    print(f"Loading sound from: {sound_path}")
    beep_sound = mixer.Sound(sound_path)
    print("Sound loaded successfully")
except Exception as e:
    print(f"Error initializing sound: {e}")
    exit(1)

# for YOLO model 
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "yolov3.weights")
    config_path = os.path.join(base_dir, "yolov3.cfg")
    names_path = os.path.join(base_dir, "coco.names")
    
    print(f"Loading YOLO model from: {weights_path} and {config_path}")
    net = cv2.dnn.readNet(weights_path, config_path)
    
    print(f"Loading class names from: {names_path}")
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(classes)} classes")
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error initializing model: {e}")
    exit(1)

def distance_to_camera(known_width, focal_length, per_width):
    # Prevent division by zero
    if per_width == 0:
        return float('inf')
    return (known_width * focal_length) / per_width

# Dictionary of known object widths in cm
KNOWN_WIDTHS = {
    "person": 40,       
    "car": 180,         
    "bottle": 8,        
    "cup": 8,           
    "cell phone": 7,    
    "book": 15,         
    "laptop": 35,       
    "chair": 50,        
    "tv": 100,          
    "mouse": 6,         
    "keyboard": 35,     }
# Default width for objects not in the dictionary
DEFAULT_WIDTH = 15

def main():
    # Focal length (can be calibrated for your specific camera)
    focal_length = 800
    
    # Create output directory if it doesn't exist
    output_dir = "detection_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV file for real-time detection data
    csv_filename = os.path.join(output_dir, f"detection_data_{timestamp}.csv")
    
    # Create JSON file for summary data
    json_filename = os.path.join(output_dir, f"detection_summary_{timestamp}.json")
    
    # Initialize CSV file with headers
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'Object', 'Distance (cm)', 'Confidence', 'Position (x,y,w,h)', 'Frame Number'])
    
    # Dictionary to store detection statistics
    detection_stats = {
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_frames_processed': 0,
        'objects_detected': {},
        'closest_detections': {},
        'detection_frequency': {}
    }
    
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
            
    except Exception as e:
        print(f"Error with video source: {e}")
        exit(1)

    # Dictionary to store detected objects and their distances
    objects_distances = {}
    
    # Variables for sound control
    last_beep_time = 0
    beep_cooldown = 2  # seconds between beeps
    proximity_threshold = 30  # cm
    
    # Create a dictionary to track object IDs and their distances
    tracked_objects = {}
    
    print("Starting main detection loop...")
    frame_count = 0
    
    try:
        while True:
            start_time = time.time()
            frame_count += 1
            detection_stats['total_frames_processed'] = frame_count
            
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Processing frame {frame_count}")
                
            ret, frame = cap.read()
            if not ret:
                if video_path:
                    print("End of video file reached")
                else:
                    print("Error: Can't receive frame from camera")
                break
            
            height, width, channels = frame.shape
            
            # Prepare image for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            
            # Collect all detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.6:  # Increased confidence threshold
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Group boxes by class
            boxes_by_class = {}
            for i, class_id in enumerate(class_ids):
                if class_id not in boxes_by_class:
                    boxes_by_class[class_id] = []
                boxes_by_class[class_id].append((i, boxes[i], confidences[i]))
            
            # Create a new list that will store the indices of boxes to keep after class-wise filtering
            filtered_indices = []
            
            # Process each class separately
            for class_id, class_boxes in boxes_by_class.items():
                # Sort boxes by confidence (highest first)
                class_boxes.sort(key=lambda x: x[2], reverse=True)
                
                # Get indices of boxes for this class (sorted by confidence)
                indices_for_class = [box[0] for box in class_boxes]
                boxes_for_class = [box[1] for box in class_boxes]
                
                # Apply class-specific NMS with stricter threshold
                nms_results = cv2.dnn.NMSBoxes(
                    boxes_for_class, 
                    [confidences[i] for i in indices_for_class], 
                    0.6, 0.2  # Higher confidence threshold, lower NMS threshold
                )
                
                if len(nms_results) > 0:
                    if isinstance(nms_results[0], list):  # Handle older OpenCV versions
                        nms_results = [item[0] for item in nms_results]
                    
                    # Add the filtered indices for this class
                    for idx in nms_results:
                        filtered_indices.append(indices_for_class[idx])
            
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Detected {len(boxes)} objects, {len(filtered_indices)} after filtering")
            
            # Flag to check if any object is too close
            close_object_detected = False
            
            # Current tracked objects in this frame
            current_objects = {}
            
            # Process the filtered boxes
            for i in filtered_indices:
                box = boxes[i]
                x, y, w, h = box
                class_id = class_ids[i]
                label = str(classes[class_id])
                confidence = confidences[i]
                
                # Skip very small detections (likely false positives)
                min_size = 20  # Minimum width/height in pixels
                if w < min_size or h < min_size:
                    continue
                
                # Get appropriate width for distance calculation based on the object type
                known_width = KNOWN_WIDTHS.get(label, DEFAULT_WIDTH)
                
                # Calculate distance from camera
                distance = distance_to_camera(known_width, focal_length, w)
                
                # Smooth distance with previous measurements if available
                obj_id = f"{label}_{i}"
                if obj_id in tracked_objects:
                    # Apply simple moving average for smoother distance estimates
                    prev_distance = tracked_objects[obj_id]
                    distance = 0.7 * prev_distance + 0.3 * distance  # 70% previous, 30% new
                
                # Store the current distance
                tracked_objects[obj_id] = distance
                current_objects[obj_id] = True
                
                # Update overall object distances dictionary
                objects_distances[label] = distance
                
                # Save detection data to CSV
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                with open(csv_filename, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([current_time, label, f"{distance:.2f}", f"{confidence:.2f}", f"{x},{y},{w},{h}", frame_count])
                
                # Color based on distance (green if far, yellow if medium, red if close)
                color = (0, 255, 0)  # Default green
                if distance < 50:
                    color = (0, 255, 255)  # Yellow
                if distance < proximity_threshold:
                    color = (0, 0, 255)  # Red
                    close_object_detected = True
                    if frame_count % 10 == 0:  # Print less frequently
                        print(f"Close object detected: {label} at {distance:.1f} cm")
                
                # Display distance on the frame
                cv2.putText(frame, f"{label}: {distance:.1f} cm", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw rectangle around the object
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
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
            
            # Calculate and display FPS
            end_time = time.time()
            fps = 1/(end_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save frame to output video if processing a video file
            if video_path:
                out.write(frame)
            
            # Display the result
            cv2.imshow('Object Detection', frame)
            
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
        
        # Update final statistics
        detection_stats['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detection_stats['duration_seconds'] = (datetime.now() - datetime.strptime(detection_stats['start_time'], "%Y-%m-%d %H:%M:%S")).total_seconds()
        
        # Calculate detection frequencies
        for obj, count in detection_stats['objects_detected'].items():
            detection_stats['detection_frequency'][obj] = count / detection_stats['total_frames_processed']
        
        # Save summary statistics to JSON
        with open(json_filename, 'w') as jsonfile:
            json.dump(detection_stats, jsonfile, indent=4)
        
        print("\nDetection Summary:")
        print(f"Total frames processed: {detection_stats['total_frames_processed']}")
        print(f"Duration: {detection_stats['duration_seconds']:.2f} seconds")
        print("\nObjects detected:")
        for obj, count in detection_stats['objects_detected'].items():
            print(f"{obj}: {count} times (closest: {detection_stats['closest_detections'][obj]['distance']:.2f} cm)")
        
        print(f"\nDetailed detection data saved to: {csv_filename}")
        print(f"Summary statistics saved to: {json_filename}")
        if video_path:
            print(f"Processed video saved to: {output_path}")

if __name__ == "__main__":
    main()
