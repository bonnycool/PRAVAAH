from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model = YOLO('yolov10x.pt')  # Make sure this path is correct or the model is already downloaded

# Open the webcam
cap = cv2.VideoCapture(0)  # Change 0 to the webcam index if you have multiple webcams

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error opening webcam")
    exit()

# Get webcam properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
out = cv2.VideoWriter('output_webcam.mp4', fourcc, fps, (width, height))

# Process each frame of the webcam feed
frame_count = 0
car_counts = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from webcam")
        break

    frame_count += 1

    # Convert frame to RGB for YOLO processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Additional check for uint8 data type which is expected by YOLO
    if frame_rgb.dtype != np.uint8:
        frame_rgb = frame_rgb.astype(np.uint8)

    # Perform object detection with try-except for error handling
    try:
        results = model(frame_rgb)
    except Exception as e:
        print(f"Error during object detection for frame {frame_count}: {e}")
        continue  # Skip this frame and continue with the next one

    # Count cars in the frame
    car_count = 0
    car_class_id = 2  # Assuming car class ID is 2 for COCO dataset

    for r in results:
        for detection in r.boxes:
            cls = int(detection.cls)
            if cls == car_class_id and detection.conf > 0.5:  # Confidence threshold
                car_count += 1

    car_counts.append(car_count)

    # Draw detections on the frame
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if cls == car_class_id and conf > 0.5:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw in BGR space
                # Add text for class and confidence
                label = f'Car {conf:.2f}'
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Webcam Feed', frame)

    # Write the frame with detections to the output video
    out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and writer, close windows
cap.release()
out.release()
cv2.destroyAllWindows()

# Print summary of car counts per frame
print(f"Total frames processed: {frame_count}")
print(f"Car counts per frame: {car_counts}")

# Compute total cars in the webcam feed
total_cars = sum(car_counts)
print(f"Total number of cars in the webcam feed: {total_cars}")

# Optionally, compute average cars per frame
avg_cars_per_frame = total_cars / frame_count if frame_count else 0
