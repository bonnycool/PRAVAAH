from ultralytics import YOLO
import cv2

# Load a lightweight YOLO model
model = YOLO('yolov5n.pt')  # Lightweight YOLO model (or use yolov5m.pt for better accuracy)
model.classes = [2]  # Only detect cars (COCO class ID = 2)

# Open the webcam
cap = cv2.VideoCapture(1)  # Use the correct index for your webcam (0, 1, etc.)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Set video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get webcam FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

# Define VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('optimized_output.mp4', fourcc, fps, (width, height))

# Frame processing loop
frame_count = 0
car_counts = []
frame_skip = 3  # Skip every 3rd frame to reduce processing load

print("Starting car detection... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    frame_count += 1

    # Skip frames for faster processing
    if frame_count % frame_skip != 0:
        continue

    # Convert frame to RGB (YOLO expects RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform car detection
    results = model(frame_rgb, conf=0.5)  # Confidence threshold of 50%
    car_count = 0

    # Process detection results
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf.item())  # Convert tensor to a standard Python float
            if cls == 2 and conf > 0.5:  # Only count cars with confidence > 50%
                car_count += 1
                # Convert tensor to list before extracting coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Ensure proper extraction of tensor values
                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add label
                label = f"Car {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    car_counts.append(car_count)

    # Display the frame with detections
    cv2.imshow('Car Detection', frame)

    # Write the frame to the output video
    out.write(frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting car detection...")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Print detection summary
print(f"Total frames processed: {frame_count}")
print(f"Car counts per frame: {car_counts}")
total_cars = sum(car_counts)
print(f"Total cars detected: {total_cars}")
avg_cars_per_frame = total_cars / len(car_counts) if car_counts else 0
print(f"Average cars per frame: {avg_cars_per_frame:.2f}")
