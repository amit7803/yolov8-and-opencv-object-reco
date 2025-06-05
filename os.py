from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model (default is yolov8n - nano, fast and light)
model = YOLO("yolov8n.pt")  # you can use yolov8s.pt, yolov8m.pt, etc.

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Show the result
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
