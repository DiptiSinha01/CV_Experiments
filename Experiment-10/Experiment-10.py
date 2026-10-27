import cv2
from ultralytics import YOLO

VIDEO_PATH = "traffic.mp4"        # change to your video file name
CONF_THRESHOLD = 0.4            # ignore detections below this confidence

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "Car", 5: "Bus", 7: "Truck"}
COLORS = {"Car": (0, 255, 0), "Bus": (255, 0, 0), "Truck": (0, 0, 255)}

# Loads a pre-trained YOLO model (downloads yolov8n.pt the first time)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open {VIDEO_PATH}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    results = model(frame, verbose=False)[0]
    counts = {"Car": 0, "Bus": 0, "Truck": 0}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id not in VEHICLE_CLASSES or conf < CONF_THRESHOLD:
            continue

        name = VEHICLE_CLASSES[cls_id]
        counts[name] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[name], 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[name], 2)

    # Show vehicle counts in the top left corner
    y = 30
    for name, c in counts.items():
        cv2.putText(frame, f"{name}s: {c}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[name], 2)
        y += 30

    cv2.imshow("YOLO Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()