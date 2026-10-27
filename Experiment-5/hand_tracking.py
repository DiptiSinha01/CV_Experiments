import cv2
import numpy as np
import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

print("Hand Tracking + Gesture Recognition")
print("=====================================")


model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Trying alternative download...")
        # Alternative URL
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Failed to download model: {e}")
            exit()


print("Initializing hand detection model...")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


tip_ids = [4, 8, 12, 16, 20]   
pip_ids = [3, 6, 10, 14, 18]   

def classify_gesture(landmarks, handedness):
    
    if not landmarks:
        return "No hand", 0
    
    fingers = []
    
    
    thumb_up = abs(landmarks[4].x - landmarks[2].x) > abs(landmarks[4].y - landmarks[2].y) * 0.5
    fingers.append(1 if thumb_up else 0)
    
    
    for tip_idx, pip_idx in zip([8, 12, 16, 20], [6, 10, 14, 18]):
      
        finger_up = landmarks[tip_idx].y < (landmarks[pip_idx].y - 0.01)
        fingers.append(1 if finger_up else 0)
    
    total = sum(fingers)
   
    if total == 0:
        return "Fist", 0
    elif total == 5:
        return "Open Palm", 5
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace/Victory", 2
    elif fingers == [0, 1, 0, 0, 0]:
        return "Pointing", 1
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up", 1
    elif fingers == [1, 1, 0, 0, 1]:
        return "Rock On", 3
    elif fingers == [1, 0, 0, 0, 1]:
        return "Call Me", 2
    elif total == 4 and fingers[0] == 0:
        return "Four Fingers", 4
    elif total == 3:
        return "Three Fingers", 3
    elif total == 2 and fingers[0] == 1:
        return "Two Fingers + Thumb", 2
    else:
        return f"Gesture", total

def draw_landmarks(image, landmarks, handedness):
  
    if not landmarks:
        return
    
    h, w, c = image.shape
    
   
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Hand center connections
    ]
    
    for connection in connections:
        start = landmarks[connection[0]]
        end = landmarks[connection[1]]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
   
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(image, (x, y), 5, (255, 0, 0), 2)

print("Starting webcam capture...")
print("Press 'q' to quit")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened successfully!")
print("Hand detection active - move your hand in front of camera\n")

frame_count = 0
while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame from webcam")
        break

    frame = cv2.flip(frame, 1) 
    h, w, c = frame.shape
    
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
   
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    
    
    detection_result = detector.detect(mp_image)
    
    
    cv2.putText(frame, "Hand Detection Active", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw detected hands
    if detection_result.hand_landmarks and detection_result.handedness:
        for landmarks, handedness_list in zip(detection_result.hand_landmarks, detection_result.handedness):
            # handedness is a list of classifications
            if handedness_list:
                handedness = handedness_list[0].category_name
            else:
                handedness = "Unknown"
            
            # Draw landmarks
            draw_landmarks(frame, landmarks, handedness)
            
            # Get gesture
            gesture, finger_count = classify_gesture(landmarks, handedness)
            
            # Display gesture near wrist (without handedness label)
            wrist = landmarks[0]
            x, y = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, f"{gesture}", (x - 50, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f"Fingers: {finger_count}", (x - 50, y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No hands detected", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    cv2.imshow("Hand Tracking + Gesture Recognition", frame)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nQuitting...")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Webcam closed. Processed {frame_count} frames total.")