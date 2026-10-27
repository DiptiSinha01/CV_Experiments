import os
import urllib.request

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_lite.task"


def download_model(model_path: str) -> None:
    if os.path.exists(model_path):
        return

    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    print(f"Downloading pose model to {model_path}...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Pose model downloaded successfully.")
    except Exception as exc:
        raise RuntimeError(f"Could not download MediaPipe pose model: {exc}") from exc


def initialize_pose_detector():
    download_model(MODEL_PATH)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def draw_pose(frame, landmarks):
    h, w, _ = frame.shape

    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]

        x1 = int(start.x * w)
        y1 = int(start.y * h)
        x2 = int(end.x * w)
        y2 = int(end.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    detector = initialize_pose_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check that your camera is connected.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        if result and result.pose_landmarks:
            draw_pose(frame, result.pose_landmarks[0])

        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()