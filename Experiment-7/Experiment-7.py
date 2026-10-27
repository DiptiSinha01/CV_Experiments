import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face Detection
    detection_results = face_detection.process(img_rgb)

    if detection_results.detections:
        for detection in detection_results.detections:

            bbox = detection.location_data.relative_bounding_box

            h, w, c = img.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            cv2.rectangle(img,
                          (x, y),
                          (x + bw, y + bh),
                          (0, 255, 0),
                          2)

    # Face Mesh
    # mesh_results = face_mesh.process(img_rgb)

    # if mesh_results.multi_face_landmarks:
    #     for face_landmarks in mesh_results.multi_face_landmarks:

    #         mp_draw.draw_landmarks(
    #             img,
    #             face_landmarks,
    #             mp_face_mesh.FACEMESH_TESSELATION
    #         )

        # Face Mesh
    # mesh_results = face_mesh.process(img_rgb)
    # h, w, _ = img.shape

    # if mesh_results.multi_face_landmarks:
    #     for face_landmarks in mesh_results.multi_face_landmarks:
    #         # Draw every 4th landmark (change 4 to 5 or 6 for even fewer points)
    #         for i, lm in enumerate(face_landmarks.landmark):
    #             if i % 4 == 0:
    #                 x = int(lm.x * w)
    #                 y = int(lm.y * h)
    #                 cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    # Face Mesh
    mesh_results = face_mesh.process(img_rgb)
    h, w, _ = img.shape

    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            lms = face_landmarks.landmark

            # Connect landmarks with thin white lines
            for start, end in mp_face_mesh.FACEMESH_TESSELATION:
                x1, y1 = int(lms[start].x * w), int(lms[start].y * h)
                x2, y2 = int(lms[end].x * w), int(lms[end].y * h)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # Draw every 4th landmark as a red dot
            for i, lm in enumerate(lms):
                if i % 4 == 0:
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 2, (0, 0, 255), -1)
    cv2.imshow("Face Detection and Face Mesh", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()