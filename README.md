# Computer Vision Experiments

A collection of Computer Vision laboratory experiments implemented primarily with **Python, OpenCV, MediaPipe, and related tools** as part of the Computer Vision course (BE CSE AI-ML).

## Experiments

| No. | Experiment | Main concepts / tools |
|---:|---|---|
| 1 | Find the frame rate (FPS) of a video and display it on the video | Python, OpenCV, video properties, `cv2.putText()` |
| 2 | Implement basic image-processing functions | OpenCV, NumPy, grayscale conversion, resizing, rotation |
| 3 | Extract a sub-image using perspective transformation | Perspective transform, `cv2.getPerspectiveTransform()`, `cv2.warpPerspective()` |
| 4 | Detect contours and shapes in an image | Grayscale conversion, Gaussian blur, thresholding, contours |
| 5 | Build a real-time hand-tracking system and track hand landmarks | OpenCV, MediaPipe Hands, landmark visualization |
| 6 | Implement a real-time pose-detection system | OpenCV, MediaPipe Pose, pose landmarks |
| 7 | Detect human faces and extract facial-mesh landmarks | OpenCV, MediaPipe Face Mesh / Face Landmarker |
| 8 | Implement camera calibration and find the real length of an input object | Camera calibration, image measurements, focal length |
| 9 | Implement face detection using Haar Cascade Classifier and MediaPipe | Haar Cascade, MediaPipe, face detection |
| 10 | Use the YOLO object-detection algorithm to detect vehicles | YOLO, object detection, bounding boxes, class labels |

> Experiment titles are based on the lab index. The exact implementation and dependencies may vary by experiment folder.

## Repository structure

```text
CV_Experiment/
├── Experiment-1/
├── Experiment-2/
├── Experiment-3/
├── Experiment-4/
├── Experiment-5/
├── Experiment-6/
├── Experiment-7/
├── Experiment-8/
├── Experiment-9/
├── Experiment-10/
├── .gitignore
└── README.md
```

Each `Experiment-N` directory contains the code and supporting files for that experiment. Input images, videos, trained model weights, or other assets may be required by individual scripts; check the relevant folder before running a program.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- MediaPipe for the hand, pose, and facial-landmark experiments
- Any additional package required by a specific experiment (for example, a YOLO implementation or model-specific dependency)

Install common dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the common packages:

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python numpy mediapipe
```

Some experiments may require extra packages or model files. Install those only when specified by the code or the experiment's own instructions.

## Running an experiment

1. Open a terminal in the repository root.
2. Activate the virtual environment.
3. Open the required `Experiment-N` folder.
4. Check the script for its expected input paths, camera index, and required model files.
5. Run the relevant Python script, for example:

   ```bash
   python your_script.py
   ```

Replace `your_script.py` with the actual script filename in that experiment folder. For webcam-based experiments, allow camera access and close the OpenCV preview window using the key specified in the script (often `q` or `Esc`).

## Notes

- Keep experiment code and small, permitted sample assets in version control.
- Virtual environments (`.venv`, `cv_env`, and `cv_env312`) are ignored by Git; recreate them locally rather than committing them.
- Do not commit passwords, API keys, private configuration, or machine-specific paths.
- Large videos and model-weight files can make a repository difficult to clone. The `.gitignore` includes commented-out patterns for these files; enable them only if you intend to exclude those assets from Git.
- If a script expects an input file such as `input.jpg`, `input_video.mp4`, or `shapes.png`, make sure the file is present at the expected path.

## Learning objectives

These experiments provide practice with video processing, basic image transformations, geometric perspective correction, contour and shape analysis, real-time hand and pose tracking, facial landmarks, camera calibration, face detection, and object detection.
