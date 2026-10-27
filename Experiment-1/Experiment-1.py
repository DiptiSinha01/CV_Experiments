import sys
from pathlib import Path
import cv2


def main():
    script_dir = Path(__file__).resolve().parent
    video_path = script_dir / "video.mp4"

    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return 1

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print(f"Unable to open video path: {video_path}")
        return 1

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video loaded:", video_path)
    print("Frame Rate (FPS):", fps)
    print("Frame count:", frame_count)

    output_dir = script_dir / "output_frames"
    output_dir.mkdir(exist_ok=True)

    window_name = "Video with FPS"
    max_frames = 10
    saved_frames = 0

    while saved_frames < max_frames:
        ret, frame = video.read()
        if not ret:
            break

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        output_path = output_dir / f"frame_{saved_frames + 1:02d}.png"
        ok = cv2.imwrite(str(output_path), frame)
        if ok:
            print(f"Saved frame: {output_path}")
        else:
            print(f"Failed to save frame: {output_path}")

        try:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
        except cv2.error:
            print("GUI display is unavailable; continuing with saved frames.")

        saved_frames += 1

    video.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())