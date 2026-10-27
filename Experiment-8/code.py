import cv2
from math import sqrt
# import os

SCALE_LENGTH_MM = 150   # real distance between the two points you click on the scale image

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_image(name):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(SCRIPT_DIR, name + ext)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
    raise FileNotFoundError(f"{name} image not found in {SCRIPT_DIR}")
def distance(p1, p2):
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def click_points(img, title, labels):
    """Lets you click len(labels) points on the image. Returns the list of points."""
    points = []
    display = img.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display, str(len(points)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"  {labels[len(points) - 1]} -> {(x, y)}")

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_click)
    print(f"Click in this order: {', '.join(labels)}  (ESC to quit)")

    while len(points) < len(labels):
        cv2.imshow(title, display)
        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            raise SystemExit

    # Draw the line between the points
    cv2.line(display, points[0], points[1], (255, 0, 0), 2)
    cv2.imshow(title, display)
    cv2.waitKey(800)
    cv2.destroyWindow(title)
    return points

# 1. Calibrate using the scale image
print("\n--- Calibration from scale image ---")
scale_img = load_image("scale")
scale_pts = click_points(scale_img, "Scale", ["Scale start", "Scale end"])
scale_px = distance(scale_pts[0], scale_pts[1])
pixels_per_mm = scale_px / SCALE_LENGTH_MM
print(f"Scale: {scale_px:.1f} px = {SCALE_LENGTH_MM} mm -> {pixels_per_mm:.2f} px/mm")

# 2. Measure the pen and bottle using the same pixels_per_mm
results = {}
for name in ["pen", "bottle"]:
    print(f"\n--- Measuring {name} ---")
    img = load_image(name)
    pts = click_points(img, name, [f"{name} bottom", f"{name} top"])
    px = distance(pts[0], pts[1])
    mm = px / pixels_per_mm
    results[name] = mm
    print(f"{name}: {px:.1f} px -> {mm:.2f} mm")

# 3. Summary
print("\n=== RESULTS ===")
print(f"Pixels per mm (from scale): {pixels_per_mm:.2f}")
for name, mm in results.items():
    print(f"{name.capitalize():8}: {mm:.2f} mm  ({mm / 10:.2f} cm)")