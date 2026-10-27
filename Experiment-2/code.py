from pathlib import Path

import cv2
import numpy as np

base_dir = Path(__file__).resolve().parent
image_path = base_dir / "apple.webp"
output_path = base_dir / "gray_image.jpg"

# Read Image
img = cv2.imread(str(image_path))

# Check if image is loaded successfully
if img is None:
    print(f"Error: Image not found at {image_path}")
    raise SystemExit(1)

# -------------------------------
# 1. Display Original Image
# -------------------------------
cv2.imshow("Original Image", img)

# -------------------------------
# 2. Convert to Grayscale
# -------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)

# -------------------------------
# 3. Resize Image
# -------------------------------
resized = cv2.resize(img, (400, 300))
cv2.imshow("Resized", resized)

# -------------------------------
# 4. Rotate Image (90° Clockwise)
# -------------------------------
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Rotated", rotated)

# -------------------------------
# 5. Save Grayscale Image
# -------------------------------
cv2.imwrite(str(output_path), gray)

# =====================================================
# ADDITIONAL IMAGE PROCESSING OPERATIONS
# =====================================================

# -------------------------------
# 6. Flip Image Horizontally
# -------------------------------
flip_horizontal = cv2.flip(img, 1)
cv2.imshow("Flip Horizontal", flip_horizontal)

# -------------------------------
# 7. Flip Image Vertically
# -------------------------------
flip_vertical = cv2.flip(img, 0)
cv2.imshow("Flip Vertical", flip_vertical)

# -------------------------------
# 8. Blur Image
# -------------------------------
blur = cv2.GaussianBlur(img, (7,7), 0)
cv2.imshow("Blurred Image", blur)

# -------------------------------
# 9. Edge Detection
# -------------------------------
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("Edge Detection", edges)

# -------------------------------
# 10. Binary Thresholding
# -------------------------------
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold", threshold)

# -------------------------------
# 11. Crop Image
# -------------------------------
crop = img[50:300, 100:350]
cv2.imshow("Cropped Image", crop)

# -------------------------------
# 12. Draw Rectangle
# -------------------------------
rectangle = img.copy()
cv2.rectangle(rectangle, (50,50), (250,200), (0,255,0), 2)
cv2.imshow("Rectangle", rectangle)

# -------------------------------
# 13. Draw Circle
# -------------------------------
circle = img.copy()
cv2.circle(circle, (250,150), 70, (255,0,0), 3)
cv2.imshow("Circle", circle)

# -------------------------------
# 14. Draw Line
# -------------------------------
line = img.copy()
cv2.line(line, (0,0), (400,300), (0,0,255), 2)
cv2.imshow("Line", line)

# -------------------------------
# 15. Add Text
# -------------------------------
text = img.copy()
cv2.putText(text,
            "Computer Vision",
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2)

cv2.imshow("Text on Image", text)

# -------------------------------
# Wait for key press
# -------------------------------
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()