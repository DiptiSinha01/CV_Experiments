import cv2
import numpy as np
from pathlib import Path

# ----------------------------------------------------------
# Load Image
# ----------------------------------------------------------
image_path = Path(__file__).parent / "img.jpeg"

img = cv2.imread(str(image_path))

if img is None:
    print("Error: Could not load img.jpeg")
    exit()

original = img.copy()

# ----------------------------------------------------------
# Source Points (Top Left, Top Right, Bottom Left, Bottom Right)
# Coordinates adjusted for your image
# ----------------------------------------------------------
pts1 = np.float32([
    [118, 56],    # Top Left
    [368, 56],    # Top Right
    [33, 472],    # Bottom Left
    [433, 472]    # Bottom Right
])

# ----------------------------------------------------------
# Destination Points
# ----------------------------------------------------------
width = 600
height = 800

pts2 = np.float32([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
])

# ----------------------------------------------------------
# Perspective Transformation
# ----------------------------------------------------------
matrix = cv2.getPerspectiveTransform(pts1, pts2)

warped = cv2.warpPerspective(
    img,
    matrix,
    (width, height),
    flags=cv2.INTER_CUBIC
)

# ----------------------------------------------------------
# Add Title to Warped Image
# ----------------------------------------------------------
cv2.putText(
    warped,
    "Warped Perspective",
    (10, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

# ----------------------------------------------------------
# Resize Images for Side-by-Side Display
# ----------------------------------------------------------
display_height = 700

original_display = cv2.resize(
    original,
    (
        int(original.shape[1] * display_height / original.shape[0]),
        display_height
    )
)

warped_display = cv2.resize(
    warped,
    (
        original_display.shape[1],
        display_height
    )
)

# ----------------------------------------------------------
# Stack Images
# ----------------------------------------------------------
stacked = np.hstack((original_display, warped_display))

# ----------------------------------------------------------
# Save Output
# ----------------------------------------------------------
output_path = Path(__file__).parent / "WarpPerspective_Output.jpg"
cv2.imwrite(str(output_path), stacked)

print("Output saved as:", output_path)

# ----------------------------------------------------------
# Display
# ----------------------------------------------------------
cv2.imshow("Original Image", original)
cv2.imshow("Warped Perspective", warped)
cv2.imshow("Stacked Output", stacked)

cv2.waitKey(0)
cv2.destroyAllWindows()