import cv2
import numpy as np
import os

# Read input image from the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "image.png")
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

# Create output copy and move everything upward
output = image.copy()
output = np.pad(output, ((60, 0), (0, 0), (0, 0)), mode='constant', constant_values=255)

# Convert to grayscale
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Optional debug
SHOW_DEBUG = True

# Detect circles using Hough as a fallback (helps when polygon approx gives many vertices)
hough_circles = []
detected = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=40,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=0
)
if detected is not None:
    detected = np.uint16(np.around(detected))
    for c in detected[0, :]:
        hough_circles.append((int(c[0]), int(c[1]), int(c[2])))

# Threshold shapes against the white background and close small gaps
_, mask = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours including nested shapes so bottom-row inner contours can be detected
contours, hierarchy = cv2.findContours(
    mask,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

shape_count = 0
contour_info = []
image_height = image.shape[0]
bottom_threshold = int(image_height * 0.65)
for index, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 1200:
        continue

    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    x, y, w, h = cv2.boundingRect(approx)
    parent = -1
    if hierarchy is not None and len(hierarchy) > 0:
        parent = int(hierarchy[0][index][3])

    contour_info.append({
        'index': index,
        'cnt': cnt,
        'approx': approx,
        'area': area,
        'perimeter': perimeter,
        'vertices': len(approx),
        'convex': cv2.isContourConvex(approx),
        'parent': parent,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'bottom_row': y + h >= bottom_threshold
    })

def is_circle_shape(info):
    area = info['area']
    perimeter = info['perimeter']
    convex = info['convex']
    x, y, w, h = cv2.boundingRect(info['approx'])
    aspect_ratio = float(w) / h if h > 0 else 0
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    return circularity > 0.85 and convex and 0.85 <= aspect_ratio <= 1.15


def classify_shape(info):
    vertices = info['vertices']
    area = info['area']
    perimeter = info['perimeter']
    convex = info['convex']

    # compute circularity for rounded shapes
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    x, y, w, h = cv2.boundingRect(info['approx'])
    aspect_ratio = float(w) / h if h > 0 else 0

    # Strong circle check should happen before polygon checks so the round shape is not missed.
    if is_circle_shape(info):
        return 'Circle'

    # polygon-first: exact vertex-based recognition
    if vertices == 3:
        return 'Triangle'
    if vertices == 4:
        return 'Square' if 0.95 <= aspect_ratio <= 1.05 else 'Rectangle'
    if vertices == 5:
        return 'Pentagon'
    if vertices == 6:
        return 'Hexagon'

    # many-vertex shapes that are somewhat round may be circles
    if vertices >= 8:
        if not convex or vertices == 10:
            return 'Star'
        if circularity > 0.70:
            return 'Circle'
        return 'Unknown'

    return 'Unknown'

outer_contours = [info for info in contour_info if info['parent'] == -1]
rectangle_boxes = []
for info in outer_contours:
    x, y, w, h = info['x'], info['y'], info['w'], info['h']
    shape = classify_shape(info)
    # compute circularity for possible Hough override
    perimeter = info['perimeter']
    area = info['area']
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    if SHOW_DEBUG:
        print(f"Contour {info['index']}: verts={info['vertices']} circ={circularity:.3f} area={area:.1f}")

    # Hough-based override: only when Hough circle overlaps, area matches closely,
    # and contour is already smooth (many vertices) or very round
    if hough_circles:
        cx = x + w / 2
        cy = y + h / 2
        for (hc_x, hc_y, hc_r) in hough_circles:
            dist = np.hypot(cx - hc_x, cy - hc_y)
            if dist < hc_r * 0.6:
                circle_area = np.pi * (hc_r ** 2)
                area_ratio_diff = abs(area - circle_area) / max(circle_area, 1)
                # require either many vertices (smooth contour) OR very high circularity
                if ((info['vertices'] >= 8 and circularity > 0.6) or circularity > 0.82) and area_ratio_diff < 0.25 and info['convex']:
                    shape = 'Circle'
                    if SHOW_DEBUG:
                        print(f"  Hough match: r={hc_r} area_diff={area_ratio_diff:.2f}")
                    break
    shape_count += 1

    text = shape
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    if shape == 'Circle':
        text_x = x + w + 20
        text_y = y + h // 2 + 10
    else:
        text_x = x
        text_y = y - 25 if y - 25 > text_size[1] + 10 else y + h + text_size[1] + 10

    rect_start = (text_x - 5, text_y - text_size[1] - 5)
    rect_end = (text_x + text_size[0] + 5, text_y + 5)
    cv2.rectangle(output, rect_start, rect_end, (255, 255, 255), -1)
    cv2.drawContours(output, [info['approx']], -1, (0, 255, 0), 2)
    cv2.putText(
        output,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    if shape == 'Rectangle':
        rectangle_boxes.append(info)

# Label bottom-row inner shapes only inside bottom-row rectangles/circles
for info in contour_info:
    if info['parent'] == -1:
        continue
    parent = next((p for p in outer_contours if p['index'] == info['parent']), None)
    if parent is None or not parent['bottom_row']:
        continue

    child_shape = classify_shape(info)
    if child_shape not in ('Circle', 'Rectangle'):
        continue
    if info['area'] < parent['area'] * 0.05:
        continue

    x, y, w, h = info['x'], info['y'], info['w'], info['h']
    if x < parent['x'] or y < parent['y'] or x + w > parent['x'] + parent['w'] or y + h > parent['y'] + parent['h']:
        continue

    label = 'Inner Circle' if child_shape == 'Circle' else 'Inner Rectangle'
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = x
    text_y = y - 20 if y - 20 > text_size[1] + 8 else y + h + text_size[1] + 8

    rect_start = (text_x - 4, text_y - text_size[1] - 4)
    rect_end = (text_x + text_size[0] + 4, text_y + 4)
    cv2.rectangle(output, rect_start, rect_end, (255, 255, 255), -1)
    cv2.drawContours(output, [info['approx']], -1, (0, 255, 0), 2)
    cv2.putText(
        output,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )
    shape_count += 1

# Display total number of detected shapes
cv2.putText(
    output,
    f"Total Shapes: {shape_count}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

# Resize display windows to fit the screen when the image is larger than the monitor
max_display_width = 1200
max_display_height = 700
height, width = output.shape[:2]
scale = min(1.0, max_display_width / width, max_display_height / height)
if scale < 1.0:
    display_output = cv2.resize(output, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    display_mask = cv2.resize(mask, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
else:
    display_output = output
    display_mask = mask

cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Detected Shapes and Contours", cv2.WINDOW_NORMAL)
cv2.imshow("Binary Image", display_mask)
cv2.imshow("Detected Shapes and Contours", display_output)

cv2.waitKey(0)
cv2.destroyAllWindows()