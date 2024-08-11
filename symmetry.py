import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the image
image = cv2.imread("test2.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to gray

# Threshold the gray image to get a binary image
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# List of colors for different shapes
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Image dimensions
height, width = image.shape[:2]

# Function to check if a contour is on the edge of the image
def is_contour_on_edge(contour, width, height):
    for point in contour:
        x, y = point[0]
        if x <= 1 or y <= 1 or x >= width-2 or y >= height-2:
            return True
    return False

# Function to draw symmetry lines with line numbers
def draw_symmetry_lines(image, x, y, w, h, color, approx, line_num):
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Vertical symmetry line
    cv2.line(image, (x + w // 2, y), (x + w // 2, y + h), color, 2)
    cv2.putText(image, f'L{line_num}', (x + w // 2 + 5, y + h // 2 - 5), font, 0.5, color, 2)
    line_num += 1

    # Horizontal symmetry line
    cv2.line(image, (x, y + h // 2), (x + w, y + h // 2), color, 2)
    cv2.putText(image, f'L{line_num}', (x + w // 2 + 5, y + h // 2 + 15), font, 0.5, color, 2)
    line_num += 1

    # Diagonal symmetry lines (only for squares and circles)
    if len(approx) == 4 or len(approx) > 6:
        cv2.line(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f'L{line_num}', (x + w // 2 - 20, y + h // 2 - 20), font, 0.5, color, 2)
        line_num += 1
        cv2.line(image, (x, y + h), (x + w, y), color, 2)
        cv2.putText(image, f'L{line_num}', (x + w // 2 - 20, y + h // 2 + 20), font, 0.5, color, 2)
        line_num += 1

    return line_num

# Iterate through each contour
line_num = 1
for i, contour in enumerate(contours):
    if is_contour_on_edge(contour, width, height):
        continue  # Skip this contour if it's on the edge

    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Select a color for each detected shape
    color = colors[i % len(colors)]  # Cycle through the color list

    # Drawing the contours onto the image
    cv2.drawContours(image, [contour], 0, color, 5)

    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + (w / 3))
    y_mid = int(y + (h / 1.5))
    coords = (x_mid, y_mid)

    font = cv2.FONT_HERSHEY_DUPLEX

    # Identifying the shape based on the number of vertices
    if len(approx) == 3:
        cv2.putText(image, "", coords, font, 0.5, color, 2)
        # Vertical symmetry line only for triangles
        line_num = draw_symmetry_lines(image, x, y, w, h, color, approx, line_num)
    elif len(approx) == 4:
        cv2.putText(image, "", coords, font, 0.5, color, 2)
        # Draw all symmetry lines for rectangles
        line_num = draw_symmetry_lines(image, x, y, w, h, color, approx, line_num)
    elif len(approx) == 5:
        cv2.putText(image, "", coords, font, 0.5, color, 2)
        # Vertical symmetry line for pentagons
        line_num = draw_symmetry_lines(image, x, y, w, h, color, approx, line_num)
    elif len(approx) == 6:
        cv2.putText(image, "", coords, font, 0.5, color, 2)
        # Draw all symmetry lines for hexagons
        line_num = draw_symmetry_lines(image, x, y, w, h, color, approx, line_num)
    else:
        cv2.putText(image, "", coords, font, 0.5, color, 2)
        # Draw all symmetry lines for circles
        line_num = draw_symmetry_lines(image, x, y, w, h, color, approx, line_num)

# Convert BGR to RGB for displaying with Matplotlib/Seaborn
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Use Seaborn to create a plot
sns.set_theme(style="white")
plt.figure(figsize=(8, 8))
plt.imshow(image_rgb)
plt.axis('off')  # Hide the axes
plt.show()
