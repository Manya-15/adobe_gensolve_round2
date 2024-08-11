import cv2
import numpy as np

# Load the input image
input_image_path = "test5.jpg"
input_image = cv2.imread(input_image_path)

# Convert to grayscale
gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Threshold the image
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank canvas with the same dimensions as the input image
canvas = np.ones_like(input_image) * 255  # White background

# Draw the contours on the canvas
for contour in contours:
    cv2.drawContours(canvas, [contour], -1, (0, 0, 0), -1)

# Fill in the ellipse to complete the shape
cv2.ellipse(canvas, (int(input_image.shape[1]/2), int(input_image.shape[0]/2)), (200, 300), 0, 0, 360, (0, 165, 255), -1)

# Draw the ring (elliptical orbit)
cv2.ellipse(canvas, (int(input_image.shape[1]/2), int(input_image.shape[0]/2)), (250, 120), 30, 0, 360, (255, 255, 0), -1)

# Save the output image
output_image_path = "output_image4.png"
cv2.imwrite(output_image_path, canvas)

# Show the output image (optional)
# cv2.imshow('Output', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(f"Output image saved to {output_image_path}")
