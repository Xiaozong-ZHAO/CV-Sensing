

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random
# Fixed HSV thresholds
lower_bound = np.array([102, 3, 0])  # Lower HSV bounds
upper_bound = np.array([179, 255, 255])  # Upper HSV bounds
# Read image and ground truth
img = cv2.imread('dataset/images/000065.png')
gnd = cv2.imread('dataset/masks/000065.png', cv2.IMREAD_GRAYSCALE)
# Binarize the ground truth
_, gnd_binary = cv2.threshold(gnd, 127, 255, cv2.THRESH_BINARY)
# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Gaussian filtering
blurred = cv2.GaussianBlur(hsv, (5, 5), 0)  # Kernel size (5, 5) can be adjusted
# Generate the mask
mask = cv2.inRange(blurred, lower_bound, upper_bound)
# Generate kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# Erosion and dilation
mask_eroded = cv2.erode(mask, kernel, iterations=1)  # Erosion
mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=1)  # Dilation
# Find contours
contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Set the minimum contour area
min_contour_area = 1000
# Filter out small contours
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]
# Merge all the points of the contours
# and calculate the convex hull for the enclosing ellipse
result_with_shapes = img.copy()
if len(filtered_contours) > 0:
    # Merge all the points of the contours
    all_points = np.vstack(filtered_contours)
    # Draw individual contours
    for contour in filtered_contours:
        cv2.drawContours(result_with_shapes, [contour], -1, (0, 255, 0), 2)  # Green for filtered contours
    # Calculate the convex hull
    hull = cv2.convexHull(all_points)
    # Fit the enclosing ellipse
    if len(hull) >= 5:
        ellipse = cv2.fitEllipse(hull)
        cv2.ellipse(result_with_shapes, ellipse, (0, 255, 255), 2)  # Yellow enclosing ellipse
    # Calculate and draw the enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(all_points)
    cv2.circle(result_with_shapes, (int(x), int(y)), int(radius), (0, 0, 255), 2)  # Red enclosing circle
    cv2.circle(result_with_shapes, (int(x), int(y)), 4, (255, 255, 255), -1)  # White center point
# Generate the ground truth contours for comparison
gt_contours, _ = cv2.findContours(gnd_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw Ground Truth Contours
for gt_contour in gt_contours:
    cv2.drawContours(result_with_shapes, [gt_contour], -1, (255, 0, 0), 2)  # Blue ground truth contours
# Display results
combined_mask = np.hstack((mask, mask_eroded, mask_dilated))  # Display original mask, eroded, and dilated masks
cv2.imshow("Contours, Enclosing Circle, Ellipse, and Ground Truth", result_with_shapes)
# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()

# def nothing(x):
#     pass
# # Load the image
# img = cv2.imread('dataset/images/000099.png')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # Create a resizable window for trackbars
# cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Trackbars", 600, 300)
# # Create trackbars for HSV
# cv2.createTrackbar("Lower-H", "Trackbars", 0, 179, nothing)
# cv2.createTrackbar("Lower-S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("Lower-V", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("Upper-H", "Trackbars", 179, 179, nothing)
# cv2.createTrackbar("Upper-S", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("Upper-V", "Trackbars", 255, 255, nothing)
# # Structuring element for morphological operations
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# while True:
#     # Get current positions of trackbars
#     lower_h = cv2.getTrackbarPos("Lower-H", "Trackbars")
#     lower_s = cv2.getTrackbarPos("Lower-S", "Trackbars")
#     lower_v = cv2.getTrackbarPos("Lower-V", "Trackbars")
#     upper_h = cv2.getTrackbarPos("Upper-H", "Trackbars")
#     upper_s = cv2.getTrackbarPos("Upper-S", "Trackbars")
#     upper_v = cv2.getTrackbarPos("Upper-V", "Trackbars")
#     # Set the HSV bounds based on trackbar positions
#     lower_bound = np.array([lower_h, lower_s, lower_v])
#     upper_bound = np.array([upper_h, upper_s, upper_v])
#     # Apply Gaussian blur to smooth the image
#     blurred = cv2.GaussianBlur(hsv, (5, 5), 0)  # Kernel size (5, 5) can be adjusted
#     # Create a mask based on current HSV bounds
#     mask = cv2.inRange(blurred, lower_bound, upper_bound)
#     # Apply morphological operations
#     mask_morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
#     mask_morphed = cv2.morphologyEx(mask_morphed, cv2.MORPH_OPEN, kernel)  # Remove noise
#     # Find contours on the morphed mask
#     contours, _ = cv2.findContours(mask_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     result_with_contours = img.copy()
#     # Draw contours and fit ellipses
#     for contour in contours:
#         if len(contour) >= 50:  # Ensure sufficient points for fitting
#             # Fit ellipse
#             ellipse = cv2.fitEllipse(contour)
#             cv2.ellipse(result_with_contours, ellipse, (0, 0, 255), 2, cv2.LINE_AA)  # Draw the ellipse
#             x, y = ellipse[0]
#             cv2.circle(result_with_contours, (int(x), int(y)), 4, (255, 0, 0), -1, 8, 0)  # Mark the center
            
#             # Draw the contour
#             cv2.drawContours(result_with_contours, [contour], -1, (0, 255, 0), 2)  # Green for contour
#     # Display results
#     combined_mask = np.hstack((mask, mask_morphed))
#     cv2.imshow("Masks (Original | Morphed)", combined_mask)
#     cv2.imshow("Contours and Ellipses", result_with_contours)
#     # Break the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
