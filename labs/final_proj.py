import cv2
import numpy as np
import random

def nothing(x):
    pass

# Load the image
img = cv2.imread('dataset/images/000099.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create a resizable window
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 600, 300)  # Resize the window

# Create trackbars for HSV
cv2.createTrackbar("Lower-H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Lower-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Lower-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Upper-H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("Upper-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "Trackbars", 255, 255, nothing)

while True:
    # Get current positions of trackbars
    lower_h = cv2.getTrackbarPos("Lower-H", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower-S", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower-V", "Trackbars")
    upper_h = cv2.getTrackbarPos("Upper-H", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper-S", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper-V", "Trackbars")

    # Set the HSV bounds based on trackbar positions
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # Create a mask based on current HSV bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Convert the mask to grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw each contour with a different color
    contour_image = np.zeros_like(img)
    for contour in contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(contour_image, [contour], -1, color, thickness=2)

    # Detect circles using Hough Circle Transform on edges
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,                # Inverse ratio of accumulator resolution
        minDist=150,          # Minimum distance between detected circles
        param1=50,           # Upper threshold for Canny edge detection
        param2=60,           # Accumulator threshold for circle detection
        minRadius=100,        # Minimum circle radius
        maxRadius=300        # Maximum circle radius
    )

    # Draw detected circles on the original image
    hough_circle_image = edges.copy()
    hough_circle_image = cv2.cvtColor(hough_circle_image, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(hough_circle_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(hough_circle_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Display the mask, result, edges, contours, and Hough Circles
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Result", result)
    # cv2.imshow("Edges", edges)
    cv2.imshow("Contours", contour_image)
    cv2.imshow("Hough Circles", hough_circle_image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
