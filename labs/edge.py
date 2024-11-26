import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded image
image_path = "C:/Users/Lenovo/comp0241_24/dataset/candy2.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply edge detection (Canny method)
edges = cv2.Canny(image, 100, 200)

# Display the original and edge-detected images
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Edge-detected image
plt.subplot(1, 2, 2)
plt.title("Edge-Detected Image")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()