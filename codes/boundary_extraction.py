# Boundary extraction - original minus eroded = boundary
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(binary, kernel, iterations=1)
out = binary - erosion

plt.subplot(1, 2, 1); plt.imshow(binary, cmap='gray'); plt.title('Binary')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Boundary')
plt.show()
