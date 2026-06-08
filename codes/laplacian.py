# Laplacian - 2nd order derivative, detects edges (high-pass)
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
out = cv2.Laplacian(img, cv2.CV_64F)
out = cv2.convertScaleAbs(out)            # convert back to 0-255

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Laplacian')
plt.show()
