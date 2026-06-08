# Global thresholding - pixel > 127 becomes white, else black
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
_, out = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Threshold')
plt.show()
