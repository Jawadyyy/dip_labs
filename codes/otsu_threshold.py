# Otsu threshold - automatically picks best threshold value
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
_, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Otsu Threshold')
plt.show()
