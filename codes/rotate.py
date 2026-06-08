# Rotate image by 45 degrees
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
h, w = img.shape
M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1)   # center, angle, scale
out = cv2.warpAffine(img, M, (w, h))

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Rotated 45')
plt.show()
