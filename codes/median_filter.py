# Median filter - best for salt & pepper noise
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
out = cv2.medianBlur(img, 3)              # 3 = kernel size (odd)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Median Filter')
plt.show()
