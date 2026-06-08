# Canny edge detection - clean thin edges
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lena.png', 0)
out = cv2.Canny(img, 100, 200)                # 100, 200 = thresholds

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Canny Edges')
plt.show()
