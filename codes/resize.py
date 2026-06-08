# Resize image (and interpolation example)
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
out = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Resized 200x200')
plt.show()
