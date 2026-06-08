# Mean (average) filter - blurs / smooths image
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)          # 0 = grayscale
out = cv2.blur(img, (5, 5))               # 5x5 average filter

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Mean Filter')
plt.show()
