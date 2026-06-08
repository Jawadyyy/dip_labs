# Histogram - shows distribution of pixel intensities
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lena.png', 0)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Image')
plt.subplot(1, 2, 2); plt.hist(img.ravel(), 256, [0, 256]); plt.title('Histogram')
plt.show()

print('Mean:', img.mean())
print('Std:', img.std())
