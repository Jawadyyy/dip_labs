# Negative - inverts pixel values (255 - pixel)
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
out = 255 - img

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Negative')
plt.show()
