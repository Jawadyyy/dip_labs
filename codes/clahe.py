# CLAHE - adaptive (local) histogram equalization
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
out = clahe.apply(img)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('CLAHE')
plt.show()
