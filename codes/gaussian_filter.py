# Gaussian filter - smooth blur, keeps edges better than mean
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
out = cv2.GaussianBlur(img, (5, 5), 0)    # (5,5) kernel, 0 = auto sigma

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Gaussian Filter')
plt.show()
