# Sobel - 1st order derivative, finds edges in x and y
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)   # x edges
sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)   # y edges
out = cv2.convertScaleAbs(cv2.magnitude(sx, sy))

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Sobel')
plt.show()
