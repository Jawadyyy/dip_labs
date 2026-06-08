# Unsharp masking - sharpen by adding (original - blurred)
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
blur = cv2.GaussianBlur(img, (5, 5), 0)
out = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Unsharp Mask')
plt.show()
