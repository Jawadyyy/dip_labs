# Adaptive threshold - threshold changes per region (good for uneven light)
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

plt.subplot(1, 2, 1); 
plt.imshow(img, cmap='gray'); 
plt.title('Original')

plt.subplot(1, 2, 2); 
plt.imshow(out, cmap='gray'); 
plt.title('Adaptive Threshold')

plt.show()
