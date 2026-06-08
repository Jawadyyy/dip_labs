# Convert color image to grayscale
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')                  # color
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Color')
plt.subplot(1, 2, 2); plt.imshow(gray, cmap='gray'); plt.title('Grayscale')
plt.show()
