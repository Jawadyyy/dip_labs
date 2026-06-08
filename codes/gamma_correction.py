# Gamma correction - adjusts brightness (gamma<1 brighter, >1 darker)
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
gamma = 0.5
out = np.array(255 * (img / 255) ** gamma, dtype=np.uint8)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Gamma 0.5')
plt.show()
