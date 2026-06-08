# Add Gaussian noise to an image
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
noise = np.random.normal(0, 25, img.shape)       # mean 0, std 25
out = np.clip(img + noise, 0, 255).astype(np.uint8)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Gaussian Noise')
plt.show()
