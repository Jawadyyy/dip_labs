# Add salt & pepper noise (random black and white dots)
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
out = img.copy()
prob = 0.02
rnd = np.random.rand(*img.shape)
out[rnd < prob] = 0           # pepper (black)
out[rnd > 1 - prob] = 255     # salt (white)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Salt & Pepper')
plt.show()
