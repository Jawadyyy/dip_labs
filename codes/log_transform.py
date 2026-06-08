# Log transform - brightens dark areas
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)
c = 255 / np.log(1 + img.max())
out = (c * np.log(1 + img.astype(float))).astype(np.uint8)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Log Transform')
plt.show()
