# Frequency domain HIGH-PASS filter - keeps edges
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)

f = np.fft.fftshift(np.fft.fft2(img))
rows, cols = img.shape
cr, cc = rows // 2, cols // 2

mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (cc, cr), 30, 0, -1)     # block low freq (center)
out = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('High Pass')
plt.show()
