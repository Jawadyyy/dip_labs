# Frequency domain LOW-PASS filter - smooths image
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)

f = np.fft.fftshift(np.fft.fft2(img))     # FFT + shift center
rows, cols = img.shape
cr, cc = rows // 2, cols // 2

mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (cc, cr), 30, 1, -1)     # keep low freq (center)
out = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(out, cmap='gray'); plt.title('Low Pass')
plt.show()
