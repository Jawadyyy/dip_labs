# FFT magnitude spectrum - shows frequencies in image
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)

f = np.fft.fftshift(np.fft.fft2(img))
spectrum = 20 * np.log(np.abs(f) + 1)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(spectrum, cmap='gray'); plt.title('FFT Spectrum')
plt.show()
