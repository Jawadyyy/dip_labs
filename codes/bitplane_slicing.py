# Bit-plane slicing - shows each bit layer of the image
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', 0)

plt.figure(figsize=(10, 5))
for i in range(8):
    plane = ((img >> i) & 1) * 255         # bit 0 = LSB, bit 7 = MSB
    plt.subplot(2, 4, i + 1)
    plt.imshow(plane, cmap='gray')
    plt.title('Bit ' + str(i))
    plt.axis('off')
plt.show()
