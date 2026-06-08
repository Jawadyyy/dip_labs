import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "image.jpg"

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

kernel = np.ones((5,5), np.uint8)

gradient = cv2.morphologyEx(
    binary,
    cv2.MORPH_GRADIENT,
    kernel
)

plt.imshow(gradient, cmap="gray")
plt.title("Morphological Gradient")
plt.axis("off")
plt.show()