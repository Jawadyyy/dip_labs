# K-Means clustering segmentation - groups pixels by color
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')                 # color image

Z = img.reshape((-1, 3)).astype(np.float32)   # list of pixels
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3                                         # number of clusters/colors

_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
out = center[label.flatten()].reshape(img.shape)

plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); plt.title('K-Means K=3')
plt.show()
