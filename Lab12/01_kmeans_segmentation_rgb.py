"""
PART 1: Segmentation as Clustering (K-Means)
Objective: Segment image pixels into K color clusters using RGB features.
"""
import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data

image = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = img.reshape((-1,3)).astype(np.float32)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
k=4
_,labels,centers=cv2.kmeans(pixel_values,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
centers=np.uint8(centers)
segmented=centers[labels.flatten()].reshape(img.shape)

plt.imshow(segmented); plt.title("K-Means RGB Segmentation"); plt.axis("off"); plt.show()
