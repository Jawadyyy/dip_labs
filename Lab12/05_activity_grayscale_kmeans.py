import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data
gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
pixels=gray.reshape(-1,1).astype(np.float32)
_,labels,centers=cv2.kmeans(pixels,3,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2),10,cv2.KMEANS_RANDOM_CENTERS)
seg=centers[labels.flatten()].reshape(gray.shape)
plt.imshow(seg,cmap='gray'); plt.title("Grayscale Clustering"); plt.axis('off'); plt.show()
