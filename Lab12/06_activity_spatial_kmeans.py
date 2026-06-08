import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data
image=data.astronaut()
h,w,_=image.shape
X=[]
for i in range(h):
    for j in range(w):
        X.append([image[i,j][0],image[i,j][1],image[i,j][2],i,j])
X=np.array(X,np.float32)
_,labels,centers=cv2.kmeans(X,4,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2),10,cv2.KMEANS_RANDOM_CENTERS)
seg=centers[:,:3][labels.flatten()].reshape(image.shape).astype(np.uint8)
plt.imshow(seg); plt.title("Spatial K-Means"); plt.axis('off'); plt.show()
