import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data
img=data.astronaut()
pixels=img.reshape((-1,3)).astype(np.float32)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
dist=[]
for k in range(2,8):
    _,_,centers=cv2.kmeans(pixels,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    dist.append(np.sum((pixels-centers.mean(axis=0))**2))
plt.plot(range(2,8),dist,marker='o')
plt.title("Elbow Method")
plt.xlabel("K"); plt.ylabel("Distortion"); plt.show()
