import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data
gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
noise=np.random.normal(0,25,gray.shape).astype(np.uint8)
noisy=cv2.add(gray,noise)
blur=cv2.GaussianBlur(noisy,(5,5),0)
edges=cv2.Canny(blur,100,200)
plt.imshow(edges,cmap='gray'); plt.axis('off'); plt.show()
