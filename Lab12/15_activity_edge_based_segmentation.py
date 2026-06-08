import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data
gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
edges=cv2.Canny(gray,100,200)
kernel=np.ones((3,3),np.uint8)
dilated=cv2.dilate(edges,kernel,iterations=1)
plt.imshow(dilated,cmap='gray'); plt.axis('off'); plt.show()
