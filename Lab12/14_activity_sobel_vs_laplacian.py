import cv2, matplotlib.pyplot as plt
from skimage import data
gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
lap=cv2.Laplacian(gray,cv2.CV_64F)
plt.imshow(lap,cmap='gray'); plt.axis('off'); plt.show()
