import cv2, matplotlib.pyplot as plt
from skimage import data
gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
edges=cv2.Canny(gray,100,200)
plt.imshow(edges,cmap='gray'); plt.title("Canny Edge Detection"); plt.axis('off'); plt.show()
