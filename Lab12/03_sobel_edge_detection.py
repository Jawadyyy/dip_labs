import cv2, matplotlib.pyplot as plt
from skimage import data
gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
sx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
sobel=cv2.magnitude(sx,sy)
plt.imshow(sobel,cmap='gray'); plt.title("Sobel Edge Detection"); plt.axis('off'); plt.show()
