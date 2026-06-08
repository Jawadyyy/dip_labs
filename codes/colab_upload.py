# Colab only - upload image at runtime instead of imread('image.jpg')
from google.colab import files
import cv2
import matplotlib.pyplot as plt

up = files.upload()                    # pick file in popup
name = list(up.keys())[0]
img = cv2.imread(name, 0)

plt.imshow(img, cmap='gray'); plt.title('Uploaded')
plt.show()
