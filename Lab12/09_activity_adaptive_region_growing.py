import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data
# Adaptive threshold based on image std deviation
gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
print("Adaptive threshold =", np.std(gray))
