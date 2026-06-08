import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data
def split_region(img,threshold=10):
    if np.std(img)<threshold or min(img.shape)<2:
        return np.ones_like(img)*np.mean(img)
    h,w=img.shape
    return np.vstack((np.hstack((split_region(img[:h//2,:w//2]),split_region(img[:h//2,w//2:]))),
                      np.hstack((split_region(img[h//2:,:w//2]),split_region(img[h//2:,w//2:])))))
gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
small=cv2.resize(gray,(128,128))
out=split_region(small)
plt.imshow(out,cmap='gray'); plt.axis('off'); plt.show()
