import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data

def region_growing(img, seed, threshold=15):
    h,w=img.shape
    seg=np.zeros((h,w),np.uint8)
    seed_val=img[seed]
    stack=[seed]
    while stack:
        x,y=stack.pop()
        if seg[x,y]==0:
            seg[x,y]=255
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx,ny=x+dx,y+dy
                if 0<=nx<h and 0<=ny<w and abs(int(img[nx,ny])-int(seed_val))<threshold:
                    stack.append((nx,ny))
    return seg

gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
result=region_growing(gray,(100,100))
plt.imshow(result,cmap='gray'); plt.title("Region Growing"); plt.axis('off'); plt.show()
