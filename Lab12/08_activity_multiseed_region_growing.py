# Multi-seed Region Growing
import cv2, numpy as np, matplotlib.pyplot as plt
from skimage import data
def rg(img,seed,t=10):
    h,w=img.shape; s=np.zeros((h,w),np.uint8); sv=img[seed]; st=[seed]
    while st:
        x,y=st.pop()
        if s[x,y]==0:
            s[x,y]=255
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx,ny=x+dx,y+dy
                if 0<=nx<h and 0<=ny<w and abs(int(img[nx,ny])-int(sv))<t: st.append((nx,ny))
    return s
gray=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2GRAY)
res=np.zeros_like(gray)
for seed in [(50,50),(150,150),(200,50)]: res+=rg(gray,seed)
plt.imshow(res,cmap='gray'); plt.axis('off'); plt.show()
