import numpy as np
import cv2
import matplotlib.pyplot as plt


def show(title, im, cmap='gray'):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.imshow(im, cmap=cmap)
    plt.axis('off')
    plt.show()


img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ==================================================================
# 1) FILTERING
# ==================================================================

# ---- Smoothing (low-pass) ----
mean3 = cv2.blur(gray, (3, 3))
mean5 = cv2.blur(gray, (5, 5))                 # bigger kernel = more blur
gauss = cv2.GaussianBlur(gray, (5, 5), 0)
median = cv2.medianBlur(gray, 3)               # best for salt & pepper
show("Mean 3x3", mean3)
show("Gaussian", gauss)
show("Median", median)

# ---- HIGH-PASS / sharpening (high-order = 2nd derivative) ----
# Laplacian = 2nd order derivative -> detects edges/fine detail
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)
show("Laplacian (2nd order high-pass)", laplacian)

# Sharpen = original + Laplacian high-freq
sharp_k = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
sharpen = cv2.filter2D(gray, -1, sharp_k)
show("Sharpened", sharpen)

# Sobel = 1st order derivative (edges in x / y)
sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
show("Sobel (1st order edges)", sobel)

# Unsharp masking (boost high-freq detail)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
show("Unsharp Masking", unsharp)

# ---- Frequency-domain HIGH-PASS ----
f = np.fft.fftshift(np.fft.fft2(gray))
rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), 30, 0, -1)      # block low freq -> high-pass
hp = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))
show("Frequency High-Pass", hp)


# ==================================================================
# 2) CLUSTERING SEGMENTATION
# ==================================================================

# ---- K-Means (color clustering) ----
Z = img.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3                                           # number of clusters
_, label, center = cv2.kmeans(Z, K, None, criteria, 10,
                              cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
seg = center[label.flatten()].reshape(img.shape)
show("K-Means K=3", cv2.cvtColor(seg, cv2.COLOR_BGR2RGB), cmap=None)

# ---- Otsu thresholding (auto global threshold) ----
_, otsu = cv2.threshold(gray, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show("Otsu Segmentation", otsu)


# ==================================================================
# 3) MORPHOLOGICAL OPERATIONS  (work on BINARY image)
# ==================================================================
_, binary = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

erosion = cv2.erode(binary, kernel, iterations=1)    # shrink white
dilation = cv2.dilate(binary, kernel, iterations=1)  # grow white
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # erode->dilate, remove noise
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # dilate->erode, fill holes
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)  # outline
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

show("Binary", binary)
show("Erosion", erosion)
show("Dilation", dilation)
show("Opening (removes noise)", opening)
show("Closing (fills holes)", closing)
show("Gradient (boundary)", gradient)

# Boundary extraction = original - eroded
boundary = binary - erosion
show("Boundary Extraction", boundary)
