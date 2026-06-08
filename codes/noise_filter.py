import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("lena.png")

# Convert to RGB and Gray
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# ADD GAUSSIAN NOISE
# -----------------------------

# Generate random noise
mean = 0
std = 25

noise = np.random.normal(mean, std, img_gray.shape)

# Add noise to image
noisy_img = img_gray + noise

# Keep values between 0 and 255
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# -----------------------------
# APPLY GAUSSIAN FILTER
# -----------------------------

gaussian_filtered = cv2.GaussianBlur(noisy_img, (7,7), 0)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Original RGB")

plt.subplot(1,4,2)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Gray Image")

plt.subplot(1,4,3)
plt.imshow(noisy_img, cmap="gray")
plt.axis("off")
plt.title("Noisy Image")

plt.subplot(1,4,4)
plt.imshow(gaussian_filtered, cmap="gray")
plt.axis("off")
plt.title("Gaussian Filtered")

plt.show()