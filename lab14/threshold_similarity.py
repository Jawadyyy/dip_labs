import cv2
import numpy as np

IMAGE_PATH = "image.jpg"

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

kernel = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

def similarity(img1, img2):

    img1 = cv2.resize(img1, (100,100))
    img2 = cv2.resize(img2, (100,100))

    matching_pixels = np.sum(img1 == img2)
    total_pixels = img1.size

    return matching_pixels / total_pixels

score = similarity(binary, opening)

print(f"Similarity Score: {score:.4f}")

if score > 0.90:
    print("Images are highly similar")
elif score > 0.70:
    print("Images are moderately similar")
else:
    print("Images are different")