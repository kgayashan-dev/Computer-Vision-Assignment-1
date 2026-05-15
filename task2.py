import cv2
import numpy as np
from matplotlib import pyplot as plt

img2 = cv2.imread("img2.tif", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("img3.jpg", cv2.IMREAD_GRAYSCALE)

# a) 3x3 averaging filter
avg_3x3 = cv2.blur(img2, (3, 3))

# b) Gaussian filtering 5x5
gaussian_5x5 = cv2.GaussianBlur(img2, (5, 5), 0)

# c) Median filtering kernel size 5
median_5 = cv2.medianBlur(img2, 5)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img2, cmap="gray")
plt.title("Original img2")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(avg_3x3, cmap="gray")
plt.title("3x3 Averaging Filter")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(gaussian_5x5, cmap="gray")

plt.title("5x5 Gaussian Filter")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(median_5, cmap="gray")
plt.title("Median Filter k=5")
plt.axis("off")


plt.show()