
import cv2 

import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("img1.jpg")

kernel_5x5 = np.ones((5, 5), np.float32) / 25

# Apply the 5x5 averaging filter to the image
smoothed_img = cv2.filter2D(img1, -1, kernel_5x5)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(smoothed_img, cv2.COLOR_BGR2RGB))
plt.title("5x5 Averaging Filter")
plt.axis("off")

plt.show()

