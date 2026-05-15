import cv2
import matplotlib.pyplot as plt

img3 = cv2.imread("img3.jpg", cv2.IMREAD_GRAYSCALE)

if img3 is None:
    print("img3.jpg not found. Check file name and folder.")
    exit()

sobel_x = cv2.Sobel(img3, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img3, cv2.CV_64F, 0, 1, ksize=3)

sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img3, cmap="gray")
plt.title("Original img3")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sobel_x_abs, cmap="gray")
plt.title("Sobel X-direction")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sobel_y_abs, cmap="gray")
plt.title("Sobel Y-direction")
plt.axis("off")

plt.show()