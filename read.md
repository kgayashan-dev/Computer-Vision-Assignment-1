========================================
COMPUTER VISION ASSIGNMENT - FULL GUIDE
========================================

1. OPEN PROJECT FOLDER
----------------------

cd ~/Documents/GitHub/COMPUTER\ VISION/Computer-Vision-Assignment-1

Check files:

ls

Required files:
- img1.jpg
- img2.tif
- img3.jpg


2. CREATE VIRTUAL ENVIRONMENT
-----------------------------

python3 -m venv .venv


3. ACTIVATE VIRTUAL ENVIRONMENT
-------------------------------

source .venv/bin/activate

You should see:

(.venv)


4. INSTALL REQUIRED PACKAGES
----------------------------

python3 -m ensurepip --upgrade

python3 -m pip install --upgrade pip

python3 -m pip install opencv-python numpy matplotlib


5. VERIFY INSTALLATION
----------------------

python3 -m pip list

You should see:
- opencv-python
- numpy
- matplotlib


6. CREATE PYTHON FILE
---------------------

touch worksheet1.py


7. PASTE THIS CODE INTO worksheet1.py
-------------------------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt


# =========================
# PART 2 - img1.jpg
# 5x5 Averaging Filter
# =========================

img1 = cv2.imread("img1.jpg")

if img1 is None:
    print("img1.jpg not found")
else:
    kernel_5x5 = np.ones((5, 5), np.float32) / 25
    smoothed_img1 = cv2.filter2D(img1, -1, kernel_5x5)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Original img1")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(smoothed_img1, cv2.COLOR_BGR2RGB))
    plt.title("5x5 Averaging Filter")
    plt.axis("off")

    plt.savefig("part2_img1_averaging.png")
    plt.show()


# =========================
# PART 3 - img2.tif
# Averaging, Gaussian, Median
# =========================

img2 = cv2.imread("img2.tif", cv2.IMREAD_GRAYSCALE)

if img2 is None:
    print("img2.tif not found")
else:
    avg_3x3 = cv2.blur(img2, (3, 3))
    gaussian_5x5 = cv2.GaussianBlur(img2, (5, 5), 0)
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

    plt.savefig("part3_img2_filters.png")
    plt.show()


# =========================
# PART 3 - img3.jpg
# Sobel X and Y
# =========================

img3 = cv2.imread("img3.jpg", cv2.IMREAD_GRAYSCALE)

if img3 is None:
    print("img3.jpg not found")
else:
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

    plt.savefig("part3_img3_sobel.png")
    plt.show()


8. RUN THE PROGRAM
------------------

python3 worksheet1.py


9. OUTPUT FILES GENERATED
-------------------------

- part2_img1_averaging.png
- part3_img2_filters.png
- part3_img3_sobel.png


10. IF MATPLOTLIB SECURITY ERROR OCCURS
---------------------------------------

python3 -m pip uninstall matplotlib

python3 -m pip install --no-cache-dir matplotlib


11. IF STILL NOT WORKING
------------------------

deactivate

rm -rf .venv

python3 -m venv .venv

source .venv/bin/activate

python3 -m ensurepip --upgrade

python3 -m pip install opencv-python numpy matplotlib


12. FINAL PDF STRUCTURE
-----------------------

PART 1
- Theory answers

PART 2
- Code
- Output screenshot
- Explanation

PART 3
- Code
- Output screenshots
- Comparison
- Sobel explanation


13. FINAL PDF NAME
------------------

worksheet1_<registration_number>.pdf

Example:
worksheet1_E1234567.pdf