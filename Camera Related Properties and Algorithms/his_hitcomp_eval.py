import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

image_dir = "images"

target_hist = np.array([10] * 256, dtype=np.uint8)

for filename in os.listdir(image_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        imgpath = os.path.join(image_dir, filename)

        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])

            equalized_image = cv2.equalizeHist(img)

            hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

            matched_image = cv2.LUT(equalized_image, target_hist)

            hist_matched = cv2.calcHist([matched_image], [0], None, [256], [0, 256])

            plt.figure(figsize=(15, 4))
            
            plt.subplot(131)
            plt.plot(hist_original)
            plt.title("Original Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")

            plt.subplot(132)
            plt.plot(hist_equalized)
            plt.title("Histogram Equalization")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")

            plt.subplot(133)
            plt.plot(hist_matched)
            plt.title("Histogram Matching")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.show()
