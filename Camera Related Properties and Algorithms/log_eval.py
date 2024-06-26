import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

image_dir = "images"

for filename in os.listdir(image_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        imgpath = os.path.join(image_dir, filename)

        img = cv2.imread(imgpath)

        if img is not None:
            c = 255 / np.log(1 + np.max(img))
            log_image = c * (np.log(img + 1))
            log_image = np.array(log_image, dtype=np.uint8)

            plt.figure()
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.subplot(122)
            plt.imshow(log_image, cmap='gray')
            plt.title("Log-Transformed Image")
            plt.show()
