import cv2
import numpy as np
import matplotlib.pyplot as plt

imgpath = "eggimg1.jpeg"
img = cv2.imread(imgpath)
original = img.copy()

xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
img = cv2.LUT(img, table)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binarized_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

b1 = cv2.resize(original, (900, 900))
b2 = cv2.resize(img, (900, 900))

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(b1, cv2.COLOR_BGR2RGB))
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(b2, cv2.COLOR_BGR2RGB))
plt.title('Output')

plt.subplot(1, 3, 3)
plt.imshow(binarized_img, cmap='gray')
plt.title('Binarized')

plt.show()
