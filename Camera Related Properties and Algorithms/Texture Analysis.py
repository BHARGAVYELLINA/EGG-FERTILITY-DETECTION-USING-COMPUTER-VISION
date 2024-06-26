import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

def extract_texture_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    
    return lbp, image

image_path = "eggimg1.jpeg"
texture_features, original_image = extract_texture_features(image_path)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(texture_features, cmap='gray')
plt.title('Texture Features')
plt.axis('off')

plt.show()
