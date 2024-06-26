import cv2
import numpy as np
import matplotlib.pyplot as plt

def perspective_transform(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_points = np.float32([[0, 0], [width, 0], [width*0.8, height], [width*0.2, height]])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    transformed_image = cv2.warpPerspective(image, M, (width, height))
    
    return transformed_image, image

image_path = "eggimg1.jpeg"
transformed_image, original_image = perspective_transform(image_path)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
plt.title('Perspective Transformed Image')
plt.axis('off')

plt.show()
