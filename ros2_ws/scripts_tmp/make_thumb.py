import cv2
img = cv2.imread('data/rgb/DJI_20240626113157_0064_V.JPG')
h, w = img.shape[:2]
small = cv2.resize(img, (w // 6, h // 6))
cv2.imwrite('ros2_ws/scripts_tmp/dji_thumb.png', small)
print(small.shape)
