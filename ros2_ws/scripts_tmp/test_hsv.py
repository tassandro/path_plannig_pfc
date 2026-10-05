import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
from data_ingestion import ImageData
from preprocessing import PreProcessorFactory

img = cv2.imread('data/rgb/DJI_20240626113157_0064_V.JPG')
data = ImageData(source_path='data/rgb/DJI_20240626113157_0064_V.JPG', pixels=img, resolution_m_per_px=1.0)

pre = PreProcessorFactory.create('hsv_threshold')
seg = pre.process(data)
mask = seg.get_walkable_mask()
print('walkable pct:', mask.mean() * 100)

mask_img = (mask.astype(np.uint8) * 255)
h, w = mask_img.shape
small = cv2.resize(mask_img, (w // 6, h // 6))
cv2.imwrite('ros2_ws/scripts_tmp/dji_mask_thumb.png', small)

overlay = img.copy()
overlay[mask] = (0, 255, 0)
blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
small2 = cv2.resize(blended, (w // 6, h // 6))
cv2.imwrite('ros2_ws/scripts_tmp/dji_overlay_thumb.png', small2)
