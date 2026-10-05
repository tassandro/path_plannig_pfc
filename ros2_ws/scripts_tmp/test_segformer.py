import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
from data_ingestion import ImageData
from preprocessing import PreProcessorFactory

img = cv2.imread('data/rgb/DJI_20240626113157_0064_V_small.jpg')
data = ImageData(source_path='data/rgb/DJI_20240626113157_0064_V_small.jpg', pixels=img, resolution_m_per_px=1.0)

pre = PreProcessorFactory.create('segformer', confidence_threshold=0.3, min_area=200)
seg = pre.process(data)
mask = seg.get_walkable_mask()
print('walkable pct:', mask.mean() * 100)

overlay = img.copy()
overlay[mask] = (0, 255, 0)
blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
cv2.imwrite('ros2_ws/scripts_tmp/dji_segformer_overlay.png', blended)
