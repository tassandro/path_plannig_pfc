import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
from data_ingestion import ImageData
from preprocessing import PreProcessorFactory

img = cv2.imread('data/rgb/DJI_20240626113157_0064_V.JPG')
h, w = img.shape[:2]
scale = 5
small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
cv2.imwrite('data/rgb/DJI_20240626113157_0064_V_small.jpg', small)
print('resized:', small.shape)

data = ImageData(source_path='data/rgb/DJI_20240626113157_0064_V_small.jpg', pixels=small, resolution_m_per_px=1.0)
pre = PreProcessorFactory.create('hsv_threshold')
seg = pre.process(data)
mask = seg.get_walkable_mask()
print('walkable pct:', mask.mean() * 100)

candidates = [(336, 24), (372, 576), (330, 30), (340, 20), (300, 250)]
for x, y in candidates:
    print((x, y), 'walkable' if mask[y, x] else 'NOT walkable')

overlay = small.copy()
overlay[mask] = (0, 255, 0)
blended = cv2.addWeighted(small, 0.5, overlay, 0.5, 0)
for x, y in candidates:
    cv2.circle(blended, (x, y), 5, (0, 0, 255), -1)
cv2.imwrite('ros2_ws/scripts_tmp/dji_small_overlay.png', blended)
