import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
from data_ingestion import ImageData
from preprocessing import PreProcessorFactory

img = cv2.imread('data/rgb/DJI_20240626113157_0064_V_small.jpg')
data = ImageData(source_path='data/rgb/DJI_20240626113157_0064_V_small.jpg', pixels=img, resolution_m_per_px=1.0)

pre = PreProcessorFactory.create('drone_mobilenet_v2', min_area=200)
seg = pre.process(data)
mask = seg.get_walkable_mask().astype(np.uint8)

candidates = [(320, 40), (500, 550), (330, 60), (480, 520), (350, 300)]
for x, y in candidates:
    print((x, y), 'walkable' if mask[y, x] else 'NOT walkable')

num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
for x, y in candidates:
    lbl = labels[y, x]
    print((x, y), 'component', lbl, 'area', stats[lbl, cv2.CC_STAT_AREA] if lbl > 0 else 0)

overlay = img.copy()
overlay[mask.astype(bool)] = (0, 255, 0)
blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
for x, y in candidates:
    cv2.circle(blended, (x, y), 6, (0, 0, 255), -1)
cv2.imwrite('ros2_ws/scripts_tmp/dji_drone_candidates.png', blended)
