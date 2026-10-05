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
mask = seg.get_walkable_mask().astype(np.uint8)

num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
# sort components by area descending (skip label 0 = background)
order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
print("Top components (label, area, x, y, w, h):")
for lbl in order[:8]:
    x, y, w, h, area = stats[lbl]
    print(lbl, area, x, y, w, h)

# candidate street component likely spans nearly full image height
h_img, w_img = mask.shape
for lbl in order[:8]:
    x, y, w, h, area = stats[lbl]
    if h > 0.8 * h_img:
        print("Street-like component:", lbl, "bbox:", x, y, w, h, "area:", area)
        ys, xs = np.where(labels == lbl)
        top_idx = np.argmin(ys)
        bot_idx = np.argmax(ys)
        print("  top point:", xs[top_idx], ys[top_idx])
        print("  bottom point:", xs[bot_idx], ys[bot_idx])
