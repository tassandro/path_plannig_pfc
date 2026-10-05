"""One-off helper: segment DJI_..._small.jpg and suggest a valid start/end
pixel pair (same connected walkable component) for main.py --start/--end."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_ingestion import ImageData
from preprocessing import PreProcessorFactory

IMAGE_PATH = "data/rgb/DJI_20240626113157_0064_V_small.jpg"

pixels = cv2.imread(IMAGE_PATH)
image_data = ImageData(source_path=IMAGE_PATH, pixels=pixels)

pre = PreProcessorFactory.create("drone_mobilenet_v2")
segmented = pre.process(image_data)
mask = segmented.get_walkable_mask().astype(np.uint8)

n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
# label 0 is background; find the largest non-background component
areas = stats[1:, cv2.CC_STAT_AREA]
biggest = 1 + int(np.argmax(areas))
ys, xs = np.where(labels == biggest)
print(f"Biggest walkable component: label={biggest}, area={areas[biggest - 1]} px")

# Pick the two points that are farthest apart (in image space) within it.
pts = np.stack([xs, ys], axis=1)
# Sample to keep it fast if huge
if len(pts) > 4000:
    idx = np.random.default_rng(0).choice(len(pts), 4000, replace=False)
    sample = pts[idx]
else:
    sample = pts
# approx farthest pair via extremes on x+y and x-y
s = sample[:, 0] + sample[:, 1]
d = sample[:, 0] - sample[:, 1]
cand_idx = [np.argmin(s), np.argmax(s), np.argmin(d), np.argmax(d)]
best_pair = None
best_dist = -1
for i in cand_idx:
    for j in cand_idx:
        p1, p2 = sample[i], sample[j]
        dist = np.hypot(*(p1 - p2))
        if dist > best_dist:
            best_dist = dist
            best_pair = (p1, p2)

p1, p2 = best_pair
print(f"Suggested --start {p1[0]},{p1[1]}  --end {p2[0]},{p2[1]}  (dist={best_dist:.1f}px)")

overlay = pixels.copy()
overlay[labels == biggest] = (0, 255, 0)
blended = cv2.addWeighted(pixels, 0.6, overlay, 0.4, 0)
cv2.circle(blended, tuple(int(v) for v in p1), 6, (0, 0, 255), -1)
cv2.circle(blended, tuple(int(v) for v in p2), 6, (255, 0, 0), -1)
os.makedirs("output", exist_ok=True)
cv2.imwrite("output/waypoint_check.png", blended)
print("Saved output/waypoint_check.png (red=start, blue=end)")
