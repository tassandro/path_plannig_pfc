#!/usr/bin/env python3
import sys
import cv2
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "output/jackal_dji_demo_teleport.mp4"
cap = cv2.VideoCapture(path)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"frames={n} fps={fps} duration={n/fps if fps else 0:.1f}s")

idxs = sorted(set([0, n // 8, n // 4, n // 2, 3 * n // 4, max(0, n - 5)]))
frames = []
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        continue
    frames.append((i, frame))

# find the orange marker (BGR ~ (0,165,255)) position in each frame
for i, frame in frames:
    mask = cv2.inRange(frame, (0, 140, 230), (40, 190, 255))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        print(f"frame {i}: marker not found")
        continue
    cx, cy = int(xs.mean()), int(ys.mean())
    print(f"frame {i}: marker pixel ~ ({cx},{cy})")
    out = f"/tmp/frames/frame_{i}.png"
    cv2.imwrite(out, frame)
    print(f"  saved {out}")
cap.release()
