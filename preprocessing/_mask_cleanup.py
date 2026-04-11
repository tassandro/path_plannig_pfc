"""Shared morphological cleanup and connected-component filtering for
walkability masks.

Any ``PreProcessor`` can call ``cleanup_mask()`` after producing a raw
boolean mask to remove noise, bridge small gaps, and discard tiny
disconnected islands that are useless for path planning.
"""

from __future__ import annotations

import cv2
import numpy as np


def cleanup_mask(
    mask: np.ndarray,
    *,
    close_kernel_size: int = 15,
    open_kernel_size: int = 5,
    min_area: int = 1000,
) -> np.ndarray:
    """Clean a boolean walkability mask.

    Parameters
    ----------
    mask :
        (H, W) boolean or uint8 array.
    close_kernel_size :
        Kernel for morphological *close* (bridges small gaps between
        walkable regions).
    open_kernel_size :
        Kernel for morphological *open* (removes thin spurs / noise).
    min_area :
        Connected components with fewer pixels than this are removed.
        Set to 0 to skip island filtering.

    Returns
    -------
    np.ndarray
        Cleaned boolean mask with the same shape as *mask*.
    """
    m = mask.astype(np.uint8) if mask.dtype == bool else mask.copy()

    if close_kernel_size > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

    if open_kernel_size > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            m, connectivity=8
        )
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < min_area:
                m[labels == label_id] = 0

    return m.astype(bool)
