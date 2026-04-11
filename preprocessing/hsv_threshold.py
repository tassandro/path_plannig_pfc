from __future__ import annotations

import cv2
import numpy as np

from data_ingestion.interfaces import ImageIngestedData

from ._mask_cleanup import cleanup_mask
from .entities import SegmentationMask, SegmentedImageData
from .interfaces import PreProcessor


class HSVThresholdPreProcessor(PreProcessor):
    """Segment walkable terrain by colour ranges in HSV space.

    Default ranges target green vegetation and light-grey surfaces
    (roads / concrete) commonly seen in aerial drone imagery.

    Parameters
    ----------
    walkable_hsv_ranges :
        List of ``(lower_hsv, upper_hsv)`` tuples.  A pixel is
        considered walkable if it falls inside *any* of the ranges.
    morph_kernel_size :
        Size of the square kernel used for morphological open/close
        cleanup (removes small holes and speckles).
    """

    def __init__(
        self,
        walkable_hsv_ranges: (
            list[tuple[tuple[int, int, int], tuple[int, int, int]]] | None
        ) = None,
        morph_kernel_size: int = 5,
        min_area: int = 1000,
    ):
        if walkable_hsv_ranges is None:
            self._ranges = [
                ((25, 30, 30), (95, 255, 255)),   # green vegetation
                ((0, 0, 120), (180, 50, 255)),     # light grey (roads / concrete)
            ]
        else:
            self._ranges = walkable_hsv_ranges
        self._kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        self._min_area = min_area

    def name(self) -> str:
        return "hsv_threshold"

    def process(self, data: ImageIngestedData) -> SegmentedImageData:
        pixels = data.get_pixels()
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)

        mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
        for lower, upper in self._ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        mask = cleanup_mask(mask, close_kernel_size=0, open_kernel_size=0,
                            min_area=self._min_area)

        segmentation = SegmentationMask(mask=mask)

        return SegmentedImageData(
            source_path=data.get_source(),
            pixels=pixels,
            segmentation=segmentation,
            resolution_m_per_px=getattr(data, "resolution_m_per_px", None),
            metadata=getattr(data, "metadata", {}),
        )
