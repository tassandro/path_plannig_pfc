from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data_ingestion.interfaces import ImageIngestedData


@dataclass
class SegmentationMask:
    """Per-pixel classification produced by a preprocessor.

    Attributes
    ----------
    mask : np.ndarray
        (H, W) boolean array — ``True`` means *walkable*.
    confidence : np.ndarray | None
        (H, W) float [0, 1] softmax confidence for the predicted class.
    class_map : dict[int, str]
        Maps integer class IDs to human-readable labels.
    """

    mask: np.ndarray
    confidence: np.ndarray | None = None
    class_map: dict[int, str] = field(default_factory=dict)


@dataclass
class SegmentedImageData(ImageIngestedData):
    """An image enriched with a walkability segmentation mask.

    Extends ``ImageIngestedData`` so it can be consumed by any graph
    parser typed against that interface, while also exposing the
    segmentation for parsers that know about it.
    """

    source_path: str
    pixels: np.ndarray
    segmentation: SegmentationMask
    resolution_m_per_px: float | None = None
    metadata: dict = field(default_factory=dict)

    # --- IngestedData ---------------------------------------------------

    def get_source(self) -> str:
        return self.source_path

    def get_dimensions(self) -> tuple[int, ...]:
        return self.pixels.shape

    def get_raw_data(self) -> np.ndarray:
        return self.pixels

    def validate(self) -> bool:
        if self.pixels.ndim != 3 or self.pixels.shape[2] not in (1, 3, 4):
            return False
        h, w = self.pixels.shape[:2]
        if self.segmentation.mask.shape != (h, w):
            return False
        return True

    # --- ImageIngestedData ----------------------------------------------

    def get_pixels(self) -> np.ndarray:
        return self.pixels

    # --- Segmentation-specific ------------------------------------------

    def get_walkable_mask(self) -> np.ndarray:
        """Boolean mask where ``True`` = walkable."""
        return self.segmentation.mask
