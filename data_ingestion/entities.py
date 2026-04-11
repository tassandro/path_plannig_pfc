from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interfaces import ImageIngestedData, PointCloudIngestedData


@dataclass
class PointCloudData(PointCloudIngestedData):
    """Concrete representation of a 3-D point cloud."""

    source_path: str
    points: np.ndarray                    # (N, 3) XYZ
    normals: np.ndarray | None = None     # (N, 3) optional surface normals
    colors: np.ndarray | None = None      # (N, 3) optional RGB [0-255]
    intensity: np.ndarray | None = None   # (N,)   optional scalar intensity
    metadata: dict = field(default_factory=dict)

    # --- IngestedData interface ------------------------------------------

    def get_source(self) -> str:
        return self.source_path

    def get_dimensions(self) -> tuple[int, ...]:
        return self.points.shape

    def get_raw_data(self) -> np.ndarray:
        return self.points

    def validate(self) -> bool:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            return False
        if self.normals is not None and self.normals.shape != self.points.shape:
            return False
        if self.colors is not None and self.colors.shape != self.points.shape:
            return False
        if self.intensity is not None and self.intensity.shape[0] != self.points.shape[0]:
            return False
        return True

    # --- PointCloudIngestedData interface --------------------------------

    def get_points(self) -> np.ndarray:
        return self.points


@dataclass
class ImageData(ImageIngestedData):
    """Concrete representation of a satellite / aerial image."""

    source_path: str
    pixels: np.ndarray                          # (H, W, C)
    resolution_m_per_px: float | None = None    # ground sampling distance
    metadata: dict = field(default_factory=dict)

    # --- IngestedData interface ------------------------------------------

    def get_source(self) -> str:
        return self.source_path

    def get_dimensions(self) -> tuple[int, ...]:
        return self.pixels.shape

    def get_raw_data(self) -> np.ndarray:
        return self.pixels

    def validate(self) -> bool:
        if self.pixels.ndim != 3:
            return False
        if self.pixels.shape[2] not in (1, 3, 4):
            return False
        return True

    # --- ImageIngestedData interface -------------------------------------

    def get_pixels(self) -> np.ndarray:
        return self.pixels
