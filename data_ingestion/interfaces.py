from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class IngestedData(ABC):
    """Base interface for all ingested data types.

    Every data source (satellite image, point cloud, etc.) must be wrapped
    in a concrete subclass that exposes a uniform API for downstream
    consumers (graph parsers, visualizers, etc.).
    """

    @abstractmethod
    def get_source(self) -> str:
        """Return the source identifier (file path, URL, sensor ID, …)."""
        ...

    @abstractmethod
    def get_dimensions(self) -> tuple[int, ...]:
        """Return the shape / dimensions of the underlying data."""
        ...

    @abstractmethod
    def get_raw_data(self) -> np.ndarray:
        """Return the raw data as a numpy array."""
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Return True if the ingested data is well-formed."""
        ...


class PointCloudIngestedData(IngestedData):
    """Marker interface for point-cloud-specific ingested data.

    Graph parsers that only work with point clouds should type-hint
    against this interface rather than the generic ``IngestedData``.
    """

    @abstractmethod
    def get_points(self) -> np.ndarray:
        """Return the (N, 3) array of XYZ coordinates."""
        ...


class ImageIngestedData(IngestedData):
    """Marker interface for image-specific ingested data.

    Graph parsers that only work with images should type-hint
    against this interface rather than the generic ``IngestedData``.
    """

    @abstractmethod
    def get_pixels(self) -> np.ndarray:
        """Return the (H, W, C) pixel array."""
        ...
