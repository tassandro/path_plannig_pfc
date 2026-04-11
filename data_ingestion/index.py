from __future__ import annotations

from .entities import ImageData, PointCloudData
from .interfaces import IngestedData


class DataIngestionFactory:
    """Registry-based factory for creating ``IngestedData`` instances.

    Register new data types once::

        DataIngestionFactory.register("lidar", LidarPointCloudData)

    Then create instances anywhere::

        data = DataIngestionFactory.create("lidar", source_path="scan.las", ...)
    """

    _registry: dict[str, type[IngestedData]] = {}

    @classmethod
    def register(cls, name: str, data_class: type[IngestedData]) -> None:
        cls._registry[name] = data_class

    @classmethod
    def create(cls, name: str, **kwargs) -> IngestedData:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown data type: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())


DataIngestionFactory.register("point_cloud", PointCloudData)
DataIngestionFactory.register("image", ImageData)
