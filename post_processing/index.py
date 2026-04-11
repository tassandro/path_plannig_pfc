from __future__ import annotations

from .image_drawing_processor import ImageDrawingPostProcessor
from .interfaces import PostProcessor
from .simple_waypoint_processor import SimpleWaypointProcessor


class PostProcessorFactory:
    """Registry-based factory for ``PostProcessor`` implementations.

    Register a processor once::

        PostProcessorFactory.register("simple_waypoint", SimpleWaypointProcessor)

    Then create instances anywhere::

        processor  = PostProcessorFactory.create("simple_waypoint")
        robot_path = processor.process(path_result, graph)
    """

    _registry: dict[str, type[PostProcessor]] = {}

    @classmethod
    def register(cls, name: str, processor_class: type[PostProcessor]) -> None:
        cls._registry[name] = processor_class

    @classmethod
    def create(cls, name: str, **kwargs) -> PostProcessor:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown post-processor: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())


PostProcessorFactory.register("simple_waypoint", SimpleWaypointProcessor)
PostProcessorFactory.register("image_drawing", ImageDrawingPostProcessor)
