from __future__ import annotations

from .entities import RosBridgeConfig
from .interfaces import RosPublisher


class RosBridgeFactory:
    """Registry-based factory for ``RosPublisher`` implementations.

    Register a bridge once::

        RosBridgeFactory.register("ros2", Ros2ImagePathPublisher)

    Then create instances anywhere::

        bridge = RosBridgeFactory.create("ros2", config=RosBridgeConfig(...))
        with bridge:
            bridge.publish(image=annotated_bgr, robot_path=robot_path)
    """

    _registry: dict[str, type[RosPublisher]] = {}

    @classmethod
    def register(cls, name: str, publisher_class: type[RosPublisher]) -> None:
        cls._registry[name] = publisher_class

    @classmethod
    def create(cls, name: str, **kwargs) -> RosPublisher:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown ROS bridge: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())


# ROS 2 requires rclpy / cv_bridge — register only when a sourced ROS 2
# environment is importable, so the rest of the pipeline keeps working
# unchanged on machines without ROS installed (e.g. during development).
try:
    from .ros2_image_path_publisher import Ros2ImagePathPublisher

    RosBridgeFactory.register("ros2", Ros2ImagePathPublisher)
except ImportError:
    Ros2ImagePathPublisher = None  # type: ignore[assignment,misc]
