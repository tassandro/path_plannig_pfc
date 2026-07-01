from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from post_processing.entities import RobotPath

from .entities import RosBridgeConfig


class RosPublisher(ABC):
    """Base interface for publishing pipeline outputs to a ROS network.

    Implementations translate the pipeline's internal representations
    (annotated OpenCV image + ``RobotPath``) into ROS messages and
    publish them, so a ground robot subscribed to the corresponding
    topics can receive both the visual identification of the path and
    its navigable waypoints.

    Instances are meant to be used as context managers so the
    underlying ROS node lifecycle (init/shutdown) is always cleaned
    up::

        with RosBridgeFactory.create("ros2", config=config) as bridge:
            bridge.publish(image=annotated_bgr, robot_path=robot_path)
    """

    def __init__(self, config: RosBridgeConfig | None = None):
        self._config = config or RosBridgeConfig()

    @property
    def config(self) -> RosBridgeConfig:
        return self._config

    @abstractmethod
    def connect(self) -> None:
        """Initialize the ROS node and publishers."""
        ...

    @abstractmethod
    def publish(self, image: np.ndarray, robot_path: RobotPath | None = None) -> None:
        """Publish *image* (BGR) as ``sensor_msgs/Image`` and, when
        provided, *robot_path* as ``nav_msgs/Path``."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Tear down the ROS node and release resources."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this bridge implementation."""
        ...

    def __enter__(self) -> "RosPublisher":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
