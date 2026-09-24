from __future__ import annotations

import time
from math import cos, sin

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image

from post_processing.entities import RobotPath

from .entities import RosBridgeConfig
from .interfaces import RosPublisher


class Ros2ImagePathPublisher(RosPublisher):
    """Publishes the annotated image and the computed path over ROS 2.

    The ground robot subscribes to ``config.image_topic``
    (``sensor_msgs/Image``) to receive the aerial image with the
    identified path drawn on top of it, and optionally to
    ``config.path_topic`` (``nav_msgs/Path``) to receive the
    corresponding navigable waypoints.

    Requires a sourced ROS 2 environment (``rclpy``, ``sensor_msgs``,
    ``nav_msgs``, ``geometry_msgs`` and ``cv_bridge`` importable), e.g.::

        source /opt/ros/<distro>/setup.bash
    """

    def __init__(self, config: RosBridgeConfig | None = None):
        super().__init__(config)
        self._node: Node | None = None
        self._bridge: CvBridge | None = None
        self._image_pub = None
        self._path_pub = None
        self._owns_rclpy_init = False

    def name(self) -> str:
        return "ros2"

    def connect(self) -> None:
        cfg = self._config

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy_init = True

        self._node = rclpy.create_node(cfg.node_name)
        self._bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=cfg.queue_size,
            durability=(
                QoSDurabilityPolicy.TRANSIENT_LOCAL
                if cfg.latched
                else QoSDurabilityPolicy.VOLATILE
            ),
        )

        self._image_pub = self._node.create_publisher(Image, cfg.image_topic, qos)
        self._path_pub = self._node.create_publisher(Path, cfg.path_topic, qos)

        self._wait_for_subscribers(cfg.wait_for_subscribers_sec)

    def _wait_for_subscribers(self, timeout_sec: float) -> None:
        """Poll DDS discovery briefly so early publishes aren't lost.

        This process connects, publishes once, and shuts down right
        away, so without this the ground robot's subscribers may not
        have finished matching yet when ``publish()`` is called — even
        latched (transient local) messages are lost in that case,
        because the publisher node gets destroyed before it can resend.
        """
        if timeout_sec <= 0 or self._node is None:
            return

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if (
                self._image_pub.get_subscription_count() > 0
                or self._path_pub.get_subscription_count() > 0
            ):
                return
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def publish(self, image: np.ndarray, robot_path: RobotPath | None = None) -> None:
        if self._node is None or self._bridge is None:
            raise RuntimeError(
                "Bridge not connected — call connect() or use it as a context manager"
            )

        cfg = self._config
        stamp = self._node.get_clock().now().to_msg()

        img_msg = self._bridge.cv2_to_imgmsg(image, encoding=cfg.image_encoding)
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = cfg.frame_id
        self._image_pub.publish(img_msg)

        if robot_path is not None and robot_path.is_valid:
            path_msg = Path()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = cfg.frame_id

            res = cfg.resolution_m_per_px
            for wp in robot_path.waypoints:
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = cfg.frame_id
                pose.pose.position.x = float(wp.x) * res + cfg.origin_x_m
                pose.pose.position.y = float(-wp.y if cfg.flip_y else wp.y) * res + cfg.origin_y_m
                pose.pose.position.z = float(wp.z) * res
                heading = -wp.heading if cfg.flip_y else wp.heading
                pose.pose.orientation.z = sin(heading / 2.0)
                pose.pose.orientation.w = cos(heading / 2.0)
                path_msg.poses.append(pose)

            self._path_pub.publish(path_msg)

        # Keep spinning briefly so DDS actually flushes the message to
        # already-matched subscribers before shutdown() destroys the
        # node (see _wait_for_subscribers for why this matters).
        deadline = time.monotonic() + cfg.publish_grace_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        if self._owns_rclpy_init and rclpy.ok():
            rclpy.shutdown()
            self._owns_rclpy_init = False
