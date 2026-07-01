"""ROS 2 node run on the ground robot side.

Receives the annotated image (source image with the identified path
drawn on top of it) published by the path-planning system
(``path_plannig_pfc``'s ``ros_bridge`` module) and stores the latest
frame to disk, so an operator / HMI can inspect what path the robot
was given.

Parameters
----------
image_topic : str
    Topic where ``sensor_msgs/Image`` is received (default matches
    ``ros_bridge``'s ``RosBridgeConfig.image_topic``).
output_dir : str
    Directory where received frames are saved.
save_latest_only : bool
    When ``True`` (default) every new frame overwrites ``latest.png``.
    When ``False`` each frame is saved with a timestamped filename.
"""

from __future__ import annotations

import os
from datetime import datetime

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image


class ImageMonitorNode(Node):
    """Subscribes to the annotated path image and saves received frames."""

    def __init__(self) -> None:
        super().__init__("image_monitor_node")

        self.declare_parameter("image_topic", "/path_planning/annotated_image")
        self.declare_parameter(
            "output_dir", os.path.expanduser("~/ugv_bridge/received_images")
        )
        self.declare_parameter("save_latest_only", True)

        image_topic = self.get_parameter("image_topic").value
        self._output_dir = self.get_parameter("output_dir").value
        self._save_latest_only = bool(self.get_parameter("save_latest_only").value)

        os.makedirs(self._output_dir, exist_ok=True)

        # Transient-local so a late-starting node still receives the last
        # image already published by ros_bridge (see RosBridgeConfig.latched).
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._bridge = CvBridge()
        self._sub = self.create_subscription(Image, image_topic, self._on_image, qos)

        self.get_logger().info(
            f"ImageMonitorNode ready — listening on '{image_topic}', "
            f"saving to '{self._output_dir}'"
        )

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        filename = (
            "latest.png"
            if self._save_latest_only
            else datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + ".png"
        )
        path = os.path.join(self._output_dir, filename)
        cv2.imwrite(path, frame)
        self.get_logger().info(f"Received annotated path image -> {path}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ImageMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
