"""ROS 2 node that republishes a TF transform as ``nav_msgs/Odometry``.

Some Gazebo simulation stacks (notably Clearpath's ``clearpath_gz``, used
to simulate the Jackal UGV) only bridge the robot's pose as a
``odom`` -> ``base_link`` transform on ``/tf`` and do **not** publish a
``nav_msgs/Odometry`` topic directly. ``path_follower_node`` (this
package) needs ``nav_msgs/Odometry`` to know the robot's current pose,
so this node bridges the gap: it looks up the TF transform at a fixed
rate and republishes it as ``nav_msgs/Odometry`` (pose only — linear/
angular velocity fields are left zeroed, since ``path_follower_node``
only reads the pose).

Parameters
----------
odom_frame : str
    Parent TF frame (default ``odom``). Prefix with a namespace if the
    simulator publishes namespaced frames, e.g. ``a200_0000/odom``.
base_frame : str
    Child TF frame — the robot's base link (default ``base_link``).
odom_topic : str
    Topic where the resulting ``nav_msgs/Odometry`` is published
    (default ``/odom``, matching ``path_follower_node``'s default).
publish_rate : float
    Rate (Hz) at which the TF lookup is attempted and republished.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros


class TfOdomBridgeNode(Node):
    """Looks up an ``odom`` -> ``base_link`` TF transform and republishes
    it as ``nav_msgs/Odometry``, for simulators that only provide TF."""

    def __init__(self) -> None:
        super().__init__("tf_odom_bridge_node")

        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("publish_rate", 20.0)

        self._odom_frame = self.get_parameter("odom_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        odom_topic = self.get_parameter("odom_topic").value
        publish_rate = float(self.get_parameter("publish_rate").value)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self._timer = self.create_timer(1.0 / publish_rate, self._on_timer)
        self._warned_once = False

        self.get_logger().info(
            f"TfOdomBridgeNode ready — '{self._odom_frame}' -> "
            f"'{self._base_frame}' republished as Odometry on '{odom_topic}'"
        )

    def _on_timer(self) -> None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._odom_frame, self._base_frame, Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            if not self._warned_once:
                self.get_logger().warn(
                    f"Waiting for TF '{self._odom_frame}' -> "
                    f"'{self._base_frame}': {exc}"
                )
                self._warned_once = True
            return

        self._warned_once = False
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        msg = Odometry()
        msg.header.stamp = transform.header.stamp
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame
        msg.pose.pose = Pose(
            position=Point(x=translation.x, y=translation.y, z=translation.z),
            orientation=Quaternion(
                x=rotation.x, y=rotation.y, z=rotation.z, w=rotation.w
            ),
        )
        msg.twist.twist = Twist()  # not tracked — path_follower_node only needs pose
        self._odom_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TfOdomBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
