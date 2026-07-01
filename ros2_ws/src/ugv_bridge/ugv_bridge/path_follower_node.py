"""ROS 2 node that drives a differential-drive UGV along a ``nav_msgs/Path``.

Subscribes to the path published by the path-planning system
(``path_plannig_pfc``'s ``ros_bridge`` module) and to the robot's
odometry, and publishes velocity commands so the robot follows the
waypoints in order — a simple waypoint / "carrot chasing" follower,
suitable for a differential-drive UGV simulated in Gazebo.

Parameters
----------
path_topic : str
    Topic where ``nav_msgs/Path`` is received (default matches
    ``ros_bridge``'s ``RosBridgeConfig.path_topic``).
odom_topic : str
    Topic where ``nav_msgs/Odometry`` is read from (default ``/odom``).
cmd_vel_topic : str
    Topic where ``geometry_msgs/Twist`` commands are published
    (default ``/cmd_vel``).
linear_speed : float
    Cruise linear speed (m/s) when facing the next waypoint.
max_angular_speed : float
    Maximum angular speed (rad/s).
waypoint_tolerance : float
    Distance (m) under which an intermediate waypoint is considered
    reached and the follower advances to the next one.
goal_tolerance : float
    Distance (m) under which the *final* waypoint is considered
    reached (usually tighter than ``waypoint_tolerance``).
control_period : float
    Control loop period, in seconds.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


def _yaw_from_quaternion(q) -> float:
    """Extract yaw (rotation about Z) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PathFollowerNode(Node):
    """Waypoint follower: consumes ``nav_msgs/Path`` and drives the robot."""

    def __init__(self) -> None:
        super().__init__("path_follower_node")

        self.declare_parameter("path_topic", "/path_planning/path")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("linear_speed", 0.4)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("waypoint_tolerance", 0.3)
        self.declare_parameter("goal_tolerance", 0.15)
        self.declare_parameter("control_period", 0.1)

        self._path_topic = self.get_parameter("path_topic").value
        self._odom_topic = self.get_parameter("odom_topic").value
        self._cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self._linear_speed = float(self.get_parameter("linear_speed").value)
        self._max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self._waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self._goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        control_period = float(self.get_parameter("control_period").value)

        # Transient-local so the follower still receives the path if it
        # starts after ros_bridge already published it (see RosBridgeConfig.latched).
        path_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._waypoints: list[tuple[float, float]] = []
        self._current_index = 0
        self._current_pose: tuple[float, float, float] | None = None  # x, y, yaw

        self._path_sub = self.create_subscription(
            Path, self._path_topic, self._on_path, path_qos,
        )
        self._odom_sub = self.create_subscription(
            Odometry, self._odom_topic, self._on_odom, 10,
        )
        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)

        self._timer = self.create_timer(control_period, self._control_loop)

        self.get_logger().info(
            f"PathFollowerNode ready — path: '{self._path_topic}', "
            f"odom: '{self._odom_topic}', cmd_vel: '{self._cmd_vel_topic}'"
        )

    # ------------------------------------------------------------------

    def _on_path(self, msg: Path) -> None:
        self._waypoints = [
            (pose.pose.position.x, pose.pose.position.y) for pose in msg.poses
        ]
        self._current_index = 0
        self.get_logger().info(f"New path received: {len(self._waypoints)} waypoints")

    def _on_odom(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        self._current_pose = (pos.x, pos.y, yaw)

    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        if not self._waypoints or self._current_pose is None:
            return

        if self._current_index >= len(self._waypoints):
            return

        x, y, yaw = self._current_pose
        tx, ty = self._waypoints[self._current_index]
        dx, dy = tx - x, ty - y
        distance = math.hypot(dx, dy)

        is_last = self._current_index == len(self._waypoints) - 1
        tolerance = self._goal_tolerance if is_last else self._waypoint_tolerance

        if distance < tolerance:
            self._current_index += 1
            if self._current_index >= len(self._waypoints):
                self._cmd_pub.publish(Twist())  # stop — path complete
                self.get_logger().info("Path complete — goal reached")
            return

        target_heading = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(target_heading - yaw), math.cos(target_heading - yaw)
        )

        cmd = Twist()
        cmd.angular.z = max(
            -self._max_angular_speed,
            min(self._max_angular_speed, 2.0 * heading_error),
        )
        # Slow down while turning sharply, move at cruise speed otherwise.
        turn_penalty = max(0.0, 1.0 - abs(heading_error) / (math.pi / 2))
        cmd.linear.x = self._linear_speed * turn_penalty
        self._cmd_pub.publish(cmd)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
