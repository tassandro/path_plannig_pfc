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
import subprocess

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


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class PathFollowerNode(Node):
    """Waypoint follower: consumes ``nav_msgs/Path`` and drives the robot."""

    def __init__(self) -> None:
        super().__init__("path_follower_node")

        self.declare_parameter("path_topic", "/path_planning/path")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("linear_speed", 0.4)
        self.declare_parameter("max_angular_speed", 1.0)
        # Kept well below typical wall/aisle thickness so the "waypoint
        # reached" Euclidean-distance check can't jump across a nearby
        # obstacle to a further-along waypoint that only *looks* close in a
        # straight line (was cutting corners straight into shelves/walls).
        self.declare_parameter("waypoint_tolerance", 0.12)
        self.declare_parameter("goal_tolerance", 0.15)
        self.declare_parameter("control_period", 0.1)
        self.declare_parameter("teleport_to_start", False)
        self.declare_parameter("gz_world", "warehouse")
        self.declare_parameter("gz_entity_name", "")
        self.declare_parameter("teleport_z", 0.1)
        self._path_topic = self.get_parameter("path_topic").value
        self._odom_topic = self.get_parameter("odom_topic").value
        self._cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self._linear_speed = float(self.get_parameter("linear_speed").value)
        self._max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self._waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self._goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        control_period = float(self.get_parameter("control_period").value)
        self._teleport_to_start = bool(self.get_parameter("teleport_to_start").value)
        self._gz_world = self.get_parameter("gz_world").value
        self._gz_entity_name = self.get_parameter("gz_entity_name").value
        self._teleport_z = float(self.get_parameter("teleport_z").value)
        self._teleporting = False
        # (offset_yaw, tx, ty) rigid transform applied to raw odom readings
        # after a teleport, so navigation stays consistent with the world
        # pose Gazebo was snapped to (odom itself keeps integrating from
        # wherever it was before the teleport — see _teleport_to_first_waypoint).
        self._odom_offset: tuple[float, float, float] | None = None
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
        # Detects a re-published but identical path (e.g. the DDS-drop
        # workaround of publishing twice) so we don't teleport a second time.
        self._last_waypoints_key: tuple[tuple[float, float], ...] | None = None

        self._path_sub = self.create_subscription(
            Path, self._path_topic, self._on_path, path_qos,
        )
        self._odom_sub = self.create_subscription(
            Odometry, self._odom_topic, self._on_odom, 10,
        )
        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        # World-frame-consistent pose (raw odom + teleport offset applied) —
        # subscribe to this instead of odom_topic if teleport_to_start is on,
        # since raw odom then diverges from the path's coordinate frame.
        self._corrected_odom_pub = self.create_publisher(Odometry, "corrected_odom", 10)

        self._timer = self.create_timer(control_period, self._control_loop)

        self.get_logger().info(
            f"PathFollowerNode ready — path: '{self._path_topic}', "
            f"odom: '{self._odom_topic}', cmd_vel: '{self._cmd_vel_topic}'"
        )

    # ------------------------------------------------------------------

    def _on_path(self, msg: Path) -> None:
        new_waypoints = [
            (pose.pose.position.x, pose.pose.position.y) for pose in msg.poses
        ]
        waypoints_key = tuple(new_waypoints)
        is_new_path = waypoints_key != self._last_waypoints_key
        self._waypoints = new_waypoints
        self._current_index = 0
        self.get_logger().info(f"New path received: {len(self._waypoints)} waypoints")

        if self._teleport_to_start and self._waypoints and is_new_path:
            self._teleport_to_first_waypoint()
        self._last_waypoints_key = waypoints_key

    def _on_odom(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        cx, cy, cyaw = self._apply_odom_offset(pos.x, pos.y, yaw)
        self._current_pose = (cx, cy, cyaw)

        corrected = Odometry()
        corrected.header = msg.header
        corrected.child_frame_id = msg.child_frame_id
        corrected.pose.pose.position.x = cx
        corrected.pose.pose.position.y = cy
        corrected.pose.pose.position.z = msg.pose.pose.position.z
        corrected.pose.pose.orientation.z = math.sin(cyaw / 2.0)
        corrected.pose.pose.orientation.w = math.cos(cyaw / 2.0)
        corrected.twist = msg.twist
        self._corrected_odom_pub.publish(corrected)

    def _apply_odom_offset(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        """Map a raw odom reading to the world frame using ``self._odom_offset``.

        Gazebo's ``set_pose`` teleport moves the robot in the world but the
        EKF (wheel + IMU integration) has no way to know that happened, so
        it keeps publishing odom relative to wherever it was before the
        teleport (confirmed: calling robot_localization's own ``set_pose``
        service does not make it stick). Instead, we record the rigid
        transform between "odom" and "world" at teleport time and apply it
        here on every reading, so navigation stays correct afterwards.
        """
        if self._odom_offset is None:
            return (x, y, yaw)
        offset_yaw, tx, ty = self._odom_offset
        c, s = math.cos(offset_yaw), math.sin(offset_yaw)
        return (c * x - s * y + tx, s * x + c * y + ty, _normalize_angle(yaw + offset_yaw))

    # ------------------------------------------------------------------

    def _teleport_to_first_waypoint(self) -> None:
        """Snap the robot to the path's first waypoint via Gazebo's set_pose service.

        Avoids the robot driving from its spawn pose to wherever the
        planned path happens to start before the "real" trajectory begins.
        """
        if not self._gz_entity_name:
            self.get_logger().warn(
                "teleport_to_start is enabled but gz_entity_name is empty — skipping teleport"
            )
            return

        x, y = self._waypoints[0]
        if len(self._waypoints) > 1:
            nx, ny = self._waypoints[1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = 0.0

        self._teleporting = True
        self._cmd_pub.publish(Twist())
        req = (
            f'name: "{self._gz_entity_name}", '
            f"position: {{x: {x}, y: {y}, z: {self._teleport_z}}}, "
            f"orientation: {{x: 0, y: 0, z: {math.sin(yaw / 2.0)}, w: {math.cos(yaw / 2.0)}}}"
        )
        try:
            result = subprocess.run(
                [
                    "ign", "service", "-s", f"/world/{self._gz_world}/set_pose",
                    "--reqtype", "ignition.msgs.Pose",
                    "--reptype", "ignition.msgs.Boolean",
                    "--timeout", "2000",
                    "--req", req,
                ],
                capture_output=True, text=True, timeout=5.0,
            )
            if result.returncode == 0:
                self.get_logger().info(
                    f"Teleported '{self._gz_entity_name}' to path start "
                    f"({x:.2f}, {y:.2f})"
                )
                self._set_odom_offset(x, y, yaw)
            else:
                self.get_logger().warn(f"Teleport failed: {result.stderr.strip()}")
        except (OSError, subprocess.SubprocessError) as exc:
            self.get_logger().warn(f"Teleport failed: {exc}")
        finally:
            self._teleporting = False

    def _set_odom_offset(self, world_x: float, world_y: float, world_yaw: float) -> None:
        """Compute the odom->world rigid transform so ``world_x/y/yaw`` is
        where the *current* (pre-transform) odom reading should map to."""
        ox, oy, oyaw = self._current_pose if self._current_pose is not None else (0.0, 0.0, 0.0)
        offset_yaw = _normalize_angle(world_yaw - oyaw)
        c, s = math.cos(offset_yaw), math.sin(offset_yaw)
        tx = world_x - (c * ox - s * oy)
        ty = world_y - (s * ox + c * oy)
        self._odom_offset = (offset_yaw, tx, ty)
        self._current_pose = (world_x, world_y, world_yaw)

    def _control_loop(self) -> None:
        if self._teleporting or not self._waypoints or self._current_pose is None:
            return

        if self._current_index >= len(self._waypoints):
            return

        x, y, yaw = self._current_pose

        # Advance past every waypoint already within tolerance in this same
        # tick, instead of returning early and letting the robot coast on
        # a stale cmd_vel until the next tick — that coasting was exactly
        # what let it cut corners into obstacles at sharp turns (the raw
        # grid path packs waypoints only a few cm apart, so several can be
        # "reached" between control ticks).
        while self._current_index < len(self._waypoints):
            tx, ty = self._waypoints[self._current_index]
            distance = math.hypot(tx - x, ty - y)
            is_last = self._current_index == len(self._waypoints) - 1
            tolerance = self._goal_tolerance if is_last else self._waypoint_tolerance
            if distance >= tolerance:
                break
            self._current_index += 1

        if self._current_index >= len(self._waypoints):
            self._cmd_pub.publish(Twist())  # stop — path complete
            self.get_logger().info("Path complete — goal reached")
            return

        tx, ty = self._waypoints[self._current_index]
        dx, dy = tx - x, ty - y
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
