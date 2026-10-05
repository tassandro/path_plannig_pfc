#!/usr/bin/env python3
"""Record robot odometry during a run and render it as an MP4 replay video.

The Gazebo/RViz GUI can't be reliably displayed in this WSL2 setup
(Xwayland glamor/GBM failure at the WSLg level), so instead of showing
the robot live in a 3D window, this script records its real odometry
while the simulation runs and renders a 2D "replay" video: the annotated
path image (already produced by ``main.py --publish-ros``) with the
robot's actual recorded trajectory animated on top of it.

Workflow:
    1. Start this script FIRST — it just waits, subscribed to the path
       and odometry topics, until data arrives.
    2. Run ``main.py --publish-ros ...`` as usual to publish the path.
    3. Let the robot drive. This script auto-stops a couple of seconds
       after the goal is reached (or press Ctrl+C anytime to stop early
       and render whatever was captured so far).
    4. The MP4 is written to ``--output``.

Usage::

    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash
    python3 record_path_video.py \\
        --map-image ../../output/2026_08_10_12_00_00.png \\
        --output ../../output/robot_run.mp4 \\
        --resolution-m-per-px 0.05
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OdomRecorder(Node):
    """Records (t, x, y, yaw) odometry samples until the goal is reached."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("odom_video_recorder")
        self.args = args
        self.samples: list[tuple[float, float, float, float]] = []
        self.done = False
        self._start_time = time.monotonic()
        self._goal_xy: tuple[float, float] | None = None
        self._goal_reached_time: float | None = None
        self.path_xy: list[tuple[float, float]] = []

        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        path_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(Odometry, args.odom_topic, self._on_odom, odom_qos)
        self.create_subscription(Path, args.path_topic, self._on_path, path_qos)
        self.create_timer(0.5, self._check_stop)

        self.get_logger().info(
            f"Recording odom from '{args.odom_topic}', waiting for the path on "
            f"'{args.path_topic}' to know where the goal is…"
        )

    def _on_path(self, msg: Path) -> None:
        if not msg.poses:
            return
        self.path_xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        last = msg.poses[-1].pose.position
        self._goal_xy = (last.x, last.y)
        self.get_logger().info(f"Goal received: ({last.x:.2f}, {last.y:.2f}) m")

    def _on_odom(self, msg: Odometry) -> None:
        t = time.monotonic() - self._start_time
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        self.samples.append((t, x, y, yaw))

        if self._goal_xy is not None and self._goal_reached_time is None and t > 1.0:
            dist = math.hypot(x - self._goal_xy[0], y - self._goal_xy[1])
            if dist < self.args.goal_tolerance:
                self._goal_reached_time = time.monotonic()
                self.get_logger().info(
                    f"Goal reached at t={t:.1f}s — recording "
                    f"{self.args.tail_sec:.1f}s more, then rendering…"
                )

    def _check_stop(self) -> None:
        now = time.monotonic()
        if (
            self._goal_reached_time is not None
            and now - self._goal_reached_time > self.args.tail_sec
        ):
            self.done = True
        elif now - self._start_time > self.args.max_duration:
            self.get_logger().warn("Max duration reached — stopping.")
            self.done = True


def render_video(samples: list[tuple[float, float, float, float]], args: argparse.Namespace) -> str:
    if len(samples) < 2:
        raise RuntimeError("Not enough odometry samples recorded to render a video.")

    image = cv2.imread(args.map_image)
    if image is None:
        raise RuntimeError(f"Could not load map image: {args.map_image}")
    h, w = image.shape[:2]
    res = args.resolution_m_per_px

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        # Inverse of ros2_image_path_publisher.py's pixel->meters conversion:
        # x_world = wp.x*res + origin_x_m, y_world = (-wp.y if flip_y else wp.y)*res + origin_y_m
        px = (x_m - args.origin_x_m) / res
        py = (args.origin_y_m - y_m) / res if args.flip_y else (y_m - args.origin_y_m) / res
        return int(round(px)), int(round(py))

    sample_t = np.array([s[0] for s in samples])
    sample_x = np.array([s[1] for s in samples])
    sample_y = np.array([s[2] for s in samples])
    sample_yaw = np.unwrap(np.array([s[3] for s in samples]))

    n_frames = max(2, int(round((sample_t[-1] - sample_t[0]) * args.fps)))
    times = np.linspace(sample_t[0], sample_t[-1], n_frames)
    xs = np.interp(times, sample_t, sample_x)
    ys = np.interp(times, sample_t, sample_y)
    yaws = np.interp(times, sample_t, sample_yaw)

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )

    trail_px: list[tuple[int, int]] = []
    for t, x_m, y_m, yaw in zip(times, xs, ys, yaws):
        frame = image.copy()
        px, py = to_px(x_m, y_m)
        trail_px.append((px, py))
        if len(trail_px) > args.trail_length:
            trail_px.pop(0)

        for j in range(1, len(trail_px)):
            cv2.line(frame, trail_px[j - 1], trail_px[j], (255, 0, 255), thickness=2)

        arrow_len = 12
        tip = (int(px + arrow_len * math.cos(yaw)), int(py - arrow_len * math.sin(yaw)))
        cv2.circle(frame, (px, py), 6, (0, 165, 255), thickness=-1)
        cv2.arrowedLine(frame, (px, py), tip, (0, 165, 255), thickness=2, tipLength=0.5)

        cv2.putText(
            frame, f"t = {t:.1f}s", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-image", required=True, help="Annotated path image produced by main.py")
    parser.add_argument("--output", default="output/robot_run.mp4")
    parser.add_argument("--odom-topic", default="/j100_0000/platform/odom/filtered")
    parser.add_argument("--path-topic", default="/path_planning/path")
    parser.add_argument("--resolution-m-per-px", type=float, default=0.05)
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false", default=True)
    parser.add_argument("--origin-x-m", type=float, default=0.0,
                         help="World X that pixel (0,0) of --map-image maps to (see RosBridgeConfig.origin_x_m)")
    parser.add_argument("--origin-y-m", type=float, default=0.0,
                         help="World Y that pixel (0,0) of --map-image maps to (see RosBridgeConfig.origin_y_m)")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--trail-length", type=int, default=80)
    parser.add_argument("--goal-tolerance", type=float, default=0.3)
    parser.add_argument("--tail-sec", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=600.0)
    args = parser.parse_args()

    rclpy.init()
    node = OdomRecorder(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        samples = node.samples
        path_xy = node.path_xy
        node.destroy_node()
        rclpy.shutdown()

    if not samples:
        print("\nNo odometry samples recorded — nothing to render.")
        sys.exit(1)

    print(f"\nRecorded {len(samples)} odometry samples ({samples[-1][0] - samples[0][0]:.1f}s)")
    base = args.output.rsplit(".", 1)[0]
    np.savetxt(f"{base}_odom.csv", np.array(samples), delimiter=",",
               header="t,x,y,yaw", comments="")
    if path_xy:
        np.savetxt(f"{base}_path.csv", np.array(path_xy), delimiter=",",
                   header="x,y", comments="")
    out = render_video(samples, args)
    print(f"Video saved to: {out}")


if __name__ == "__main__":
    main()
