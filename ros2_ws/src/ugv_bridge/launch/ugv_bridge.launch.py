"""Launch both ugv_bridge nodes: the path follower (drives the robot) and
the image monitor (receives/saves the annotated path image).

This launch file assumes Gazebo and the UGV (providing ``odom_topic``
and consuming ``cmd_vel_topic``) are already running — e.g. started
separately via your robot's own simulation launch file, or ``ros_gz``.

Usage::

    ros2 launch ugv_bridge ugv_bridge.launch.py \\
        odom_topic:=/odom cmd_vel_topic:=/cmd_vel
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    path_topic_arg = DeclareLaunchArgument(
        "path_topic", default_value="/path_planning/path",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic", default_value="/path_planning/annotated_image",
    )
    odom_topic_arg = DeclareLaunchArgument("odom_topic", default_value="/odom")
    cmd_vel_topic_arg = DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel")

    path_follower = Node(
        package="ugv_bridge",
        executable="path_follower_node",
        name="path_follower_node",
        output="screen",
        parameters=[{
            "path_topic": LaunchConfiguration("path_topic"),
            "odom_topic": LaunchConfiguration("odom_topic"),
            "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
        }],
    )

    image_monitor = Node(
        package="ugv_bridge",
        executable="image_monitor_node",
        name="image_monitor_node",
        output="screen",
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
        }],
    )

    return LaunchDescription([
        path_topic_arg,
        image_topic_arg,
        odom_topic_arg,
        cmd_vel_topic_arg,
        path_follower,
        image_monitor,
    ])
