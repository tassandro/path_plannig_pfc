#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
echo "--- odom before ---"
timeout 2 ros2 topic echo /j100_0000/platform/odom --field pose.pose.position --once 2>&1
timeout 4 ros2 topic pub --rate 10 /j100_0000/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}" > /dev/null 2>&1
echo "--- odom after 4s of cmd_vel ---"
timeout 2 ros2 topic echo /j100_0000/platform/odom --field pose.pose.position --once 2>&1
