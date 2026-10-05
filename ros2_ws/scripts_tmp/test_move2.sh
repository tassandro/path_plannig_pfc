#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
echo "=== subscriber counts ==="
ros2 topic info /j100_0000/cmd_vel --verbose
echo "--- publishing for 3s and echoing twist_mux output ---"
ros2 topic pub --rate 10 /j100_0000/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}" > /dev/null 2>&1 &
PUBPID=$!
timeout 3 ros2 topic echo /j100_0000/platform/cmd_vel_unstamped
kill $PUBPID 2>/dev/null
echo "--- odom after ---"
timeout 2 ros2 topic echo /j100_0000/platform/odom --field pose.pose.position --once 2>&1
