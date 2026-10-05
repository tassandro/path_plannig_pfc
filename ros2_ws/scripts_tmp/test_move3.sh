#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic pub --rate 10 /j100_0000/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}" > /dev/null 2>&1 &
PUBPID=$!
sleep 2
echo "--- cmd_vel publisher count now ---"
ros2 topic info /j100_0000/cmd_vel --verbose | grep -A3 "Publisher count"
echo "--- echoing twist_mux output for 4s ---"
timeout 4 ros2 topic echo /j100_0000/platform/cmd_vel_unstamped
echo "--- odom after ---"
timeout 2 ros2 topic echo /j100_0000/platform/odom --field pose.pose.position --once 2>&1
kill $PUBPID 2>/dev/null
