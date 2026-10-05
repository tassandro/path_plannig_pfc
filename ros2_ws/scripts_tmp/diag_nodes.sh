#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
echo "=== all node-ish processes ==="
ps aux | grep -E 'ros2|gz|ign|twist_mux|ekf|controller' | grep -v grep
echo "=== ros2 node list ==="
timeout 5 ros2 node list
echo "=== ros2 topic list (cmd_vel/odom related) ==="
timeout 5 ros2 topic list | grep -iE 'cmd_vel|odom'
