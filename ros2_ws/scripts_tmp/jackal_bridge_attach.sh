#!/bin/bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
exec ros2 launch ugv_bridge jackal_bridge.launch.py \
    launch_jackal_sim:=false \
    cmd_vel_topic:=j100_0000/cmd_vel \
    odom_topic:=j100_0000/platform/odom/filtered
