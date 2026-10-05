#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
# Gazebo Fortress only publishes the sim clock under /world/<world>/clock,
# NOT a bare /clock — bridge that GZ topic and remap the ROS side to /clock
# (bare /clock never has any GZ publisher, so use_sim_time nodes like
# twist_mux never get a tick and their periodic timers silently never fire).
WORLD="${1:-warehouse}"
exec ros2 run ros_gz_bridge parameter_bridge "/world/${WORLD}/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" \
    --ros-args -r "/world/${WORLD}/clock:=/clock"
