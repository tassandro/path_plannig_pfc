#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source ~/clearpath_ws/install/setup.bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
exec ros2 launch clearpath_gz robot_spawn.launch.py rviz:=false use_sim_time:=true world:=warehouse setup_path:=$HOME/clearpath/
