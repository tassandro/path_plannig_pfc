#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source ~/clearpath_ws/install/setup.bash
CP_GZ_SHARE="$(ros2 pkg prefix clearpath_gz)/share/clearpath_gz"
EXTRA_SHARE="$(echo "$AMENT_PREFIX_PATH" | tr ':' '\n' | sed 's#$#/share#' | tr '\n' ':')"
export GZ_SIM_RESOURCE_PATH="${CP_GZ_SHARE}/worlds:${CP_GZ_SHARE}/meshes:${EXTRA_SHARE}"
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
echo "GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"
exec ros2 launch ros_gz_sim gz_sim.launch.py gz_args:="warehouse.sdf -s -r -v 4"
