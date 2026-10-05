#!/bin/bash
source /opt/ros/humble/setup.bash
source ~/clearpath_ws/install/setup.bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
exec ign gazebo -g
