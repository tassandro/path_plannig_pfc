#!/bin/bash
source /opt/ros/humble/setup.bash
timeout 5 ign topic -e -t /world/warehouse/dynamic_pose/info -n 1 > /tmp/dynpose.txt 2>&1
grep -B2 -A 12 'j100_0000/robot' /tmp/dynpose.txt
