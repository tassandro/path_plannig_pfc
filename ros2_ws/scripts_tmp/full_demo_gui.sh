#!/bin/bash
# One-shot demo WITH the live Gazebo GUI (based on the user's own proven
# recipe — LIBGL_ALWAYS_SOFTWARE + MESA_LOADER_DRIVER_OVERRIDE=llvmpipe lets
# the GUI render in this WSL2 setup when launched from a real interactive
# terminal session; automated/agent-spawned terminals hit a black/frozen
# window instead, see repo memory). Unlike full_demo.sh (headless + MP4
# replay), this one shows the robot driving live in the Gazebo window.
#
# Difference from a plain drone-photo run: the path is planned BEFORE the
# sim even needs to be up, straight from the world's own geometry via
# scripts/gazebo_world_to_occupancy.py — no camera/photo involved.
#
# Assumes the same one-time environment setup as full_demo.sh (~/clearpath_ws,
# ~/clearpath/robot.yaml, ~/ros2_ws, ~/venvs/pfc all already built/present).
#
# Run this directly in YOUR OWN interactive WSL terminal (not through an
# automation tool) so the GUI window actually renders.
set -o pipefail

REPO="/mnt/c/Users/Admin/Documents/pfc_novo/path_plannig_pfc"
WORLD="warehouse"

echo "=== Cleaning up any previous run ==="
pkill -9 -f 'ign gazebo' 2>/dev/null
pkill -9 -f 'ros2 launch' 2>/dev/null
pkill -9 -f 'parameter_bridge' 2>/dev/null
sleep 3

source /opt/ros/humble/setup.bash
source ~/clearpath_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

echo "=== [1/4] Starting Gazebo (GUI) + spawning the Jackal ==="
ros2 launch clearpath_gz simulation.launch.py rviz:=false setup_path:=$HOME/clearpath/ &

ODOM_TOPIC=""
echo "Waiting for the robot to be spawned…"
for i in $(seq 1 90); do
    ODOM_TOPIC=$(ros2 topic list 2>/dev/null | grep -m1 -E 'platform/odom(/filtered)?$')
    [ -n "$ODOM_TOPIC" ] && break
    sleep 1
done
if [ -z "$ODOM_TOPIC" ]; then
    echo "! robot odom topic never appeared — check the Gazebo window/log for spawn errors"
    exit 1
fi
echo "Robot spawned, real odom topic: $ODOM_TOPIC"

echo "=== [2/4] Activating ros2_control controllers ==="
ros2 run controller_manager spawner \
    joint_state_broadcaster platform_velocity_controller \
    --controller-manager /j100_0000/controller_manager \
    --ros-args -r __ns:=/j100_0000

echo "=== [3/4] Starting ugv_bridge (path follower, teleport-to-start on) ==="
ros2 launch ugv_bridge jackal_bridge.launch.py \
    launch_jackal_sim:=false \
    cmd_vel_topic:=j100_0000/cmd_vel \
    odom_topic:="${ODOM_TOPIC}" \
    use_tf_odom_bridge:=false \
    teleport_to_start:=true gz_world:=${WORLD} gz_entity_name:=j100_0000/robot &
echo "Esperando a ponte ficar pronta…"; sleep 8

echo "=== [4/4] Planning the path from the known Gazebo world, then publishing ==="
(
    source ~/venvs/pfc/bin/activate
    cd "$REPO"

    python3 scripts/gazebo_world_to_occupancy.py \
        ~/clearpath_ws/src/clearpath_simulator/clearpath_gz/worlds/${WORLD}.sdf \
        --output output/${WORLD}_occupancy.png

    # Start at the map's center (open hall between the 3 shelf blocks) and
    # end near the top-right corner — a longer, more complex route that
    # winds around several shelf/wall obstacles instead of the earlier
    # straight-corridor shortcut. --robot-size 12 (=0.6m) inflates obstacles
    # with real margin: with 7 (0.35m) the path's tightest pinch measured
    # only 0.354m clearance — barely 2cm over the Jackal's ~0.33m
    # half-diagonal, which is exactly what made it graze/collide and stall
    # in Gazebo despite "fitting" on paper. 12px raises that same pinch's
    # clearance to 0.6m (verified with a distance-transform diagnostic),
    # a real ~0.27m safety buffer, for basically the same path (621 vs 614
    # waypoints).
    PUBLISH_ARGS="output/${WORLD}_occupancy.png --preprocessor none --bgr 84,1,68 \
        --start 431,621 --end 700,160 --algorithm a_star --publish-ros \
        --robot-size 12 \
        --resolution-m-per-px 0.05 --origin-x-m -23.061 --origin-y-m 32.061"

    # Publish twice (a known DDS quirk in this setup can drop the very first
    # publish for one subscriber) — republishing a second time is a reliable
    # fix, per repo memory.
    python3 main.py $PUBLISH_ARGS
    sleep 3
    python3 main.py $PUBLISH_ARGS
)

echo
echo "Watch the Gazebo window — the Jackal should teleport to the path start"
echo "and drive to the goal. Ctrl+C here to stop everything."
wait
