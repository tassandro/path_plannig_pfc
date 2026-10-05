#!/bin/bash
# One-shot demo: headless Gazebo warehouse + Jackal (J100) + a path planned
# BEFORE the sim runs (straight from the world's own geometry, via
# scripts/gazebo_world_to_occupancy.py) + video replay of the robot driving it.
#
# There is NO live Gazebo GUI here on purpose: this WSL2 install's
# Xwayland/GBM stack cannot render one (confirmed dead end — black/frozen
# window even with software rendering forced, see repo memory). Instead the
# robot's real recorded odometry is rendered into an MP4 replay at the end,
# which is the actually-working way to "see" the run in this environment.
#
# Assumes the one-time environment setup already documented/verified in this
# repo's session history is in place on THIS machine:
#   - ROS 2 Humble (source /opt/ros/humble/setup.bash)
#   - ~/clearpath_ws (clearpath_simulator, humble branch, built)
#   - ~/clearpath/robot.yaml (J100 sample, namespace j100_0000)
#   - ~/ros2_ws (this repo's ros2_ws/src/ugv_bridge, symlinked in, built)
#   - ~/venvs/pfc (--system-site-packages venv with this repo's requirements.txt)
# If any of these are missing, run this from a fresh clone and it WILL fail —
# it does not perform first-time installs.
# (no 'set -u': ROS 2's own setup.bash references unbound vars internally)
set -o pipefail

REPO="/mnt/c/Users/Admin/Documents/pfc_novo/path_plannig_pfc"
WORLD="warehouse"
LOGDIR="/tmp/full_demo_logs_rev"
mkdir -p "$LOGDIR"

echo "=== [0/7] Cleaning up any previous run ==="
pkill -9 -f "ign gazebo" 2>/dev/null
pkill -9 -f "ros_gz_bridge" 2>/dev/null
pkill -9 -f "ros2 launch" 2>/dev/null
pkill -9 -f "record_path_video.py" 2>/dev/null
sleep 2

source /opt/ros/humble/setup.bash
source ~/clearpath_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/venvs/pfc/bin/activate

export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost

echo "=== [1/7] Starting headless Gazebo server (world: $WORLD) ==="
CP_GZ_SHARE="$(ros2 pkg prefix clearpath_gz)/share/clearpath_gz"
EXTRA_SHARE="$(echo "$AMENT_PREFIX_PATH" | tr ':' '\n' | sed 's#$#/share#' | tr '\n' ':')"
export GZ_SIM_RESOURCE_PATH="${CP_GZ_SHARE}/worlds:${CP_GZ_SHARE}/meshes:${EXTRA_SHARE}"
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:="${WORLD}.sdf -s -r -v 4" \
    > "$LOGDIR/gz_server.log" 2>&1 &

for i in $(seq 1 30); do
    ign topic -l 2>/dev/null | grep -q "/world/${WORLD}/stats" && break
    sleep 1
done
echo "  Gazebo server up"

echo "=== [2/7] Bridging the sim clock ==="
ros2 run ros_gz_bridge parameter_bridge "/world/${WORLD}/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" \
    --ros-args -r "/world/${WORLD}/clock:=/clock" \
    > "$LOGDIR/clock_bridge.log" 2>&1 &

for i in $(seq 1 15); do
    timeout 1 ros2 topic echo /clock --once >/dev/null 2>&1 && break
    sleep 1
done
echo "  /clock alive"

echo "=== [3/7] Spawning the Jackal (J100) ==="
ros2 launch clearpath_gz robot_spawn.launch.py rviz:=false use_sim_time:=true \
    world:=${WORLD} setup_path:=$HOME/clearpath/ \
    > "$LOGDIR/robot_spawn.log" 2>&1 &

ODOM_TOPIC=""
for i in $(seq 1 60); do
    ODOM_TOPIC=$(ros2 topic list 2>/dev/null | grep -m1 -E 'platform/odom(/filtered)?$')
    [ -n "$ODOM_TOPIC" ] && break
    sleep 1
done
if [ -z "$ODOM_TOPIC" ]; then
    echo "  ! robot odom topic never appeared — check $LOGDIR/robot_spawn.log"
    exit 1
fi
echo "  Robot spawned, real odom topic: $ODOM_TOPIC"

echo "=== [4/7] Activating ros2_control controllers ==="
for i in $(seq 1 30); do
    ros2 service list 2>/dev/null | grep -q "/j100_0000/controller_manager/list_controllers" && break
    sleep 1
done
ros2 run controller_manager spawner joint_state_broadcaster platform_velocity_controller \
    --controller-manager /j100_0000/controller_manager --ros-args -r __ns:=/j100_0000 \
    > "$LOGDIR/controller_spawn.log" 2>&1
echo "  Controllers active"

echo "=== [5/7] Starting ugv_bridge (path follower, teleport-to-start on) ==="
ros2 launch ugv_bridge jackal_bridge.launch.py \
    launch_jackal_sim:=false \
    cmd_vel_topic:=j100_0000/cmd_vel \
    odom_topic:="${ODOM_TOPIC}" \
    use_tf_odom_bridge:=false \
    teleport_to_start:=true gz_world:=${WORLD} gz_entity_name:=j100_0000/robot \
    > "$LOGDIR/ugv_bridge.log" 2>&1 &
sleep 5
echo "  ugv_bridge nodes started"

echo "=== [6/7] Planning the path from the known Gazebo world (before publishing) ==="
cd "$REPO"
python scripts/gazebo_world_to_occupancy.py \
    ~/clearpath_ws/src/clearpath_simulator/clearpath_gz/worlds/${WORLD}.sdf \
    --output output/${WORLD}_occupancy.png \
    | tee "$LOGDIR/occupancy_gen.log"

# Dry run (no --publish-ros yet) just to get the annotated preview image the
# video recorder will use as its background. Start/end pixels are inside the
# building's own footprint (verified against the sealed occupancy map),
# not in the open exterior margin.
python main.py output/${WORLD}_occupancy.png --preprocessor none --bgr 84,1,68 \
    --start 200,200 --end 700,1100 --algorithm a_star \
    --resolution-m-per-px 0.05 --origin-x-m -23.061 --origin-y-m 32.061 \
    --output-dir output > "$LOGDIR/main_plan_dry.log" 2>&1

ANNOTATED=$(ls -t output/*.png | grep -v occupancy | head -1)
echo "  Planned path image: $ANNOTATED"

echo "=== [7/7] Recording the drive, then publishing the path ==="
# Recorder must be started BEFORE the real (--publish-ros) run below: a known
# DDS quirk in this setup means only the first path message after a fresh
# subscriber start is reliably delivered.
python ros2_ws/scripts_tmp/record_path_video.py \
    --map-image "$ANNOTATED" \
    --output output/${WORLD}_run_interno.mp4 \
    --odom-topic /corrected_odom \
    --resolution-m-per-px 0.05 --origin-x-m -23.061 --origin-y-m 32.061 \
    --max-duration 480 \
    > "$LOGDIR/recorder.log" 2>&1 &
REC_PID=$!
sleep 3

# Start at the map's center (open hall between the 3 shelf blocks) and end
# near the top-right corner — a longer, more complex route that winds
# around several shelf/wall obstacles instead of the earlier straight-
# corridor shortcut. --robot-size 12 (=0.6m) inflates obstacles with real
# margin: with 7 (0.35m) the path's tightest pinch measured only 0.354m
# clearance — barely 2cm over the Jackal's ~0.33m half-diagonal, which is
# exactly what made it graze/collide and stall in Gazebo despite "fitting"
# on paper. 12px raises that same pinch's clearance to 0.6m (verified with
# a distance-transform diagnostic), a real ~0.27m safety buffer, for
# basically the same path (621 vs 614 waypoints).
PUBLISH_ARGS="output/${WORLD}_occupancy.png --preprocessor none --bgr 84,1,68 \
    --start 431,621 --end 700,160 --algorithm a_star --publish-ros \
    --robot-size 12 \
    --resolution-m-per-px 0.05 --origin-x-m -23.061 --origin-y-m 32.061 \
    --output-dir output"

# A known DDS quirk in this setup occasionally drops the FIRST path publish
# for exactly ONE fresh subscriber (often path_follower_node) even though
# others (recorder, image_monitor_node) receive it fine — republish if
# path_follower_node doesn't confirm receipt within a few seconds.
for attempt in 1 2 3; do
    echo "  Publishing path (attempt $attempt)…"
    python main.py $PUBLISH_ARGS >> "$LOGDIR/main_plan_dry.log" 2>&1
    for i in $(seq 1 8); do
        grep -q "New path received" "$LOGDIR/ugv_bridge.log" 2>/dev/null && break 2
        sleep 1
    done
done
if ! grep -q "New path received" "$LOGDIR/ugv_bridge.log" 2>/dev/null; then
    echo "  ! path_follower_node never confirmed receiving the path after 3 attempts — check $LOGDIR/ugv_bridge.log"
fi

echo "Waiting for the robot to finish driving (recorder stops itself on goal)…"
wait $REC_PID

echo
echo "=== DONE ==="
echo "Video:  $REPO/output/${WORLD}_run_interno.mp4"
echo "Logs:   $LOGDIR"
echo
echo "(Sim/bridge processes are still running in the background — rerun this"
echo " script to clean up and start fresh, or: pkill -9 -f 'ign gazebo|ros_gz_bridge|ros2 launch')"
