"""Integration launch file: Jackal UGV (Clearpath) simulation + ugv_bridge.

Starts (by default) the Jackal ("J100") Gazebo simulation via Clearpath's
``clearpath_gz`` package, plus the three ``ugv_bridge`` nodes needed to
close the loop with ``path_plannig_pfc``'s ``ros_bridge`` output:

- ``tf_odom_bridge_node``: ``clearpath_gz`` only bridges the robot pose as
  a TF transform (``odom`` -> ``base_link``), not as ``nav_msgs/Odometry``.
  This node republishes that transform as Odometry so ``path_follower_node``
  can consume it.
- ``path_follower_node``: drives the robot (via ``/cmd_vel``) along the
  ``nav_msgs/Path`` published by ``ros_bridge``.
- ``image_monitor_node``: saves the annotated path image published by
  ``ros_bridge``.

Prerequisites
-------------
1. Install & build ``clearpath_gz`` (part of the ``clearpath_simulator``
   repo — clone the branch matching your ROS distro, e.g. ``humble``):
   https://github.com/clearpathrobotics/clearpath_simulator
2. Create a minimal Clearpath robot description at ``~/clearpath/robot.yaml``
   selecting the Jackal platform, e.g.::

       serial_number: j100-0000
       system:
         hosts:
           - hostname: localhost
             ip: 127.0.0.1
       platform: j100

   (see clearpath_config / clearpath_simulator docs for the full schema).

Usage
-----
::

    ros2 launch ugv_bridge jackal_bridge.launch.py

    # Reuse an already-running simulation instead of starting a new one:
    ros2 launch ugv_bridge jackal_bridge.launch.py launch_jackal_sim:=false

    # Pick a different world:
    ros2 launch ugv_bridge jackal_bridge.launch.py world:=office

Then, from ``path_plannig_pfc``, run e.g.::

    python main.py --publish-ros --resolution-m-per-px 0.05 ...

Note on topic/frame defaults (verified against a real run)
------------------------------------------------------------
clearpath_gz's real sample ``robot.yaml`` (``clearpath_config``'s
``sample/j100/j100_default.yaml``) sets ``system.ros2.namespace:
j100_0000``, which namespaces the platform topics/frames. With that
sample, the real Jackal sim exposes:

- ``j100_0000/cmd_vel`` (Twist in).
- ``j100_0000/platform/odom/filtered`` — a **real ``nav_msgs/Odometry``**
  published by ``robot_localization``'s EKF (fusing wheel odom + IMU).
  Prefer pointing ``odom_topic`` straight at this instead of using
  ``tf_odom_bridge_node``/``odom_frame``/``base_frame`` — the TF bridge
  is only needed as a fallback for robots/sims that don't expose a real
  Odometry topic::

    ros2 launch ugv_bridge jackal_bridge.launch.py \\
        cmd_vel_topic:=j100_0000/cmd_vel \\
        odom_topic:=j100_0000/platform/odom/filtered

Only set an empty namespace in ``robot.yaml`` if you want the
un-namespaced defaults below (``/cmd_vel``, ``/odom``) to apply as-is.

Known gaps found while verifying this against a real Gazebo run
------------------------------------------------------------------
1. **No GPU / WSL2 without GPU passthrough**: ``simulation.launch.py``'s
   GUI (Ogre2/GL3Plus) can crash with
   ``Ogre::UnimplementedException`` in ``GL3PlusTextureGpu::copyTo`` on
   software-rendering-only setups. Workaround: run headless — start
   ``ros2 launch ros_gz_sim gz_sim.launch.py
   gz_args:="<world>.sdf -s -r"`` (server only) with
   ``GZ_SIM_RESOURCE_PATH`` set to clearpath_gz's ``worlds``/``meshes``
   dirs + every sourced package's ``share`` dir, plus a manual clock
   bridge, then ``ros2 launch clearpath_gz robot_spawn.launch.py`` —
   instead of the full ``simulation.launch.py`` used here.
2. **``ros2_control`` controllers aren't auto-activated.** After the
   robot spawns, ``/<namespace>/controller_manager/list_controllers``
   may come back empty — cmd_vel gets forwarded by twist_mux but the
   robot won't physically move. Fix: spawn the controllers once, e.g.::

    ros2 run controller_manager spawner \\
        joint_state_broadcaster platform_velocity_controller \\
        --controller-manager /j100_0000/controller_manager \\
        --ros-args -r __ns:=/j100_0000

   (controller names come from the generated
   ``<setup_path>/platform/config/control.yaml``).
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    # -- Jackal / clearpath_gz simulation args -----------------------------
    launch_jackal_sim_arg = DeclareLaunchArgument(
        "launch_jackal_sim", default_value="true",
        description=(
            "Start the Jackal Gazebo simulation (clearpath_gz) alongside "
            "the bridge nodes. Set to false to attach to a simulation "
            "that is already running (e.g. started separately)."
        ),
    )
    world_arg = DeclareLaunchArgument(
        "world", default_value="warehouse",
        description="clearpath_gz simulation world.",
    )
    setup_path_arg = DeclareLaunchArgument(
        "setup_path", default_value="~/clearpath/",
        description="Directory containing clearpath_gz's robot.yaml.",
    )
    rviz_arg = DeclareLaunchArgument("rviz", default_value="false")
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="true")

    # -- path_plannig_pfc <-> robot topics ----------------------------------
    path_topic_arg = DeclareLaunchArgument(
        "path_topic", default_value="/path_planning/path",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic", default_value="/path_planning/annotated_image",
    )
    cmd_vel_topic_arg = DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel")
    odom_topic_arg = DeclareLaunchArgument("odom_topic", default_value="/odom")

    # TF frames clearpath_gz publishes for the Jackal's pose. Prefix with a
    # namespace if robot.yaml sets one, e.g. "j100_0000/odom".
    odom_frame_arg = DeclareLaunchArgument("odom_frame", default_value="odom")
    base_frame_arg = DeclareLaunchArgument("base_frame", default_value="base_link")

    jackal_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("clearpath_gz"), "launch", "simulation.launch.py"]
            )
        ),
        launch_arguments={
            "world": LaunchConfiguration("world"),
            "setup_path": LaunchConfiguration("setup_path"),
            "rviz": LaunchConfiguration("rviz"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("launch_jackal_sim")),
    )

    tf_odom_bridge = Node(
        package="ugv_bridge",
        executable="tf_odom_bridge_node",
        name="tf_odom_bridge_node",
        output="screen",
        parameters=[{
            "odom_frame": LaunchConfiguration("odom_frame"),
            "base_frame": LaunchConfiguration("base_frame"),
            "odom_topic": LaunchConfiguration("odom_topic"),
        }],
    )

    path_follower = Node(
        package="ugv_bridge",
        executable="path_follower_node",
        name="path_follower_node",
        output="screen",
        parameters=[{
            "path_topic": LaunchConfiguration("path_topic"),
            "odom_topic": LaunchConfiguration("odom_topic"),
            "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
        }],
    )

    image_monitor = Node(
        package="ugv_bridge",
        executable="image_monitor_node",
        name="image_monitor_node",
        output="screen",
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
        }],
    )

    return LaunchDescription([
        launch_jackal_sim_arg,
        world_arg,
        setup_path_arg,
        rviz_arg,
        use_sim_time_arg,
        path_topic_arg,
        image_topic_arg,
        cmd_vel_topic_arg,
        odom_topic_arg,
        odom_frame_arg,
        base_frame_arg,
        jackal_sim,
        tf_odom_bridge,
        path_follower,
        image_monitor,
    ])
