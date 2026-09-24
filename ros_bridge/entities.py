from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RosBridgeConfig:
    """Configuration for a ROS publisher bridge.

    Attributes
    ----------
    node_name : str
        Name of the ROS node created to publish messages.
    image_topic : str
        Topic where the annotated image (source image with the
        identified path drawn on top of it) is published, as a
        ``sensor_msgs/Image`` message.
    path_topic : str
        Topic where the computed path is published as a
        ``nav_msgs/Path`` message.
    frame_id : str
        Reference frame used in message headers (e.g. ``"map"``).
    image_encoding : str
        OpenCV → ROS image encoding. Images produced by this pipeline
        are BGR (OpenCV convention), so the default is ``"bgr8"``.
    queue_size : int
        Publisher queue depth.
    latched : bool
        When ``True`` the last published message is kept for
        late-joining subscribers (QoS *transient local* on ROS 2) —
        useful because the ground robot may connect to the network
        after the image has already been generated and published.
    resolution_m_per_px : float
        Ground sampling distance — meters represented by one pixel of
        the source image (same convention as
        ``ImageData.resolution_m_per_px``). ``RobotPath`` waypoints are
        expressed in *pixel* coordinates; this factor converts them to
        meters before they are published as ``nav_msgs/Path``, so the
        waypoints line up with a Gazebo/world frame instead of landing
        hundreds of "meters" away. Defaults to ``1.0`` (no conversion)
        when the ground sampling distance of the source image is
        unknown.
    flip_y : bool
        Image row coordinates grow downward, while ROS world frames
        (REP 103) grow "up"/left-handed. When ``True`` (default) the Y
        axis is negated during the pixel→meters conversion.
    origin_x_m : float
    origin_y_m : float
        World-frame coordinates (meters, in ``frame_id``) that pixel
        ``(0, 0)`` of the source image maps to. Added *after* the
        ``resolution_m_per_px``/``flip_y`` conversion. Needed whenever
        the image does not represent a patch of the world centered on
        its own origin (e.g. an occupancy image generated from a known
        Gazebo world, where pixel (0, 0) corresponds to some arbitrary
        world coordinate, not ``(0, 0)``). Defaults to ``0.0`` (no
        offset), matching the previous behaviour.
    wait_for_subscribers_sec : float
        How long to wait (polling for DDS discovery) for at least one
        subscriber to match before publishing. The pipeline is a
        short-lived process that connects, publishes once and
        shuts down right away, so without this wait the messages can
        easily be lost if the subscriber (ground robot) hasn't finished
        matching yet — even with a *transient local* / latched QoS,
        because the publisher node is destroyed before the resend can
        happen. Set to ``0`` to disable the wait.
    publish_grace_sec : float
        How long to keep spinning after publishing (before the node is
        destroyed) to give DDS time to actually flush the message to
        already-matched subscribers.
    extra : dict
        Implementation-specific extra options.
    """

    node_name: str = "path_planning_bridge"
    image_topic: str = "/path_planning/annotated_image"
    path_topic: str = "/path_planning/path"
    frame_id: str = "map"
    image_encoding: str = "bgr8"
    queue_size: int = 10
    latched: bool = True
    resolution_m_per_px: float = 1.0
    flip_y: bool = True
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    wait_for_subscribers_sec: float = 3.0
    publish_grace_sec: float = 0.5
    extra: dict = field(default_factory=dict)
