from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MoveType(Enum):
    """Semantic labels for different movement segments."""
    LINEAR = "linear"
    ARC = "arc"
    STOP = "stop"


@dataclass
class Waypoint:
    """A single waypoint in real-world coordinates."""

    x: float
    y: float
    z: float = 0.0
    heading: float = 0.0        # orientation in radians
    speed: float = 0.0          # suggested speed (m/s)
    move_type: MoveType = MoveType.LINEAR

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class RobotPath:
    """Robot-consumable path — the final output of the pipeline.

    This is the data structure that gets serialized and sent to the
    robot's navigation controller.
    """

    waypoints: list[Waypoint]
    total_distance: float = 0.0
    estimated_time: float = 0.0     # seconds
    metadata: dict = field(default_factory=dict)

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)

    @property
    def is_valid(self) -> bool:
        return len(self.waypoints) >= 2

    def to_coordinate_list(self) -> list[tuple[float, float, float]]:
        """Flat list of (x, y, z) tuples — convenient for serialization."""
        return [wp.as_tuple() for wp in self.waypoints]
