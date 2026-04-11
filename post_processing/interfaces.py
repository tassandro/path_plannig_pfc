from __future__ import annotations

from abc import ABC, abstractmethod

from algorithm.entities import PathResult
from graph_parsing.entities import Graph

from .entities import RobotPath


class PostProcessor(ABC):
    """Base interface for converting an algorithm ``PathResult`` into a
    ``RobotPath`` that can be consumed by a robot's navigation stack.

    Implementations may apply smoothing, coordinate transforms,
    velocity profiling, etc.
    """

    @abstractmethod
    def process(self, path: PathResult, graph: Graph) -> RobotPath:
        """Transform *path* (node indices on *graph*) into a ``RobotPath``."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this post-processor."""
        ...
