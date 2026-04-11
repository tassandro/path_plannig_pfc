from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PathResult:
    """Standardized output of any path-finding algorithm.

    Attributes
    ----------
    node_indices : list[int]
        Ordered sequence of node indices from start to goal (inclusive).
    total_cost : float
        Cumulative traversal cost along the path.
    algorithm_name : str
        Name of the algorithm that produced this result.
    metadata : dict
        Algorithm-specific extras — execution time, nodes explored, etc.
    """

    node_indices: list[int]
    total_cost: float
    algorithm_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of nodes in the path."""
        return len(self.node_indices)

    @property
    def is_valid(self) -> bool:
        """A valid path has at least a start and a goal node."""
        return len(self.node_indices) >= 2

    @property
    def start(self) -> int:
        return self.node_indices[0]

    @property
    def goal(self) -> int:
        return self.node_indices[-1]
