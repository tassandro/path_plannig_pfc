from __future__ import annotations

from math import atan2, sqrt

from algorithm.entities import PathResult
from graph_parsing.entities import Graph

from .entities import RobotPath, Waypoint
from .interfaces import PostProcessor


class SimpleWaypointProcessor(PostProcessor):
    """Direct conversion of path node-indices into spatial waypoints.

    Reads the (x, y) position from ``node_embeddings[:, :2]`` and
    computes heading from consecutive waypoint deltas.
    """

    def name(self) -> str:
        return "simple_waypoint"

    def process(self, path: PathResult, graph: Graph) -> RobotPath:
        if not path.is_valid:
            return RobotPath(waypoints=[], metadata={"error": "invalid input path"})

        waypoints: list[Waypoint] = []
        total_distance = 0.0

        for i, node_idx in enumerate(path.node_indices):
            x = float(graph.node_embeddings[node_idx, 0])
            y = float(graph.node_embeddings[node_idx, 1])

            heading = 0.0
            if i > 0:
                prev = waypoints[-1]
                dx = x - prev.x
                dy = y - prev.y
                total_distance += sqrt(dx * dx + dy * dy)
                heading = atan2(dy, dx)

            waypoints.append(Waypoint(x=x, y=y, heading=heading))

        return RobotPath(
            waypoints=waypoints,
            total_distance=total_distance,
            metadata={
                "algorithm": path.algorithm_name,
                "path_cost": path.total_cost,
                "num_waypoints": len(waypoints),
            },
        )
