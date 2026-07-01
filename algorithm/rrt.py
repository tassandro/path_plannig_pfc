"""RRT (Rapidly-exploring Random Tree) path planner.

Sampling-based planner adapted from Major Fábio Luiz Junior's reference
implementation (https://github.com/fabioluiz1987/RRT_sobre_mapa_de_navegacao).
Unlike the graph-search algorithms (BFS/DFS/Dijkstra/A*), RRT works in
**continuous pixel space**: it grows a tree from the start point by repeatedly
sampling a random point, stepping ``step_size`` pixels from the nearest tree
node toward it, and keeping the new node when the connecting segment is
collision-free. The path is found once a new node has an unobstructed straight
line to the goal.

To stay compatible with the rest of the pipeline (``PathResult`` +
post-processing index into the graph), the free space and the start/goal
coordinates are read from the ``Graph`` produced by ``graph_parsing``:

* ``metadata["node_index_map"]`` — (H, W) int map, ``>= 0`` where walkable.
* ``metadata["image_shape"]`` — (H, W) of the source image.
* ``node_embeddings[:, :2]`` — (x, y) pixel coordinates of every node.
* ``node_embeddings[:, 2]`` — obstacle-proximity weight (``inf`` when the
  pixel is within ``robot_size`` of an obstacle → treated as non-free, so RRT
  respects the same clearance as the graph search algorithms).

Collision checking follows Junior's convention: sample ~100 points along the
segment between two positions and require every one to land on free space.
Each resulting tree waypoint is snapped to the nearest graph node so the
existing waypoint / drawing post-processors work unchanged.
"""

from __future__ import annotations

import random
from math import atan2, cos, hypot, sin
from time import perf_counter

import numpy as np

from graph_parsing.entities import Graph

from .entities import PathResult
from .interfaces import PathAlgorithm


class RRTAlgorithm(PathAlgorithm):
    """Rapidly-exploring Random Tree planner (pixel-space, sampling-based).

    Parameters
    ----------
    step_size :
        Expansion step in pixels (Junior's default: 20).
    max_iterations :
        Maximum number of sampling iterations before giving up.
    goal_sample_rate :
        Probability of sampling the goal itself instead of a random point
        (goal biasing — speeds up convergence; ``0`` reproduces Junior's
        purely random sampling).
    collision_samples :
        Number of points sampled along a segment for the collision check.
    seed :
        RNG seed for reproducible runs (``None`` = nondeterministic).
    weighted :
        Accepted for interface symmetry with the graph-search algorithms and
        ignored — RRT is a geometric planner, its cost is the Euclidean path
        length.
    """

    def __init__(
        self,
        step_size: float = 20.0,
        max_iterations: int = 5000,
        goal_sample_rate: float = 0.05,
        collision_samples: int = 100,
        seed: int | None = None,
        weighted: bool = True,
    ):
        self._step_size = float(step_size)
        self._max_iterations = int(max_iterations)
        self._goal_sample_rate = float(goal_sample_rate)
        self._collision_samples = int(collision_samples)
        self._seed = seed
        self._weighted = weighted  # unused; kept for factory symmetry

    def name(self) -> str:
        return "RRT"

    # ------------------------------------------------------------------

    def find_path(self, graph: Graph, start: int, end: int) -> PathResult:
        rng = random.Random(self._seed)

        h, w = graph.metadata["image_shape"]
        node_index_map = graph.metadata["node_index_map"]

        # Free-space map: walkable AND far enough from obstacles (finite weight).
        free = node_index_map >= 0
        weights = graph.node_embeddings[:, 2]
        if not np.all(np.isfinite(weights)):
            pixel_coords = graph.metadata["pixel_coords"]
            blocked = ~np.isfinite(weights)
            free[pixel_coords[blocked, 1], pixel_coords[blocked, 0]] = False

        sx = int(round(graph.node_embeddings[start, 0]))
        sy = int(round(graph.node_embeddings[start, 1]))
        ex = int(round(graph.node_embeddings[end, 0]))
        ey = int(round(graph.node_embeddings[end, 1]))
        # The endpoints are valid nodes even if within clearance — allow them.
        free[sy, sx] = True
        free[ey, ex] = True

        def collision(x1: float, y1: float, x2: float, y2: float) -> bool:
            """True if the segment (x1,y1)->(x2,y2) crosses non-free space."""
            n = self._collision_samples
            for t in np.linspace(0.0, 1.0, n):
                xi = int(round(x1 + (x2 - x1) * t))
                yi = int(round(y1 + (y2 - y1) * t))
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    return True
                if not free[yi, xi]:
                    return True
            return False

        # ── Grow the tree ──────────────────────────────────────────────
        tree_x: list[float] = [float(sx)]
        tree_y: list[float] = [float(sy)]
        parent: list[int] = [-1]

        t0 = perf_counter()
        found = False
        goal_leaf = -1
        iterations = 0

        for iterations in range(1, self._max_iterations + 1):
            if rng.random() < self._goal_sample_rate:
                rx, ry = ex, ey
            else:
                rx, ry = rng.randint(0, w - 1), rng.randint(0, h - 1)

            # Nearest existing tree node
            best_i, best_d = 0, float("inf")
            for i in range(len(tree_x)):
                d = hypot(rx - tree_x[i], ry - tree_y[i])
                if d < best_d:
                    best_i, best_d = i, d
            nx, ny = tree_x[best_i], tree_y[best_i]

            # Step step_size toward the sample
            theta = atan2(ry - ny, rx - nx)
            tx = nx + self._step_size * cos(theta)
            ty = ny + self._step_size * sin(theta)
            if tx < 0 or tx >= w or ty < 0 or ty >= h:
                continue

            # Keep the new node only if the branch is collision-free
            if collision(tx, ty, nx, ny):
                continue

            tree_x.append(tx)
            tree_y.append(ty)
            parent.append(best_i)
            new_i = len(tree_x) - 1

            # Finished if we can now see the goal directly
            if not collision(tx, ty, ex, ey):
                found = True
                goal_leaf = new_i
                break

        elapsed = perf_counter() - t0

        base_meta = {
            "iterations": iterations,
            "tree_size": len(tree_x),
            "elapsed_seconds": elapsed,
            "step_size": self._step_size,
            "goal_sample_rate": self._goal_sample_rate,
        }

        if not found:
            return PathResult(
                node_indices=[],
                total_cost=float("inf"),
                algorithm_name=self.name(),
                metadata={**base_meta, "error": "no path found"},
            )

        # Backtrack the tree, then append the goal point
        pts: list[tuple[float, float]] = []
        i = goal_leaf
        while i != -1:
            pts.append((tree_x[i], tree_y[i]))
            i = parent[i]
        pts.reverse()
        pts.append((float(ex), float(ey)))

        # Geometric path length (matches Junior's calcular_comprimento_caminho)
        total_cost = sum(
            hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
            for k in range(len(pts) - 1)
        )

        # Snap each waypoint to a graph node so post-processing can index it
        node_indices = self._snap_path(pts, node_index_map, start, end)

        return PathResult(
            node_indices=node_indices,
            total_cost=float(total_cost),
            algorithm_name=self.name(),
            metadata=base_meta,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _snap_path(
        pts: list[tuple[float, float]],
        node_index_map: np.ndarray,
        start: int,
        end: int,
        search_radius: int = 6,
    ) -> list[int]:
        """Map continuous (x, y) waypoints to graph node indices.

        Each point is rounded to the nearest pixel; if that pixel is not a
        node, a small spiral search finds the closest walkable pixel. The
        exact start/end node indices are forced at the extremities, and
        consecutive duplicates are collapsed.
        """
        h, w = node_index_map.shape
        indices: list[int] = []

        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            idx = -1
            if 0 <= xi < w and 0 <= yi < h:
                idx = int(node_index_map[yi, xi])
            if idx < 0:
                idx = RRTAlgorithm._nearest_walkable(
                    node_index_map, xi, yi, search_radius
                )
            if idx >= 0 and (not indices or indices[-1] != idx):
                indices.append(idx)

        # Guarantee correct extremities
        if not indices or indices[0] != start:
            indices.insert(0, start)
        if indices[-1] != end:
            indices.append(end)
        return indices

    @staticmethod
    def _nearest_walkable(
        node_index_map: np.ndarray, cx: int, cy: int, radius: int
    ) -> int:
        """Spiral outward from (cx, cy) for the closest node pixel."""
        h, w = node_index_map.shape
        for r in range(1, radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue  # only the current ring
                    x, y = cx + dx, cy + dy
                    if 0 <= x < w and 0 <= y < h and node_index_map[y, x] >= 0:
                        return int(node_index_map[y, x])
        return -1
