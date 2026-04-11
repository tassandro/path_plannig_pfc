from __future__ import annotations

from collections import deque
from time import perf_counter

import numpy as np

from graph_parsing.entities import Graph

from ._utils import compute_path_cost, reconstruct_path
from .entities import PathResult
from .interfaces import PathAlgorithm


class BFSAlgorithm(PathAlgorithm):
    """Breadth-first search — finds the shortest-hop path."""

    def __init__(self, weighted: bool = True):
        self._weighted = weighted

    def name(self) -> str:
        return "BFS"

    def find_path(self, graph: Graph, start: int, end: int) -> PathResult:
        cost_column = 0 if self._weighted else 1
        adj = graph.build_adjacency_list(cost_column)

        n = graph.num_nodes
        parent = np.full(n, -1, dtype=np.int64)
        visited = np.zeros(n, dtype=bool)
        visited[start] = True

        queue: deque[int] = deque([start])
        nodes_explored = 0
        found = False

        t0 = perf_counter()

        while queue:
            u = queue.popleft()
            nodes_explored += 1

            if u == end:
                found = True
                break

            for v, _ in adj.get(u, []):
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)

        elapsed = perf_counter() - t0

        if not found:
            return PathResult(
                node_indices=[],
                total_cost=float("inf"),
                algorithm_name=self.name(),
                metadata={
                    "nodes_explored": nodes_explored,
                    "elapsed_seconds": elapsed,
                    "error": "no path found",
                },
            )

        path = reconstruct_path(parent, end)
        total_cost = compute_path_cost(path, adj)

        return PathResult(
            node_indices=path,
            total_cost=total_cost,
            algorithm_name=self.name(),
            metadata={
                "nodes_explored": nodes_explored,
                "elapsed_seconds": elapsed,
                "weighted": self._weighted,
            },
        )
