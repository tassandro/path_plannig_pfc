from __future__ import annotations

import heapq
from math import sqrt
from time import perf_counter

import numpy as np

from graph_parsing.entities import Graph

from ._utils import reconstruct_path
from .entities import PathResult


def priority_queue_search(
    graph: Graph,
    start: int,
    end: int,
    *,
    weighted: bool = True,
    use_heuristic: bool = True,
    algorithm_name: str = "",
) -> PathResult:
    """Shared Dijkstra / A* core.

    When *use_heuristic* is ``True`` the Euclidean distance to *end*
    (read from ``node_embeddings[:, :2]``) is used as the A* heuristic.
    """
    cost_column = 0 if weighted else 1
    adj = graph.build_adjacency_list(cost_column)

    n = graph.num_nodes
    g_cost = np.full(n, np.inf)
    g_cost[start] = 0.0
    parent = np.full(n, -1, dtype=np.int64)

    end_x = float(graph.node_embeddings[end, 0])
    end_y = float(graph.node_embeddings[end, 1])

    open_set: list[tuple[float, int]] = [(0.0, start)]
    closed = np.zeros(n, dtype=bool)
    nodes_explored = 0

    t0 = perf_counter()

    while open_set:
        _, u = heapq.heappop(open_set)
        if closed[u]:
            continue
        closed[u] = True
        nodes_explored += 1

        if u == end:
            return PathResult(
                node_indices=reconstruct_path(parent, end),
                total_cost=float(g_cost[end]),
                algorithm_name=algorithm_name,
                metadata={
                    "nodes_explored": nodes_explored,
                    "elapsed_seconds": perf_counter() - t0,
                    "weighted": weighted,
                },
            )

        for v, edge_cost in adj.get(u, []):
            if closed[v]:
                continue
            ng = g_cost[u] + edge_cost
            if ng < g_cost[v]:
                g_cost[v] = ng
                parent[v] = u
                h = 0.0
                if use_heuristic:
                    vx = float(graph.node_embeddings[v, 0])
                    vy = float(graph.node_embeddings[v, 1])
                    h = sqrt((vx - end_x) ** 2 + (vy - end_y) ** 2)
                heapq.heappush(open_set, (ng + h, v))

    return PathResult(
        node_indices=[],
        total_cost=float("inf"),
        algorithm_name=algorithm_name,
        metadata={
            "nodes_explored": nodes_explored,
            "elapsed_seconds": perf_counter() - t0,
            "weighted": weighted,
            "error": "no path found",
        },
    )
