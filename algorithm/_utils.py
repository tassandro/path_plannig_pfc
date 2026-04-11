from __future__ import annotations

import numpy as np


def reconstruct_path(parent: np.ndarray, end: int) -> list[int]:
    """Walk the *parent* array backwards from *end* to build the path."""
    path: list[int] = []
    node = end
    while node != -1:
        path.append(node)
        node = int(parent[node])
    path.reverse()
    return path


def compute_path_cost(
    path: list[int],
    adjacency_list: dict[int, list[tuple[int, float]]],
) -> float:
    """Sum edge costs along *path* using a pre-built adjacency list."""
    total = 0.0
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        for neighbor, cost in adjacency_list.get(src, []):
            if neighbor == dst:
                total += cost
                break
    return total
