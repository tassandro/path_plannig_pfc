"""Shared logic for building a grid-based graph from a boolean mask.

Both ``GridImageGraphParser`` (exact colour) and
``SegmentedImageGraphParser`` (ML mask) delegate here so the weight
computation, edge building, and adjacency construction live in one
place.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix

from .entities import Graph

MOVEMENTS_8D = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]
MOVEMENTS_4D = MOVEMENTS_8D[:4]


def build_grid_graph(
    transitable_mask: np.ndarray,
    *,
    robot_size: float = 0,
    connectivity: int = 8,
    weight_limit: int = 100,
) -> Graph:
    """Turn a boolean **(H, W)** mask into a full ``Graph``.

    Parameters
    ----------
    transitable_mask :
        ``True`` where a pixel is walkable.
    robot_size :
        Minimum clearance (px) to obstacles.  Nodes closer than this
        are blocked.
    connectivity :
        4 or 8 movement directions.
    weight_limit :
        Radius (px) for the obstacle-proximity weight field.
    """
    h, w = transitable_mask.shape
    movements = MOVEMENTS_8D if connectivity == 8 else MOVEMENTS_4D

    # 1. Assign node indices --------------------------------------------
    node_index_map = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.where(transitable_mask)
    n_nodes = len(ys)
    node_index_map[ys, xs] = np.arange(n_nodes, dtype=np.int64)

    # 2. Obstacle-proximity weights (distance transform) ----------------
    dist_to_obstacle = distance_transform_edt(transitable_mask)
    obstacle_dists = dist_to_obstacle[ys, xs]

    raw_weights = np.zeros(n_nodes, dtype=np.float64)
    blocked = obstacle_dists <= robot_size
    raw_weights[blocked] = np.inf

    active = ~blocked & (obstacle_dists < weight_limit)
    raw_weights[active] = weight_limit - obstacle_dists[active]

    # 3. Build edges (vectorised per direction) -------------------------
    all_src: list[np.ndarray] = []
    all_dst: list[np.ndarray] = []
    all_weighted: list[np.ndarray] = []
    all_dist: list[np.ndarray] = []

    for dx, dy in movements:
        sy0, sy1 = max(0, -dy), h - max(0, dy)
        sx0, sx1 = max(0, -dx), w - max(0, dx)

        src = node_index_map[sy0:sy1, sx0:sx1].ravel()
        dst = node_index_map[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx].ravel()

        valid = (src >= 0) & (dst >= 0)
        src_v, dst_v = src[valid], dst[valid]

        not_blocked = np.isfinite(raw_weights[src_v]) & np.isfinite(raw_weights[dst_v])
        src_f, dst_f = src_v[not_blocked], dst_v[not_blocked]

        dist = 1.0 if dx == 0 or dy == 0 else sqrt(2)
        all_src.append(src_f)
        all_dst.append(dst_f)
        all_weighted.append(dist + raw_weights[dst_f])
        all_dist.append(np.full(len(src_f), dist))

    if all_src:
        edge_src = np.concatenate(all_src)
        edge_dst = np.concatenate(all_dst)
        edge_index = np.stack([edge_src, edge_dst])
        edge_embeddings = np.column_stack(
            [np.concatenate(all_weighted), np.concatenate(all_dist)]
        )
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_embeddings = np.zeros((0, 2), dtype=np.float64)

    # 4. Node embeddings: [x, y, weight] --------------------------------
    node_embeddings = np.column_stack(
        [xs.astype(np.float64), ys.astype(np.float64), raw_weights]
    )

    # 5. Sparse adjacency matrix ----------------------------------------
    n_edges = edge_index.shape[1]
    adjacency_matrix = csr_matrix(
        (np.ones(n_edges, dtype=np.float64), (edge_index[0], edge_index[1])),
        shape=(n_nodes, n_nodes),
    )

    max_finite = raw_weights[np.isfinite(raw_weights)]
    max_weight = float(max_finite.max()) if max_finite.size > 0 else 0.0

    return Graph(
        adjacency_matrix=adjacency_matrix,
        node_embeddings=node_embeddings,
        edge_index=edge_index,
        edge_embeddings=edge_embeddings,
        metadata={
            "node_index_map": node_index_map,
            "pixel_coords": np.column_stack([xs, ys]),
            "image_shape": (h, w),
            "max_weight": max_weight,
        },
    )
