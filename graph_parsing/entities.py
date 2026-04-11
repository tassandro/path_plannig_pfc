from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    """Generic graph representation used across the entire pipeline.

    Attributes
    ----------
    adjacency_matrix : np.ndarray
        Dense (N, N) connectivity / weight matrix.
    node_embeddings : np.ndarray
        (N, D_node) feature matrix — one row per node.
        Examples: spatial coordinates, elevation, terrain type, RGB, …
    edge_index : np.ndarray
        (2, E) array where ``edge_index[0]`` are source nodes and
        ``edge_index[1]`` are target nodes (PyG convention).
    edge_embeddings : np.ndarray
        (E, D_edge) feature matrix — one row per edge.
        Examples: Euclidean distance, slope, traversal cost, …
    """

    adjacency_matrix: np.ndarray
    node_embeddings: np.ndarray
    edge_index: np.ndarray
    edge_embeddings: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.adjacency_matrix.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def node_feature_dim(self) -> int:
        return self.node_embeddings.shape[1] if self.node_embeddings.ndim == 2 else 0

    @property
    def edge_feature_dim(self) -> int:
        return self.edge_embeddings.shape[1] if self.edge_embeddings.ndim == 2 else 0

    def validate(self) -> bool:
        n = self.num_nodes
        if self.adjacency_matrix.shape != (n, n):
            return False
        if self.node_embeddings.shape[0] != n:
            return False
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            return False
        e = self.num_edges
        if self.edge_embeddings.shape[0] != e:
            return False
        return True

    def neighbors(self, node: int) -> np.ndarray:
        """Return the indices of all neighbors of *node*."""
        return np.nonzero(self.adjacency_matrix[node])[0]
