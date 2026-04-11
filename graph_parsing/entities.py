from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Graph:
    """Generic graph representation used across the entire pipeline.

    Attributes
    ----------
    adjacency_matrix : np.ndarray
        (N, N) connectivity matrix.  For large graphs this may be a
        ``scipy.sparse`` matrix (duck-typed — ``.shape`` is the only
        property used internally).
    node_embeddings : np.ndarray
        (N, D_node) feature matrix — one row per node.
        **Convention:** columns 0 and 1 are spatial *x, y* coordinates.
    edge_index : np.ndarray
        (2, E) array — row 0 is source nodes, row 1 is target nodes.
    edge_embeddings : np.ndarray
        (E, D_edge) feature matrix — one row per edge.
        **Convention:** column 0 is the *weighted* cost, column 1 is
        the *unweighted* (pure distance) cost.
    metadata : dict
        Arbitrary extra data (pixel↔node maps for grid graphs, etc.).
    """

    adjacency_matrix: np.ndarray
    node_embeddings: np.ndarray
    edge_index: np.ndarray
    edge_embeddings: np.ndarray
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return int(self.node_embeddings.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def node_feature_dim(self) -> int:
        return self.node_embeddings.shape[1] if self.node_embeddings.ndim == 2 else 0

    @property
    def edge_feature_dim(self) -> int:
        return self.edge_embeddings.shape[1] if self.edge_embeddings.ndim == 2 else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        n = self.num_nodes
        if self.adjacency_matrix.shape != (n, n):
            return False
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            return False
        if self.edge_embeddings.shape[0] != self.num_edges:
            return False
        return True

    def neighbors(self, node: int) -> np.ndarray:
        """Return indices of all neighbors of *node* (via ``edge_index``)."""
        mask = self.edge_index[0] == node
        return self.edge_index[1, mask]

    def build_adjacency_list(
        self, cost_column: int = 0
    ) -> dict[int, list[tuple[int, float]]]:
        """Build ``{node: [(neighbor, cost), …]}`` from edge data.

        Parameters
        ----------
        cost_column :
            Which column of ``edge_embeddings`` to use as the edge cost.
            0 = weighted cost (default), 1 = pure distance.
        """
        adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        src = self.edge_index[0]
        dst = self.edge_index[1]
        costs = self.edge_embeddings[:, cost_column]
        for e in range(self.num_edges):
            adj[int(src[e])].append((int(dst[e]), float(costs[e])))
        return dict(adj)
