from __future__ import annotations

from abc import ABC, abstractmethod

from graph_parsing.entities import Graph

from .entities import PathResult


class PathAlgorithm(ABC):
    """Base interface for path-finding algorithms.

    Every algorithm receives a generic ``Graph`` (adjacency matrix +
    node/edge embeddings) and a start/end pair, and must return a
    ``PathResult`` with the ordered node sequence and total cost.
    """

    @abstractmethod
    def find_path(self, graph: Graph, start: int, end: int) -> PathResult:
        """Compute the optimal path from *start* to *end* on *graph*."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this algorithm."""
        ...
