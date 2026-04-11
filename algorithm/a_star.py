from __future__ import annotations

from graph_parsing.entities import Graph

from ._priority_queue_search import priority_queue_search
from .entities import PathResult
from .interfaces import PathAlgorithm


class AStarAlgorithm(PathAlgorithm):
    """A* shortest-path with Euclidean heuristic."""

    def __init__(self, weighted: bool = True):
        self._weighted = weighted

    def name(self) -> str:
        suffix = "weighted" if self._weighted else "unweighted"
        return f"A* ({suffix})"

    def find_path(self, graph: Graph, start: int, end: int) -> PathResult:
        return priority_queue_search(
            graph,
            start,
            end,
            weighted=self._weighted,
            use_heuristic=True,
            algorithm_name=self.name(),
        )
