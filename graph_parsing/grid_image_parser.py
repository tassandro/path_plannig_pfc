from __future__ import annotations

import numpy as np

from data_ingestion.interfaces import ImageIngestedData

from ._grid_utils import build_grid_graph
from .entities import Graph
from .interfaces import GraphParser


class GridImageGraphParser(GraphParser[ImageIngestedData]):
    """Grid-based graph parser using **exact BGR colour matching**.

    Every pixel whose colour equals *transitable_color* becomes a node.
    Use this for pre-processed maps where walkable areas are painted
    with a known colour.
    """

    def __init__(
        self,
        transitable_color: tuple[int, int, int] = (84, 1, 68),
        robot_size: float = 0,
        connectivity: int = 8,
        weight_limit: int = 100,
    ):
        self._transitable_color = np.array(transitable_color, dtype=np.uint8)
        self._robot_size = robot_size
        self._connectivity = connectivity
        self._weight_limit = weight_limit

    def name(self) -> str:
        return f"grid_image_{self._connectivity}d"

    def parse(self, data: ImageIngestedData) -> Graph:
        pixels = data.get_pixels()
        transitable = np.all(pixels == self._transitable_color, axis=2)
        return build_grid_graph(
            transitable,
            robot_size=self._robot_size,
            connectivity=self._connectivity,
            weight_limit=self._weight_limit,
        )
