from __future__ import annotations

from preprocessing.entities import SegmentedImageData

from ._grid_utils import build_grid_graph
from .entities import Graph
from .interfaces import GraphParser


class SegmentedImageGraphParser(GraphParser[SegmentedImageData]):
    """Grid-based graph parser that reads a pre-computed segmentation mask.

    Pair this with any ``PreProcessor`` that produces
    ``SegmentedImageData`` (e.g. ``HSVThresholdPreProcessor`` or
    ``DeepLabV3PreProcessor``).
    """

    def __init__(
        self,
        robot_size: float = 0,
        connectivity: int = 8,
        weight_limit: int = 100,
    ):
        self._robot_size = robot_size
        self._connectivity = connectivity
        self._weight_limit = weight_limit

    def name(self) -> str:
        return f"segmented_image_{self._connectivity}d"

    def parse(self, data: SegmentedImageData) -> Graph:
        return build_grid_graph(
            data.get_walkable_mask(),
            robot_size=self._robot_size,
            connectivity=self._connectivity,
            weight_limit=self._weight_limit,
        )
