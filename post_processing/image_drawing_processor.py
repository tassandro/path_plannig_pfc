from __future__ import annotations

import os

import cv2
import numpy as np

from algorithm.entities import PathResult
from graph_parsing.entities import Graph

from .entities import RobotPath
from .interfaces import PostProcessor
from .simple_waypoint_processor import SimpleWaypointProcessor

ALGORITHM_COLORS: dict[str, tuple[int, int, int]] = {
    "A* (weighted)": (255, 0, 0),
    "A* (unweighted)": (255, 169, 64),
    "Dijkstra (weighted)": (24, 173, 46),
    "Dijkstra (unweighted)": (96, 247, 62),
    "BFS": (255, 42, 245),
    "DFS": (0, 255, 255),
    "RRT": (255, 128, 0),
}


class ImageDrawingPostProcessor(PostProcessor):
    """Draw the computed path onto the original image.

    Each call to ``process()`` adds a new path (auto-coloured by
    algorithm name).  Call ``save()`` once all paths are drawn.

    Parameters
    ----------
    image :
        Source image (BGR, will be copied internally).
    output_path :
        Default file path for ``save()``.
    color :
        Override colour for *every* path.  When ``None`` the colour
        is chosen automatically from ``ALGORITHM_COLORS``.
    """

    def __init__(
        self,
        image: np.ndarray,
        output_path: str | None = None,
        color: tuple[int, int, int] | None = None,
    ):
        self._result_image = image.copy()
        self._output_path = output_path
        self._default_color = color
        self._waypoint_processor = SimpleWaypointProcessor()

    def name(self) -> str:
        return "image_drawing"

    @property
    def result_image(self) -> np.ndarray:
        """The annotated image (modified in-place across ``process`` calls)."""
        return self._result_image

    def process(self, path: PathResult, graph: Graph) -> RobotPath:
        robot_path = self._waypoint_processor.process(path, graph)

        if not path.is_valid:
            return robot_path

        pixel_coords = graph.metadata["pixel_coords"]
        color = self._default_color or ALGORITHM_COLORS.get(
            path.algorithm_name, (255, 255, 255)
        )

        # Draw path segments
        for i, node_idx in enumerate(path.node_indices):
            x = int(pixel_coords[node_idx, 0])
            y = int(pixel_coords[node_idx, 1])
            cv2.circle(self._result_image, (x, y), 2, color, thickness=3, lineType=8)
            if i > 0:
                prev = path.node_indices[i - 1]
                px = int(pixel_coords[prev, 0])
                py = int(pixel_coords[prev, 1])
                cv2.line(
                    self._result_image, (px, py), (x, y), color, thickness=1, lineType=8
                )

        # Start (red) / end (green) markers
        s = path.node_indices[0]
        e = path.node_indices[-1]
        cv2.circle(
            self._result_image,
            (int(pixel_coords[s, 0]), int(pixel_coords[s, 1])),
            6, (0, 0, 255), thickness=-1, lineType=8,
        )
        cv2.circle(
            self._result_image,
            (int(pixel_coords[e, 0]), int(pixel_coords[e, 1])),
            6, (0, 255, 0), thickness=-1, lineType=8,
        )

        robot_path.metadata["output_image_path"] = self._output_path
        return robot_path

    def save(self, path: str | None = None) -> str:
        """Write the annotated image to disk and return the file path."""
        out = path or self._output_path
        if out is None:
            raise ValueError("No output path provided")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        cv2.imwrite(out, self._result_image)
        return out
