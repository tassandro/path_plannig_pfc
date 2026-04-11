"""
Path Planning Pipeline
======================

End-to-end skeleton showing the four stages:

    1. Data Ingestion   — load a satellite image or point cloud
    2. Graph Parsing     — convert raw data into a graph representation
    3. Algorithm         — find the optimal path on the graph
    4. Post-Processing   — translate the path into robot-ready waypoints

Each stage is backed by an interface + factory so concrete strategies
can be swapped in without touching the orchestration code.
"""

from __future__ import annotations

from algorithm import AlgorithmFactory, PathResult
from data_ingestion import DataIngestionFactory, IngestedData
from graph_parsing import Graph, GraphParserFactory
from post_processing import PostProcessorFactory, RobotPath


def run_pipeline(
    *,
    data_type: str,
    data_kwargs: dict,
    parser_name: str,
    algorithm_name: str,
    post_processor_name: str,
    start: int,
    end: int,
) -> RobotPath:
    """Execute the full path-planning pipeline.

    Parameters
    ----------
    data_type : str
        Key registered in ``DataIngestionFactory`` (e.g. "image", "point_cloud").
    data_kwargs : dict
        Keyword arguments forwarded to the data constructor.
    parser_name : str
        Key registered in ``GraphParserFactory``.
    algorithm_name : str
        Key registered in ``AlgorithmFactory``.
    post_processor_name : str
        Key registered in ``PostProcessorFactory``.
    start, end : int
        Source and target node indices for the path search.

    Returns
    -------
    RobotPath
        The final robot-consumable path.
    """

    # 1. Ingest -----------------------------------------------------------
    data: IngestedData = DataIngestionFactory.create(data_type, **data_kwargs)
    assert data.validate(), f"Ingested data failed validation ({data.get_source()})"

    # 2. Parse to graph ---------------------------------------------------
    parser = GraphParserFactory.create(parser_name)
    graph: Graph = parser.parse(data)
    assert graph.validate(), "Graph failed validation"

    # 3. Find path --------------------------------------------------------
    algo = AlgorithmFactory.create(algorithm_name)
    path: PathResult = algo.find_path(graph, start, end)
    assert path.is_valid, "Algorithm returned an invalid path"

    # 4. Post-process for robot -------------------------------------------
    processor = PostProcessorFactory.create(post_processor_name)
    robot_path: RobotPath = processor.process(path, graph)

    return robot_path


if __name__ == "__main__":
    print("Available data types:    ", DataIngestionFactory.available())
    print("Available graph parsers: ", GraphParserFactory.available())
    print("Available algorithms:    ", AlgorithmFactory.available())
    print("Available post-processors:", PostProcessorFactory.available())
    print()
    print(
        "Register concrete implementations in each module's index.py, "
        "then call run_pipeline() to execute the full flow."
    )
