"""
Path Planning Pipeline — CLI
=============================

Usage examples::

    # Real drone image — all 4 algorithms, DeepLabV3+ segmentation
    python main.py data/rgb/gloria/perspectiva.png \\
        --preprocessor deeplabv3 --algorithm a_star dijkstra bfs dfs

    # HSV baseline (fast, no GPU)
    python main.py data/rgb/gloria/perspectiva.png \\
        --preprocessor hsv_threshold --algorithm a_star

    # Pre-processed map with known transitable colour
    python main.py map.png --preprocessor none --bgr 84,1,68

    # Save segmentation mask for debugging
    python main.py image.jpg --preprocessor deeplabv3 --save-mask
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np

from algorithm import AlgorithmFactory
from data_ingestion import ImageData
from graph_parsing import GraphParserFactory
from post_processing import ImageDrawingPostProcessor
from preprocessing import PreProcessorFactory


def pixel_to_node(node_index_map: np.ndarray, x: int, y: int) -> int:
    idx = int(node_index_map[y, x])
    if idx < 0:
        raise ValueError(f"Pixel ({x}, {y}) is not a walkable node")
    return idx


def save_mask(segmented_data, output_dir: str) -> str:
    mask = segmented_data.get_walkable_mask()
    mask_img = mask.astype(np.uint8) * 255
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = os.path.join(output_dir, f"{ts}_mask.png")
    cv2.imwrite(path, mask_img)
    return path


# ------------------------------------------------------------------


def main() -> None:
    all_preprocessors = PreProcessorFactory.available()

    parser = argparse.ArgumentParser(description="Path Planning Pipeline")
    parser.add_argument("image", help="Path to the satellite / aerial image")
    parser.add_argument("--start", default="100,100", help="Start pixel x,y")
    parser.add_argument("--end", default="500,500", help="End pixel x,y")
    parser.add_argument(
        "--preprocessor",
        default="hsv_threshold",
        choices=["none"] + all_preprocessors,
        help="Segmentation method (default: hsv_threshold)",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        default=["a_star"],
        choices=AlgorithmFactory.available(),
    )
    parser.add_argument("--weighted", action="store_true", default=True)
    parser.add_argument("--no-weighted", dest="weighted", action="store_false")
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--robot-size", type=float, default=3.0)
    parser.add_argument("--bgr", default="84,1,68", help="Transitable BGR (preprocessor=none)")
    parser.add_argument("--weight-limit", type=int, default=100)
    parser.add_argument("--save-mask", action="store_true", help="Save segmentation mask")
    parser.add_argument("--output-dir", default="output")

    args = parser.parse_args()

    sx, sy = (int(v) for v in args.start.split(","))
    ex, ey = (int(v) for v in args.end.split(","))

    # ── 1. Ingest ─────────────────────────────────────────────────────
    print(f"Loading image: {args.image}")
    pixels = cv2.imread(args.image)
    if pixels is None:
        print(f"Error: could not load '{args.image}'")
        sys.exit(1)
    data = ImageData(source_path=args.image, pixels=pixels)
    print(f"  Image size: {pixels.shape[1]}×{pixels.shape[0]}")

    # ── 2. Preprocess + Parse graph ───────────────────────────────────
    if args.preprocessor == "none":
        bgr = tuple(int(v) for v in args.bgr.split(","))
        print(f"Skipping preprocessing — fixed BGR {bgr}")
        graph_parser = GraphParserFactory.create(
            "grid_image",
            transitable_color=bgr,
            robot_size=args.robot_size,
            connectivity=args.connectivity,
            weight_limit=args.weight_limit,
        )
        graph = graph_parser.parse(data)
    else:
        print(f"Preprocessing with '{args.preprocessor}' …")
        preprocessor = PreProcessorFactory.create(args.preprocessor)
        segmented = preprocessor.process(data)
        walkable_pct = segmented.get_walkable_mask().mean() * 100
        print(f"  Walkable pixels: {walkable_pct:.1f}%")

        if args.save_mask:
            mask_path = save_mask(segmented, args.output_dir)
            print(f"  Mask saved to: {mask_path}")

        print("Parsing segmented image to graph …")
        graph_parser = GraphParserFactory.create(
            "segmented_image",
            robot_size=args.robot_size,
            connectivity=args.connectivity,
            weight_limit=args.weight_limit,
        )
        graph = graph_parser.parse(segmented)

    print(f"  Nodes: {graph.num_nodes:,}  |  Edges: {graph.num_edges:,}")

    if graph.num_nodes == 0:
        print("Error: graph has zero nodes — no walkable area found.")
        sys.exit(1)

    # ── 3. Map pixel coords → node indices ────────────────────────────
    node_map = graph.metadata["node_index_map"]
    try:
        start_node = pixel_to_node(node_map, sx, sy)
        end_node = pixel_to_node(node_map, ex, ey)
    except ValueError as exc:
        print(f"Error: {exc}")
        print("Tip: pick start/end coordinates inside a walkable region.")
        sys.exit(1)

    print(f"  Start: pixel ({sx},{sy}) → node {start_node}")
    print(f"  End:   pixel ({ex},{ey}) → node {end_node}")

    # ── 4. Run algorithms + post-process ──────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_path = os.path.join(args.output_dir, f"{ts}.png")

    drawer = ImageDrawingPostProcessor(image=pixels, output_path=out_path)

    for algo_key in args.algorithm:
        algo = AlgorithmFactory.create(algo_key, weighted=args.weighted)
        print(f"\n{'=' * 50}")
        print(f"Running {algo.name()} …")

        path_result = algo.find_path(graph, start_node, end_node)

        if not path_result.is_valid:
            print("  ✗ No path found!")
            continue

        robot_path = drawer.process(path_result, graph)

        meta = path_result.metadata
        print(f"  Path nodes:      {path_result.length}")
        print(f"  Total cost:      {path_result.total_cost:.2f}")
        print(f"  Nodes explored:  {meta.get('nodes_explored', '?')}")
        print(f"  Time:            {meta.get('elapsed_seconds', 0):.4f}s")
        print(f"  Robot waypoints: {robot_path.num_waypoints}")
        print(f"  Real distance:   {robot_path.total_distance:.2f}")

    # ── 5. Save annotated image ───────────────────────────────────────
    saved = drawer.save()
    print(f"\nAnnotated image saved to: {saved}")


if __name__ == "__main__":
    main()
