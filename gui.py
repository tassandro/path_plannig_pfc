"""
Path Planning Pipeline — Interactive Test GUI
==============================================

Launch with::

    python gui.py

Then open http://localhost:7860 in a browser.

Workflow:
  1. Pick a folder and select an image from the dropdown.
  2. Click on the displayed image to set **Start** (1st click) then **End**
     (2nd click).  Click again to reset to a new Start.
  3. Tune preprocessing / graph / algorithm parameters on the right panel.
  4. Press **Run Pipeline** and inspect the annotated output + mask + log.
"""

from __future__ import annotations

import os
import time
import traceback
from datetime import datetime

import cv2
import gradio as gr
import numpy as np

from algorithm import AlgorithmFactory
from data_ingestion import ImageData
from graph_parsing import GraphParserFactory
from post_processing import ImageDrawingPostProcessor
from preprocessing import PreProcessorFactory
from ros_bridge import RosBridgeConfig, RosBridgeFactory

DEFAULT_DATA_DIR = os.path.join("data", "rgb", "gloria")
OUTPUT_DIR = "output"


# ── Helpers ──────────────────────────────────────────────────────────────


def _scan_images(folder: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts)


def _load_bgr(folder: str, filename: str) -> np.ndarray | None:
    if not filename:
        return None
    return cv2.imread(os.path.join(folder, filename))


def _draw_markers(rgb: np.ndarray, start, end) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    r = max(10, min(w, h) // 200)
    thick = max(2, r // 4)
    font_scale = r / 12

    for pt, fill, label in [
        (start, (0, 220, 0), "START"),
        (end, (220, 0, 0), "END"),
    ]:
        if pt is None:
            continue
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(out, (x, y), r, fill, -1)
        cv2.circle(out, (x, y), r + 2, (255, 255, 255), thick)
        cv2.putText(
            out, label, (x + r + 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, fill, thick,
        )
    return out


# ── Event handlers ───────────────────────────────────────────────────────


def refresh_images(folder):
    choices = _scan_images(folder)
    return gr.update(choices=choices, value=None)


def on_image_selected(folder, filename):
    bgr = _load_bgr(folder, filename)
    if bgr is None:
        return None, None, None, "", "", ""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    return rgb, None, None, "", "", f"{w} x {h}"


def on_click(folder, filename, start_pt, end_pt, evt: gr.SelectData):
    bgr = _load_bgr(folder, filename)
    if bgr is None:
        return None, start_pt, end_pt, "", ""

    x, y = int(evt.index[0]), int(evt.index[1])

    if start_pt is None or end_pt is not None:
        start_pt = [x, y]
        end_pt = None
    else:
        end_pt = [x, y]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    display = _draw_markers(rgb, start_pt, end_pt)

    s_txt = f"{start_pt[0]}, {start_pt[1]}" if start_pt else ""
    e_txt = f"{end_pt[0]}, {end_pt[1]}" if end_pt else ""
    return display, start_pt, end_pt, s_txt, e_txt


def apply_manual_coords(folder, filename, start_str, end_str):
    start_pt = end_pt = None
    try:
        parts = [int(v.strip()) for v in start_str.split(",")]
        if len(parts) == 2:
            start_pt = parts
    except (ValueError, AttributeError):
        pass
    try:
        parts = [int(v.strip()) for v in end_str.split(",")]
        if len(parts) == 2:
            end_pt = parts
    except (ValueError, AttributeError):
        pass

    bgr = _load_bgr(folder, filename)
    if bgr is None:
        return None, start_pt, end_pt
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _draw_markers(rgb, start_pt, end_pt), start_pt, end_pt


def reset_points(folder, filename):
    bgr = _load_bgr(folder, filename)
    if bgr is None:
        return None, None, None, "", ""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, None, None, "", ""


# ── Pipeline runner ──────────────────────────────────────────────────────


def run_pipeline(
    folder, filename, start_pt, end_pt,
    preprocessor, algorithms, connectivity, weighted,
    robot_size, min_area, confidence, max_input_size, save_mask,
    publish_ros, ros_image_topic, ros_path_topic, ros_frame_id, resolution_m_per_px,
):
    if not filename:
        return None, None, "No image selected."
    if not start_pt or not end_pt:
        return None, None, "Set start and end points first (click on the image)."
    if not algorithms:
        return None, None, "Select at least one algorithm."

    lines: list[str] = []

    def log(msg: str):
        lines.append(msg)

    try:
        path = os.path.join(folder, filename)
        bgr = cv2.imread(path)
        if bgr is None:
            return None, None, f"Could not load '{path}'"

        h, w = bgr.shape[:2]
        sx, sy = int(start_pt[0]), int(start_pt[1])
        ex, ey = int(end_pt[0]), int(end_pt[1])

        log(f"Image: {filename} ({w} x {h})")
        log(f"Start: ({sx}, {sy})   End: ({ex}, {ey})")

        data = ImageData(source_path=path, pixels=bgr)

        # ── Preprocess + parse graph ─────────────────────────────────
        t0 = time.time()
        mask_rgb = None

        if preprocessor == "none":
            log("Preprocessor: none (fixed BGR)")
            gp = GraphParserFactory.create(
                "grid_image",
                transitable_color=(84, 1, 68),
                robot_size=robot_size,
                connectivity=int(connectivity),
                weight_limit=100,
            )
            graph = gp.parse(data)
        else:
            log(f"Preprocessor: {preprocessor}")
            kwargs: dict = {}
            if preprocessor == "segformer":
                kwargs = dict(
                    confidence_threshold=confidence,
                    max_input_size=int(max_input_size),
                    min_area=int(min_area),
                )
            elif preprocessor == "deeplabv3":
                kwargs = dict(
                    confidence_threshold=confidence,
                    max_input_size=int(max_input_size),
                )
            elif preprocessor == "hsv_threshold":
                kwargs = dict(min_area=int(min_area))

            pre = PreProcessorFactory.create(preprocessor, **kwargs)
            segmented = pre.process(data)

            walk_mask = segmented.get_walkable_mask()
            walkable_pct = walk_mask.mean() * 100
            log(f"Walkable: {walkable_pct:.1f}%")

            # Build an overlay visualisation of the mask
            mask_vis = np.zeros((*walk_mask.shape, 3), dtype=np.uint8)
            mask_vis[walk_mask] = [0, 200, 0]
            rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask_rgb = cv2.addWeighted(rgb_orig, 0.6, mask_vis, 0.4, 0)

            if save_mask:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                mp = os.path.join(OUTPUT_DIR, f"{ts}_mask.png")
                cv2.imwrite(mp, walk_mask.astype(np.uint8) * 255)
                log(f"Mask saved: {mp}")

            gp = GraphParserFactory.create(
                "segmented_image",
                robot_size=robot_size,
                connectivity=int(connectivity),
                weight_limit=100,
            )
            graph = gp.parse(segmented)

        t_parse = time.time() - t0
        log(f"Graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges ({t_parse:.1f}s)")

        if graph.num_nodes == 0:
            log("\nNo walkable area found — try different preprocessor settings.")
            return None, mask_rgb, "\n".join(lines)

        # ── Map pixels to nodes ──────────────────────────────────────
        node_map = graph.metadata["node_index_map"]
        si = int(node_map[sy, sx])
        ei = int(node_map[ey, ex])

        if si < 0:
            log(f"\nStart pixel ({sx}, {sy}) is not walkable. Click inside a green region.")
            return None, mask_rgb, "\n".join(lines)
        if ei < 0:
            log(f"\nEnd pixel ({ex}, {ey}) is not walkable. Click inside a green region.")
            return None, mask_rgb, "\n".join(lines)

        log(f"Mapped nodes: start={si}, end={ei}")

        # ── Run algorithms ───────────────────────────────────────────
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_path = os.path.join(OUTPUT_DIR, f"{ts}.png")
        drawer = ImageDrawingPostProcessor(image=bgr, output_path=out_path)
        last_robot_path = None

        for algo_key in algorithms:
            algo = AlgorithmFactory.create(algo_key, weighted=weighted)
            log(f"\n{'─' * 40}")
            log(f" {algo.name()}")
            log(f"{'─' * 40}")

            result = algo.find_path(graph, si, ei)

            if not result.is_valid:
                log("  No path found!")
                continue

            rp = drawer.process(result, graph)
            last_robot_path = rp
            meta = result.metadata
            log(f"  Path length:   {result.length} nodes")
            log(f"  Total cost:    {result.total_cost:.2f}")
            log(f"  Explored:      {meta.get('nodes_explored', '?')}")
            log(f"  Time:          {meta.get('elapsed_seconds', 0):.4f}s")
            log(f"  Waypoints:     {rp.num_waypoints}")
            log(f"  Real distance: {rp.total_distance:.2f}")

        saved = drawer.save()
        log(f"\nOutput saved: {saved}")

        if publish_ros:
            if last_robot_path is None:
                log("\nSkipping ROS publish — no algorithm produced a valid path.")
            else:
                log("\nPublishing to ROS 2 …")
                try:
                    ros_config = RosBridgeConfig(
                        image_topic=ros_image_topic or "/path_planning/annotated_image",
                        path_topic=ros_path_topic or "/path_planning/path",
                        frame_id=ros_frame_id or "map",
                        resolution_m_per_px=float(resolution_m_per_px or 1.0),
                    )
                    bridge = RosBridgeFactory.create("ros2", config=ros_config)
                    with bridge:
                        bridge.publish(image=drawer.result_image, robot_path=last_robot_path)
                    log(f"  Image published to: {ros_config.image_topic}")
                    log(f"  Path published to:  {ros_config.path_topic}")
                except (RuntimeError, ValueError) as exc:
                    log(f"  Could not publish to ROS: {exc}")

        result_bgr = cv2.imread(saved)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        return result_rgb, mask_rgb, "\n".join(lines)

    except Exception as exc:
        lines.append(f"\nError: {exc}")
        lines.append(traceback.format_exc())
        return None, None, "\n".join(lines)


# ── Build Gradio app ─────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    available_pre = PreProcessorFactory.available()
    default_pre = "segformer" if "segformer" in available_pre else (
        available_pre[0] if available_pre else "none"
    )

    with gr.Blocks(title="Path Planning Pipeline") as demo:
        gr.Markdown("# Path Planning Pipeline")
        gr.Markdown(
            "Select an image, click to set **start** (1st click, green) and "
            "**end** (2nd click, red), adjust parameters, then run."
        )

        start_state = gr.State(None)
        end_state = gr.State(None)

        with gr.Row(equal_height=False):
            # ── Left column: image ──
            with gr.Column(scale=3):
                with gr.Row():
                    folder_input = gr.Textbox(
                        value=DEFAULT_DATA_DIR, label="Image Folder", scale=3,
                    )
                    refresh_btn = gr.Button("Refresh", scale=1, min_width=80)

                image_dd = gr.Dropdown(
                    choices=_scan_images(DEFAULT_DATA_DIR),
                    label="Select Image",
                    interactive=True,
                )
                image_info = gr.Textbox(label="Dimensions", interactive=False, max_lines=1)

                image_display = gr.Image(
                    label="Click to place Start then End",
                    type="numpy",
                    height=550,
                )

                with gr.Row():
                    start_tb = gr.Textbox(
                        value="", label="Start (x, y)",
                        placeholder="click or type x, y", scale=2,
                    )
                    end_tb = gr.Textbox(
                        value="", label="End (x, y)",
                        placeholder="click or type x, y", scale=2,
                    )
                    reset_btn = gr.Button("Reset Points", variant="secondary", scale=1)

            # ── Right column: controls ──
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Preprocessing")
                pre_dd = gr.Dropdown(
                    choices=["none"] + available_pre,
                    value=default_pre,
                    label="Preprocessor",
                )
                confidence_sl = gr.Slider(
                    0.0, 1.0, 0.5, step=0.05,
                    label="Confidence Threshold",
                    info="Min softmax probability (segformer / deeplabv3)",
                )
                max_input_sl = gr.Slider(
                    256, 2048, 1024, step=128,
                    label="Max Input Size (px)",
                    info="Resize large images before inference",
                )
                min_area_sl = gr.Slider(
                    0, 10000, 1000, step=100,
                    label="Min Island Area (px)",
                    info="Connected components smaller than this are removed",
                )
                save_mask_cb = gr.Checkbox(True, label="Save mask to disk")

                gr.Markdown("### Graph Parsing")
                conn_radio = gr.Radio([4, 8], value=8, label="Connectivity")
                robot_sl = gr.Slider(
                    1.0, 20.0, 3.0, step=0.5, label="Robot Size (px)",
                )

                gr.Markdown("### Algorithm")
                algo_dd = gr.Dropdown(
                    choices=AlgorithmFactory.available(),
                    value=["a_star"],
                    multiselect=True,
                    label="Algorithms",
                )
                weighted_cb = gr.Checkbox(True, label="Weighted search")

                gr.Markdown("### ROS 2 (ground robot)")
                publish_ros_cb = gr.Checkbox(
                    False, label="Publish to ROS 2",
                    info="Sends the annotated image (sensor_msgs/Image) and "
                    "the path (nav_msgs/Path) to the ground robot",
                )
                ros_image_topic_tb = gr.Textbox(
                    value="/path_planning/annotated_image", label="Image Topic",
                )
                ros_path_topic_tb = gr.Textbox(
                    value="/path_planning/path", label="Path Topic",
                )
                ros_frame_tb = gr.Textbox(value="map", label="Frame ID")
                ros_resolution_sl = gr.Slider(
                    0.01, 5.0, 1.0, step=0.01, label="Resolution (m/px)",
                    info="Meters per pixel — converts the path to real-world "
                    "units so it lines up with the Gazebo/world frame",
                )

                run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

        gr.Markdown("---")
        gr.Markdown("### Results")

        with gr.Row():
            result_img = gr.Image(label="Path Output", type="numpy", height=500)
            mask_img = gr.Image(label="Walkability Mask Overlay", type="numpy", height=500)

        result_log = gr.Textbox(label="Pipeline Log", lines=18, max_lines=50)

        # ── Wire events ──────────────────────────────────────────────

        refresh_btn.click(refresh_images, [folder_input], [image_dd])
        folder_input.submit(refresh_images, [folder_input], [image_dd])

        image_dd.change(
            on_image_selected,
            [folder_input, image_dd],
            [image_display, start_state, end_state, start_tb, end_tb, image_info],
        )

        image_display.select(
            on_click,
            [folder_input, image_dd, start_state, end_state],
            [image_display, start_state, end_state, start_tb, end_tb],
        )

        start_tb.submit(
            apply_manual_coords,
            [folder_input, image_dd, start_tb, end_tb],
            [image_display, start_state, end_state],
        )
        end_tb.submit(
            apply_manual_coords,
            [folder_input, image_dd, start_tb, end_tb],
            [image_display, start_state, end_state],
        )

        reset_btn.click(
            reset_points,
            [folder_input, image_dd],
            [image_display, start_state, end_state, start_tb, end_tb],
        )

        run_btn.click(
            run_pipeline,
            [
                folder_input, image_dd, start_state, end_state,
                pre_dd, algo_dd, conn_radio, weighted_cb,
                robot_sl, min_area_sl, confidence_sl, max_input_sl, save_mask_cb,
                publish_ros_cb, ros_image_topic_tb, ros_path_topic_tb, ros_frame_tb,
                ros_resolution_sl,
            ],
            [result_img, mask_img, result_log],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(theme=gr.themes.Soft())
