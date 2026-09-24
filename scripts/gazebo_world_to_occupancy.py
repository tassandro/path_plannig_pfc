"""Generate a 2D top-down occupancy image from a Gazebo world file.

Lets you plan a path *before* the simulation runs, using the known layout of
a Gazebo world (e.g. ``clearpath_gz``'s ``warehouse.sdf``) instead of a real
aerial photo. The output PNG uses the same convention as
``main.py --preprocessor none --bgr ...``: a flat walkable colour with
obstacles painted black, so it can be fed straight into the existing
pipeline (``GraphParserFactory.create("grid_image", ...)``).

Handles, for every ``<model>`` defined inline in the world file and every
``<include>`` (resolved either as a local ``model://`` path or a cached Fuel
model under ``~/.ignition/fuel`` / ``~/.gz/fuel``):

- ``<box>`` / ``<cylinder>`` collision geometry — rasterized directly.
- ``<mesh><uri>....stl</uri></mesh>`` collision geometry — triangles are
  parsed from the STL and projected top-down; near-horizontal triangles
  (floors/ceilings, controlled by ``--wall-min-height``) are skipped so only
  wall-like geometry becomes an obstacle.
- Anything else (``<plane>``, non-STL meshes such as ``.dae``, unresolved
  includes) is skipped with a warning — it is not treated as an obstacle.

Only 2D (x, y, yaw) is used for rasterization; z/roll/pitch only matter for
deciding whether an STL triangle is "wall-like".

Usage::

    python scripts/gazebo_world_to_occupancy.py \\
        ~/clearpath_ws/src/clearpath_simulator/clearpath_gz/worlds/warehouse.sdf \\
        --resolution 0.05 --output data/rgb/warehouse_occupancy.png

The script prints the exact ``--resolution-m-per-px``, ``--origin-x-m`` and
``--origin-y-m`` values to pass to ``main.py --publish-ros`` (or the
matching gui.py fields) so the planned path lines up with the real Gazebo
world frame.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

_FUEL_ROOTS = [
    os.path.expanduser("~/.ignition/fuel"),
    os.path.expanduser("~/.gz/fuel"),
]


# ------------------------------------------------------------------ pose math

def parse_pose(text: str | None) -> np.ndarray:
    """Parse an SDF ``<pose>x y z roll pitch yaw</pose>`` into a 4x4 matrix."""
    if not text or not text.strip():
        vals = [0.0] * 6
    else:
        vals = [float(v) for v in text.split()]
        vals += [0.0] * (6 - len(vals))
    x, y, z, roll, pitch, yaw = vals
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    r = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = [x, y, z]
    return m


# ------------------------------------------------------------------ STL parsing

def read_stl_triangles(path: str) -> np.ndarray:
    """Return an (N, 3, 3) array of triangle vertices from an STL file."""
    with open(path, "rb") as f:
        data = f.read()

    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if 84 + count * 50 == len(data):
            tris = np.zeros((count, 3, 3), dtype=np.float64)
            offset = 84
            for i in range(count):
                v = struct.unpack("<9f", data[offset + 12: offset + 48])
                tris[i, 0] = v[0:3]
                tris[i, 1] = v[3:6]
                tris[i, 2] = v[6:9]
                offset += 50
            return tris

    # Fallback: ASCII STL.
    verts: list[list[list[float]]] = []
    tri: list[list[float]] = []
    for line in data.decode("ascii", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            _, x, y, z = line.split()
            tri.append([float(x), float(y), float(z)])
            if len(tri) == 3:
                verts.append(tri)
                tri = []
    return np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3, 3))


# ------------------------------------------------------------------ model resolution

def resolve_fuel_uri(uri: str, extra_roots: list[str]) -> str | None:
    """Resolve a ``fuel.gazebosim.org`` model URI to a cached local ``model.sdf``."""
    parts = [p for p in uri.strip().split("/") if p]
    try:
        idx = parts.index("models")
        owner, name = parts[idx - 1], parts[idx + 1]
    except (ValueError, IndexError):
        return None

    for root in [*extra_roots, *_FUEL_ROOTS]:
        base = os.path.join(root, "fuel.gazebosim.org", owner.lower(), "models", name.lower())
        if not os.path.isdir(base):
            continue
        versions = sorted((int(v) for v in os.listdir(base) if v.isdigit()), reverse=True)
        for v in versions:
            candidate = os.path.join(base, str(v), "model.sdf")
            if os.path.isfile(candidate):
                return candidate
    return None


def resolve_model_uri(uri: str, search_paths: list[str]) -> str | None:
    uri = uri.strip()
    if uri.startswith("model://"):
        name = uri[len("model://"):]
        for sp in search_paths:
            candidate = os.path.join(sp, name, "model.sdf")
            if os.path.isfile(candidate):
                return candidate
        return None
    if "fuel.gazebosim.org" in uri or "fuel.ignitionrobotics.org" in uri:
        return resolve_fuel_uri(uri, search_paths)
    return None


def load_model_root(path: str) -> ET.Element | None:
    root = ET.parse(path).getroot()
    return root if root.tag == "model" else root.find("model")


# ------------------------------------------------------------------ geometry extraction

def parse_model_geometries(model_el: ET.Element, world_xform: np.ndarray, base_dir: str):
    """Yield (kind, params, xform) for every collision geometry under a <model>."""
    model_xform = world_xform @ parse_pose(model_el.findtext("pose"))
    for link_el in model_el.findall("link"):
        link_xform = model_xform @ parse_pose(link_el.findtext("pose"))
        for coll_el in link_el.findall("collision"):
            coll_xform = link_xform @ parse_pose(coll_el.findtext("pose"))
            geom_el = coll_el.find("geometry")
            if geom_el is None:
                continue

            box_el = geom_el.find("box")
            if box_el is not None:
                size = [float(v) for v in box_el.findtext("size").split()]
                yield "box", size, coll_xform
                continue

            cyl_el = geom_el.find("cylinder")
            if cyl_el is not None:
                radius = float(cyl_el.findtext("radius"))
                yield "cylinder", [radius], coll_xform
                continue

            mesh_el = geom_el.find("mesh")
            if mesh_el is not None:
                uri = (mesh_el.findtext("uri") or "").strip()
                scale_txt = mesh_el.findtext("scale")
                scale = (
                    np.array([float(v) for v in scale_txt.split()])
                    if scale_txt else np.ones(3)
                )
                mesh_path = os.path.join(base_dir, uri)
                if uri.lower().endswith(".stl") and os.path.isfile(mesh_path):
                    yield "mesh", (mesh_path, scale), coll_xform
                else:
                    print(f"  ! skipping unsupported/missing mesh geometry: {uri}")
                continue
            # <plane>, <sphere>, etc. — not treated as an obstacle.


def collect_obstacles(world_path: str, search_paths: list[str]):
    tree = ET.parse(world_path)
    world_el = tree.getroot().find("world")
    if world_el is None:
        raise ValueError(f"'{world_path}' has no <world> element")
    base_dir = os.path.dirname(os.path.abspath(world_path))

    obstacles = []

    for model_el in world_el.findall("model"):
        obstacles.extend(parse_model_geometries(model_el, np.eye(4), base_dir))

    for include_el in world_el.findall("include"):
        uri = include_el.findtext("uri", "")
        pose = parse_pose(include_el.findtext("pose"))
        model_path = resolve_model_uri(uri, search_paths)
        if model_path is None:
            print(f"  ! could not resolve include '{uri.strip()}' — skipping (missing obstacle)")
            continue
        sub_model = load_model_root(model_path)
        if sub_model is None:
            continue
        obstacles.extend(
            parse_model_geometries(sub_model, pose, os.path.dirname(model_path))
        )

    return obstacles


# ------------------------------------------------------------------ rasterization

def rasterize(obstacles, resolution: float, bounds, wall_min_height: float):
    xmin, ymin, xmax, ymax = bounds
    width = max(1, int(math.ceil((xmax - xmin) / resolution)))
    height = max(1, int(math.ceil((ymax - ymin) / resolution)))
    canvas = np.ones((height, width), dtype=np.uint8)  # 1 = walkable

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (x - xmin) / resolution, height - (y - ymin) / resolution

    for kind, params, xform in obstacles:
        if kind == "box":
            sx, sy, _ = params
            local = np.array([
                [sx / 2, sy / 2, 0, 1],
                [sx / 2, -sy / 2, 0, 1],
                [-sx / 2, -sy / 2, 0, 1],
                [-sx / 2, sy / 2, 0, 1],
            ]).T
            world = xform @ local
            pts = np.array([to_px(x, y) for x, y in zip(world[0], world[1])], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts, 0)

        elif kind == "cylinder":
            radius = params[0]
            cx, cy = xform[0, 3], xform[1, 3]
            px, py = to_px(cx, cy)
            r_px = max(1, int(round(radius / resolution)))
            cv2.circle(canvas, (int(round(px)), int(round(py))), r_px, 0, -1)

        elif kind == "mesh":
            mesh_path, scale = params
            try:
                tris = read_stl_triangles(mesh_path)
            except OSError as exc:
                print(f"  ! could not read mesh '{mesh_path}': {exc}")
                continue
            if tris.size == 0:
                continue
            tris = tris * scale
            ones = np.ones((tris.shape[0], tris.shape[1], 1))
            tris_h = np.concatenate([tris, ones], axis=-1)
            tris_world = np.einsum("ij,nkj->nki", xform, tris_h)
            for tri in tris_world:
                if tri[:, 2].max() - tri[:, 2].min() < wall_min_height:
                    continue  # near-horizontal face (floor/ceiling) -> not an obstacle
                pts = np.array([to_px(v[0], v[1]) for v in tri], dtype=np.int32)
                cv2.fillConvexPoly(canvas, pts, 0)

    return canvas


def seal_exterior(canvas: np.ndarray) -> np.ndarray:
    """Mark any walkable area reachable from the image border as an obstacle.

    Without this, the auto-computed ``--margin`` around a building leaves an
    unbounded "outside" area marked walkable too (nothing paints it as an
    obstacle), so A* can shortcut straight through the exterior instead of
    staying inside the building's rooms/aisles. Assumes the outer wall mesh
    fully encloses the interior (no gaps) — if it has a real door/opening in
    its collision geometry, sealing would incorrectly cut off the interior
    too; check the output image if that happens and use --no-seal-exterior.
    """
    num_labels, labels = cv2.connectedComponents((canvas == 1).astype(np.uint8), connectivity=8)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border_labels.discard(0)  # label 0 is the (already-obstacle) background
    sealed = canvas.copy()
    for label in border_labels:
        sealed[labels == label] = 0
    return sealed


def compute_auto_bounds(obstacles, margin: float):
    xs, ys = [], []
    for kind, params, xform in obstacles:
        if kind == "box":
            sx, sy, _ = params
            half = math.hypot(sx, sy) / 2
        elif kind == "cylinder":
            half = params[0]
        else:  # mesh
            mesh_path, scale = params
            try:
                tris = read_stl_triangles(mesh_path) * scale
            except OSError:
                continue
            if tris.size == 0:
                continue
            ones = np.ones((tris.shape[0], tris.shape[1], 1))
            tris_h = np.concatenate([tris, ones], axis=-1)
            world = np.einsum("ij,nkj->nki", xform, tris_h)
            xs.extend(world[..., 0].ravel().tolist())
            ys.extend(world[..., 1].ravel().tolist())
            continue
        cx, cy = xform[0, 3], xform[1, 3]
        xs.extend([cx - half, cx + half])
        ys.extend([cy - half, cy + half])

    if not xs:
        raise ValueError("No obstacles found and no --bounds given — cannot infer canvas size")
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


# ------------------------------------------------------------------ CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("world", help="Path to the top-level Gazebo .world/.sdf file")
    parser.add_argument("--resolution", type=float, default=0.05, help="Meters per pixel (default: 0.05)")
    parser.add_argument("--bounds", default=None, help="xmin,ymin,xmax,ymax in meters (default: auto, from obstacle extents + --margin)")
    parser.add_argument("--margin", type=float, default=5.0, help="Extra free space (meters) around auto-computed bounds")
    parser.add_argument("--wall-min-height", type=float, default=0.3, help="Min vertical extent (m) for a mesh triangle to count as a wall (filters out floor/ceiling faces)")
    parser.add_argument("--walkable-bgr", default="84,1,68", help="BGR colour for walkable area (matches 'main.py --preprocessor none --bgr')")
    parser.add_argument("--model-search-path", action="append", default=[], help="Extra directory to resolve model:// URIs (repeatable)")
    parser.add_argument("--no-seal-exterior", dest="seal_exterior", action="store_false", default=True,
                         help="Don't block off the area outside the outer wall outline (default: sealed, so A* can't shortcut through the unbounded exterior margin)")
    parser.add_argument("--output", default=None, help="Output PNG path (default: <world_name>_occupancy.png)")
    args = parser.parse_args()

    search_paths = list(args.model_search_path)
    for env_var in ("GZ_SIM_RESOURCE_PATH", "GAZEBO_MODEL_PATH"):
        value = os.environ.get(env_var, "")
        search_paths.extend(p for p in value.split(os.pathsep) if p)

    print(f"Parsing world: {args.world}")
    obstacles = collect_obstacles(args.world, search_paths)
    print(f"  Obstacles found: {len(obstacles)}")

    if args.bounds:
        bounds = tuple(float(v) for v in args.bounds.split(","))
    else:
        bounds = compute_auto_bounds(obstacles, args.margin)
    xmin, ymin, xmax, ymax = bounds
    print(f"  Bounds (m): x=[{xmin:.2f}, {xmax:.2f}]  y=[{ymin:.2f}, {ymax:.2f}]")

    canvas = rasterize(obstacles, args.resolution, bounds, args.wall_min_height)
    if args.seal_exterior:
        canvas = seal_exterior(canvas)

    walkable_bgr = tuple(int(v) for v in args.walkable_bgr.split(","))
    image = np.zeros((*canvas.shape, 3), dtype=np.uint8)
    image[canvas == 1] = walkable_bgr

    output = args.output or f"{os.path.splitext(os.path.basename(args.world))[0]}_occupancy.png"
    cv2.imwrite(output, image)

    origin_x_m, origin_y_m = xmin, ymax
    print(f"\nSaved: {output}  ({image.shape[1]}x{image.shape[0]} px)")
    print("\nUse these values so the planned path lines up with the Gazebo world frame:")
    print(f"  --resolution-m-per-px {args.resolution}")
    print(f"  --origin-x-m {origin_x_m:.3f}")
    print(f"  --origin-y-m {origin_y_m:.3f}")
    print(f"\nExample:\n  python main.py {output} --preprocessor none --bgr {args.walkable_bgr} \\\n"
          f"    --start <sx>,<sy> --end <ex>,<ey> --algorithm a_star --publish-ros \\\n"
          f"    --resolution-m-per-px {args.resolution} --origin-x-m {origin_x_m:.3f} --origin-y-m {origin_y_m:.3f}")


if __name__ == "__main__":
    main()
