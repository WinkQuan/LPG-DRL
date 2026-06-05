#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Plot algorithm trajectory comparisons from validation outputs."""

from __future__ import print_function

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
except ImportError:
    plt = None
    Line2D = None
    Circle = None


SCRIPT_DIR = Path(__file__).resolve().parent
NAV_SRC_DIR = SCRIPT_DIR.parent
WORKSPACE_SRC_DIR = NAV_SRC_DIR.parents[1]
VALIDATE_DIR = NAV_SRC_DIR / "Validate"
WORLD_DIR = WORKSPACE_SRC_DIR / "mymodel_gazebo" / "worlds"

if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

try:
    import algorithm_comparison
    import validation_scenes
except ImportError:
    algorithm_comparison = None
    validation_scenes = None


FALLBACK_SCENES = ["off_road", "flat_land", "without_obstacle"]
SCENE_WORLD_FILES = {
    "off_road": WORLD_DIR / "off-road_world.world",
    "flat_land": WORLD_DIR / "flat_land_world.world",
    "without_obstacle": WORLD_DIR / "off-road_world_without_obstacle.world",
}

FALLBACK_ALGORITHM_ORDER = ["LPG-DRL", "NPE-DRL", "D3QN", "PPO", "BC", "DAgger", "APF"]
ALGORITHM_ORDER = (
    [spec["algorithm"] for spec in algorithm_comparison.ALGORITHM_SPECS]
    if algorithm_comparison is not None
    else FALLBACK_ALGORITHM_ORDER
)

COLORS = {
    "LPG-DRL": "#d62728",
    "NPE-DRL": "#ff7f0e",
    "D3QN": "#1f77b4",
    "PPO": "#9467bd",
    "BC": "#2ca02c",
    "DAgger": "#8c564b",
    "APF": "#7f7f7f",
}

DEFAULT_OBSTACLE_RADIUS = 0.5
DEFAULT_FIGURE_WIDTH_IN = 3.5
DEFAULT_FIGURE_HEIGHT_IN = 4.2


def parse_args():
    parser = argparse.ArgumentParser(description="Plot validation trajectories for multiple algorithms in one scene.")
    parser.add_argument(
        "--scene",
        choices=scene_choices(),
        help="Validation scene to plot. If omitted, an interactive scene selector is shown.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHM_ORDER,
        default=ALGORITHM_ORDER,
        help="Algorithms to include. Defaults to all comparison algorithms.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path. Defaults to Visualization/trajectory_<scene>.pdf.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "matplotlib", "builtin"],
        default="auto",
        help="Plot backend. `auto` uses matplotlib when available and otherwise falls back to builtin PDF.",
    )
    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=DEFAULT_OBSTACLE_RADIUS,
        help="Radius in meters used to draw unit_cylinder obstacles.",
    )
    parser.add_argument(
        "--figure-width-in",
        type=float,
        default=DEFAULT_FIGURE_WIDTH_IN,
        help="Output figure width in inches.",
    )
    parser.add_argument(
        "--figure-height-in",
        type=float,
        default=DEFAULT_FIGURE_HEIGHT_IN,
        help="Output figure height in inches.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Matplotlib raster DPI for non-PDF outputs.")
    return parser.parse_args()


def scene_choices():
    if validation_scenes is not None:
        return sorted(validation_scenes.SCENES.keys())
    return FALLBACK_SCENES


def normalize_scene(scene):
    if validation_scenes is not None:
        return validation_scenes.normalize_scene(scene)
    if scene not in FALLBACK_SCENES:
        raise ValueError(f"Unknown validation scene `{scene}`. Valid values: {', '.join(FALLBACK_SCENES)}")
    return scene


def select_scene(scene_arg):
    if scene_arg:
        scene = normalize_scene(scene_arg)
        if validation_scenes is not None:
            validation_scenes.print_selected_scene(scene)
        else:
            print(f"Validation scene: {scene}")
        return scene

    if validation_scenes is not None:
        return validation_scenes.select_scene()

    print("Please choose the validation scene:")
    for index, scene in enumerate(FALLBACK_SCENES, start=1):
        print(f"  {index}. {scene}")
    while True:
        value = input("Scene [1/2/3]: ")
        try:
            index = int(value)
        except ValueError:
            print("Please enter 1, 2, or 3.")
            continue
        if 1 <= index <= len(FALLBACK_SCENES):
            return FALLBACK_SCENES[index - 1]
        print("Please enter 1, 2, or 3.")


def default_output_path(scene):
    return SCRIPT_DIR / f"trajectory_{scene}.pdf"


def choose_backend(requested_backend):
    if requested_backend == "matplotlib":
        if plt is None:
            raise RuntimeError("matplotlib backend requested, but matplotlib is not installed.")
        return "matplotlib"
    if requested_backend == "builtin":
        return "builtin"
    return "matplotlib" if plt is not None else "builtin"


def trajectory_dir(scene, algorithm):
    return VALIDATE_DIR / scene / algorithm


def find_trajectory_file(scene, algorithm):
    directory = trajectory_dir(scene, algorithm)
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("*trajectory.txt"))
    if not candidates:
        return None
    return candidates[0]


def read_trajectory(path):
    if np is not None:
        data = np.loadtxt(str(path), dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            raise ValueError(f"Trajectory file must contain at least x and y columns: {path}")
        return [(float(row[0]), float(row[1]), float(row[2]) if data.shape[1] > 2 else 0.0) for row in data]

    points = []
    with path.open("r") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid trajectory row {line_number} in {path}: {line.rstrip()}")
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2]) if len(parts) > 2 else 0.0
            points.append((x, y, z))
    if not points:
        raise ValueError(f"Trajectory file is empty: {path}")
    return points


def load_trajectories(scene, algorithms):
    trajectories = []
    for algorithm in algorithms:
        path = find_trajectory_file(scene, algorithm)
        if path is None:
            print(f"[MISSING] {algorithm}: no trajectory file under {trajectory_dir(scene, algorithm)}")
            continue
        try:
            points = read_trajectory(path)
        except Exception as err:
            print(f"[INVALID] {algorithm}: {path} ({err})")
            continue
        trajectories.append({"algorithm": algorithm, "path": path, "points": points})
        print(f"[FOUND]   {algorithm}: {path} ({len(points)} points)")
    return trajectories


def parse_pose_text(pose_text):
    if not pose_text:
        return None
    values = pose_text.split()
    if len(values) < 2:
        return None
    return float(values[0]), float(values[1])


def parse_world_elements(scene):
    world_file = SCENE_WORLD_FILES[scene]
    obstacles = []
    goal = None

    if not world_file.exists():
        print(f"[WARNING] World file not found: {world_file}")
        return obstacles, goal

    try:
        tree = ET.parse(str(world_file))
    except ET.ParseError as err:
        print(f"[WARNING] Could not parse world file {world_file}: {err}")
        return obstacles, goal

    for include in tree.findall(".//include"):
        name = (include.findtext("name") or "").strip()
        uri = (include.findtext("uri") or "").strip()
        pose = parse_pose_text(include.findtext("pose"))
        if pose is None:
            continue

        if name.startswith("unit_cylinder") or uri == "model://unit_cylinder":
            obstacles.append({"name": name or "unit_cylinder", "x": pose[0], "y": pose[1]})
        elif name == "Oak tree":
            goal = {"name": name, "x": pose[0], "y": pose[1]}
        elif name == "unit_box_real" and goal is None:
            goal = {"name": name, "x": pose[0], "y": pose[1]}

    return obstacles, goal


def point_bounds(trajectories, obstacles, goal, obstacle_radius):
    xs = []
    ys = []

    for trajectory in trajectories:
        for x, y, _ in trajectory["points"]:
            xs.append(x)
            ys.append(y)

    for obstacle in obstacles:
        xs.extend([obstacle["x"] - obstacle_radius, obstacle["x"] + obstacle_radius])
        ys.extend([obstacle["y"] - obstacle_radius, obstacle["y"] + obstacle_radius])

    if goal is not None:
        xs.append(goal["x"])
        ys.append(goal["y"])

    if not xs or not ys:
        raise RuntimeError("No trajectory points were available to determine plot bounds.")

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    margin = max(x_span, y_span) * 0.08

    return x_min - margin, x_max + margin, y_min - margin, y_max + margin


def equalized_bounds(bounds, plot_aspect):
    x_min, x_max, y_min, y_max = bounds
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)
    data_aspect = x_span / y_span

    if data_aspect > plot_aspect:
        y_span = x_span / plot_aspect
    else:
        x_span = y_span * plot_aspect

    return (
        x_center - x_span / 2.0,
        x_center + x_span / 2.0,
        y_center - y_span / 2.0,
        y_center + y_span / 2.0,
    )


def scene_title(scene):
    if validation_scenes is not None:
        return f"Trajectory Comparison - {validation_scenes.scene_display_name(scene)}"
    title_lookup = {
        "off_road": "Trajectory Comparison - Off Road",
        "flat_land": "Trajectory Comparison - Flat Land",
        "without_obstacle": "Trajectory Comparison - Without Obstacle",
    }
    return title_lookup.get(scene, f"Trajectory Comparison - {scene}")


def draw_matplotlib(output_path, scene, trajectories, obstacles, goal, bounds, obstacle_radius, args):
    if plt is None:
        raise RuntimeError("matplotlib is not installed.")

    fig, ax = plt.subplots(figsize=(args.figure_width_in, args.figure_height_in))
    ax.set_title(scene_title(scene), fontsize=9)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    obstacle_label_used = False
    for obstacle in obstacles:
        patch = Circle(
            (obstacle["x"], obstacle["y"]),
            obstacle_radius,
            facecolor="#9e9e9e",
            edgecolor="#5f5f5f",
            alpha=0.45,
            linewidth=0.6,
            label="Obstacle" if not obstacle_label_used else None,
        )
        ax.add_patch(patch)
        obstacle_label_used = True

    for trajectory in trajectories:
        algorithm = trajectory["algorithm"]
        points = trajectory["points"]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        color = COLORS.get(algorithm, "#333333")
        line_width = 2.0 if algorithm == "LPG-DRL" else 1.35
        ax.plot(xs, ys, color=color, linewidth=line_width, label=algorithm)
        ax.scatter(xs[0], ys[0], marker="^", s=22, color=color, edgecolors="black", linewidths=0.35, zorder=4)
        ax.scatter(xs[-1], ys[-1], marker="o", s=17, color=color, edgecolors="black", linewidths=0.35, zorder=4)

    if goal is not None:
        ax.scatter(
            goal["x"],
            goal["y"],
            marker="*",
            s=95,
            color="#111111",
            edgecolors="white",
            linewidths=0.45,
            label="Goal",
            zorder=5,
        )

    x_min, x_max, y_min, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    extra_handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor="black", label="Start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", label="End"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(extra_handles)
    labels.extend(["Start", "End"])
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False, fontsize=6)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def pdf_escape(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_number(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def pdf_text(x, y, text, size=7, anchor="start", bold=False, angle=0):
    approx_width = len(str(text)) * size * 0.5
    if angle == 90:
        if anchor == "middle":
            y -= approx_width / 2
        elif anchor == "end":
            y -= approx_width
    else:
        if anchor == "middle":
            x -= approx_width / 2
        elif anchor == "end":
            x -= approx_width

    font = "F2" if bold else "F1"
    escaped = pdf_escape(text)
    if angle == 90:
        matrix = f"0 1 -1 0 {pdf_number(x)} {pdf_number(y)}"
    else:
        matrix = f"1 0 0 1 {pdf_number(x)} {pdf_number(y)}"
    return f"BT /{font} {pdf_number(size)} Tf {matrix} Tm ({escaped}) Tj ET\n"


def pdf_line(x1, y1, x2, y2, width=0.5, color=(0.0, 0.0, 0.0)):
    r, g, b = color
    return (
        f"{pdf_number(r)} {pdf_number(g)} {pdf_number(b)} RG {pdf_number(width)} w "
        f"{pdf_number(x1)} {pdf_number(y1)} m {pdf_number(x2)} {pdf_number(y2)} l S\n"
    )


def pdf_polyline(points, width=0.6, color=(0.0, 0.0, 0.0)):
    if not points:
        return ""
    r, g, b = color
    commands = [f"{pdf_number(r)} {pdf_number(g)} {pdf_number(b)} RG {pdf_number(width)} w"]
    commands.append(f"{pdf_number(points[0][0])} {pdf_number(points[0][1])} m")
    for x, y in points[1:]:
        commands.append(f"{pdf_number(x)} {pdf_number(y)} l")
    commands.append("S")
    return "\n".join(commands) + "\n"


def pdf_rect(x, y, width, height):
    return f"0 0 0 RG 0.5 w {pdf_number(x)} {pdf_number(y)} " f"{pdf_number(width)} {pdf_number(height)} re S\n"


def pdf_circle(x, y, radius, stroke=(0.0, 0.0, 0.0), fill=None, width=0.5):
    segments = 24
    points = [
        (
            x + math.cos(2.0 * math.pi * index / segments) * radius,
            y + math.sin(2.0 * math.pi * index / segments) * radius,
        )
        for index in range(segments)
    ]
    stroke_cmd = f"{pdf_number(stroke[0])} {pdf_number(stroke[1])} {pdf_number(stroke[2])} RG"
    fill_cmd = ""
    if fill is not None:
        fill_cmd = f"{pdf_number(fill[0])} {pdf_number(fill[1])} {pdf_number(fill[2])} rg "
    commands = [f"{fill_cmd}{stroke_cmd} {pdf_number(width)} w"]
    commands.append(f"{pdf_number(points[0][0])} {pdf_number(points[0][1])} m")
    for px, py in points[1:]:
        commands.append(f"{pdf_number(px)} {pdf_number(py)} l")
    commands.append("h")
    commands.append("B" if fill is not None else "S")
    return "\n".join(commands) + "\n"


def pdf_triangle(x, y, radius, stroke=(0.0, 0.0, 0.0), fill=(1.0, 1.0, 1.0), width=0.5):
    points = [(x, y + radius), (x - radius * 0.9, y - radius * 0.75), (x + radius * 0.9, y - radius * 0.75)]
    return pdf_polygon(points, stroke=stroke, fill=fill, width=width)


def pdf_star(x, y, radius, stroke=(0.0, 0.0, 0.0), fill=(0.0, 0.0, 0.0), width=0.5):
    points = []
    for index in range(10):
        angle = math.pi / 2.0 + index * math.pi / 5.0
        current_radius = radius if index % 2 == 0 else radius * 0.42
        points.append((x + math.cos(angle) * current_radius, y + math.sin(angle) * current_radius))
    return pdf_polygon(points, stroke=stroke, fill=fill, width=width)


def pdf_polygon(points, stroke=(0.0, 0.0, 0.0), fill=None, width=0.5):
    stroke_cmd = f"{pdf_number(stroke[0])} {pdf_number(stroke[1])} {pdf_number(stroke[2])} RG"
    fill_cmd = ""
    if fill is not None:
        fill_cmd = f"{pdf_number(fill[0])} {pdf_number(fill[1])} {pdf_number(fill[2])} rg "
    commands = [f"{fill_cmd}{stroke_cmd} {pdf_number(width)} w"]
    commands.append(f"{pdf_number(points[0][0])} {pdf_number(points[0][1])} m")
    for px, py in points[1:]:
        commands.append(f"{pdf_number(px)} {pdf_number(py)} l")
    commands.append("h")
    commands.append("B" if fill is not None else "S")
    return "\n".join(commands) + "\n"


def write_pdf(output_path, content, width_pt, height_pt):
    stream = content.encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pdf_number(width_pt)} {pdf_number(height_pt)}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii"))
        chunks.append(obj)
        chunks.append(b"\nendobj\n")

    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f\n")
    for offset in offsets[1:]:
        chunks.append(f"{offset:010d} 00000 n\n".encode("ascii"))
    chunks.append(
        (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n" f"startxref\n{xref_offset}\n%%EOF\n").encode("ascii")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"".join(chunks))


def transform_point(x, y, bounds, plot_box):
    x_min, x_max, y_min, y_max = bounds
    left, bottom, width, height = plot_box
    px = left + (x - x_min) / max(x_max - x_min, 1e-8) * width
    py = bottom + (y - y_min) / max(y_max - y_min, 1e-8) * height
    return px, py


def builtin_ticks(min_value, max_value, count=6):
    if count <= 1:
        return [min_value]
    return [min_value + (max_value - min_value) * index / (count - 1) for index in range(count)]


def draw_builtin_pdf(output_path, scene, trajectories, obstacles, goal, bounds, obstacle_radius, args):
    if output_path.suffix.lower() != ".pdf":
        raise RuntimeError("The builtin backend only supports PDF output paths.")

    width = args.figure_width_in * 72.0
    height = args.figure_height_in * 72.0
    plot_left = 38.0
    plot_bottom = 48.0
    plot_top_reserved = 62.0
    plot_right_margin = 13.0
    plot_box = (plot_left, plot_bottom, width - plot_left - plot_right_margin, height - plot_bottom - plot_top_reserved)

    x_min, x_max, y_min, y_max = bounds
    content = []
    content.append(pdf_text(width / 2, height - 13, scene_title(scene), size=7.2, anchor="middle", bold=True))

    legend_columns = 3
    legend_column_width = plot_box[2] / legend_columns
    legend_y = height - 29
    legend_items = [trajectory["algorithm"] for trajectory in trajectories]
    if obstacles:
        legend_items.append("Obstacle")
    if goal is not None:
        legend_items.append("Goal")
    legend_items.extend(["Start", "End"])

    for index, label in enumerate(legend_items):
        x = plot_left + (index % legend_columns) * legend_column_width
        y = legend_y - (index // legend_columns) * 9.8
        if label in COLORS:
            color = hex_to_rgb(COLORS[label])
            content.append(pdf_line(x, y, x + 10, y, width=1.4 if label == "LPG-DRL" else 0.9, color=color))
        elif label == "Obstacle":
            content.append(pdf_circle(x + 5, y, 2.5, stroke=(0.45, 0.45, 0.45), fill=(0.75, 0.75, 0.75)))
        elif label == "Goal":
            content.append(pdf_star(x + 5, y, 3.5))
        elif label == "Start":
            content.append(pdf_triangle(x + 5, y, 3.3))
        elif label == "End":
            content.append(pdf_circle(x + 5, y, 2.5, stroke=(0.0, 0.0, 0.0), fill=(1.0, 1.0, 1.0)))
        content.append(pdf_text(x + 13, y - 2.0, label, size=5.3))

    grid_color = (0.88, 0.88, 0.88)
    for tick in builtin_ticks(x_min, x_max):
        x, _ = transform_point(tick, y_min, bounds, plot_box)
        content.append(pdf_line(x, plot_bottom, x, plot_bottom + plot_box[3], width=0.25, color=grid_color))
        content.append(pdf_text(x, plot_bottom - 10, f"{tick:.0f}", size=5.2, anchor="middle"))

    for tick in builtin_ticks(y_min, y_max):
        _, y = transform_point(x_min, tick, bounds, plot_box)
        content.append(pdf_line(plot_left, y, plot_left + plot_box[2], y, width=0.25, color=grid_color))
        content.append(pdf_text(plot_left - 3.5, y - 1.8, f"{tick:.0f}", size=5.2, anchor="end"))

    content.append(pdf_rect(plot_left, plot_bottom, plot_box[2], plot_box[3]))
    content.append(pdf_text(width / 2, 10.0, "X (m)", size=6.1, anchor="middle"))
    content.append(pdf_text(12.5, plot_bottom + plot_box[3] / 2.0, "Y (m)", size=6.1, anchor="middle", angle=90))

    obstacle_radius_pt = obstacle_radius / max(x_max - x_min, 1e-8) * plot_box[2]
    for obstacle in obstacles:
        x, y = transform_point(obstacle["x"], obstacle["y"], bounds, plot_box)
        content.append(
            pdf_circle(
                x,
                y,
                obstacle_radius_pt,
                stroke=(0.45, 0.45, 0.45),
                fill=(0.75, 0.75, 0.75),
                width=0.4,
            )
        )

    for trajectory in trajectories:
        algorithm = trajectory["algorithm"]
        color = hex_to_rgb(COLORS.get(algorithm, "#333333"))
        points = [transform_point(x, y, bounds, plot_box) for x, y, _ in trajectory["points"]]
        content.append(pdf_polyline(points, width=0.9 if algorithm == "LPG-DRL" else 0.6, color=color))
        sx, sy = points[0]
        ex, ey = points[-1]
        content.append(pdf_triangle(sx, sy, 2.7, stroke=(0.0, 0.0, 0.0), fill=color, width=0.35))
        content.append(pdf_circle(ex, ey, 2.2, stroke=(0.0, 0.0, 0.0), fill=color, width=0.35))

    if goal is not None:
        x, y = transform_point(goal["x"], goal["y"], bounds, plot_box)
        content.append(pdf_star(x, y, 4.2, stroke=(0.0, 0.0, 0.0), fill=(0.0, 0.0, 0.0), width=0.4))

    write_pdf(output_path, "".join(content), width, height)


def main():
    args = parse_args()
    scene = select_scene(args.scene)
    output_path = args.output or default_output_path(scene)
    backend = choose_backend(args.backend)

    trajectories = load_trajectories(scene, args.algorithms)
    if not trajectories:
        raise RuntimeError(
            f"No usable trajectory files were found for scene `{scene}`. Run validation before plotting."
        )

    obstacles, goal = parse_world_elements(scene)
    base_bounds = point_bounds(trajectories, obstacles, goal, args.obstacle_radius)
    plot_aspect = args.figure_width_in / max(args.figure_height_in, 1e-8)
    bounds = equalized_bounds(base_bounds, plot_aspect)

    print(f"Backend: {backend}")
    print(f"Scene: {scene}")
    print(f"Obstacles: {len(obstacles)}")
    print(f"Output: {output_path}")

    if backend == "matplotlib":
        draw_matplotlib(output_path, scene, trajectories, obstacles, goal, bounds, args.obstacle_radius, args)
    else:
        draw_builtin_pdf(output_path, scene, trajectories, obstacles, goal, bounds, args.obstacle_radius, args)

    print(f"Saved trajectory comparison to: {output_path}")


if __name__ == "__main__":
    main()
