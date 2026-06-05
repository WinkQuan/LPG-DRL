#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Plot multiple algorithm truth trajectories on one top-down Gazebo image."""

from __future__ import print_function

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NAV_SRC_DIR = SCRIPT_DIR.parent
VALIDATE_DIR = NAV_SRC_DIR / "Validate"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

try:
    import validation_scenes
except ImportError:
    validation_scenes = None

from draw_trajectory_on_video import (  # noqa: E402
    ALGORITHM_COLORS_RGB,
    DEFAULT_EXTENT,
    IMAGE_TOPIC,
    convert_image_message,
    save_bgr_frame_as_pdf,
    world_to_pixel,
)

SCENES = ("off_road", "flat_land", "without_obstacle")
DEFAULT_ALGORITHMS = ["LPG-DRL", "NPE-DRL", "D3QN-LfD", "BC", "DAgger", "APF"]
DEFAULT_FIGURE_WIDTH_IN = 7.16
DEFAULT_BACKGROUND_TIMEOUT = 5.0
LEGEND_AREA_FRACTION = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay first-success Gazebo truth trajectories for multiple algorithms "
            "on one fixed top-down camera background."
        )
    )
    parser.add_argument(
        "--scene",
        choices=SCENES,
        help="Validation scene to plot. If omitted, the same interactive selector as validation scripts is shown.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=DEFAULT_ALGORITHMS,
        default=DEFAULT_ALGORITHMS,
        help="Algorithms to include. Defaults to all non-ablation comparison algorithms.",
    )
    parser.add_argument(
        "--background",
        type=Path,
        help="Top-down background image. Defaults to Visualization/topdown_background_<scene>.png.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PDF path. Defaults to Visualization/topdown_trajectory_comparison_<scene>.pdf.",
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=DEFAULT_EXTENT,
        help=(
            "World-coordinate extent matching the top-down image. "
            "The landscape mapping uses world y for image x and world x for image y."
        ),
    )
    parser.add_argument(
        "--image-topic",
        default=IMAGE_TOPIC,
        help="Top-down camera topic used when the background image must be captured.",
    )
    parser.add_argument(
        "--background-timeout",
        type=float,
        default=DEFAULT_BACKGROUND_TIMEOUT,
        help="Seconds to wait for a top-down image when auto-capturing the background.",
    )
    parser.add_argument(
        "--figure-width-in",
        type=float,
        default=DEFAULT_FIGURE_WIDTH_IN,
        help="Figure width in inches. Height follows the background aspect ratio.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output raster DPI.")
    parser.add_argument("--line-width", type=float, default=1.1, help="Trajectory line width in points.")
    parser.add_argument("--point-size", type=float, default=18.0, help="Start marker and arrow-head size.")
    return parser.parse_args()


def default_background_path(scene):
    return SCRIPT_DIR / f"topdown_background_{scene}.png"


def default_output_path(scene):
    return SCRIPT_DIR / f"topdown_trajectory_comparison_{scene}.pdf"


def print_scene(scene):
    if validation_scenes is not None:
        validation_scenes.print_selected_scene(scene)
    else:
        print(f"Validation scene: {scene}")


def select_scene(scene_arg):
    if validation_scenes is not None:
        if scene_arg:
            scene = validation_scenes.normalize_scene(scene_arg)
            validation_scenes.print_selected_scene(scene)
            return scene
        return validation_scenes.select_scene()

    if scene_arg:
        return scene_arg

    print("Please choose the validation scene:")
    for index, scene in enumerate(SCENES, start=1):
        print(f"  {index}. {scene}")

    while True:
        try:
            value = input("Scene [1/2/3]: ")
        except EOFError as exc:
            raise RuntimeError("Validation scene is required. Pass --scene or provide interactive input.") from exc

        try:
            index = int(value)
        except ValueError:
            print("Please enter 1, 2, or 3.")
            continue

        if 1 <= index <= len(SCENES):
            scene = SCENES[index - 1]
            print_scene(scene)
            return scene

        print("Please enter 1, 2, or 3.")


def rgb_tuple_to_mpl(rgb):
    return tuple(channel / 255.0 for channel in rgb)


def color_for_algorithm(algorithm):
    return rgb_tuple_to_mpl(ALGORITHM_COLORS_RGB.get(algorithm, (51, 51, 51)))


def color_for_algorithm_bgr(algorithm):
    rgb = ALGORITHM_COLORS_RGB.get(algorithm, (51, 51, 51))
    return (rgb[2], rgb[1], rgb[0])


def safe_algorithm_name(algorithm):
    return algorithm.replace(" ", "_").replace("/", "_")


def trajectory_dir(scene, algorithm):
    return VALIDATE_DIR / scene / algorithm


def find_truth_csv(scene, algorithm):
    directory = trajectory_dir(scene, algorithm)
    if not directory.exists():
        return None

    preferred = directory / f"{safe_algorithm_name(algorithm)}_first_success_topdown_truth.csv"
    if preferred.exists():
        return preferred

    candidates = sorted(directory.glob("*first_success_topdown_truth.csv"))
    if candidates:
        return candidates[0]
    return None


def read_truth_csv(path):
    samples = []
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing = {"x", "y"} - fieldnames
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        for row in reader:
            try:
                samples.append((float(row["x"]), float(row["y"])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid x/y row in {path}") from exc

    if not samples:
        raise ValueError("empty trajectory")
    return samples


def load_trajectories(scene, algorithms):
    trajectories = []
    for algorithm in algorithms:
        path = find_truth_csv(scene, algorithm)
        if path is None:
            print(f"[MISSING] {algorithm}: no first-success truth CSV under {trajectory_dir(scene, algorithm)}")
            continue
        try:
            samples = read_truth_csv(path)
        except ValueError as exc:
            print(f"[INVALID] {algorithm}: {path} ({exc})")
            continue

        trajectories.append({"algorithm": algorithm, "path": path, "samples": samples})
        print(f"[FOUND]   {algorithm}: {path} ({len(samples)} samples)")

    return trajectories


def load_capture_dependencies():
    try:
        import cv2
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError(
            "Background image is missing and automatic capture requires ROS Python, "
            "cv_bridge, and OpenCV. Start Gazebo with top_down_camera or pass --background."
        ) from exc
    return cv2, rospy, CvBridge, Image


def capture_background(background_path, image_topic, timeout):
    cv2, rospy, CvBridge, Image = load_capture_dependencies()

    if not rospy.core.is_initialized():
        rospy.init_node("topdown_trajectory_comparison_background", anonymous=True, disable_signals=True)

    print(f"[CAPTURE] Waiting for one frame from {image_topic} ...")
    try:
        msg = rospy.wait_for_message(image_topic, Image, timeout=timeout)
    except rospy.ROSException as exc:
        raise RuntimeError(
            f"Could not capture top-down background from {image_topic}. "
            "Please start the selected Gazebo world or pass --background."
        ) from exc

    bridge = CvBridge()
    frame = convert_image_message(cv2, bridge, msg)
    background_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(background_path), frame):
        raise RuntimeError(f"Failed to write captured background image: {background_path}")
    print(f"[CAPTURE] Saved background image: {background_path}")


def ensure_background(background_path, image_topic, timeout):
    if background_path.exists():
        print(f"[BACKGROUND] {background_path}")
        return background_path

    print(f"[MISSING] Background image not found: {background_path}")
    capture_background(background_path, image_topic, timeout)
    return background_path


def trajectory_to_pixels(samples, extent, width, height):
    return [world_to_pixel(x, y, extent, width, height) for x, y in samples]


def arrow_tail_pixel(pixels, min_distance):
    if len(pixels) < 2:
        return pixels[0]

    end_x, end_y = pixels[-1]
    min_distance_sq = min_distance * min_distance
    for tail_x, tail_y in reversed(pixels[:-1]):
        dx = end_x - tail_x
        dy = end_y - tail_y
        if dx * dx + dy * dy >= min_distance_sq:
            return tail_x, tail_y
    return pixels[0]


def load_matplotlib_backend():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
    except ImportError:
        return None, None
    return plt, path_effects


def load_cv2_backend():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib, or OpenCV+numpy as a fallback. "
            "Install matplotlib in the active environment, or run with a Python that has cv2/numpy."
        ) from exc
    return cv2, np


def draw_trajectory_matplotlib(ax, path_effects, trajectory, extent, width, height, args):
    algorithm = trajectory["algorithm"]
    pixels = trajectory_to_pixels(trajectory["samples"], extent, width, height)
    xs = [point[0] for point in pixels]
    ys = [point[1] for point in pixels]
    color = color_for_algorithm(algorithm)
    line_width = args.line_width * (1.18 if algorithm == "LPG-DRL" else 1.0)

    (line,) = ax.plot(xs, ys, color=color, linewidth=line_width, label=algorithm, solid_capstyle="round")
    line.set_path_effects(
        [
            path_effects.Stroke(linewidth=line_width + 0.65, foreground="white", alpha=0.65),
            path_effects.Normal(),
        ]
    )

    start_x, start_y = pixels[0]
    end_x, end_y = pixels[-1]
    tail_x, tail_y = arrow_tail_pixel(pixels, max(12.0, args.point_size))
    ax.scatter(
        [start_x],
        [start_y],
        marker="o",
        s=args.point_size * 0.82,
        color=color,
        edgecolors="white",
        linewidths=0.65,
        zorder=5,
    )
    annotation = ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(tail_x, tail_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            linewidth=line_width,
            mutation_scale=max(7.0, args.point_size * 0.78),
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )
    if annotation.arrow_patch is not None:
        annotation.arrow_patch.set_path_effects(
            [
                path_effects.Stroke(linewidth=line_width + 0.65, foreground="white", alpha=0.65),
                path_effects.Normal(),
            ]
        )


def draw_comparison_matplotlib(plt, path_effects, background_path, output_path, trajectories, args):
    image = plt.imread(str(background_path))
    height, width = image.shape[:2]
    image_height = args.figure_width_in * float(height) / float(width)
    figure_height = image_height / (1.0 - LEGEND_AREA_FRACTION)
    fig, ax = plt.subplots(figsize=(args.figure_width_in, figure_height))
    ax.set_position([0.0, LEGEND_AREA_FRACTION, 1.0, 1.0 - LEGEND_AREA_FRACTION])

    ax.imshow(image, origin="upper")
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.axis("off")

    for trajectory in trajectories:
        draw_trajectory_matplotlib(ax, path_effects, trajectory, args.extent, width, height, args)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_ax = fig.add_axes([0.0, 0.0, 1.0, LEGEND_AREA_FRACTION])
        legend_ax.axis("off")
        legend = legend_ax.legend(
            handles,
            labels,
            loc="center",
            ncol=len(handles),
            frameon=False,
            fontsize=12,
            handlelength=1.8,
            columnspacing=0.82,
            handletextpad=0.45,
        )
        for line in legend.get_lines():
            line.set_linewidth(1.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, pad_inches=0.0)
    plt.close(fig)
    print(f"Saved top-down trajectory comparison: {output_path}")


def draw_trajectory_cv2(cv2, np, frame, trajectory, extent, width, height, args):
    algorithm = trajectory["algorithm"]
    pixels = trajectory_to_pixels(trajectory["samples"], extent, width, height)
    color = color_for_algorithm_bgr(algorithm)
    line_width = max(1, int(round(args.line_width * (1.18 if algorithm == "LPG-DRL" else 1.0))))
    points = np.array(pixels, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        frame, [points], isClosed=False, color=(255, 255, 255), thickness=line_width + 1, lineType=cv2.LINE_AA
    )
    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=line_width, lineType=cv2.LINE_AA)

    marker_size = max(10, int(round(args.point_size * 0.55)))
    start_pixel = pixels[0]
    end_pixel = pixels[-1]
    cv2.circle(frame, start_pixel, max(4, marker_size // 3), (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, start_pixel, max(3, marker_size // 4), color, thickness=-1, lineType=cv2.LINE_AA)
    tail_pixel = arrow_tail_pixel(pixels, max(18.0, args.point_size * 1.2))
    cv2.arrowedLine(
        frame,
        tail_pixel,
        end_pixel,
        (255, 255, 255),
        thickness=line_width + 2,
        line_type=cv2.LINE_AA,
        tipLength=0.38,
    )
    cv2.arrowedLine(
        frame,
        tail_pixel,
        end_pixel,
        color,
        thickness=line_width,
        line_type=cv2.LINE_AA,
        tipLength=0.38,
    )


def draw_cv2_legend(cv2, np, frame, trajectories):
    height, width = frame.shape[:2]
    legend_height = max(72, int(round(width * 0.035)))
    canvas = np.full((height + legend_height, width, 3), 255, dtype=np.uint8)
    canvas[:height, :, :] = frame

    labels = [trajectory["algorithm"] for trajectory in trajectories]
    columns = min(4, max(1, len(labels)))
    column_width = width // columns
    font_scale = max(0.58, width / 3000.0 * 0.72)
    line_y_start = height + int(legend_height * 0.38)
    text_y_offset = int(round(7 * font_scale))

    for index, label in enumerate(labels):
        row = index // columns
        col = index % columns
        x = col * column_width + 24
        y = line_y_start + row * int(legend_height * 0.42)
        color = color_for_algorithm_bgr(label)
        cv2.line(canvas, (x, y), (x + 44, y), color, thickness=4, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (x + 56, y + text_y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (35, 35, 35),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def draw_comparison_cv2(background_path, output_path, trajectories, args):
    cv2, np = load_cv2_backend()
    frame = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"OpenCV could not read background image: {background_path}")

    height, width = frame.shape[:2]
    for trajectory in trajectories:
        draw_trajectory_cv2(cv2, np, frame, trajectory, args.extent, width, height, args)

    canvas = draw_cv2_legend(cv2, np, frame, trajectories)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".pdf":
        save_bgr_frame_as_pdf(output_path, canvas, cv2)
    elif not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"OpenCV could not write output image: {output_path}")
    print(f"Saved top-down trajectory comparison: {output_path}")


def draw_comparison(background_path, output_path, trajectories, args):
    plt, path_effects = load_matplotlib_backend()
    if plt is not None:
        draw_comparison_matplotlib(plt, path_effects, background_path, output_path, trajectories, args)
        return

    print("[INFO] matplotlib is not available; falling back to OpenCV raster PDF output.")
    draw_comparison_cv2(background_path, output_path, trajectories, args)


def main():
    args = parse_args()
    scene = select_scene(args.scene)
    output_path = args.output or default_output_path(scene)
    background_path = args.background or default_background_path(scene)

    trajectories = load_trajectories(scene, args.algorithms)
    if not trajectories:
        raise RuntimeError(
            f"No usable first-success truth CSV files were found for scene `{scene}`. "
            "Run the validation scripts first."
        )

    background_path = ensure_background(background_path, args.image_topic, args.background_timeout)
    print(f"World extent: x=[{args.extent[0]}, {args.extent[1]}], y=[{args.extent[2]}, {args.extent[3]}]")
    print(f"Output: {output_path}")
    draw_comparison(background_path, output_path, trajectories, args)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        sys.stdout.flush()
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
