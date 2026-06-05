#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Draw Gazebo truth trajectories on fixed top-down validation videos."""

from __future__ import print_function

import argparse
import csv
import math
import re
import sys
import zlib
from bisect import bisect_right
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NAV_SRC_DIR = SCRIPT_DIR.parent

IMAGE_TOPIC = "/top_down_camera/image_raw"
STATE_TOPIC = "/gazebo/model_states"
DEFAULT_EXTENT = (-8.3043, 8.3043, -16.5205, 16.5205)
SCENES = ("off_road", "flat_land", "without_obstacle")

ALGORITHM_COLORS_RGB = {
    "LPG-DRL": (214, 39, 40),
    "NPE-DRL": (255, 127, 14),
    "D3QN": (31, 119, 180),
    "D3QN-LfD": (31, 119, 180),
    "PPO": (148, 103, 189),
    "BC": (44, 160, 44),
    "DAgger": (140, 86, 75),
    "APF": (23, 190, 207),
}

if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

try:
    import validation_scenes
except ImportError:
    validation_scenes = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read a rosbag containing a fixed top-down image topic and /gazebo/model_states, "
            "then overlay the true mycar trajectory onto the video."
        )
    )
    parser.add_argument("--bag", type=Path, required=True, help="Input rosbag path.")
    parser.add_argument(
        "--scene",
        choices=SCENES,
        default="off_road",
        help="Scene label used in default output names.",
    )
    parser.add_argument(
        "--algorithm",
        default="LPG-DRL",
        help="Algorithm label used in annotations and default output names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path. Defaults to Visualization/trajectory_video_<scene>_<algorithm>.mp4.",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        help="Final overlaid frame PDF path. Defaults to the MP4 path with .pdf suffix.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        help="Truth trajectory CSV path. Defaults to the MP4 path with _truth.csv suffix.",
    )
    parser.add_argument(
        "--image-topic",
        default=IMAGE_TOPIC,
        help="Top-down camera image topic in the rosbag.",
    )
    parser.add_argument(
        "--state-topic",
        default=STATE_TOPIC,
        help="Gazebo model-state topic in the rosbag.",
    )
    parser.add_argument(
        "--model-name",
        default="mycar",
        help="Gazebo model name to extract from /gazebo/model_states.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Ignore image and model-state messages earlier than this ROS/bag timestamp.",
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=DEFAULT_EXTENT,
        help=(
            "World-coordinate ground-plane extent for the tight full-wall top-down view. "
            "In landscape mode, y spans image width and x spans image height."
        ),
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Output video frame rate.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec, for example mp4v or XVID.")
    parser.add_argument("--line-width", type=int, default=4, help="Trajectory line width in pixels.")
    parser.add_argument("--point-radius", type=int, default=7, help="Current vehicle point radius in pixels.")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit for quick smoke tests.",
    )
    parser.add_argument("--no-final-pdf", action="store_true", help="Do not save the final overlaid PDF frame.")
    parser.add_argument("--no-trajectory-csv", action="store_true", help="Do not save the extracted truth CSV.")
    return parser.parse_args()


def load_runtime_dependencies():
    try:
        import cv2
        import numpy as np
        import rosbag
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise RuntimeError(
            "This script requires ROS Python packages, cv_bridge, OpenCV, and numpy. "
            "Source your catkin workspace and install python3-opencv/numpy if needed."
        ) from exc
    return cv2, np, rosbag, CvBridge


def sanitize_label(label):
    return re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_") or "run"


def default_output_path(scene, algorithm):
    return SCRIPT_DIR / f"trajectory_video_{scene}_{sanitize_label(algorithm)}.mp4"


def default_truth_path(output_path):
    return output_path.with_name(f"{output_path.stem}_truth.csv")


def print_scene(scene):
    if validation_scenes is not None:
        validation_scenes.print_selected_scene(scene)
    else:
        print(f"Validation scene: {scene}")


def quaternion_to_yaw(orientation):
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def collect_pose_samples(rosbag_module, bag_path, state_topic, model_name, start_time=None):
    samples = []
    with rosbag_module.Bag(str(bag_path), "r") as bag:
        for _, msg, stamp in bag.read_messages(topics=[state_topic]):
            stamp_time = stamp.to_sec()
            if start_time is not None and stamp_time < start_time:
                continue
            try:
                index = msg.name.index(model_name)
            except ValueError:
                continue
            pose = msg.pose[index]
            samples.append(
                {
                    "time": stamp_time,
                    "x": float(pose.position.x),
                    "y": float(pose.position.y),
                    "z": float(pose.position.z),
                    "yaw": quaternion_to_yaw(pose.orientation),
                }
            )
    samples.sort(key=lambda item: item["time"])
    return samples


def write_truth_csv(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time", "x", "y", "z", "yaw"])
        for sample in samples:
            writer.writerow(
                [
                    f"{sample['time']:.9f}",
                    f"{sample['x']:.9f}",
                    f"{sample['y']:.9f}",
                    f"{sample['z']:.9f}",
                    f"{sample['yaw']:.9f}",
                ]
            )


def world_to_pixel(x, y, extent, width, height):
    x_min, x_max, y_min, y_max = extent
    u = float(width - 1) - (y - y_min) / (y_max - y_min) * float(width - 1)
    v = float(height - 1) - (x - x_min) / (x_max - x_min) * float(height - 1)
    return int(round(u)), int(round(v))


def pose_for_time(samples, sample_times, stamp):
    index = bisect_right(sample_times, stamp) - 1
    if index < 0:
        return None
    return samples[index]


def convert_image_message(cv2, bridge, msg):
    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception:
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        encoding = getattr(msg, "encoding", "").lower()
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


def color_for_algorithm(algorithm):
    rgb = ALGORITHM_COLORS_RGB.get(algorithm, (214, 39, 40))
    return (rgb[2], rgb[1], rgb[0])


def draw_overlay(cv2, frame, algorithm, pixel_history, current_pose, current_pixel, extent, line_width, point_radius):
    color = color_for_algorithm(algorithm)
    height, width = frame.shape[:2]

    if len(pixel_history) >= 2:
        import numpy as np

        points = np.array(pixel_history, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=line_width, lineType=cv2.LINE_AA)

    if current_pixel is not None:
        cv2.circle(frame, current_pixel, point_radius + 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, current_pixel, point_radius, color, thickness=-1, lineType=cv2.LINE_AA)

        heading_length = 0.9
        head_x = current_pose["x"] + heading_length * math.cos(current_pose["yaw"])
        head_y = current_pose["y"] + heading_length * math.sin(current_pose["yaw"])
        head_pixel = world_to_pixel(head_x, head_y, extent, width, height)
        cv2.arrowedLine(frame, current_pixel, head_pixel, color, 2, cv2.LINE_AA, 0, 0.35)

    label = f"{algorithm} truth trajectory"
    cv2.putText(frame, label, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(frame, label, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def draw_full_trajectory_on_frame(cv2, frame, algorithm, pixel_history, point_radius, line_width):
    color = color_for_algorithm(algorithm)
    if len(pixel_history) >= 2:
        import numpy as np

        points = np.array(pixel_history, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=line_width, lineType=cv2.LINE_AA)

    if pixel_history:
        start_pixel = pixel_history[0]
        end_pixel = pixel_history[-1]
        cv2.circle(frame, start_pixel, point_radius + 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, start_pixel, point_radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, end_pixel, point_radius + 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, end_pixel, point_radius, color, thickness=-1, lineType=cv2.LINE_AA)

    label = f"{algorithm} truth trajectory"
    cv2.putText(frame, label, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(frame, label, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def make_pdf_object(number, content):
    if isinstance(content, bytes):
        return b"%d 0 obj\n" % number + content + b"\nendobj\n"
    return f"{number} 0 obj\n{content}\nendobj\n".encode("ascii")


def save_bgr_frame_as_pdf(path, frame_bgr, cv2):
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb_frame.shape[:2]
    image_bytes = rgb_frame.tobytes()
    compressed_image = zlib.compress(image_bytes)

    page_width = 432.0
    page_height = page_width * float(height) / float(width)
    content_stream = f"q {page_width:.4f} 0 0 {page_height:.4f} 0 0 cm /Im0 Do Q\n".encode("ascii")

    objects = [
        make_pdf_object(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        make_pdf_object(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        make_pdf_object(
            3,
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width:.4f} {page_height:.4f}] "
                "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
            ),
        ),
        make_pdf_object(
            4,
            (
                f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(compressed_image)} >>\n"
            ).encode("ascii")
            + b"stream\n"
            + compressed_image
            + b"\nendstream",
        ),
        make_pdf_object(
            5,
            f"<< /Length {len(content_stream)} >>\n".encode("ascii") + b"stream\n" + content_stream + b"endstream",
        ),
    ]

    pdf_parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    for obj in objects:
        offsets.append(sum(len(part) for part in pdf_parts))
        pdf_parts.append(obj)

    xref_offset = sum(len(part) for part in pdf_parts)
    pdf_parts.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf_parts.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_parts.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf_parts.append(
        (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n" f"startxref\n{xref_offset}\n%%EOF\n").encode("ascii")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(pdf_parts))


def make_video_writer(cv2, output_path, codec, fps, width, height):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video writer for {output_path}. "
            "Try another extension/codec, for example --codec XVID --output output.avi."
        )
    return writer


def process_bag(args, cv2, rosbag_module, bridge_cls):
    if not args.bag.exists():
        raise FileNotFoundError(f"Bag file not found: {args.bag}")

    output_path = args.output or default_output_path(args.scene, args.algorithm)
    final_output = args.final_output or output_path.with_suffix(".pdf")
    trajectory_output = args.trajectory_output or default_truth_path(output_path)

    samples = collect_pose_samples(
        rosbag_module,
        args.bag,
        args.state_topic,
        args.model_name,
        start_time=args.start_time,
    )
    if not samples:
        raise RuntimeError(f"No `{args.model_name}` samples found on topic {args.state_topic}.")

    if not args.no_trajectory_csv:
        write_truth_csv(trajectory_output, samples)
        print(f"Saved truth trajectory: {trajectory_output}")

    sample_times = [sample["time"] for sample in samples]
    bridge = bridge_cls()
    writer = None
    frame_count = 0
    pixel_history = []
    first_frame = None

    with rosbag_module.Bag(str(args.bag), "r") as bag:
        for _, msg, stamp in bag.read_messages(topics=[args.image_topic]):
            if args.start_time is not None and stamp.to_sec() < args.start_time:
                continue
            frame = convert_image_message(cv2, bridge, msg)
            height, width = frame.shape[:2]
            if first_frame is None:
                first_frame = frame.copy()
            if writer is None:
                writer = make_video_writer(cv2, output_path, args.codec, args.fps, width, height)

            pose = pose_for_time(samples, sample_times, stamp.to_sec())
            current_pixel = None
            if pose is not None:
                current_pixel = world_to_pixel(pose["x"], pose["y"], args.extent, width, height)
                if not pixel_history or pixel_history[-1] != current_pixel:
                    pixel_history.append(current_pixel)

            draw_overlay(
                cv2,
                frame,
                args.algorithm,
                pixel_history,
                pose,
                current_pixel,
                args.extent,
                args.line_width,
                args.point_radius,
            )
            writer.write(frame)
            frame_count += 1

            if args.max_frames is not None and frame_count >= args.max_frames:
                break

    if writer is None:
        raise RuntimeError(f"No image frames found on topic {args.image_topic}.")

    writer.release()
    print(f"Saved trajectory video: {output_path}")
    print(f"Frames written: {frame_count}")
    print(f"Pose samples read: {len(samples)}")

    if not args.no_final_pdf and first_frame is not None:
        first_frame_overlay = first_frame.copy()
        draw_full_trajectory_on_frame(
            cv2,
            first_frame_overlay,
            args.algorithm,
            pixel_history,
            args.point_radius,
            args.line_width,
        )
        save_bgr_frame_as_pdf(final_output, first_frame_overlay, cv2)
        print(f"Saved final overlay PDF: {final_output}")


def main():
    args = parse_args()
    print_scene(args.scene)
    print(f"Input bag: {args.bag}")
    print(f"Image topic: {args.image_topic}")
    print(f"State topic: {args.state_topic}")
    if args.start_time is not None:
        print(f"Start time filter: {args.start_time:.9f}")
    print(f"World extent: x=[{args.extent[0]}, {args.extent[1]}], y=[{args.extent[2]}, {args.extent[3]}]")

    cv2, _, rosbag_module, bridge_cls = load_runtime_dependencies()
    process_bag(args, cv2, rosbag_module, bridge_cls)


if __name__ == "__main__":
    main()
