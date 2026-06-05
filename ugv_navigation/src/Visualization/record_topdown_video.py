#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Record a fixed top-down Gazebo camera together with model-state truth."""

from __future__ import print_function

import argparse
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NAV_SRC_DIR = SCRIPT_DIR.parent
DEFAULT_RECORD_DIR = SCRIPT_DIR / "Records"

IMAGE_TOPIC = "/top_down_camera/image_raw"
STATE_TOPIC = "/gazebo/model_states"
SCENES = ("off_road", "flat_land", "without_obstacle")

if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

try:
    import validation_scenes
except ImportError:
    validation_scenes = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record /top_down_camera/image_raw and /gazebo/model_states to a rosbag. "
            "Start the selected Gazebo world before running this command."
        )
    )
    parser.add_argument(
        "--scene",
        choices=SCENES,
        default="off_road",
        help="Scene label used in the default bag name.",
    )
    parser.add_argument(
        "--algorithm",
        default="LPG-DRL",
        help="Algorithm label used in the default bag name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output bag path. Defaults to Visualization/Records/<scene>_<algorithm>_topdown.bag.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional recording duration in seconds. If omitted, record until Ctrl-C.",
    )
    parser.add_argument(
        "--image-topic",
        default=IMAGE_TOPIC,
        help="Top-down camera image topic to record.",
    )
    parser.add_argument(
        "--state-topic",
        default=STATE_TOPIC,
        help="Gazebo model-state topic to record.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rosbag command and topic checks without recording.",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip rostopic availability checks before recording.",
    )
    return parser.parse_args()


def sanitize_label(label):
    chars = []
    for char in label.lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    cleaned = "".join(chars).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "run"


def default_output_path(scene, algorithm):
    return DEFAULT_RECORD_DIR / f"{scene}_{sanitize_label(algorithm)}_topdown.bag"


def format_command(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def print_scene(scene):
    if validation_scenes is not None:
        validation_scenes.print_selected_scene(scene)
    else:
        print(f"Validation scene: {scene}")


def preflight_topics(required_topics):
    try:
        result = subprocess.run(
            ["rostopic", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError("Failed to run `rostopic list`. Is ROS sourced?") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            "ROS master is not reachable. Start the Gazebo world first, for example:\n"
            "  roslaunch mymodel_gazebo myworld.launch\n"
            f"rostopic stderr: {stderr}"
        )

    available = set(topic.strip() for topic in result.stdout.splitlines() if topic.strip())
    missing = [topic for topic in required_topics if topic not in available]
    if missing:
        raise RuntimeError(
            "Required ROS topics are not available: "
            + ", ".join(missing)
            + "\nStart the world with the top-down camera model enabled, then retry."
        )


def build_record_command(output_path, image_topic, state_topic):
    return ["rosbag", "record", "-O", str(output_path), image_topic, state_topic]


def run_recording(command, duration):
    if duration is None:
        print("Recording until Ctrl-C...")
    else:
        print(f"Recording for {duration:.1f} seconds...")

    process = subprocess.Popen(command)
    try:
        if duration is None:
            process.wait()
        else:
            time.sleep(max(0.0, duration))
            process.send_signal(signal.SIGINT)
            process.wait(timeout=15)
    except KeyboardInterrupt:
        print("\nStopping rosbag recording...")
        process.send_signal(signal.SIGINT)
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)
        raise RuntimeError("rosbag did not stop cleanly after SIGINT.")

    if process.returncode not in (0, -signal.SIGINT, 128 + signal.SIGINT):
        raise RuntimeError(f"rosbag record failed with exit code {process.returncode}.")


def main():
    args = parse_args()
    output_path = args.output or default_output_path(args.scene, args.algorithm)
    required_topics = [args.image_topic, args.state_topic]
    command = build_record_command(output_path, args.image_topic, args.state_topic)

    print_scene(args.scene)
    print(f"Image topic: {args.image_topic}")
    print(f"State topic: {args.state_topic}")
    print(f"Output bag: {output_path}")

    if not args.no_preflight:
        if args.dry_run:
            print("Preflight: would check required ROS topics.")
        else:
            preflight_topics(required_topics)
            print("Preflight: required ROS topics are available.")

    print("Command:")
    print(f"  {format_command(command)}")

    if args.dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_recording(command, args.duration)
    print(f"Saved rosbag: {output_path}")


if __name__ == "__main__":
    main()
