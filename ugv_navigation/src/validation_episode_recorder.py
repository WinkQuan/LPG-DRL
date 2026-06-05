#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Record and render the first successful validation episode."""

from __future__ import print_function

import atexit
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DRAW_SCRIPT = SCRIPT_DIR / "Visualization" / "draw_trajectory_on_video.py"
IMAGE_TOPIC = "/top_down_camera/image_raw"
STATE_TOPIC = "/gazebo/model_states"
REQUIRED_TOPICS = [IMAGE_TOPIC, STATE_TOPIC]
RENDER_MODULES = ["cv2", "numpy", "rosbag", "cv_bridge"]
RECORD_START_WARMUP_SECONDS = 0.8
EPISODE_RESET_SETTLE_SECONDS = 0.5
EPISODE_START_IMAGE_TIMEOUT_SECONDS = 3.0


def safe_algorithm_name(algorithm):
    return algorithm.replace(" ", "_").replace("/", "_")


def remove_stale_outputs(*paths):
    for path in paths:
        output_path = Path(path)
        if output_path.exists():
            output_path.unlink()


def unique_candidates(candidates):
    seen = set()
    unique = []
    for candidate in candidates:
        if not candidate:
            continue
        resolved = str(Path(candidate))
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def render_dependency_check_command():
    return (
        "import cv2\n"
        "import numpy\n"
        "import rosbag\n"
        "from cv_bridge import CvBridge\n"
    )


def find_render_python():
    candidates = unique_candidates(
        [
            sys.executable,
            shutil.which("python3"),
            "/usr/bin/python3",
        ]
    )
    errors = []
    for candidate in candidates:
        completed = subprocess.run(
            [candidate, "-c", render_dependency_check_command()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            return candidate
        errors.append((candidate, completed.stderr.strip() or completed.stdout.strip()))

    details = "\n".join(f"  {candidate}: {error}" for candidate, error in errors)
    raise RuntimeError(
        "Cannot find a Python interpreter that can render first-success trajectory videos. "
        "Need cv2, numpy, rosbag, and cv_bridge.\n"
        f"Checked interpreters:\n{details}\n"
        "If you run validation from a conda env, make sure system ROS Python is available, "
        "for example /usr/bin/python3 after sourcing ROS/catkin."
    )


def require_runtime_dependencies():
    for executable in ["rostopic", "rosbag"]:
        if shutil.which(executable) is None:
            raise RuntimeError(
                f"Cannot find `{executable}`. Please source ROS and the catkin workspace before validation."
            )

    if not DRAW_SCRIPT.exists():
        raise FileNotFoundError(f"Expected trajectory video renderer not found: {DRAW_SCRIPT}")


def require_topics_available():
    completed = subprocess.run(
        ["rostopic", "list"],
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = f"\nrostopic error: {stderr}" if stderr else ""
        raise RuntimeError("ROS master is not reachable. Please start Gazebo before validation." + detail)

    available_topics = set(completed.stdout.splitlines())
    missing_topics = [topic for topic in REQUIRED_TOPICS if topic not in available_topics]
    if missing_topics:
        raise RuntimeError(
            "Cannot record first-success trajectory video because required topics are missing: "
            + ", ".join(missing_topics)
            + "\nPlease restart Gazebo with the top_down_camera model loaded."
        )


class FirstSuccessEpisodeRecorder:
    def __init__(self, scene, algorithm, validate_dir):
        self.scene = scene
        self.algorithm = algorithm
        self.validate_dir = Path(validate_dir)
        self.safe_algorithm = safe_algorithm_name(algorithm)
        self.temp_dir = self.validate_dir / "_episode_recordings"
        self.video_path = self.validate_dir / f"{self.safe_algorithm}_first_success_topdown.mp4"
        self.pdf_path = self.validate_dir / f"{self.safe_algorithm}_first_success_topdown.pdf"
        self.truth_csv_path = self.validate_dir / f"{self.safe_algorithm}_first_success_topdown_truth.csv"
        self.current_process = None
        self.current_bag_path = None
        self.current_log_path = None
        self.current_episode_start_time = None
        self.has_saved_success = False

        require_runtime_dependencies()
        self.render_python = find_render_python()
        require_topics_available()
        remove_stale_outputs(self.video_path, self.pdf_path, self.truth_csv_path)
        atexit.register(self.abort_episode)
        print(f"First-success video renderer: {self.render_python}")

    def start_episode(self, episode_index):
        self.abort_episode()
        if self.has_saved_success:
            return

        self.current_episode_start_time = None
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.current_bag_path = self.temp_dir / f"{self.safe_algorithm}_episode_{episode_index:03d}.bag"
        self.current_log_path = self.temp_dir / f"{self.safe_algorithm}_episode_{episode_index:03d}.log"
        self._remove_bag_artifacts(self.current_bag_path)

        command = [
            "rosbag",
            "record",
            "-O",
            str(self.current_bag_path),
            IMAGE_TOPIC,
            STATE_TOPIC,
        ]
        log_file = self.current_log_path.open("w")
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        try:
            self.current_process = subprocess.Popen(
                command,
                cwd=str(SCRIPT_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_file.close()

        time.sleep(0.2)
        if self.current_process.poll() is not None:
            raise RuntimeError(
                f"rosbag record failed to start for first-success video. See log: {self.current_log_path}"
            )
        time.sleep(RECORD_START_WARMUP_SECONDS)

    def mark_episode_start(self, start_time):
        self.current_episode_start_time = float(start_time)

    def mark_episode_start_from_next_image(
        self,
        settle_time=EPISODE_RESET_SETTLE_SECONDS,
        timeout=EPISODE_START_IMAGE_TIMEOUT_SECONDS,
    ):
        if self.current_process is None:
            self.current_episode_start_time = None
            return None

        try:
            import rospy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                "Cannot synchronize first-success recording with the top-down camera. "
                "Please source ROS/catkin before validation."
            ) from exc

        if settle_time > 0.0:
            rospy.sleep(settle_time)

        start_time = rospy.Time.now().to_sec()
        try:
            rospy.wait_for_message(IMAGE_TOPIC, Image, timeout=timeout)
        except rospy.ROSException as exc:
            raise RuntimeError(
                f"Timed out waiting for a fresh {IMAGE_TOPIC} frame after episode reset. "
                "Please confirm the top_down_camera is publishing before validation."
            ) from exc

        self.mark_episode_start(start_time)
        return start_time

    def finish_episode(self, success=False):
        if self.current_process is None:
            return

        bag_path = self.current_bag_path
        log_path = self.current_log_path
        start_time = self.current_episode_start_time
        return_code = self._stop_current_process()
        self.current_process = None
        self.current_bag_path = None
        self.current_log_path = None
        self.current_episode_start_time = None

        if return_code not in (0, -signal.SIGINT, 128 + signal.SIGINT):
            raise RuntimeError(
                f"rosbag record failed during episode recording with exit code {return_code}. "
                f"See log: {log_path}"
            )

        if success and not self.has_saved_success:
            self._render_success_bag(bag_path, start_time)
            self.has_saved_success = True
            self._remove_bag_artifacts(bag_path)
            if log_path and log_path.exists():
                log_path.unlink()
            print(f"First successful trajectory video saved: {self.video_path}")
            return

        self._remove_bag_artifacts(bag_path)
        if log_path and log_path.exists():
            log_path.unlink()

    def abort_episode(self):
        if self.current_process is None:
            return

        bag_path = self.current_bag_path
        log_path = self.current_log_path
        self._stop_current_process()
        self.current_process = None
        self.current_bag_path = None
        self.current_log_path = None
        self.current_episode_start_time = None
        self._remove_bag_artifacts(bag_path)
        if log_path and log_path.exists():
            log_path.unlink()

    def print_summary(self):
        if self.has_saved_success:
            print(f"First-success top-down video: {self.video_path}")
            print(f"First-success top-down PDF: {self.pdf_path}")
            print(f"First-success truth trajectory CSV: {self.truth_csv_path}")
        else:
            print("Warning: no successful episode was recorded, so no first-success top-down video was generated.")

    def _stop_current_process(self):
        if self.current_process is None:
            return 0

        if self.current_process.poll() is not None:
            return self.current_process.returncode

        self.current_process.send_signal(signal.SIGINT)
        try:
            self.current_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
                self.current_process.wait(timeout=5)
        return self.current_process.returncode

    def _render_success_bag(self, bag_path, start_time):
        if bag_path is None or not bag_path.exists():
            raise RuntimeError(f"First-success rosbag was not created: {bag_path}")

        command = [
            self.render_python,
            str(DRAW_SCRIPT),
            "--bag",
            str(bag_path),
            "--scene",
            self.scene,
            "--algorithm",
            self.algorithm,
            "--output",
            str(self.video_path),
            "--final-output",
            str(self.pdf_path),
            "--trajectory-output",
            str(self.truth_csv_path),
        ]
        if start_time is not None:
            command.extend(["--start-time", f"{start_time:.9f}"])
        print("$ " + " ".join(command))
        completed = subprocess.run(command, cwd=str(SCRIPT_DIR))
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to render first-success trajectory video. Temporary bag kept for inspection: {bag_path}"
            )

    def _remove_bag_artifacts(self, bag_path):
        if bag_path is None:
            return

        candidates = [
            Path(bag_path),
            Path(str(bag_path) + ".active"),
            Path(str(bag_path) + ".orig.active"),
        ]
        for path in candidates:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
