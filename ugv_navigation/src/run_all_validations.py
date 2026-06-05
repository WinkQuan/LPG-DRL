#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Run validation for all non-ablation algorithms."""

from __future__ import print_function

import argparse
import contextlib
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import algorithm_comparison
import validation_scenes

SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_GAZEBO_TOPICS = [
    "/gazebo/model_states",
    "/mycar/camera_right/image_raw_right",
    "/top_down_camera/image_raw",
]
MODEL_PATHS = {
    "LPG-DRL": Path("./Model/LPG-DRL/model_lpg_drl.onnx"),
    "NPE-DRL": Path("./Model/NPE-DRL/model_npe_drl.onnx"),
    "D3QN": Path("./Model/D3QN/model_d3qn.onnx"),
    "D3QN-LfD": Path("./Model/D3QN-LfD/model_d3qn_lfd.onnx"),
    "PPO": Path("./Model/PPO/model_ppo.onnx"),
    "BC": Path("./Model/BC/model_bc.onnx"),
    "DAgger": Path("./Model/DAgger/model_dagger.onnx"),
}


def algorithm_names():
    return [spec["algorithm"] for spec in algorithm_comparison.ALGORITHM_SPECS]


def default_algorithm_count():
    return len(algorithm_comparison.ALGORITHM_SPECS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run validation for all non-ablation algorithms and aggregate metrics."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the validation plan without running preflight checks or commands.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=algorithm_names(),
        default=algorithm_names(),
        help="Algorithms to validate. Defaults to all non-ablation algorithms.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip validation when the algorithm's metrics.csv already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later algorithms after a validation command fails.",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Run validations only and skip algorithm_comparison.py.",
    )
    return parser.parse_args()


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def display_command(command):
    if command and Path(command[0]).name.startswith("python"):
        return " ".join(["python"] + command[1:])
    return " ".join(command)


def specs_by_algorithm():
    return {spec["algorithm"]: spec for spec in algorithm_comparison.ALGORITHM_SPECS}


def selected_specs(algorithms):
    lookup = specs_by_algorithm()
    return [lookup[algorithm] for algorithm in algorithms]


def absolute_path(path):
    return SCRIPT_DIR / path


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def scene_spec_for(spec, scene):
    scene_spec = spec.copy()
    scene_spec["metrics_path"] = validation_scenes.scene_validate_dir(scene, spec["algorithm"]) / "metrics.csv"
    return scene_spec


def scene_specs(scene):
    return [scene_spec_for(spec, scene) for spec in algorithm_comparison.ALGORITHM_SPECS]


def metrics_file_for(spec, scene):
    return absolute_path(scene_spec_for(spec, scene)["metrics_path"])


def validate_dir_for(spec, scene):
    return metrics_file_for(spec, scene).parent


def model_file_for(spec):
    model_path = MODEL_PATHS.get(spec["algorithm"])
    if model_path is None:
        return None
    return absolute_path(model_path)


def command_from_spec(spec):
    parts = shlex.split(spec["validate_cmd"])
    if parts and Path(parts[0]).name.startswith("python"):
        parts[0] = sys.executable
    return parts


def check_ros_gazebo_ready(scene):
    try:
        completed = subprocess.run(
            ["rostopic", "list"],
            cwd=str(SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as err:
        raise RuntimeError(
            "Cannot find `rostopic`. Please source ROS and the catkin workspace before running this script."
        ) from err
    except subprocess.TimeoutExpired as err:
        raise RuntimeError("Timed out while querying ROS topics. Please confirm ROS/Gazebo is running.") from err

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = f"\nrostopic error: {stderr}" if stderr else ""
        raise RuntimeError(
            "ROS master is not reachable. Please start Gazebo first:\n"
            f"  {validation_scenes.scene_launch_command(scene)}"
            f"{detail}"
        )

    available_topics = set(completed.stdout.splitlines())
    missing_topics = [topic for topic in REQUIRED_GAZEBO_TOPICS if topic not in available_topics]
    if missing_topics:
        raise RuntimeError(
            "Gazebo is reachable, but required topics are missing:\n"
            f"  {', '.join(missing_topics)}\n"
            "Please start the selected world first:\n"
            f"  {validation_scenes.scene_launch_command(scene)}"
        )

    print("ROS/Gazebo preflight passed.")


def validate_model_files(specs):
    missing = []
    for spec in specs:
        model_file = model_file_for(spec)
        if model_file is not None and not model_file.exists():
            missing.append((spec["algorithm"], model_file))

    if missing:
        print("Missing ONNX model files:")
        for algorithm, model_file in missing:
            print(f"  {algorithm}: {model_file}")
        raise RuntimeError("Cannot run validation until all selected learned-policy models exist.")


def backup_directory(path, log_dir, relative_root):
    if not path.exists():
        return None

    relative_path = path.relative_to(SCRIPT_DIR)
    backup_path = log_dir / "backups" / relative_root / relative_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(backup_path))
    return backup_path


def maybe_backup_validation(spec, scene, log_dir, dry_run):
    validate_dir = validate_dir_for(spec, scene)
    if not validate_dir.exists():
        return

    if dry_run:
        print(f"[DRY-RUN] Would back up validation directory before revalidating: {validate_dir}")
        return

    backup_path = backup_directory(validate_dir, log_dir, "validation")
    if backup_path:
        print(f"Backed up existing validation directory to: {backup_path}")


def stream_command(command, log_path, env_vars=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {display_command(command)}")
    print(f"Log: {log_path}")

    command_env = os.environ.copy()
    if env_vars:
        command_env.update(env_vars)

    with log_path.open("w") as log_file:
        log_file.write(f"$ {display_command(command)}\n")
        if env_vars:
            for key, value in sorted(env_vars.items()):
                log_file.write(f"{key}={value}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=command_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def run_stage(stage, command, log_path, continue_on_error, env_vars=None):
    return_code = stream_command(command, log_path, env_vars=env_vars)
    if return_code == 0:
        print(f"{stage} finished successfully.")
        return True

    print(f"{stage} failed with exit code {return_code}.")
    if not continue_on_error:
        raise RuntimeError(f"{stage} failed. See log: {log_path}")
    return False


def all_default_metrics_exist(scene):
    return all(metrics_file_for(spec, scene).exists() for spec in algorithm_comparison.ALGORITHM_SPECS)


def print_plan(args, specs, scene, log_dir):
    print_header("Non-Ablation Validation Run Plan")
    print(f"Working directory: {SCRIPT_DIR}")
    print(f"Scene: {validation_scenes.scene_display_name(scene)} ({scene})")
    print(f"Result directory: {absolute_path(validation_scenes.scene_validate_root(scene))}")
    print(f"Log directory: {log_dir}")
    print(f"Algorithms: {' '.join(spec['algorithm'] for spec in specs)}")
    print(f"skip_existing={args.skip_existing}, no_compare={args.no_compare}")
    if args.dry_run:
        print("Dry run: ROS/Gazebo preflight and validation commands will not be executed.")


def print_preflight_plan(specs):
    print("\nModel checks:")
    for spec in specs:
        model_file = model_file_for(spec)
        if model_file is None:
            print(f"  {spec['algorithm']}: no ONNX model required")
        else:
            status = "FOUND" if model_file.exists() else "MISSING"
            print(f"  {spec['algorithm']}: [{status}] {model_file}")


def run_validations(args, specs, scene, log_dir):
    failures = []
    validation_env = {validation_scenes.ENV_VAR: scene}

    for spec in specs:
        algorithm = spec["algorithm"]
        metrics_file = metrics_file_for(spec, scene)
        validate_log = log_dir / f"validate_{algorithm.lower().replace('-', '_')}.log"
        validate_cmd = command_from_spec(spec)

        print_header(f"Validate {algorithm}")

        if args.skip_existing and metrics_file.exists():
            print(f"Skip validation: metrics already exists: {metrics_file}")
            continue

        maybe_backup_validation(spec, scene, log_dir, args.dry_run)

        if args.dry_run:
            print(
                f"[DRY-RUN] Would run validation: {validation_scenes.ENV_VAR}={scene} {display_command(validate_cmd)}"
            )
            continue

        try:
            ok = run_stage(
                f"Validation {algorithm}",
                validate_cmd,
                validate_log,
                args.continue_on_error,
                env_vars=validation_env,
            )
            if not ok:
                failures.append((algorithm, "validate"))
        except RuntimeError:
            failures.append((algorithm, "validate"))
            raise

    return failures


def write_scene_comparison(scene):
    available_rows = []
    invalid_specs = []

    print("=" * 100)
    print("Algorithm Comparison Based on Validation Metrics")
    print("=" * 100)
    print(f"Scene: {validation_scenes.scene_display_name(scene)} ({scene})")

    for spec in scene_specs(scene):
        row, error = algorithm_comparison.load_metrics(spec)
        if row is not None:
            available_rows.append(row)
        else:
            invalid_specs.append((spec, error))

    print("\nResult overview:")
    for spec in scene_specs(scene):
        if algorithm_comparison.is_algorithm_available(spec, available_rows):
            print(f"  [FOUND]   {spec['algorithm']}: {spec['metrics_path']}")
        else:
            print(f"  [MISSING] {spec['algorithm']}: run `{spec['validate_cmd']}`")

    if invalid_specs:
        print("\nInvalid metrics files:")
        for spec, error in invalid_specs:
            print(f"  {spec['algorithm']}: {error}")
        raise RuntimeError("Cannot write comparison because one or more scene metrics files are invalid.")

    result_rows = algorithm_comparison.sort_results(available_rows)
    result_rows = algorithm_comparison.format_results(result_rows)

    output_path = absolute_path(validation_scenes.scene_comparison_path(scene))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithm_comparison.write_comparison_csv(result_rows, output_path)

    print("\nComparison table:")
    algorithm_comparison.print_table(result_rows)
    print(f"\nComparison CSV saved to: {output_path}")


def run_comparison(args, scene, log_dir):
    print_header("Algorithm Comparison")

    if args.no_compare:
        print("Skip comparison: --no-compare was set.")
        return

    if args.dry_run:
        output_path = absolute_path(validation_scenes.scene_comparison_path(scene))
        if all_default_metrics_exist(scene):
            print(f"[DRY-RUN] Would write scene comparison: {output_path}")
        else:
            print(
                "[DRY-RUN] Would run comparison after all "
                f"{default_algorithm_count()} non-ablation metrics.csv files are complete for scene `{scene}`."
            )
        return

    if not all_default_metrics_exist(scene):
        print(
            f"Skip comparison: not all {default_algorithm_count()} non-ablation metrics.csv files exist yet "
            f"for scene `{scene}`."
        )
        print("Missing metrics:")
        for spec in algorithm_comparison.ALGORITHM_SPECS:
            metrics_file = metrics_file_for(spec, scene)
            if not metrics_file.exists():
                print(f"  {spec['algorithm']}: {metrics_file}")
        return

    compare_log = log_dir / "algorithm_comparison.log"
    compare_log.parent.mkdir(parents=True, exist_ok=True)
    with compare_log.open("w") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            write_scene_comparison(scene)


def main():
    args = parse_args()
    scene = validation_scenes.select_scene()
    specs = selected_specs(args.algorithms)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = SCRIPT_DIR / "Logs" / "Validation" / scene / timestamp

    print_plan(args, specs, scene, log_dir)
    print_preflight_plan(specs)

    if not args.dry_run:
        check_ros_gazebo_ready(scene)
        validate_model_files(specs)
        log_dir.mkdir(parents=True, exist_ok=True)

    failures = run_validations(args, specs, scene, log_dir)
    run_comparison(args, scene, log_dir)

    if failures:
        print_header("Failures")
        for algorithm, stage in failures:
            print(f"{stage}: algorithm={algorithm}")
        return 1

    print_header("Done")
    print("Non-ablation validation runner finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
