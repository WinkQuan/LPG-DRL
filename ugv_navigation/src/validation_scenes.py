#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Shared scene selection helpers for validation scripts."""

from __future__ import print_function

import os
from pathlib import Path

SCENES = {
    "off_road": {
        "label": "越野",
        "launch": "roslaunch mymodel_gazebo myworld.launch",
    },
    "flat_land": {
        "label": "平地",
        "launch": "roslaunch mymodel_gazebo myworld_flat_land.launch",
    },
    "without_obstacle": {
        "label": "无障碍物",
        "launch": "roslaunch mymodel_gazebo myworld_without_obstacle.launch",
    },
}

SCENE_CHOICES = [
    ("1", "off_road"),
    ("2", "flat_land"),
    ("3", "without_obstacle"),
]

ENV_VAR = "VALIDATION_SCENE"


def normalize_scene(value):
    value = str(value).strip()
    choice_lookup = dict(SCENE_CHOICES)
    if value in choice_lookup:
        return choice_lookup[value]
    if value in SCENES:
        return value

    valid_values = sorted(list(SCENES.keys()) + [choice for choice, _ in SCENE_CHOICES])
    raise ValueError(f"Unknown validation scene `{value}`. Valid values: {', '.join(valid_values)}")


def scene_display_name(scene):
    scene = normalize_scene(scene)
    return SCENES[scene]["label"]


def scene_launch_command(scene):
    scene = normalize_scene(scene)
    return SCENES[scene]["launch"]


def scene_validate_root(scene):
    scene = normalize_scene(scene)
    return Path("./Validate") / scene


def scene_validate_dir(scene, algorithm):
    return scene_validate_root(scene) / algorithm


def scene_comparison_path(scene):
    return scene_validate_root(scene) / "algorithm_comparison.csv"


def print_selected_scene(scene):
    print(f"Validation scene: {scene_display_name(scene)} ({scene})")
    print(f"Expected Gazebo launch: {scene_launch_command(scene)}")


def select_scene():
    env_scene = os.environ.get(ENV_VAR)
    if env_scene:
        scene = normalize_scene(env_scene)
        print_selected_scene(scene)
        return scene

    print("请选择当前正在运行的仿真场景:")
    for choice, scene in SCENE_CHOICES:
        print(f"  {choice}. {scene_display_name(scene)} ({scene})")

    while True:
        try:
            value = input("请输入场景编号 [1/2/3]: ")
        except EOFError as err:
            raise RuntimeError(
                f"Validation scene is required. Set {ENV_VAR}=off_road|flat_land|without_obstacle."
            ) from err

        try:
            scene = normalize_scene(value)
        except ValueError as err:
            print(err)
            continue

        print_selected_scene(scene)
        return scene
