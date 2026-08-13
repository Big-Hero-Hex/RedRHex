"""Load or regenerate the committed cross-runtime V2 sensor-frame fixture."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .preprocessing import build_sensor_frame_numpy


def generate_sensor_frame_golden_v2() -> dict[str, Any]:
    inputs: dict[str, list[float]] = {
        "body_gyro_rad_s": [0.1, -0.2, 0.3],
        "projected_gravity": [0.0, 0.0, -1.0],
        "main_position_rad": [
            0.0,
            1.5707963267948966,
            -1.5707963267948966,
            0.0,
            1.5707963267948966,
            -1.5707963267948966,
        ],
        "main_velocity_rad_s": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "abad_position_rad": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "abad_velocity_rad_s": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        "abad_neutral_position_rad": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    }
    expected = build_sensor_frame_numpy(**inputs)
    return {
        "fixture_id": "redrhex.sensor-frame.v2.golden.1",
        "inputs": inputs,
        "expected_sensor_frame": [float(item) for item in expected],
    }


def load_sensor_frame_golden_v2() -> dict[str, Any]:
    resource = files("redrhex_policy_io").joinpath("data/sensor_frame_v2_golden.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def write_sensor_frame_golden_v2(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(generate_sensor_frame_golden_v2(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
