from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_play_applies_requested_size_to_rgb_render_product_before_env_creation() -> None:
    source = (ROOT / "scripts/rsl_rl/play.py").read_text(encoding="utf-8")
    main_source = source[source.index("def main(") :]

    configure = main_source.index("\n    _configure_video_resolution(env_cfg)\n")
    create_env = main_source.index("gym.make(")

    assert configure < create_env
    assert "viewer.resolution = (int(width), int(height))" in source
    assert "args_cli.video_width" in source
    assert "args_cli.video_height" in source


def test_sensor_v2_export_requires_hash_bound_recorded_replay_inputs() -> None:
    source = (ROOT / "scripts/rsl_rl/play.py").read_text(encoding="utf-8")

    assert '"--sensor-v2-parity-npz"' in source
    assert '"--sensor-v2-parity-npz-sha256"' in source
    assert "Sensor V2 export requires --sensor-v2-parity-npz" in source
    assert "parity_sensor_histories=parity_histories" in source
    assert "parity_input_sha256=parity_digest" in source
