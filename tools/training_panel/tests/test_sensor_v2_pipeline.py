from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_pipeline_module():
    path = REPO_ROOT / "scripts" / "rsl_rl" / "train_sensor_v2_pipeline.py"
    spec = importlib.util.spec_from_file_location("redrhex_sensor_v2_pipeline_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CompletedChild:
    def wait(self) -> int:
        return 0

    def poll(self) -> int:
        return 0

    def send_signal(self, _signum: int) -> None:
        raise AssertionError("completed child must not receive a signal")


def test_sensor_v2_pipeline_forwards_native_backend_to_every_child_stage(tmp_path: Path) -> None:
    pipeline = _load_pipeline_module()
    args = argparse.Namespace(
        num_envs=4,
        device="cpu",
        seed=42,
        spring_backend="native",
        headless=True,
        physics_profile=None,
    )
    commands: list[list[str]] = []

    def spawn(command, **_kwargs):
        commands.append(list(command))
        return _CompletedChild()

    run_dir = tmp_path / "run"
    checkpoint = run_dir / "model_0.pt"
    with (
        patch.object(pipeline.subprocess, "Popen", side_effect=spawn),
        patch.object(pipeline.signal, "signal", return_value=pipeline.signal.SIG_DFL),
        patch.object(pipeline, "_find_run_dir", return_value=run_dir),
        patch.object(pipeline, "_latest_checkpoint", return_value=checkpoint),
    ):
        for stage in ("teacher", "distillation", "ppo"):
            pipeline._run_stage(
                args,
                stage=stage,
                iterations=1,
                run_name=f"test_{stage}",
                bootstrap_flag="--bootstrap" if stage != "teacher" else None,
                bootstrap_checkpoint=checkpoint if stage != "teacher" else None,
            )

    assert len(commands) == 3
    for command in commands:
        backend_index = command.index("--spring-backend")
        assert command[backend_index + 1] == "native"
