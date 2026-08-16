from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_pipeline_module():
    path = REPO_ROOT / "scripts" / "rsl_rl" / "train_sensor_v2_pipeline.py"
    spec = importlib.util.spec_from_file_location("redrhex_sensor_v2_pipeline_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ungated_debug_pipeline_requires_explicit_cli_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _load_pipeline_module()
    monkeypatch.setattr(sys, "argv", ["train_sensor_v2_pipeline.py"])
    with pytest.raises(SystemExit):
        pipeline._arguments()


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
        isaaclab_launcher=tmp_path / "isaaclab.sh",
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


def test_sensor_v2_pipeline_runs_f0_before_training(tmp_path: Path) -> None:
    pipeline = _load_pipeline_module()
    args = argparse.Namespace(
        num_envs=4,
        teacher_iterations=1,
        distillation_iterations=1,
        ppo_iterations=1,
        device="cpu",
        seed=42,
        spring_backend="native",
        headless=True,
        pipeline_id="f0_order",
        physics_profile=None,
        isaaclab_launcher=tmp_path / "isaaclab.sh",
    )
    events: list[str] = []

    def run_f0(_args: argparse.Namespace, _pipeline_id: str) -> Path:
        events.append("f0")
        return tmp_path / "f0.json"

    def run_stage(_args, *, stage: str, **_kwargs):
        events.append(stage)
        run_dir = tmp_path / stage
        run_dir.mkdir()
        checkpoint = run_dir / "model_0.pt"
        checkpoint.touch()
        return run_dir, checkpoint

    with (
        patch.object(pipeline, "_arguments", return_value=args),
        patch.object(pipeline, "_run_f0", side_effect=run_f0),
        patch.object(pipeline, "_run_stage", side_effect=run_stage),
    ):
        assert pipeline.main() == 0

    assert events == ["f0", "teacher", "distillation", "ppo"]
    result = json.loads((tmp_path / "ppo" / "sensor_v2_pipeline.json").read_text())
    assert result["f0_status"] == "passed"
    assert result["f0_report"] == str(tmp_path / "f0.json")
    assert result["promotion_eligible"] is False
    assert result["acceptance_screening"] == "not_run_debug_only"


def test_run_f0_requires_both_pass_exit_and_json_report(tmp_path: Path) -> None:
    pipeline = _load_pipeline_module()
    pipeline.REPO_ROOT = tmp_path
    pipeline.F0_SCRIPT = tmp_path / "validate_forward_gait_baseline.py"
    args = argparse.Namespace(
        isaaclab_launcher=tmp_path / "isaaclab.sh",
        num_envs=4,
        seed=42,
        spring_backend="native",
        headless=True,
    )
    observed: list[list[str]] = []

    def completed(command, **kwargs):
        observed.append(list(command))
        assert kwargs == {"cwd": tmp_path, "check": False}
        report = Path(command[command.index("--json") + 1])
        report.parent.mkdir(parents=True)
        report.write_text(
            '{"overall_status":"PASS","simulator_rollout":{"status":"PASS",'
            '"commands":[{"status":"PASS"}]}}\n',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    with patch.object(pipeline.subprocess, "run", side_effect=completed):
        report_path = pipeline._run_f0(args, "test_pipeline")

    assert report_path.is_file()
    assert observed[0][0] == str(args.isaaclab_launcher)
    assert observed[0][2] == str(pipeline.F0_SCRIPT)
    assert "--isaac" in observed[0]

    def false_pass(command, **_kwargs):
        report = Path(command[command.index("--json") + 1])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"overall_status":"FAIL"}\n', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    with (
        patch.object(pipeline.subprocess, "run", side_effect=false_pass),
        pytest.raises(RuntimeError, match="must attest structural, simulator"),
    ):
        pipeline._run_f0(args, "false_pass_pipeline")

    with (
        patch.object(
            pipeline.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=2),
        ),
        pytest.raises(RuntimeError, match="structural/simulator gate failed"),
    ):
        pipeline._run_f0(args, "failing_pipeline")
