from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.sim2real.student_teacher_gap import (
    REQUIRED_ABLATIONS,
    StudentTeacherGapError,
    evaluate_student_teacher_gap,
)


def _write_run(root: Path, policy: str, seed: int, *, accepted: bool = True) -> dict[str, object]:
    command = root / f"{policy}-{seed}.csv"
    with command.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("command", "cmd_vx", "cmd_vy", "cmd_wz", "mae_vx", "accept_pass"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "command": "forward",
                "cmd_vx": 0.35,
                "cmd_vy": 0.0,
                "cmd_wz": 0.0,
                "mae_vx": 0.05 + seed * 0.001,
                "accept_pass": accepted,
            }
        )
    summary = root / f"{policy}-{seed}-summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerow({"metric": "tracking.mean_abs_vx", "value": 0.05})
        writer.writerow({"metric": "acceptance.overall_status", "value": "PASS" if accepted else "FAIL"})
    provenance = {}
    if policy == "v2_ppo":
        provenance = {
            "checkpoint_kind": "student_ppo_v2",
            "observation_contract_sha256": "a" * 64,
            "action_contract_sha256": "b" * 64,
            "calibration_sha256": "c" * 64,
        }
    return {
        "policy": policy,
        "seed": seed,
        "domain": "flat",
        "command_csv": command.name,
        "summary_csv": summary.name,
        "provenance": provenance,
    }


def _manifest(tmp_path: Path) -> Path:
    policies = sorted(set(REQUIRED_ABLATIONS) | {"teacher_a"})
    runs = [_write_run(tmp_path, policy, seed) for policy in policies for seed in (1, 2, 3)]
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": runs,
                "gates": {
                    "no_privileged_leak": True,
                    "torch_onnx_parity": True,
                    "sensor_replay": True,
                    "contract_provenance": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_gap_report_requires_and_passes_all_promotion_evidence(tmp_path: Path) -> None:
    result = evaluate_student_teacher_gap(_manifest(tmp_path))
    assert result["promotion"]["pass"] is True
    assert result["contact_supervision"]["status"] == "blocked"
    assert set(result["teacher_gap"]) == {"legacy_student", "v2_distilled", "v2_ppo"}


def test_gap_report_rejects_nonidentical_command_set(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    target = next(run for run in payload["runs"] if run["policy"] == "v2_ppo" and run["seed"] == 1)
    command = tmp_path / target["command_csv"]
    text = command.read_text(encoding="utf-8").replace("0.35", "0.36")
    command.write_text(text, encoding="utf-8")

    with pytest.raises(StudentTeacherGapError, match="command set differs"):
        evaluate_student_teacher_gap(manifest)
