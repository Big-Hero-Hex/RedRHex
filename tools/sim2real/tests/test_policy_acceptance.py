from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from tools.sim2real.contracts import ContractError


ROOT = Path(__file__).parents[3]
EXPECTED_SEEDS = (42, 43, 44)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(
    root: Path,
    *,
    seed: int,
    stage: str,
    forward_speed: float = 0.20,
    lateral_leak: float = 0.10,
    yaw_leak: float = 0.20,
    max_fall_rate: float = 0.10,
    command_pass_ratio: float = 1.0,
    skill_pass_ratio: float = 1.0,
    backend: str = "native",
    profile_id: str = "measured-spring",
    profile_sha256: str = "a" * 64,
    calibration_status: str = "calibrated",
) -> tuple[Path, Path]:
    directory = root / f"seed-{seed}-{len(list(root.glob('seed-*'))) }"
    directory.mkdir()
    command_path = directory / "commands.csv"
    if stage == "forwardfast":
        command_rows = [
            {
                "command": "forward",
                "skill": "forward",
                "actual_forward_speed_mean": forward_speed,
                "actual_lateral_leak_mean": lateral_leak,
                "actual_yaw_leak_mean": yaw_leak,
                "success_duration_s": 2.5,
                "fall_rate": max_fall_rate,
                "accept_pass": "True",
            }
        ]
        eval_profile = "stage1"
        skill_metrics = {"forward": 1.0}
    else:
        command_rows = [
            {
                "command": skill,
                "skill": skill,
                "actual_forward_speed_mean": 0.0,
                "actual_lateral_leak_mean": 0.0,
                "actual_yaw_leak_mean": 0.0,
                "success_duration_s": 2.5,
                "fall_rate": max_fall_rate if skill == "yaw" else 0.05,
                "accept_pass": "True",
            }
            for skill in ("forward", "lateral", "diagonal", "yaw")
        ]
        eval_profile = "stage5"
        skill_metrics = {
            skill: skill_pass_ratio
            for skill in ("forward", "lateral", "diagonal", "yaw")
        }
    _write_csv(command_path, command_rows)

    summary_path = directory / "commands_summary.csv"
    metrics: list[dict[str, object]] = [
        {"metric": "evaluation.seed", "value": seed},
        {"metric": "eval.profile", "value": eval_profile},
        {"metric": "spring.backend", "value": backend},
        {"metric": "spring.calibration_status", "value": calibration_status},
        {
            "metric": "spring.checkpoint_calibration_status",
            "value": calibration_status,
        },
        {"metric": "spring.profile_id", "value": profile_id},
        {"metric": "spring.profile_sha256", "value": profile_sha256},
        {"metric": "artifact.command_csv_sha256", "value": _sha256(command_path)},
        {"metric": "acceptance.command_pass_ratio", "value": command_pass_ratio},
        {"metric": "acceptance.max_command_fall_rate", "value": max_fall_rate},
    ]
    metrics.extend(
        {"metric": f"acceptance.skill_pass_ratio.{skill}", "value": ratio}
        for skill, ratio in skill_metrics.items()
    )
    _write_csv(summary_path, metrics)
    return command_path, summary_path


def test_forwardfast_accepts_exactly_two_passing_calibrated_seeds(
    tmp_path: Path,
) -> None:
    from tools.sim2real.policy_acceptance import evaluate_policy_acceptance

    runs = [
        _artifacts(tmp_path, seed=42, stage="forwardfast"),
        _artifacts(tmp_path, seed=43, stage="forwardfast"),
        _artifacts(
            tmp_path,
            seed=44,
            stage="forwardfast",
            forward_speed=0.149,
        ),
    ]

    report = evaluate_policy_acceptance(stage="forwardfast", runs=runs)

    assert report["eligible"] is True
    assert report["status"] == "accepted"
    assert report["passing_seed_count"] == 2
    assert report["passing_seeds"] == [42, 43]
    assert report["thresholds"] == {
        "required_passing_seeds": 2,
        "forward_speed_m_s": 0.15,
        "lateral_leak_m_s": 0.12,
        "yaw_leak_rad_s": 0.30,
        "per_command_fall_rate": 0.20,
    }
    assert all(len(seed["command_csv_sha256"]) == 64 for seed in report["seeds"])
    assert all(len(seed["summary_csv_sha256"]) == 64 for seed in report["seeds"])


def test_forwardfast_uses_only_the_four_planned_physical_gates(
    tmp_path: Path,
) -> None:
    from tools.sim2real.policy_acceptance import evaluate_policy_acceptance

    runs = [
        _artifacts(tmp_path, seed=seed, stage="forwardfast")
        for seed in EXPECTED_SEEDS
    ]
    command_path, summary_path = runs[1]
    rows = list(csv.DictReader(command_path.open(encoding="utf-8")))
    rows[0]["accept_pass"] = "False"
    rows[0]["success_duration_s"] = "0.0"
    _write_csv(command_path, rows)
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    for row in summary_rows:
        if row["metric"] == "artifact.command_csv_sha256":
            row["value"] = _sha256(command_path)
    _write_csv(summary_path, summary_rows)

    report = evaluate_policy_acceptance(stage="forwardfast", runs=runs)

    seed_43 = next(seed for seed in report["seeds"] if seed["seed"] == 43)
    assert seed_43["passed"] is True
    assert set(seed_43["gates"]) == {
        "forward_speed",
        "lateral_leak",
        "yaw_leak",
        "per_command_fall_rate",
    }


def test_direct_requires_ratios_all_four_skills_and_max_per_command_fall(
    tmp_path: Path,
) -> None:
    from tools.sim2real.policy_acceptance import evaluate_policy_acceptance

    accepted = [
        _artifacts(tmp_path, seed=42, stage="direct"),
        _artifacts(tmp_path, seed=43, stage="direct"),
        _artifacts(
            tmp_path,
            seed=44,
            stage="direct",
            max_fall_rate=0.201,
        ),
    ]

    report = evaluate_policy_acceptance(stage="direct", runs=accepted)

    assert report["eligible"] is True
    assert report["passing_seeds"] == [42, 43]
    failed = next(seed for seed in report["seeds"] if seed["seed"] == 44)
    assert failed["gates"]["per_command_fall_rate"] is False
    assert report["thresholds"] == {
        "required_passing_seeds": 2,
        "command_pass_ratio": 0.70,
        "every_skill_pass_ratio": 0.60,
        "per_command_fall_rate": 0.20,
    }


def test_direct_rejects_missing_skill_even_when_summary_minimum_would_pass(
    tmp_path: Path,
) -> None:
    from tools.sim2real.policy_acceptance import evaluate_policy_acceptance

    runs = [_artifacts(tmp_path, seed=seed, stage="direct") for seed in EXPECTED_SEEDS]
    for _, summary_path in runs[:2]:
        rows = [
            row
            for row in csv.DictReader(summary_path.open(encoding="utf-8"))
            if row["metric"] != "acceptance.skill_pass_ratio.yaw"
        ]
        _write_csv(summary_path, rows)

    with pytest.raises(ContractError, match="all four skill pass ratios"):
        evaluate_policy_acceptance(stage="direct", runs=runs)


def test_direct_rejects_summary_ratios_that_do_not_match_command_rows(
    tmp_path: Path,
) -> None:
    from tools.sim2real.policy_acceptance import evaluate_policy_acceptance

    runs = [_artifacts(tmp_path, seed=seed, stage="direct") for seed in EXPECTED_SEEDS]
    _, summary_path = runs[-1]
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    next(row for row in rows if row["metric"] == "acceptance.command_pass_ratio")[
        "value"
    ] = "0.75"
    _write_csv(summary_path, rows)

    with pytest.raises(ContractError, match="command pass ratio does not match"):
        evaluate_policy_acceptance(stage="direct", runs=runs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_seed", "exactly seeds 42, 43, and 44"),
        ("nonfinite", "finite"),
        ("uncalibrated", "calibrated"),
        ("profile_mismatch", "identical spring backend/profile binding"),
        ("hash_mismatch", "command CSV hash"),
    ],
)
def test_acceptance_fails_closed_on_invalid_provenance_or_values(
    tmp_path: Path, mutation: str, message: str
) -> None:
    from tools.sim2real.policy_acceptance import evaluate_policy_acceptance

    runs = [_artifacts(tmp_path, seed=seed, stage="direct") for seed in EXPECTED_SEEDS]
    command_path, summary_path = runs[-1]
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    if mutation == "duplicate_seed":
        next(row for row in summary_rows if row["metric"] == "evaluation.seed")["value"] = "43"
    elif mutation == "nonfinite":
        command_rows = list(csv.DictReader(command_path.open(encoding="utf-8")))
        command_rows[0]["fall_rate"] = "nan"
        _write_csv(command_path, command_rows)
        next(row for row in summary_rows if row["metric"] == "artifact.command_csv_sha256")["value"] = _sha256(
            command_path
        )
    elif mutation == "uncalibrated":
        next(row for row in summary_rows if row["metric"] == "spring.calibration_status")["value"] = "uncalibrated"
    elif mutation == "profile_mismatch":
        next(row for row in summary_rows if row["metric"] == "spring.profile_sha256")["value"] = "b" * 64
    elif mutation == "hash_mismatch":
        next(row for row in summary_rows if row["metric"] == "artifact.command_csv_sha256")["value"] = "0" * 64
    _write_csv(summary_path, summary_rows)

    with pytest.raises(ContractError, match=message):
        evaluate_policy_acceptance(stage="direct", runs=runs)


def test_eval_command_sweep_exports_acceptance_evidence_and_fall_gate() -> None:
    source = (ROOT / "scripts/rsl_rl/eval_command_sweep.py").read_text(
        encoding="utf-8"
    )

    for field in (
        "actual_forward_speed_mean",
        "actual_lateral_leak_mean",
        "actual_yaw_leak_mean",
    ):
        assert f'"{field}"' in source
    assert '{"metric": "evaluation.seed", "value": int(agent_cfg.seed)}' in source
    assert '"metric": "artifact.command_csv_sha256"' in source
    assert '{"metric": "acceptance.max_command_fall_rate", "value": max_command_fall_rate}' in source
    assert "and (max_command_fall_rate <= args_cli.accept_max_fall_rate)" in source


def test_policy_acceptance_cli_requires_three_seed_artifact_pairs() -> None:
    from tools.sim2real.cli import build_parser

    args = build_parser().parse_args(
        [
            "validate-policy-acceptance",
            "--stage",
            "forwardfast",
            "--seed-42-command",
            "42.csv",
            "--seed-42-summary",
            "42_summary.csv",
            "--seed-43-command",
            "43.csv",
            "--seed-43-summary",
            "43_summary.csv",
            "--seed-44-command",
            "44.csv",
            "--seed-44-summary",
            "44_summary.csv",
            "--output",
            "acceptance.json",
        ]
    )

    assert args.command == "validate-policy-acceptance"
    assert args.stage == "forwardfast"
    assert args.seed_42_command == Path("42.csv")
    assert args.seed_44_summary == Path("44_summary.csv")
