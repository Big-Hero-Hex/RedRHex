from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ContractError
from .traces import sha256_file


STAGES = ("forwardfast", "direct")
EXPECTED_SEEDS = (42, 43, 44)
DIRECT_SKILLS = ("forward", "lateral", "diagonal", "yaw")

REQUIRED_PASSING_SEEDS = 2
FORWARD_SPEED_LIMIT_M_S = 0.15
FORWARD_LATERAL_LEAK_LIMIT_M_S = 0.12
FORWARD_YAW_LEAK_LIMIT_RAD_S = 0.30
PER_COMMAND_FALL_RATE_LIMIT = 0.20
DIRECT_COMMAND_PASS_RATIO_LIMIT = 0.70
DIRECT_SKILL_PASS_RATIO_LIMIT = 0.60

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _read_rows(path: Path, *, kind: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ContractError(f"{kind} CSV has no header: {path}")
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise ContractError(f"cannot read {kind} CSV {path}: {exc}") from exc
    if not rows:
        raise ContractError(f"{kind} CSV has no rows: {path}")
    return rows


def _summary_metrics(path: Path) -> dict[str, str]:
    rows = _read_rows(path, kind="summary")
    if not {"metric", "value"}.issubset(rows[0]):
        raise ContractError("summary CSV requires metric and value columns")
    metrics: dict[str, str] = {}
    for row in rows:
        name = str(row.get("metric", "")).strip()
        if not name:
            raise ContractError("summary CSV contains a missing metric name")
        if name in metrics:
            raise ContractError(f"summary CSV contains duplicate metric: {name}")
        metrics[name] = str(row.get("value", "")).strip()
    return metrics


def _required(metrics: Mapping[str, str], name: str) -> str:
    value = metrics.get(name)
    if value is None or not value.strip() or value.strip().lower() == "none":
        raise ContractError(f"summary CSV is missing {name}")
    return value.strip()


def _finite(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric and finite") from exc
    if not math.isfinite(parsed):
        raise ContractError(f"{name} must be finite")
    return parsed


def _ratio(value: object, name: str) -> float:
    parsed = _finite(value, name)
    if parsed < 0.0 or parsed > 1.0:
        raise ContractError(f"{name} must be between zero and one")
    return parsed


def _boolean(value: object, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ContractError(f"{name} must be true or false")


def _seed(metrics: Mapping[str, str]) -> int:
    raw = _required(metrics, "evaluation.seed")
    try:
        seed = int(raw)
    except ValueError as exc:
        raise ContractError("evaluation.seed must be an integer") from exc
    if str(seed) != raw:
        raise ContractError("evaluation.seed must be an integer")
    return seed


def _validate_binding(metrics: Mapping[str, str]) -> tuple[str, str, str]:
    backend = _required(metrics, "spring.backend")
    if backend not in {"explicit", "native"}:
        raise ContractError("spring.backend must be explicit or native")
    for name in (
        "spring.calibration_status",
        "spring.checkpoint_calibration_status",
    ):
        if _required(metrics, name) != "calibrated":
            raise ContractError(f"{name} must be calibrated for policy acceptance")
    profile_id = _required(metrics, "spring.profile_id")
    profile_sha256 = _required(metrics, "spring.profile_sha256")
    if _SHA256_PATTERN.fullmatch(profile_sha256) is None:
        raise ContractError("spring.profile_sha256 must be a lowercase SHA-256")
    return backend, profile_id, profile_sha256


def _load_command_rows(path: Path, *, stage: str) -> list[dict[str, Any]]:
    raw_rows = _read_rows(path, kind="command")
    common = {"command", "skill", "fall_rate", "accept_pass"}
    forward = {
        "actual_forward_speed_mean",
        "actual_lateral_leak_mean",
        "actual_yaw_leak_mean",
    }
    required = common | (forward if stage == "forwardfast" else set())
    missing = required - set(raw_rows[0])
    if missing:
        raise ContractError(
            f"command CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    rows: list[dict[str, Any]] = []
    command_names: set[str] = set()
    for index, raw in enumerate(raw_rows, start=2):
        command = str(raw.get("command", "")).strip()
        skill = str(raw.get("skill", "")).strip()
        if not command or not skill:
            raise ContractError(f"command CSV row {index} has missing command metadata")
        if command in command_names:
            raise ContractError(f"command CSV contains duplicate command: {command}")
        command_names.add(command)
        row: dict[str, Any] = {
            "command": command,
            "skill": skill,
            "fall_rate": _ratio(raw.get("fall_rate"), f"{command}.fall_rate"),
            "accept_pass": _boolean(
                raw.get("accept_pass"), f"{command}.accept_pass"
            ),
        }
        if stage == "forwardfast":
            row.update(
                {
                    "actual_forward_speed_mean": _finite(
                        raw.get("actual_forward_speed_mean"),
                        f"{command}.actual_forward_speed_mean",
                    ),
                    "actual_lateral_leak_mean": _finite(
                        raw.get("actual_lateral_leak_mean"),
                        f"{command}.actual_lateral_leak_mean",
                    ),
                    "actual_yaw_leak_mean": _finite(
                        raw.get("actual_yaw_leak_mean"),
                        f"{command}.actual_yaw_leak_mean",
                    ),
                }
            )
        rows.append(row)

    skills = {row["skill"] for row in rows}
    expected_skills = {"forward"} if stage == "forwardfast" else set(DIRECT_SKILLS)
    if skills != expected_skills:
        if stage == "direct":
            raise ContractError("Direct command CSV must contain all four skills")
        raise ContractError("ForwardFast command CSV must contain only forward commands")
    return rows


def _load_run(
    command_csv: str | Path, summary_csv: str | Path, *, stage: str
) -> dict[str, Any]:
    command_path = Path(command_csv)
    summary_path = Path(summary_csv)
    metrics = _summary_metrics(summary_path)
    seed = _seed(metrics)
    binding = _validate_binding(metrics)

    expected_profile = "stage1" if stage == "forwardfast" else {"stage5", "full"}
    eval_profile = _required(metrics, "eval.profile")
    if stage == "forwardfast" and eval_profile != expected_profile:
        raise ContractError("ForwardFast acceptance requires eval.profile=stage1")
    if stage == "direct" and eval_profile not in expected_profile:
        raise ContractError("Direct acceptance requires eval.profile=stage5 or full")

    command_sha256 = sha256_file(command_path)
    recorded_sha256 = _required(metrics, "artifact.command_csv_sha256")
    if recorded_sha256 != command_sha256:
        raise ContractError("summary command CSV hash does not match the command artifact")
    rows = _load_command_rows(command_path, stage=stage)
    computed_max_fall = max(float(row["fall_rate"]) for row in rows)
    summary_max_fall = _ratio(
        _required(metrics, "acceptance.max_command_fall_rate"),
        "acceptance.max_command_fall_rate",
    )
    if not math.isclose(computed_max_fall, summary_max_fall, rel_tol=0.0, abs_tol=1.0e-12):
        raise ContractError(
            "acceptance.max_command_fall_rate does not match the command CSV"
        )

    return {
        "seed": seed,
        "binding": binding,
        "command_csv": str(command_path.resolve()),
        "summary_csv": str(summary_path.resolve()),
        "command_csv_sha256": command_sha256,
        "summary_csv_sha256": sha256_file(summary_path),
        "rows": rows,
        "metrics": metrics,
    }


def _evaluate_forwardfast(run: Mapping[str, Any]) -> dict[str, Any]:
    rows = run["rows"]
    minimum_speed = min(float(row["actual_forward_speed_mean"]) for row in rows)
    maximum_lateral_leak = max(
        float(row["actual_lateral_leak_mean"]) for row in rows
    )
    maximum_yaw_leak = max(float(row["actual_yaw_leak_mean"]) for row in rows)
    maximum_fall_rate = max(float(row["fall_rate"]) for row in rows)
    gates = {
        "forward_speed": minimum_speed >= FORWARD_SPEED_LIMIT_M_S,
        "lateral_leak": maximum_lateral_leak
        <= FORWARD_LATERAL_LEAK_LIMIT_M_S,
        "yaw_leak": maximum_yaw_leak <= FORWARD_YAW_LEAK_LIMIT_RAD_S,
        "per_command_fall_rate": maximum_fall_rate <= PER_COMMAND_FALL_RATE_LIMIT,
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "minimum_forward_speed_m_s": minimum_speed,
        "maximum_lateral_leak_m_s": maximum_lateral_leak,
        "maximum_yaw_leak_rad_s": maximum_yaw_leak,
        "maximum_command_fall_rate": maximum_fall_rate,
    }


def _evaluate_direct(run: Mapping[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    command_pass_ratio = _ratio(
        _required(metrics, "acceptance.command_pass_ratio"),
        "acceptance.command_pass_ratio",
    )
    rows = run["rows"]
    computed_command_pass_ratio = sum(
        1 for row in rows if row["accept_pass"]
    ) / len(rows)
    if not math.isclose(
        command_pass_ratio,
        computed_command_pass_ratio,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ContractError(
            "summary command pass ratio does not match the command CSV"
        )
    skill_ratios: dict[str, float] = {}
    for skill in DIRECT_SKILLS:
        name = f"acceptance.skill_pass_ratio.{skill}"
        if name not in metrics:
            raise ContractError("Direct summary CSV must contain all four skill pass ratios")
        skill_ratios[skill] = _ratio(metrics[name], name)
        skill_rows = [row for row in rows if row["skill"] == skill]
        computed_skill_ratio = sum(
            1 for row in skill_rows if row["accept_pass"]
        ) / len(skill_rows)
        if not math.isclose(
            skill_ratios[skill],
            computed_skill_ratio,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ContractError(
                f"summary {skill} skill pass ratio does not match the command CSV"
            )
    maximum_fall_rate = max(float(row["fall_rate"]) for row in rows)
    gates = {
        "command_pass_ratio": command_pass_ratio
        >= DIRECT_COMMAND_PASS_RATIO_LIMIT,
        "every_skill_pass_ratio": min(skill_ratios.values())
        >= DIRECT_SKILL_PASS_RATIO_LIMIT,
        "per_command_fall_rate": maximum_fall_rate <= PER_COMMAND_FALL_RATE_LIMIT,
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "command_pass_ratio": command_pass_ratio,
        "skill_pass_ratios": skill_ratios,
        "maximum_command_fall_rate": maximum_fall_rate,
    }


def evaluate_policy_acceptance(
    *,
    stage: str,
    runs: Sequence[tuple[str | Path, str | Path]],
) -> dict[str, Any]:
    """Evaluate the fixed 42/43/44 retraining gate from hash-bound sweep CSVs."""

    normalized_stage = str(stage).lower()
    if normalized_stage not in STAGES:
        raise ContractError("policy acceptance stage must be forwardfast or direct")
    if len(runs) != len(EXPECTED_SEEDS):
        raise ContractError("policy acceptance requires exactly seeds 42, 43, and 44")

    loaded = [
        _load_run(command_csv, summary_csv, stage=normalized_stage)
        for command_csv, summary_csv in runs
    ]
    seeds = [int(run["seed"]) for run in loaded]
    if sorted(seeds) != list(EXPECTED_SEEDS):
        raise ContractError("policy acceptance requires exactly seeds 42, 43, and 44")
    bindings = {run["binding"] for run in loaded}
    if len(bindings) != 1:
        raise ContractError(
            "all seeds must use an identical spring backend/profile binding"
        )

    evaluator = (
        _evaluate_forwardfast if normalized_stage == "forwardfast" else _evaluate_direct
    )
    seed_reports: list[dict[str, Any]] = []
    for run in sorted(loaded, key=lambda value: value["seed"]):
        evidence = evaluator(run)
        seed_reports.append(
            {
                "seed": run["seed"],
                "command_csv": run["command_csv"],
                "summary_csv": run["summary_csv"],
                "command_csv_sha256": run["command_csv_sha256"],
                "summary_csv_sha256": run["summary_csv_sha256"],
                **evidence,
            }
        )

    passing_seeds = [
        int(report["seed"]) for report in seed_reports if report["passed"]
    ]
    eligible = len(passing_seeds) >= REQUIRED_PASSING_SEEDS
    backend, profile_id, profile_sha256 = next(iter(bindings))
    if normalized_stage == "forwardfast":
        thresholds = {
            "required_passing_seeds": REQUIRED_PASSING_SEEDS,
            "forward_speed_m_s": FORWARD_SPEED_LIMIT_M_S,
            "lateral_leak_m_s": FORWARD_LATERAL_LEAK_LIMIT_M_S,
            "yaw_leak_rad_s": FORWARD_YAW_LEAK_LIMIT_RAD_S,
            "per_command_fall_rate": PER_COMMAND_FALL_RATE_LIMIT,
        }
    else:
        thresholds = {
            "required_passing_seeds": REQUIRED_PASSING_SEEDS,
            "command_pass_ratio": DIRECT_COMMAND_PASS_RATIO_LIMIT,
            "every_skill_pass_ratio": DIRECT_SKILL_PASS_RATIO_LIMIT,
            "per_command_fall_rate": PER_COMMAND_FALL_RATE_LIMIT,
        }
    return {
        "schema_version": 1,
        "stage": normalized_stage,
        "status": "accepted" if eligible else "rejected",
        "eligible": eligible,
        "passing_seed_count": len(passing_seeds),
        "passing_seeds": passing_seeds,
        "spring_binding": {
            "backend": backend,
            "calibration_status": "calibrated",
            "profile_id": profile_id,
            "profile_sha256": profile_sha256,
        },
        "thresholds": thresholds,
        "seeds": seed_reports,
    }
