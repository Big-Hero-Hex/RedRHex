from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
CORE_POLICIES = ("teacher_a", "legacy_student", "v2_distilled", "v2_ppo")
REQUIRED_ABLATIONS = (
    "legacy_student",
    "v2_no_aux",
    "v2_velocity",
    "v2_velocity_dynamics",
    "v2_distilled",
    "v2_ppo",
)
PROMOTION_GATES = (
    "no_privileged_leak",
    "torch_onnx_parity",
    "sensor_replay",
    "contract_provenance",
)


class StudentTeacherGapError(ValueError):
    """Raised when evaluation artifacts cannot support a valid comparison."""


def _reject_constant(value: str) -> None:
    raise StudentTeacherGapError(f"non-finite JSON constant {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise StudentTeacherGapError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudentTeacherGapError("evaluation manifest must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise StudentTeacherGapError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StudentTeacherGapError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise StudentTeacherGapError(f"{label} must be finite")
    return number


def _read_command_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise StudentTeacherGapError(f"cannot read command CSV {path}: {exc}") from exc
    required = {"command", "cmd_vx", "cmd_vy", "cmd_wz", "accept_pass"}
    if not rows:
        raise StudentTeacherGapError(f"command CSV is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise StudentTeacherGapError(
            f"command CSV {path} lacks columns: {', '.join(sorted(missing))}"
        )
    commands = [row["command"] for row in rows]
    if len(commands) != len(set(commands)):
        raise StudentTeacherGapError(f"command CSV has duplicate command names: {path}")
    return rows


def _read_summary_csv(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise StudentTeacherGapError(f"cannot read summary CSV {path}: {exc}") from exc
    result: dict[str, str] = {}
    for row in rows:
        metric = row.get("metric")
        if not metric or metric in result:
            raise StudentTeacherGapError(f"invalid or duplicate summary metric in {path}")
        result[metric] = row.get("value", "")
    return result


def _is_pass(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "pass", "yes"}


def _command_signature(rows: Iterable[Mapping[str, str]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            row["command"],
            _finite_number(row["cmd_vx"], f"{row['command']}.cmd_vx"),
            _finite_number(row["cmd_vy"], f"{row['command']}.cmd_vy"),
            _finite_number(row["cmd_wz"], f"{row['command']}.cmd_wz"),
        )
        for row in rows
    )


def _numeric_metrics(
    command_rows: Iterable[Mapping[str, str]], summary: Mapping[str, str]
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    ignored = {"command", "skill", "accept_pass"}
    for row in command_rows:
        for name, raw in row.items():
            if name in ignored or raw in (None, ""):
                continue
            try:
                values[f"command.{name}"].append(_finite_number(raw, name))
            except StudentTeacherGapError:
                continue
    result = {
        name: statistics.fmean(items) for name, items in values.items() if items
    }
    for name, raw in summary.items():
        try:
            result[f"summary.{name}"] = _finite_number(raw, name)
        except StudentTeacherGapError:
            continue
    return result


def _resolve_run(root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"policy", "seed", "domain", "command_csv", "summary_csv"}
    missing = required - set(value)
    if missing:
        raise StudentTeacherGapError(f"run lacks keys: {', '.join(sorted(missing))}")
    policy = str(value["policy"])
    seed = int(value["seed"])
    domain = str(value["domain"])
    command_path = (root / str(value["command_csv"])).resolve()
    summary_path = (root / str(value["summary_csv"])).resolve()
    rows = _read_command_csv(command_path)
    summary = _read_summary_csv(summary_path)
    accepted = sorted(row["command"] for row in rows if _is_pass(row["accept_pass"]))
    provenance = value.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise StudentTeacherGapError("run provenance must be an object")
    return {
        "policy": policy,
        "seed": seed,
        "domain": domain,
        "key": (seed, domain),
        "signature": _command_signature(rows),
        "accepted_commands": accepted,
        "metrics": _numeric_metrics(rows, summary),
        "overall_pass": summary.get("acceptance.overall_status", "").upper() == "PASS",
        "provenance": dict(provenance),
        "command_csv": str(command_path),
        "command_csv_sha256": _sha256(command_path),
        "summary_csv": str(summary_path),
        "summary_csv_sha256": _sha256(summary_path),
    }


def _aggregate(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_policy_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for metric, value in run["metrics"].items():
            by_policy_metric[str(run["policy"])][metric].append(float(value))
    result: dict[str, Any] = {}
    for policy, metrics in sorted(by_policy_metric.items()):
        result[policy] = {
            metric: {
                "count": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
            }
            for metric, values in sorted(metrics.items())
        }
    return result


def evaluate_student_teacher_gap(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    manifest = _load_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise StudentTeacherGapError(f"schema_version must be {SCHEMA_VERSION}")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise StudentTeacherGapError("manifest runs must be a non-empty list")
    if any(not isinstance(run, Mapping) for run in raw_runs):
        raise StudentTeacherGapError("each run must be an object")

    runs = [_resolve_run(path.parent, run) for run in raw_runs]
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_policy[run["policy"]].append(run)

    missing_core = sorted(set(CORE_POLICIES) - set(by_policy))
    if missing_core:
        raise StudentTeacherGapError(
            "comparison lacks core policies: " + ", ".join(missing_core)
        )
    reference_keys = {run["key"] for run in by_policy["teacher_a"]}
    if len(reference_keys) < 3:
        raise StudentTeacherGapError("Teacher A comparison requires at least three seed/domain runs")
    reference_signatures = {
        run["key"]: run["signature"] for run in by_policy["teacher_a"]
    }
    for policy in CORE_POLICIES:
        keys = {run["key"] for run in by_policy[policy]}
        if keys != reference_keys:
            raise StudentTeacherGapError(
                f"{policy} seed/domain set differs from Teacher A: {sorted(keys)}"
            )
        for run in by_policy[policy]:
            if run["signature"] != reference_signatures[run["key"]]:
                raise StudentTeacherGapError(
                    f"{policy} command set differs for seed/domain {run['key']}"
                )

    aggregates = _aggregate(runs)
    teacher_metrics = aggregates["teacher_a"]
    teacher_gap: dict[str, dict[str, float]] = {}
    for policy in ("legacy_student", "v2_distilled", "v2_ppo"):
        teacher_gap[policy] = {}
        for metric in sorted(set(teacher_metrics) & set(aggregates[policy])):
            teacher_gap[policy][metric] = (
                aggregates[policy][metric]["mean"] - teacher_metrics[metric]["mean"]
            )

    gates = manifest.get("gates", {})
    if not isinstance(gates, Mapping):
        raise StudentTeacherGapError("manifest gates must be an object")
    failed_gates = [name for name in PROMOTION_GATES if gates.get(name) is not True]
    missing_ablations = sorted(set(REQUIRED_ABLATIONS) - set(by_policy))
    ppo_by_key = {run["key"]: run for run in by_policy["v2_ppo"]}
    teacher_by_key = {run["key"]: run for run in by_policy["teacher_a"]}
    failed_ppo_runs = [list(key) for key, run in ppo_by_key.items() if not run["overall_pass"]]
    accepted_set_mismatches = [
        list(key)
        for key in sorted(reference_keys)
        if ppo_by_key[key]["accepted_commands"] != teacher_by_key[key]["accepted_commands"]
    ]
    provenance_failures: list[list[Any]] = []
    for key, run in ppo_by_key.items():
        provenance = run["provenance"]
        required_hashes = (
            "observation_contract_sha256",
            "action_contract_sha256",
            "calibration_sha256",
        )
        if provenance.get("checkpoint_kind") != "student_ppo_v2" or any(
            not isinstance(provenance.get(name), str) or len(provenance[name]) != 64
            for name in required_hashes
        ):
            provenance_failures.append(list(key))

    promotion_pass = not any(
        (failed_gates, missing_ablations, failed_ppo_runs, accepted_set_mismatches, provenance_failures)
    )
    public_runs = []
    for run in runs:
        public_runs.append({key: value for key, value in run.items() if key not in {"key", "signature"}})
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(path),
        "source_manifest_sha256": _sha256(path),
        "contact_supervision": {"status": "blocked", "reason": "validated simulator contact labels unavailable"},
        "runs": public_runs,
        "aggregate": aggregates,
        "teacher_gap": teacher_gap,
        "promotion": {
            "pass": promotion_pass,
            "failed_gates": failed_gates,
            "missing_ablations": missing_ablations,
            "failed_v2_ppo_runs": failed_ppo_runs,
            "teacher_command_set_mismatches": accepted_set_mismatches,
            "provenance_failures": provenance_failures,
        },
    }


def write_gap_report(result: Mapping[str, Any], json_path: str | Path, csv_path: str | Path) -> None:
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{json_target.name}-", dir=json_target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json_payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, json_target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    handle, temporary = tempfile.mkstemp(prefix=f".{csv_target.name}-", dir=csv_target.parent)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("policy", "metric", "count", "mean", "std", "gap_from_teacher"))
            aggregate = result["aggregate"]
            gaps = result["teacher_gap"]
            for policy, metrics in aggregate.items():
                for metric, stats in metrics.items():
                    writer.writerow(
                        (
                            policy,
                            metric,
                            stats["count"],
                            stats["mean"],
                            stats["std"],
                            gaps.get(policy, {}).get(metric, ""),
                        )
                    )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, csv_target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
