from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .compare import compare_traces
from .contracts import CalibrationProfileV1, ContractError, ScenarioSpecV1, load_profile
from .metrics import compute_subsystem_metrics
from .provenance import validate_real_trace_provenance
from .runtime_provenance import production_runtime_provenance
from .scenarios import load_scenario
from .sweep import candidate_cache_key, validate_sweep_candidates
from .traces import LoadedTrace, load_trace, sha256_json


_SWEEP_MODES = {"one-factor", "coarse-grid"}
_SCENE_MODES = {"fixed-base", "free-root", "contact"}
_CACHEABLE_STATUSES = {"completed", "cached"}
_RUNTIME_PROVENANCE_FIELDS = {
    "git_sha",
    "asset_sha256",
    "config_sha256",
    "redrhex_module_path",
    "redrhex_module_sha256",
    "isaaclab_version",
    "isaacsim_version",
    "characterization_runner_sha256",
    "sweep_runner_sha256",
    "runtime_bundle_sha256",
}
_DERIVED_PROVENANCE_FIELDS = _RUNTIME_PROVENANCE_FIELDS | {
    "real_trace_sha256",
    "real_metadata_sha256",
    "known_load_trace_sha256",
    "known_load_metadata_sha256",
    "audit_artifact_sha256",
    "audit_report_sha256",
}


def _main_effort_limit(profile: CalibrationProfileV1) -> Any:
    section = profile.simulation_physics.get("main_drive", {})
    return section.get("effort_limit") if isinstance(section, Mapping) else None


def _validate_prefit_audit(
    artifact: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    profile: CalibrationProfileV1,
) -> dict[str, Any]:
    # Promotion imports sweep cache helpers, so keep this dependency lazy.
    from .promotion import _derive_audit

    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ContractError("pre-fit audit artifact_root must be an existing directory")
    report = _derive_audit(root, artifact, profile)
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise ContractError("pre-fit audit report is missing derived checks")
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise ContractError("pre-fit audit failed: " + ", ".join(failed))
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is missing or invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return payload


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ContractError(f"{label} must be a {qualifier} integer")
    return value


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ContractError(f"{label} must be a safe relative path")
    return value


def _command_prefix(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)) or not value:
        raise ContractError("command_prefix is required when sweep execution is enabled")
    command = tuple(value)
    if any(not isinstance(part, str) or not part for part in command):
        raise ContractError("command_prefix entries must be non-empty strings")
    return command


def _execution_provenance(
    provenance: Mapping[str, Any],
    *,
    scene_mode: str,
    headless: bool,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ContractError("provenance must be a JSON object")
    result = dict(provenance)
    result.update(
        {
            "sweep_runner_schema_version": 1,
            "scene_mode": scene_mode,
            "headless": headless,
            "seed": seed,
            "device": device,
        }
    )
    try:
        sha256_json(result)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"provenance must contain finite JSON values: {exc}") from exc
    return result


def _bind_runtime_provenance(
    provenance: Mapping[str, Any],
    provider: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ContractError("provenance must be a JSON object")
    conflicts = sorted(set(provenance).intersection(_DERIVED_PROVENANCE_FIELDS))
    if conflicts:
        raise ContractError(
            "derived provenance fields cannot be overridden: " + ", ".join(conflicts)
        )
    if not callable(provider):
        raise ContractError("provenance_provider must be callable")
    try:
        derived = provider()
    except ContractError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"runtime provenance provider failed: {exc}") from exc
    if not isinstance(derived, Mapping):
        raise ContractError("runtime provenance provider must return a JSON object")
    missing = sorted(_RUNTIME_PROVENANCE_FIELDS - set(derived))
    if missing:
        raise ContractError("runtime provenance missing fields: " + ", ".join(missing))
    git_sha = derived["git_sha"]
    if (
        not isinstance(git_sha, str)
        or len(git_sha) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in git_sha)
    ):
        raise ContractError("runtime provenance git_sha must be a Git digest")
    module_path = derived["redrhex_module_path"]
    if not isinstance(module_path, str) or not Path(module_path).is_absolute():
        raise ContractError("runtime provenance redrhex_module_path must be absolute")
    for field in ("isaaclab_version", "isaacsim_version"):
        if not isinstance(derived[field], str) or not derived[field]:
            raise ContractError(f"runtime provenance {field} must be non-empty")
    for field in sorted(
        _RUNTIME_PROVENANCE_FIELDS
        - {
            "git_sha",
            "redrhex_module_path",
            "isaaclab_version",
            "isaacsim_version",
        }
    ):
        value = derived[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ContractError(f"runtime provenance {field} must be a SHA-256 digest")
    if derived["redrhex_module_sha256"] != derived["config_sha256"]:
        raise ContractError(
            "runtime provenance RedRhex module hash must match config_sha256"
        )
    return {**dict(provenance), **dict(derived)}


def _candidate_status(
    entry: Mapping[str, Any],
    *,
    status: str,
    attempt: int = 0,
    run_output: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "index": entry["index"],
        "cache_key": entry["cache_key"],
        "profile": entry["profile"],
        "profile_sha256": entry["profile_sha256"],
        "status": status,
        "attempt": attempt,
        "run_output": run_output,
    }
    payload.update(details)
    return payload


def _load_status(path: Path, entry: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _json_object(path, f"candidate {entry['index']} status")
    expected = {
        "schema_version": 1,
        "index": entry["index"],
        "cache_key": entry["cache_key"],
        "profile": entry["profile"],
        "profile_sha256": entry["profile_sha256"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ContractError(
                f"candidate {entry['index']} status {field} does not match this sweep"
            )
    if payload.get("status") not in {
        "pending",
        "generated",
        "running",
        "completed",
        "cached",
        "failed",
    }:
        raise ContractError(f"candidate {entry['index']} status is unsupported")
    _positive_int(payload.get("attempt"), "candidate status attempt", allow_zero=True)
    run_output = payload.get("run_output")
    if run_output is not None:
        _safe_relative(run_output, "candidate status run_output")
    return payload


def _verify_result_fields(
    result: Mapping[str, Any],
    *,
    scenario: ScenarioSpecV1,
    profile: CalibrationProfileV1,
    scene_mode: str,
    trace_sha256: str,
    output: Path,
) -> None:
    expected = {
        "schema_version": 1,
        "scenario_id": scenario.scenario_id,
        "mode": scene_mode,
        "profile_id": profile.profile_id,
        "trace_sha256": trace_sha256,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ContractError(
                f"results.json {field} mismatch: expected {value!r}, got {result.get(field)!r}"
            )
    _positive_int(result.get("steps"), "results.json steps")
    physics_dt = result.get("physics_dt_s")
    if (
        isinstance(physics_dt, bool)
        or not isinstance(physics_dt, (int, float))
        or not math.isfinite(float(physics_dt))
        or float(physics_dt) <= 0.0
    ):
        raise ContractError("results.json physics_dt_s must be positive and finite")
    audit_name = _safe_relative(result.get("runtime_audit"), "results.json runtime_audit")
    audit_path = output.joinpath(*PurePosixPath(audit_name).parts)
    _json_object(audit_path, "runtime audit")


def _verify_artifact(
    output: Path,
    *,
    scenario: ScenarioSpecV1,
    profile: CalibrationProfileV1,
    scene_mode: str,
    provenance: Mapping[str, Any],
    expected_metadata_sha256: str | None = None,
) -> dict[str, Any]:
    result = _json_object(output / "results.json", "results.json")
    try:
        loaded = load_trace(
            output,
            scenario=scenario,
            profile=profile,
            expected_metadata_sha256=expected_metadata_sha256,
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"trace artifact is missing or invalid: {exc}") from exc
    if loaded.manifest.source != "sim":
        raise ContractError("cached trace source must be sim")
    for field in (
        "git_sha",
        "asset_sha256",
        "config_sha256",
        "redrhex_module_path",
        "redrhex_module_sha256",
        "isaaclab_version",
        "isaacsim_version",
        "characterization_runner_sha256",
        "runtime_bundle_sha256",
    ):
        if loaded.manifest.metadata.get(field) != provenance[field]:
            raise ContractError(
                f"artifact provenance {field} mismatch: expected {provenance[field]!r}, "
                f"got {loaded.manifest.metadata.get(field)!r}"
            )
    trace_sha256 = loaded.manifest.provenance["trace_sha256"]
    _verify_result_fields(
        result,
        scenario=scenario,
        profile=profile,
        scene_mode=scene_mode,
        trace_sha256=trace_sha256,
        output=output,
    )
    return {
        "trace_sha256": trace_sha256,
        "metadata_sha256": loaded.metadata_sha256,
        "steps": result["steps"],
        "physics_dt_s": float(result["physics_dt_s"]),
    }


def _comparison_details(
    output: Path,
    *,
    real_trace: LoadedTrace,
    scenario: ScenarioSpecV1,
    sim_trace_sha256: str,
    require_existing: bool,
) -> dict[str, Any]:
    comparison = compare_traces(real_trace, output, scenario=scenario)
    payload = {
        **comparison,
        "real_trace_sha256": real_trace.manifest.provenance["trace_sha256"],
        "sim_trace_sha256": sim_trace_sha256,
    }
    comparison_path = output / "comparison.json"
    if comparison_path.exists():
        if _json_object(comparison_path, "comparison.json") != payload:
            raise ContractError("cached comparison mismatch")
    elif require_existing:
        raise ContractError("cached comparison.json is missing")
    else:
        _atomic_json(comparison_path, payload)
    return {
        "comparison_sha256": sha256_json(payload),
        "metrics": payload["subsystems"],
    }


def _verify_candidate(
    output: Path,
    *,
    scenario: ScenarioSpecV1,
    profile: CalibrationProfileV1,
    scene_mode: str,
    provenance: Mapping[str, Any],
    real_trace: LoadedTrace,
    require_comparison: bool,
    expected_metadata_sha256: str | None = None,
) -> dict[str, Any]:
    verified = _verify_artifact(
        output,
        scenario=scenario,
        profile=profile,
        scene_mode=scene_mode,
        provenance=provenance,
        expected_metadata_sha256=expected_metadata_sha256,
    )
    return {
        **verified,
        **_comparison_details(
            output,
            real_trace=real_trace,
            scenario=scenario,
            sim_trace_sha256=verified["trace_sha256"],
            require_existing=require_comparison,
        ),
    }


def _status_entry(status: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "index": status["index"],
        "profile": status["profile"],
        "profile_sha256": status["profile_sha256"],
        "cache_key": status["cache_key"],
        "status_file": f"statuses/{int(status['index']):04d}.json",
        "status": status["status"],
        "attempt": status["attempt"],
        "run_output": status.get("run_output"),
    }
    if "trace_sha256" in status:
        result["trace_sha256"] = status["trace_sha256"]
    if "error" in status:
        result["error"] = status["error"]
    for field in ("metadata_sha256", "comparison", "comparison_sha256", "metrics"):
        if field in status:
            result[field] = status[field]
    return result


def _payloads(
    *,
    sweep_sha256: str,
    sweep_mode: str,
    scenario: ScenarioSpecV1,
    scene_mode: str,
    provenance_sha256: str,
    statuses: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = [_status_entry(status) for status in statuses]
    index = {
        "schema_version": 1,
        "sweep_sha256": sweep_sha256,
        "sweep_mode": sweep_mode,
        "scenario_id": scenario.scenario_id,
        "scene_mode": scene_mode,
        "provenance_sha256": provenance_sha256,
        "candidate_count": len(entries),
        "candidates": entries,
    }
    counts = {
        name: sum(entry["status"] == name for entry in entries)
        for name in ("cached", "completed", "failed", "generated", "pending")
    }
    # A running candidate is pending work from the sweep's point of view.
    counts["pending"] += sum(entry["status"] == "running" for entry in entries)
    results = {**index, "counts": counts}
    return index, results


def _persist(
    output: Path,
    *,
    sweep_sha256: str,
    sweep_mode: str,
    scenario: ScenarioSpecV1,
    scene_mode: str,
    provenance_sha256: str,
    statuses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index, results = _payloads(
        sweep_sha256=sweep_sha256,
        sweep_mode=sweep_mode,
        scenario=scenario,
        scene_mode=scene_mode,
        provenance_sha256=provenance_sha256,
        statuses=statuses,
    )
    _atomic_json(output / "index.json", index)
    _atomic_json(output / "results.json", results)
    return results


def _write_status(output: Path, status: Mapping[str, Any]) -> None:
    _atomic_json(output / "statuses" / f"{int(status['index']):04d}.json", status)


@dataclass
class _SweepState:
    root: Path
    sweep_sha256: str
    sweep_mode: str
    scenario: ScenarioSpecV1
    scene_mode: str
    provenance_sha256: str
    statuses: list[dict[str, Any]]

    def persist(self) -> dict[str, Any]:
        return _persist(
            self.root,
            sweep_sha256=self.sweep_sha256,
            sweep_mode=self.sweep_mode,
            scenario=self.scenario,
            scene_mode=self.scene_mode,
            provenance_sha256=self.provenance_sha256,
            statuses=self.statuses,
        )

    def record(self, offset: int, status: dict[str, Any]) -> dict[str, Any]:
        self.statuses[offset] = status
        _write_status(self.root, status)
        return self.persist()


def _attempt_path(output: Path, cache_key: str, attempt: int) -> tuple[str, Path]:
    relative = f"runs/{cache_key}/attempt-{attempt:04d}"
    return relative, output.joinpath(*PurePosixPath(relative).parts)


def _stderr_tail(value: Any, *, limit: int = 4096) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else repr(value)
    return text[-limit:]


def execute_sweep(
    *,
    output: str | Path,
    scenario: ScenarioSpecV1,
    base_profile: CalibrationProfileV1,
    candidates: Sequence[CalibrationProfileV1],
    sweep_mode: str,
    scene_mode: str,
    headless: bool,
    seed: int,
    device: str,
    provenance: Mapping[str, Any],
    provenance_provider: Callable[[], Mapping[str, Any]] = production_runtime_provenance,
    command_prefix: Sequence[str] | None = None,
    generate_only: bool = False,
    real_trace: str | Path | None = None,
    known_load_trace: str | Path | None = None,
    audit_artifact: Mapping[str, Any] | None = None,
    audit_artifact_root: str | Path | None = None,
    run_process: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Generate, execute, verify, and resume a bounded characterization sweep."""

    if sweep_mode not in _SWEEP_MODES:
        raise ContractError(f"unsupported sweep_mode: {sweep_mode}")
    if scene_mode not in _SCENE_MODES:
        raise ContractError(f"unsupported scene_mode: {scene_mode}")
    if not isinstance(headless, bool):
        raise ContractError("headless must be boolean")
    _positive_int(seed, "seed", allow_zero=True)
    if not isinstance(device, str) or not device:
        raise ContractError("device must be a non-empty string")
    if not isinstance(scenario, ScenarioSpecV1):
        raise ContractError("scenario must be a ScenarioSpecV1")
    if not isinstance(base_profile, CalibrationProfileV1):
        raise ContractError("base_profile must be a CalibrationProfileV1")
    base_profile = base_profile.validate()
    audit_report: dict[str, Any] | None = None
    if (audit_artifact is None) != (audit_artifact_root is None):
        raise ContractError(
            "audit_artifact and audit_artifact_root must be provided together"
        )
    if audit_artifact is None:
        if not generate_only:
            raise ContractError("executable sweeps require a passing pre-fit audit")
    else:
        if not isinstance(audit_artifact, Mapping):
            raise ContractError("audit_artifact must be a JSON object")
        assert audit_artifact_root is not None
        audit_report = _validate_prefit_audit(
            audit_artifact,
            artifact_root=audit_artifact_root,
            profile=base_profile,
        )
    reference: LoadedTrace | None = None
    if real_trace is None:
        if not generate_only:
            raise ContractError("real_trace is required when sweep execution is enabled")
    else:
        reference = load_trace(
            real_trace,
            scenario=scenario,
            require_managed_dataset=True,
        )
        validate_real_trace_provenance(reference, scenario)
    clean_candidates = list(candidates)
    if not all(isinstance(candidate, CalibrationProfileV1) for candidate in clean_candidates):
        raise ContractError("candidates must contain CalibrationProfileV1 values")
    validate_sweep_candidates(
        base_profile, clean_candidates, scenario, sweep_mode=sweep_mode
    )
    effort_limit_changed = any(
        _main_effort_limit(candidate) != _main_effort_limit(base_profile)
        for candidate in clean_candidates
    )
    known_load: LoadedTrace | None = None
    if effort_limit_changed and not generate_only and known_load_trace is None:
        raise ContractError(
            "managed known-load trace is required before sweeping main_drive.effort_limit"
        )
    if known_load_trace is not None:
        known_load_scenario = load_scenario("manual-load")
        known_load = load_trace(
            known_load_trace,
            scenario=known_load_scenario,
            require_managed_dataset=True,
        )
        validate_real_trace_provenance(known_load, known_load_scenario)
        known_load_metrics = compute_subsystem_metrics(
            known_load_scenario, known_load
        )
        effort_nm = known_load_metrics.get("torque_saturation_nm")
        if (
            isinstance(effort_nm, bool)
            or not isinstance(effort_nm, (int, float))
            or not math.isfinite(float(effort_nm))
            or float(effort_nm) <= 0.0
        ):
            raise ContractError("known-load trace does not identify positive effort saturation")
        if effort_limit_changed:
            directional_envelopes: list[tuple[float, float]] = []
            for direction in ("positive", "negative"):
                direction_effort = known_load_metrics.get(f"{direction}_torque_nm")
                direction_std = known_load_metrics.get(
                    f"{direction}_torque_nm_std", 0.0
                )
                if (
                    isinstance(direction_effort, bool)
                    or not isinstance(direction_effort, (int, float))
                    or not math.isfinite(float(direction_effort))
                    or float(direction_effort) <= 0.0
                    or isinstance(direction_std, bool)
                    or not isinstance(direction_std, (int, float))
                    or not math.isfinite(float(direction_std))
                    or float(direction_std) < 0.0
                ):
                    raise ContractError(
                        f"known-load trace has invalid {direction} repeat envelope"
                    )
                directional_envelopes.append(
                    (float(direction_effort), max(2.0 * float(direction_std), 1.0e-9))
                )
            candidate_limits = [_main_effort_limit(item) for item in clean_candidates]
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and all(
                    abs(float(value) - center) <= tolerance
                    for center, tolerance in directional_envelopes
                )
                for value in candidate_limits
            ):
                raise ContractError(
                    "effort-limit candidates must each remain inside both directional "
                    "known-load envelopes"
                )
    prefix = None if generate_only else _command_prefix(command_prefix)
    effective_provenance = _execution_provenance(
        _bind_runtime_provenance(provenance, provenance_provider),
        scene_mode=scene_mode,
        headless=headless,
        seed=seed,
        device=device,
    )
    if not generate_only and any(
        effective_provenance[field] == "unavailable-generate-only"
        for field in ("isaaclab_version", "isaacsim_version")
    ):
        raise ContractError(
            "executable sweeps require exact Isaac Lab and Isaac Sim versions"
        )
    effective_provenance["real_trace_sha256"] = (
        reference.manifest.provenance["trace_sha256"]
        if reference is not None
        else None
    )
    effective_provenance["real_metadata_sha256"] = (
        reference.metadata_sha256 if reference is not None else None
    )
    effective_provenance["known_load_trace_sha256"] = (
        known_load.manifest.provenance["trace_sha256"]
        if known_load is not None
        else None
    )
    effective_provenance["known_load_metadata_sha256"] = (
        known_load.metadata_sha256 if known_load is not None else None
    )
    effective_provenance["audit_artifact_sha256"] = (
        sha256_json(dict(audit_artifact)) if audit_artifact is not None else None
    )
    effective_provenance["audit_report_sha256"] = (
        sha256_json(audit_report) if audit_report is not None else None
    )
    provenance_sha256 = sha256_json(effective_provenance)

    entries: list[dict[str, Any]] = []
    for index, candidate in enumerate(clean_candidates, start=1):
        profile_sha256 = sha256_json(candidate.to_dict())
        entries.append(
            {
                "index": index,
                "profile": f"candidates/{index:04d}.json",
                "profile_sha256": profile_sha256,
                "cache_key": candidate_cache_key(
                    candidate, scenario, provenance=effective_provenance
                ),
            }
        )
    sweep_sha256 = sha256_json(
        {
            "schema_version": 1,
            "sweep_mode": sweep_mode,
            "scenario": scenario.to_dict(),
            "candidate_profiles": [candidate.to_dict() for candidate in clean_candidates],
            "provenance": effective_provenance,
        }
    )

    root = Path(output)
    if root.exists() and not root.is_dir():
        raise ContractError(f"sweep output is not a directory: {root}")
    if root.exists() and any(root.iterdir()) and not (root / "index.json").is_file():
        raise ContractError("non-empty sweep output is missing index.json")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "index.json").exists():
        existing_index = _json_object(root / "index.json", "sweep index")
        if existing_index.get("sweep_sha256") != sweep_sha256:
            raise ContractError("existing sweep index does not match this sweep")

    scenario_path = root / "scenario.json"
    if scenario_path.exists():
        if load_scenario(scenario_path).to_dict() != scenario.to_dict():
            raise ContractError("scenario snapshot does not match this sweep")
    else:
        _atomic_json(scenario_path, scenario.to_dict())
    provenance_path = root / "provenance.json"
    if provenance_path.exists():
        if _json_object(provenance_path, "sweep provenance") != effective_provenance:
            raise ContractError("provenance snapshot does not match this sweep")
    else:
        _atomic_json(provenance_path, effective_provenance)

    for entry, candidate in zip(entries, clean_candidates, strict=True):
        profile_path = root.joinpath(*PurePosixPath(entry["profile"]).parts)
        if profile_path.exists():
            try:
                existing_profile = load_profile(profile_path)
            except (OSError, ValueError) as exc:
                raise ContractError(
                    f"candidate {entry['index']} profile is invalid: {exc}"
                ) from exc
            if existing_profile.to_dict() != candidate.to_dict():
                raise ContractError(
                    f"candidate {entry['index']} profile does not match this sweep"
                )
        else:
            _atomic_json(profile_path, candidate.to_dict())

    statuses: list[dict[str, Any]] = []
    for entry in entries:
        existing = _load_status(
            root / "statuses" / f"{entry['index']:04d}.json", entry
        )
        status = existing or _candidate_status(entry, status="pending")
        statuses.append(status)
        if existing is None:
            _write_status(root, status)
    state = _SweepState(
        root=root,
        sweep_sha256=sweep_sha256,
        sweep_mode=sweep_mode,
        scenario=scenario,
        scene_mode=scene_mode,
        provenance_sha256=provenance_sha256,
        statuses=statuses,
    )
    result = state.persist()

    for offset, (entry, candidate) in enumerate(
        zip(entries, clean_candidates, strict=True)
    ):
        status = statuses[offset]
        existing_run = status.get("run_output")
        if status["status"] in _CACHEABLE_STATUSES:
            try:
                relative = _safe_relative(existing_run, "candidate status run_output")
                if reference is None:
                    raise ContractError("cached execution requires a real trace")
                expected_metadata_sha256 = status.get("metadata_sha256")
                if not isinstance(expected_metadata_sha256, str):
                    raise ContractError("cached candidate status is missing metadata_sha256")
                verified = _verify_candidate(
                    root.joinpath(*PurePosixPath(relative).parts),
                    scenario=scenario,
                    profile=candidate,
                    scene_mode=scene_mode,
                    provenance=effective_provenance,
                    real_trace=reference,
                    require_comparison=True,
                    expected_metadata_sha256=expected_metadata_sha256,
                )
            except (OSError, ValueError) as exc:
                failure = _candidate_status(
                    entry,
                    status="failed",
                    attempt=status["attempt"],
                    run_output=existing_run,
                    error=str(exc),
                )
                state.record(offset, failure)
                raise ContractError(
                    f"candidate {entry['index']} cached artifact verification failed: {exc}"
                ) from exc
            cached = _candidate_status(
                entry,
                status="cached",
                attempt=status["attempt"],
                run_output=existing_run,
                comparison=f"{relative}/comparison.json",
                **verified,
            )
            result = state.record(offset, cached)
            continue

        if generate_only:
            generated = _candidate_status(entry, status="generated")
            result = state.record(offset, generated)
            continue

        prior_attempt = int(status["attempt"])
        if status["status"] == "running" and existing_run is not None:
            prior_output = root.joinpath(*PurePosixPath(existing_run).parts)
            if (prior_output / "results.json").is_file():
                try:
                    if reference is None:
                        raise ContractError("interrupted execution requires a real trace")
                    verified = _verify_candidate(
                        prior_output,
                        scenario=scenario,
                        profile=candidate,
                        scene_mode=scene_mode,
                        provenance=effective_provenance,
                        real_trace=reference,
                        require_comparison=False,
                    )
                except (OSError, ValueError) as exc:
                    failure = _candidate_status(
                        entry,
                        status="failed",
                        attempt=prior_attempt,
                        run_output=existing_run,
                        error=str(exc),
                    )
                    state.record(offset, failure)
                    raise ContractError(
                        f"candidate {entry['index']} interrupted artifact verification failed: {exc}"
                    ) from exc
                cached = _candidate_status(
                    entry,
                    status="cached",
                    attempt=prior_attempt,
                    run_output=existing_run,
                    comparison=f"{existing_run}/comparison.json",
                    **verified,
                )
                result = state.record(offset, cached)
                continue

        attempt = prior_attempt + 1
        relative_output, run_output = _attempt_path(
            root, entry["cache_key"], attempt
        )
        if run_output.exists():
            raise ContractError(
                f"candidate {entry['index']} attempt output already exists: {run_output}"
            )
        run_output.parent.mkdir(parents=True, exist_ok=True)
        profile_path = root.joinpath(*PurePosixPath(entry["profile"]).parts).resolve()
        command = [
            *prefix,
            "run-sim",
            "--scenario",
            str(scenario_path.resolve()),
            "--mode",
            scene_mode,
            "--physics-profile",
            str(profile_path),
            "--output",
            str(run_output.resolve()),
            "--seed",
            str(seed),
            "--device",
            device,
        ]
        if headless:
            command.append("--headless")
        running = _candidate_status(
            entry,
            status="running",
            attempt=attempt,
            run_output=relative_output,
            command=command,
        )
        state.record(offset, running)

        try:
            completed = run_process(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except BaseException as exc:
            failure = _candidate_status(
                entry,
                status="failed",
                attempt=attempt,
                run_output=relative_output,
                error=f"subprocess launch failed: {exc}",
            )
            state.record(offset, failure)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ContractError(
                f"candidate {entry['index']} subprocess launch failed: {exc}"
            ) from exc

        returncode = getattr(completed, "returncode", None)
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            returncode = -1
        if returncode != 0:
            message = (
                f"candidate {entry['index']} subprocess failed with exit code "
                f"{returncode}"
            )
            failure = _candidate_status(
                entry,
                status="failed",
                attempt=attempt,
                run_output=relative_output,
                returncode=returncode,
                stderr_tail=_stderr_tail(getattr(completed, "stderr", None)),
                error=message,
            )
            state.record(offset, failure)
            raise ContractError(message)

        try:
            if reference is None:
                raise ContractError("completed execution requires a real trace")
            verified = _verify_candidate(
                run_output,
                scenario=scenario,
                profile=candidate,
                scene_mode=scene_mode,
                provenance=effective_provenance,
                real_trace=reference,
                require_comparison=False,
            )
        except (OSError, ValueError) as exc:
            failure = _candidate_status(
                entry,
                status="failed",
                attempt=attempt,
                run_output=relative_output,
                returncode=returncode,
                stderr_tail=_stderr_tail(getattr(completed, "stderr", None)),
                error=str(exc),
            )
            state.record(offset, failure)
            raise ContractError(
                f"candidate {entry['index']} artifact verification failed: {exc}"
            ) from exc
        completed_status = _candidate_status(
            entry,
            status="completed",
            attempt=attempt,
            run_output=relative_output,
            returncode=returncode,
            comparison=f"{relative_output}/comparison.json",
            **verified,
        )
        result = state.record(offset, completed_status)

    return result


run_candidates = execute_sweep
