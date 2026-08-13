from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RedRHex sim-to-real calibration tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List reviewed characterization scenarios.")

    importer = subparsers.add_parser("import-real", help="Import an immutable real trace.")
    importer.add_argument("source", type=Path)
    importer.add_argument("--scenario", required=True)
    importer.add_argument("--output", required=True, type=Path)
    importer.add_argument("--units-json", default="{}")
    importer.add_argument("--frames-json", default="{}")
    importer.add_argument(
        "--calibration-constants-json",
        default="{}",
        help=(
            "Immutable operator-approved measurement constants, including the "
            "torsion-spring rest angle, fixture approval, safe envelope, and "
            "representative alias declaration."
        ),
    )
    importer.add_argument("--latency-clock", default=None)
    importer.add_argument("--time-bases-json", default="{}")
    importer.add_argument("--dataset-id", required=True)
    importer.add_argument("--episode-id", required=True)
    importer.add_argument("--profile", type=Path, default=None)
    importer.add_argument(
        "--replay-fixture",
        type=Path,
        default=None,
        help=(
            "Operator-reviewed fixed-base fixture JSON. Required to make a real "
            "probe episode eligible for replay; import without it remains metrics-only."
        ),
    )

    comparison = subparsers.add_parser("compare", help="Compare real and simulated traces.")
    comparison.add_argument("real", type=Path)
    comparison.add_argument("sim", type=Path)
    comparison.add_argument("--scenario", default=None)
    comparison.add_argument("--output", type=Path, default=None)

    sweep = subparsers.add_parser(
        "sweep", help="Generate or execute a resumable bounded candidate sweep."
    )
    sweep.add_argument("profile", type=Path)
    sweep.add_argument("--scenario", required=True)
    sweep.add_argument("--mode", choices=("one-factor", "coarse-grid"), required=True)
    sweep.add_argument("--space-json", required=True)
    sweep.add_argument("--max-candidates", type=int, default=256)
    sweep.add_argument("--output", required=True, type=Path)
    sweep.add_argument(
        "--scene-mode", choices=("fixed-base", "free-root", "contact"), default=None
    )
    sweep.add_argument("--isaaclab-root", type=Path, default=None)
    sweep.add_argument("--device", default="cuda:0")
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument(
        "--spring-backend",
        choices=("explicit", "native"),
        default="explicit",
        help="Passive torsion-spring implementation used for every candidate run.",
    )
    sweep.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    sweep.add_argument("--provenance-json", default="{}")
    sweep.add_argument(
        "--real-trace",
        type=Path,
        default=None,
        help="Verified real reference episode required for sweep execution.",
    )
    sweep.add_argument(
        "--known-load-trace",
        type=Path,
        default=None,
        help="Managed manual-load episode required when effort_limit is swept.",
    )
    sweep.add_argument(
        "--audit-evidence",
        type=Path,
        default=None,
        help=(
            "Hash-bound audit_artifact JSON required for execution. Relative paths "
            "inside it resolve from this file's directory."
        ),
    )
    sweep.add_argument(
        "--generate-only",
        action="store_true",
        help="Write candidates and manifests without launching Isaac processes.",
    )

    validator = subparsers.add_parser("validate-profile", help="Validate a versioned profile.")
    validator.add_argument("profile", type=Path)

    promotion = subparsers.add_parser(
        "validate-promotion",
        help="Evaluate hash-bound audit and held-out evidence for a candidate profile.",
    )
    promotion.add_argument("profile", type=Path)
    promotion.add_argument("evidence", type=Path)
    promotion.add_argument("--output", type=Path, default=None)

    simulation = subparsers.add_parser(
        "run-sim", help="Run a finite Isaac Lab characterization scenario."
    )
    simulation.add_argument("--scenario", required=True)
    simulation.add_argument(
        "--mode", choices=("fixed-base", "free-root", "contact"), required=True
    )
    simulation.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Physics frames. Command scenarios require their exact declared duration; "
            "audit defaults to 240 and may use another finite length."
        ),
    )
    simulation.add_argument("--output", type=Path, required=True)
    simulation.add_argument("--physics-profile", type=Path, default=None)
    simulation.add_argument("--replay-trace", type=Path, default=None)
    simulation.add_argument("--require-contact", action="store_true", default=False)
    simulation.add_argument("--contact-threshold-n", type=float, default=0.05)
    simulation.add_argument("--seed", type=int, default=0)
    simulation.add_argument("--headless", action="store_true", default=False)
    simulation.add_argument("--device", default="cuda:0")
    simulation.add_argument(
        "--physics-hz",
        type=int,
        choices=(120, 240),
        default=120,
        help="Physics frequency for spring characterization and timestep comparison.",
    )
    simulation.add_argument(
        "--spring-backend",
        choices=("explicit", "native"),
        default="explicit",
        help="Passive torsion-spring implementation used by the characterization run.",
    )

    selection = subparsers.add_parser(
        "select-spring-backend",
        help="Apply the deterministic gates to explicit/native 120/240 Hz releases.",
    )
    selection.add_argument("--explicit-120", type=Path, required=True)
    selection.add_argument("--explicit-240", type=Path, required=True)
    selection.add_argument("--native-120", type=Path, required=True)
    selection.add_argument("--native-240", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)

    policy = subparsers.add_parser(
        "validate-policy-acceptance",
        help="Apply the fixed three-seed ForwardFast or Direct rollout gates.",
    )
    policy.add_argument("--stage", choices=("forwardfast", "direct"), required=True)
    for seed in (42, 43, 44):
        policy.add_argument(f"--seed-{seed}-command", type=Path, required=True)
        policy.add_argument(f"--seed-{seed}-summary", type=Path, required=True)
    policy.add_argument("--output", type=Path, required=True)
    return parser


def _json_object(value: str, name: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _json_file_object(path: Path, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
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


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "list":
        from .scenarios import list_scenarios

        return {"schema_version": 1, "scenarios": list_scenarios()}
    if args.command == "validate-profile":
        from .contracts import load_profile

        profile = load_profile(args.profile)
        return {"schema_version": 1, "valid": True, "profile_id": profile.profile_id}
    if args.command == "validate-promotion":
        from .contracts import load_profile
        from .promotion import evaluate_promotion, load_validation_evidence

        result = evaluate_promotion(
            load_profile(args.profile),
            load_validation_evidence(args.evidence),
            artifact_root=args.evidence.parent,
        )
        if args.output is not None:
            if args.output.exists():
                raise ValueError(f"output already exists: {args.output}")
            _write_json(args.output, result)
        return result
    if args.command == "import-real":
        from .dataset import import_real_dataset
        from .contracts import load_profile

        imported = import_real_dataset(
            args.source,
            args.output,
            dataset_id=args.dataset_id,
            episode_id=args.episode_id,
            scenario=args.scenario,
            units=_json_object(args.units_json, "--units-json"),
            frames=_json_object(args.frames_json, "--frames-json"),
            latency_clock=args.latency_clock,
            time_bases=_json_object(args.time_bases_json, "--time-bases-json"),
            metadata={
                "calibration_constants": _json_object(
                    args.calibration_constants_json,
                    "--calibration-constants-json",
                )
            },
            profile=load_profile(args.profile) if args.profile is not None else None,
            replay_fixture=(
                _json_file_object(args.replay_fixture, "--replay-fixture")
                if args.replay_fixture is not None
                else None
            ),
        )
        episode = next(
            item
            for item in imported.manifest["episodes"]
            if item["episode_id"] == args.episode_id
        )
        return {
            "schema_version": 1,
            "scenario_id": episode["scenario_id"],
            "dataset": str(imported.dataset),
            "episode": str(imported.episode),
            "trace_sha256": episode["trace_sha256"],
        }
    if args.command == "compare":
        from .compare import compare_traces

        result = compare_traces(args.real, args.sim, scenario=args.scenario)
        if args.output is not None:
            if args.output.exists():
                raise ValueError(f"output already exists: {args.output}")
            _write_json(args.output, result)
        return result
    if args.command == "sweep":
        from .contracts import load_profile
        from .scenarios import load_scenario
        from .sweep import (
            generate_coarse_grid_candidates,
            generate_one_factor_candidates,
        )
        from . import sweep_runner
        from .runtime_provenance import production_runtime_provenance

        profile = load_profile(args.profile)
        scenario = load_scenario(args.scenario)
        space = _json_object(args.space_json, "--space-json")
        if args.mode == "one-factor":
            candidates = generate_one_factor_candidates(
                profile, space, max_candidates=args.max_candidates
            )
        else:
            candidates = generate_coarse_grid_candidates(
                profile, space, max_candidates=args.max_candidates
            )
        baseline_effort = profile.simulation_physics.get("main_drive", {}).get(
            "effort_limit"
        )
        effort_limit_changed = any(
            candidate.simulation_physics.get("main_drive", {}).get("effort_limit")
            != baseline_effort
            for candidate in candidates
        )
        if (
            effort_limit_changed
            and not args.generate_only
            and args.known_load_trace is None
        ):
            raise ValueError(
                "--known-load-trace is required when sweeping main_drive.effort_limit"
            )
        scene_mode = args.scene_mode or {
            "fixed_base": "fixed-base",
            "free_root": "free-root",
        }.get(scenario.scene_mode)
        if scene_mode is None:
            raise ValueError(
                f"--scene-mode is required for scenario scene_mode={scenario.scene_mode!r}"
            )
        if not args.generate_only and args.real_trace is None:
            raise ValueError("--real-trace is required unless --generate-only is used")
        if not args.generate_only and args.audit_evidence is None:
            raise ValueError("--audit-evidence is required unless --generate-only is used")

        audit_artifact = None
        audit_artifact_root = None
        if args.audit_evidence is not None:
            audit_artifact = _json_file_object(
                args.audit_evidence, "--audit-evidence"
            )
            audit_artifact_root = args.audit_evidence.parent

        command_prefix = None
        if not args.generate_only:
            root_value = args.isaaclab_root or os.environ.get("ISAACLAB_ROOT")
            if root_value is None:
                raise ValueError(
                    "--isaaclab-root or ISAACLAB_ROOT is required unless --generate-only is used"
                )
            launcher = Path(root_value).expanduser() / "isaaclab.sh"
            if not launcher.is_file():
                raise ValueError(f"Isaac Lab launcher does not exist: {launcher}")
            command_prefix = (
                str(launcher),
                "-p",
                "-m",
                "tools.sim2real",
            )

        provenance = _json_object(args.provenance_json, "--provenance-json")
        return sweep_runner.execute_sweep(
            output=args.output,
            scenario=scenario,
            base_profile=profile,
            candidates=candidates,
            sweep_mode=args.mode,
            scene_mode=scene_mode,
            headless=args.headless,
            seed=args.seed,
            device=args.device,
            spring_backend=args.spring_backend,
            provenance=provenance,
            provenance_provider=production_runtime_provenance,
            command_prefix=command_prefix,
            generate_only=args.generate_only,
            real_trace=args.real_trace,
            known_load_trace=args.known_load_trace,
            audit_artifact=audit_artifact,
            audit_artifact_root=audit_artifact_root,
        )
    if args.command == "run-sim":
        # Importing this bootstrap is the first point at which Isaac Lab is
        # required. All CPU-only commands and parser construction stay light.
        from .isaac_main import run

        return run(args)
    if args.command == "select-spring-backend":
        from .spring_backend_selection import evaluate_backend_runs

        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        result = evaluate_backend_runs(
            explicit_120=args.explicit_120,
            explicit_240=args.explicit_240,
            native_120=args.native_120,
            native_240=args.native_240,
        )
        _write_json(args.output, result)
        return result
    if args.command == "validate-policy-acceptance":
        from .policy_acceptance import evaluate_policy_acceptance

        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        result = evaluate_policy_acceptance(
            stage=args.stage,
            runs=[
                (args.seed_42_command, args.seed_42_summary),
                (args.seed_43_command, args.seed_43_summary),
                (args.seed_44_command, args.seed_44_summary),
            ],
        )
        _write_json(args.output, result)
        return result
    raise ValueError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _run(args)
        if args.command != "run-sim":
            _emit(result)
        if args.command == "validate-promotion" and not result["eligible_for_review"]:
            return 3
        if args.command == "select-spring-backend" and not result["eligible"]:
            return 3
        if args.command == "validate-policy-acceptance" and not result["eligible"]:
            return 3
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
