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
    importer.add_argument("--latency-clock", default="bag_receive_time")
    importer.add_argument("--time-bases-json", default="{}")
    importer.add_argument("--dataset-id", required=True)
    importer.add_argument("--episode-id", required=True)
    importer.add_argument("--profile", type=Path, default=None)

    comparison = subparsers.add_parser("compare", help="Compare real and simulated traces.")
    comparison.add_argument("real", type=Path)
    comparison.add_argument("sim", type=Path)
    comparison.add_argument("--scenario", default=None)
    comparison.add_argument("--output", type=Path, default=None)

    sweep = subparsers.add_parser("sweep", help="Generate bounded deterministic candidates.")
    sweep.add_argument("profile", type=Path)
    sweep.add_argument("--scenario", required=True)
    sweep.add_argument("--mode", choices=("one-factor", "coarse-grid"), required=True)
    sweep.add_argument("--space-json", required=True)
    sweep.add_argument("--max-candidates", type=int, default=256)
    sweep.add_argument("--output", required=True, type=Path)

    validator = subparsers.add_parser("validate-profile", help="Validate a versioned profile.")
    validator.add_argument("profile", type=Path)
    return parser


def _json_object(value: str, name: str) -> dict[str, Any]:
    payload = json.loads(value)
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
            profile=load_profile(args.profile) if args.profile is not None else None,
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
            candidate_cache_key,
            generate_coarse_grid_candidates,
            generate_one_factor_candidates,
        )

        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError(f"output already exists: {args.output}")
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
        entries = []
        for index, candidate in enumerate(candidates, start=1):
            relative = Path("candidates") / f"{index:04d}.json"
            _write_json(args.output / relative, candidate.to_dict())
            entries.append(
                {
                    "index": index,
                    "profile": relative.as_posix(),
                    "cache_key": candidate_cache_key(candidate, scenario),
                }
            )
        index_payload = {
            "schema_version": 1,
            "mode": args.mode,
            "scenario_id": scenario.scenario_id,
            "candidate_count": len(entries),
            "candidates": entries,
        }
        _write_json(args.output / "index.json", index_payload)
        return index_payload
    raise ValueError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _emit(_run(args))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
