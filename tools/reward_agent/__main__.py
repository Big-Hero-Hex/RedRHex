from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import preview_candidate_trials, queue_candidate_trials
from .experiment_store import ExperimentStore, RewardAgentPaths
from .launcher import build_panel_registry
from .planner import RewardWeightSpec, generate_weight_candidates


def _parse_scale(value: str) -> RewardWeightSpec:
    try:
        name, minimum, maximum = value.split(":", 2)
        return RewardWeightSpec(name=name, minimum=float(minimum), maximum=float(maximum))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--scale must use name:min:max format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reward Agent Lab foundation commands.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show reward-agent storage status.")
    create_session = subparsers.add_parser("create-session", help="Create a reward-agent session.")
    create_session.add_argument("--objective", required=True)
    propose = subparsers.add_parser("propose-candidates", help="Generate bounded reward-weight candidates.")
    propose.add_argument("--session-id", required=True)
    propose.add_argument("--base-overrides-json", required=True)
    propose.add_argument("--scale", type=_parse_scale, action="append", required=True)
    queue = subparsers.add_parser("queue-trials", help="Preview or launch saved candidate trials.")
    queue.add_argument("--session-id", required=True)
    queue.add_argument("--base-params-json", required=True)
    queue.add_argument("--max-iterations", type=int, default=None)
    queue.add_argument("--limit", type=int, default=None)
    queue_mode = queue.add_mutually_exclusive_group(required=True)
    queue_mode.add_argument("--dry-run", action="store_true")
    queue_mode.add_argument("--launch", action="store_true")
    return parser


def main(argv: list[str] | None = None, registry_factory=None) -> int:
    args = build_parser().parse_args(argv)
    paths = RewardAgentPaths.from_repo_root(args.repo_root)
    store = ExperimentStore(paths)
    if args.command == "status":
        print(f"reward_agent_root: {paths.root}")
        print(f"sessions: {len(store.list_sessions())}")
        return 0
    if args.command == "create-session":
        session = store.create_session({"objective": args.objective})
        print(f"session_id: {session['id']}")
        return 0
    if args.command == "propose-candidates":
        base_overrides = json.loads(args.base_overrides_json)
        candidates = generate_weight_candidates(base_overrides, args.scale)
        store.save_candidates(args.session_id, candidates)
        print(f"candidates: {len(candidates)}")
        return 0
    if args.command == "queue-trials":
        base_params = json.loads(args.base_params_json)
        candidates = store.load_candidates(args.session_id)
        if args.dry_run:
            trials = preview_candidate_trials(
                store,
                args.session_id,
                base_params,
                candidates,
                max_iterations=args.max_iterations,
                limit=args.limit,
            )
            print(f"dry_run_trials: {len(trials)}")
            return 0
        registry_builder = registry_factory or build_panel_registry
        trials = queue_candidate_trials(
            store,
            args.session_id,
            registry_builder(args.repo_root),
            base_params,
            candidates,
            max_iterations=args.max_iterations,
            limit=args.limit,
        )
        print(f"queued_trials: {len(trials)}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
