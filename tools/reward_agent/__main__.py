from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment_store import ExperimentStore, RewardAgentPaths
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
    return parser


def main(argv: list[str] | None = None) -> int:
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
