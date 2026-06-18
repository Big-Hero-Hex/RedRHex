from __future__ import annotations

import argparse
from pathlib import Path

from .experiment_store import ExperimentStore, RewardAgentPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reward Agent Lab foundation commands.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show reward-agent storage status.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = RewardAgentPaths.from_repo_root(args.repo_root)
    store = ExperimentStore(paths)
    if args.command == "status":
        print(f"reward_agent_root: {paths.root}")
        print(f"sessions: {len(store.list_sessions())}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
