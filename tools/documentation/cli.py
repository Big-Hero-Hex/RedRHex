"""Command-line interface for documentation validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .validator import _discover_candidates, validate_repository


def _find_repository_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        marker = candidate / ".git"
        if marker.is_dir() or marker.is_file():
            return candidate
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.documentation",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", allow_abbrev=False)
    validate.add_argument("--all", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repo_root = _find_repository_root(Path.cwd())
    issues = validate_repository(repo_root)
    if issues:
        for issue in issues:
            print(
                f"{issue.path.as_posix()}: {issue.code}: {issue.message}",
                file=sys.stderr,
            )
        print(
            f"documentation validation failed ({len(issues)} issues)",
            file=sys.stderr,
        )
        return 1
    if arguments.command == "validate" and arguments.all:
        count = len(_discover_candidates(repo_root))
        print(f"documentation validation passed ({count} documents)")
        return 0
    raise AssertionError("unreachable command shape")
