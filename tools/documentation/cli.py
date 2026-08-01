"""Command-line interface for documentation validation."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Sequence

from .errors import DocumentationOperationError
from .inventory import build_inventory
from .selection import changed_pair_issues, select_changed_paths, select_staged_paths
from .site import stage_site
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
    selector = validate.add_mutually_exclusive_group(required=True)
    selector.add_argument("--all", action="store_true")
    selector.add_argument("--staged", action="store_true")
    selector.add_argument("--changed-from", metavar="REF")
    inventory = commands.add_parser("inventory", allow_abbrev=False)
    inventory.add_argument("--format", choices=("json",), required=True)
    stage_site = commands.add_parser("stage-site", allow_abbrev=False)
    stage_site.add_argument("--output", metavar="DIR", required=True)
    return parser


def _run_command(arguments: argparse.Namespace, repo_root: Path) -> int:
    if arguments.command == "validate":
        issues = validate_repository(repo_root)
        if arguments.staged:
            issues.extend(changed_pair_issues(select_staged_paths(repo_root)))
        elif arguments.changed_from is not None:
            issues.extend(
                changed_pair_issues(
                    select_changed_paths(repo_root, arguments.changed_from)
                )
            )
        issues = sorted(
            set(issues),
            key=lambda issue: (issue.path.as_posix(), issue.code, issue.message),
        )
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
        count = len(_discover_candidates(repo_root))
        print(f"documentation validation passed ({count} documents)")
        return 0
    if arguments.command == "inventory":
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
        inventory = build_inventory(repo_root, datetime.date.today())
        print(json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if arguments.command == "stage-site":
        count = stage_site(repo_root, Path(arguments.output))
        print(f"documentation site staged ({count} documents)")
        return 0
    raise AssertionError("unreachable command shape")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repo_root = _find_repository_root(Path.cwd())
    try:
        return _run_command(arguments, repo_root)
    except DocumentationOperationError as error:
        print(f"documentation error: {error}", file=sys.stderr)
        return 1
