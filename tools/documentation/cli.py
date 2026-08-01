"""Command-line interface for documentation validation."""

import argparse
from pathlib import Path
import sys

from .validator import document_count, validate_repository


def _repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.documentation")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--all", action="store_true", required=True)
    arguments = parser.parse_args(argv)

    root = _repository_root(Path.cwd().resolve())
    issues = validate_repository(root)
    if not issues:
        print(f"documentation validation passed ({document_count(root)} documents)")
        return 0
    for issue in issues:
        print(f"{issue.path}: {issue.code}: {issue.message}", file=sys.stderr)
    print(f"documentation validation failed ({len(issues)} issues)", file=sys.stderr)
    return 1
