"""Command-line interface for documentation validation."""

import argparse
from pathlib import Path
import sys

from .validator import _candidate_documents, validate_repository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.documentation")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    modes = validate.add_mutually_exclusive_group(required=True)
    modes.add_argument("--all", action="store_true")
    return parser


def _document_count(repo_root: Path) -> int:
    return sum(
        path.name.endswith((".en.md", ".zh-TW.md"))
        for path in _candidate_documents(repo_root)
    )


def main(argv: list[str] | None = None) -> int:
    _parser().parse_args(argv)
    repo_root = Path.cwd()
    issues = validate_repository(repo_root)
    if issues:
        for issue in issues:
            print(f"{issue.path}: {issue.code}: {issue.message}", file=sys.stderr)
        noun = "issue" if len(issues) == 1 else "issues"
        print(
            f"documentation validation failed ({len(issues)} {noun})",
            file=sys.stderr,
        )
        return 1
    print(f"documentation validation passed ({_document_count(repo_root)} documents)")
    return 0
