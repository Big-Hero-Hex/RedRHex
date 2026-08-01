import argparse
from pathlib import Path
import sys

from .validator import _discover_candidates, validate_repository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.documentation", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", allow_abbrev=False)
    mode = validate.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    return parser


def _document_count(repo_root: Path) -> int:
    return len(_discover_candidates(repo_root))


def _repository_root(start: Path) -> Path:
    current = start.resolve()
    while True:
        marker = current / ".git"
        if marker.is_dir() or marker.is_file():
            return current
        if current.parent == current:
            return start.resolve()
        current = current.parent


def main(argv=None) -> int:
    arguments = _parser().parse_args(argv)
    repo_root = _repository_root(Path.cwd())
    if arguments.command == "validate" and arguments.all:
        issues = validate_repository(repo_root)
        if issues:
            for issue in issues:
                print(f"{issue.path.as_posix()}: {issue.code}: {issue.message}", file=sys.stderr)
            noun = "issue" if len(issues) == 1 else "issues"
            print(f"documentation validation failed ({len(issues)} {noun})", file=sys.stderr)
            return 1
        print(f"documentation validation passed ({_document_count(repo_root)} documents)")
        return 0
    return 2
