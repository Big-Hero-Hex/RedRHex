import argparse
from pathlib import Path
import sys

from .validator import _discover_document_paths, validate_repository


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.documentation")
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser("validate")
    modes = validate_parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--all", action="store_true")
    return parser


def _document_count(repo_root: Path) -> int:
    return sum(1 for _path, _relative_path in _discover_document_paths(repo_root))


def _find_repository_root(start: Path) -> Path:
    resolved_start = start.resolve()
    for candidate in (resolved_start, *resolved_start.parents):
        marker = candidate / ".git"
        if marker.is_dir() or marker.is_file():
            return candidate
    return resolved_start


def main() -> int:
    arguments = _build_parser().parse_args()
    if arguments.command == "validate" and arguments.all:
        repo_root = _find_repository_root(Path.cwd())
        issues = validate_repository(repo_root)
        if issues:
            for issue in issues:
                print(f"{issue.path.as_posix()}: {issue.code}: {issue.message}", file=sys.stderr)
            issue_word = "issue" if len(issues) == 1 else "issues"
            print(f"documentation validation failed ({len(issues)} {issue_word})", file=sys.stderr)
            return 1
        print(f"documentation validation passed ({_document_count(repo_root)} documents)")
        return 0
    return 2
