"""Command-line interface for documentation validation."""

import argparse
import pathlib
import subprocess
import sys

from . import validator as _validator


def _repository_root() -> pathlib.Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=pathlib.Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return pathlib.Path(result.stdout.strip()).resolve()
    return pathlib.Path.cwd().resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.documentation")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--all", dest="validate_all", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = _repository_root()
    if args.command == "validate" and args.validate_all:
        issues = _validator.validate_repository(repo_root)
        if issues:
            for issue in issues:
                print(
                    f"{issue.path.as_posix()}: {issue.code}: {issue.message}",
                    file=sys.stderr,
                )
            noun = "issue" if len(issues) == 1 else "issues"
            print(
                f"documentation validation failed ({len(issues)} {noun})",
                file=sys.stderr,
            )
            return 1
        count = len(_validator._discover_candidates(repo_root))
        print(f"documentation validation passed ({count} documents)")
        return 0
    return 2
