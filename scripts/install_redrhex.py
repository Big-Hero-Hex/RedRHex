#!/usr/bin/env python3
"""Install RedRHex's repository-local Python distributions in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def install_commands(
    *,
    repo_root: Path = REPO_ROOT,
    python_executable: str = sys.executable,
) -> tuple[tuple[str, ...], ...]:
    """Return the exact pip commands for the two local distributions."""

    distributions = (
        repo_root / "source" / "redrhex_policy_io",
        repo_root / "source" / "RedRhex",
    )
    missing = [str(path) for path in distributions if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "RedRHex repository-local distribution directory is missing: "
            + ", ".join(missing)
        )
    return tuple(
        (
            str(python_executable),
            "-m",
            "pip",
            "install",
            "-e",
            str(distribution),
        )
        for distribution in distributions
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered pip commands without changing the environment.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = install_commands()
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
