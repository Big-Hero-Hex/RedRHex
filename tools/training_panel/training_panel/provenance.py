"""Capture the code version behind a training run.

A run record without a commit cannot be tied back to the code that produced
it. Every failure mode here returns {} — the panel must keep working in a
tarball checkout, a detached worktree, or a machine with no git installed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

GIT_TIMEOUT_SECONDS = 5


def _git_output(args: list[str], repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_provenance(repo_root: Path) -> dict:
    """Return {"commit", "short", "branch", "dirty"} for repo_root, or {}."""
    root = Path(repo_root)
    if not root.is_dir():
        return {}
    commit = _git_output(["rev-parse", "HEAD"], root)
    if not commit:
        return {}
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], root) or ""
    status = _git_output(["status", "--porcelain"], root)
    return {
        "commit": commit,
        "short": commit[:7],
        "branch": branch,
        "dirty": bool(status),
    }
