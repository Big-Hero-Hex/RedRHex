from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import ContractError
from .traces import sha256_file


_GIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_PRODUCTION_INPUTS = {
    "asset_sha256": Path("RedRhex.usd"),
    "config_sha256": Path(
        "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
    ),
    "characterization_runner_sha256": Path("tools/sim2real/isaac_runner.py"),
    "sweep_runner_sha256": Path("tools/sim2real/sweep_runner.py"),
}


def production_runtime_provenance(
    repo_root: str | Path | None = None,
    *,
    run_git: Callable[..., Any] = subprocess.run,
) -> dict[str, str]:
    """Hash the exact production inputs that determine characterization output."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    try:
        completed = run_git(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ContractError(f"cannot determine runner Git SHA: {exc}") from exc
    git_sha = str(getattr(completed, "stdout", "")).strip()
    if getattr(completed, "returncode", None) != 0 or not _GIT_SHA.fullmatch(git_sha):
        raise ContractError("cannot determine a valid runner Git SHA")

    result = {"git_sha": git_sha}
    for field, relative in _PRODUCTION_INPUTS.items():
        path = root / relative
        try:
            result[field] = sha256_file(path)
        except OSError as exc:
            raise ContractError(f"cannot hash production input {relative}: {exc}") from exc
    return result
