from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import ContractError
from .traces import sha256_file, sha256_json


_GIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_PRODUCTION_INPUTS = {
    "asset_sha256": Path("RedRhex.usd"),
    "config_sha256": Path(
        "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
    ),
    "characterization_runner_sha256": Path("tools/sim2real/isaac_runner.py"),
    "sweep_runner_sha256": Path("tools/sim2real/sweep_runner.py"),
}
_BEHAVIOR_INPUTS = (
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/abad_target_mapping.py"),
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py"),
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"),
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/target_delay.py"),
    Path("tools/sim2real/characterization.py"),
    Path("tools/sim2real/contracts.py"),
    Path("tools/sim2real/isaac_profile.py"),
    Path("tools/sim2real/isaac_runner.py"),
    Path("tools/sim2real/metrics.py"),
    Path("tools/sim2real/physics_profile.py"),
    Path("tools/sim2real/runtime_provenance.py"),
    Path("tools/sim2real/scenarios.py"),
    Path("tools/sim2real/traces.py"),
)


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
    behavior_hashes: dict[str, str] = {}
    for relative in _BEHAVIOR_INPUTS:
        try:
            behavior_hashes[relative.as_posix()] = sha256_file(root / relative)
        except OSError as exc:
            raise ContractError(f"cannot hash behavior input {relative}: {exc}") from exc
    result["runtime_bundle_sha256"] = sha256_json(behavior_hashes)
    return result
