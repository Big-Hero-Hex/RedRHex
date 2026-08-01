from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import ContractError
from .repo_binding import redrhex_config_identity
from .traces import sha256_file, sha256_json


_GIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_PRODUCTION_INPUTS = {
    "asset_sha256": Path("RedRhex.usd"),
    "config_sha256": Path(
        "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
    ),
    "characterization_runner_sha256": Path("tools/sim2real/isaac_runner.py"),
    "sweep_runner_sha256": Path("tools/sim2real/sweep_runner.py"),
    "torsion_spring_model_sha256": Path(
        "source/RedRhex/RedRhex/tasks/direct/redrhex/torsion_spring.py"
    ),
}
_BEHAVIOR_INPUTS = (
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/abad_target_mapping.py"),
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py"),
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"),
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/target_delay.py"),
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/torsion_spring.py"),
    Path("tools/sim2real/characterization.py"),
    Path("tools/sim2real/contracts.py"),
    Path("tools/sim2real/isaac_profile.py"),
    Path("tools/sim2real/isaac_runner.py"),
    Path("tools/sim2real/metrics.py"),
    Path("tools/sim2real/physics_profile.py"),
    Path("tools/sim2real/repo_binding.py"),
    Path("tools/sim2real/runtime_provenance.py"),
    Path("tools/sim2real/scenarios.py"),
    Path("tools/sim2real/traces.py"),
)


def runtime_toolchain_fingerprint() -> dict[str, str]:
    """Read exact Isaac Lab and Isaac Sim build identifiers without launching Kit."""

    try:
        isaaclab_version = importlib.metadata.version("isaaclab")
    except importlib.metadata.PackageNotFoundError:
        lab_root = os.environ.get("ISAACLAB_PATH")
        version_file = (
            Path(lab_root) / "source/isaaclab/config/extension.toml"
            if lab_root
            else None
        )
        text = version_file.read_text(encoding="utf-8") if version_file else ""
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        isaaclab_version = match.group(1) if match else ""

    sim_candidates = []
    if os.environ.get("ISAAC_PATH"):
        sim_candidates.append(Path(os.environ["ISAAC_PATH"]) / "VERSION")
    if os.environ.get("ISAACLAB_PATH"):
        sim_candidates.append(Path(os.environ["ISAACLAB_PATH"]) / "_isaac_sim/VERSION")
    sim_version = ""
    for path in sim_candidates:
        try:
            sim_version = path.resolve().read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if sim_version:
            break
    return {
        "isaaclab_version": isaaclab_version or "unavailable-generate-only",
        "isaacsim_version": sim_version or "unavailable-generate-only",
    }


def production_runtime_provenance(
    repo_root: str | Path | None = None,
    *,
    run_git: Callable[..., Any] = subprocess.run,
    toolchain_provider: Callable[[], dict[str, str]] = runtime_toolchain_fingerprint,
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
    result.update(redrhex_config_identity(root))
    try:
        toolchain = toolchain_provider()
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot fingerprint simulator toolchain: {exc}") from exc
    expected_toolchain = {"isaaclab_version", "isaacsim_version"}
    if set(toolchain) != expected_toolchain or any(
        not isinstance(toolchain[field], str) or not toolchain[field]
        for field in expected_toolchain
    ):
        raise ContractError(
            "simulator toolchain fingerprint must contain exact Isaac Lab and Isaac Sim versions"
        )
    result.update(toolchain)
    return result
