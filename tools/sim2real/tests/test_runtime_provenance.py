from __future__ import annotations

import subprocess
from pathlib import Path

from tools.sim2real.runtime_provenance import (
    _BEHAVIOR_INPUTS,
    _PRODUCTION_INPUTS,
    production_runtime_provenance,
)


def _repository_fixture(root: Path) -> None:
    for index, relative in enumerate(
        sorted(set(_PRODUCTION_INPUTS.values()) | set(_BEHAVIOR_INPUTS)), start=1
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"behavior-input-{index}".encode())


def _git(command, **_kwargs):
    return subprocess.CompletedProcess(command, 0, stdout="e" * 40 + "\n", stderr="")


def test_runtime_bundle_hash_tracks_dirty_behavior_defining_modules(
    tmp_path: Path,
) -> None:
    _repository_fixture(tmp_path)
    toolchain = lambda: {
        "isaaclab_version": "0.54.2",
        "isaacsim_version": "5.1.0-rc.19+release.26219.9c81211b.gl",
    }
    first = production_runtime_provenance(
        tmp_path, run_git=_git, toolchain_provider=toolchain
    )

    changed = tmp_path / "tools/sim2real/metrics.py"
    changed.write_bytes(changed.read_bytes() + b"\n# dirty behavior change\n")
    second = production_runtime_provenance(
        tmp_path, run_git=_git, toolchain_provider=toolchain
    )

    assert len(first["runtime_bundle_sha256"]) == 64
    assert first["runtime_bundle_sha256"] != second["runtime_bundle_sha256"]
    assert first["git_sha"] == second["git_sha"]
    assert first["asset_sha256"] == second["asset_sha256"]
    assert first["redrhex_module_path"] == str(
        (
            tmp_path
            / "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
        ).resolve()
    )
    assert first["redrhex_module_sha256"] == first["config_sha256"]
    assert first["isaaclab_version"] == "0.54.2"
    assert first["isaacsim_version"].startswith("5.1.0-rc.19+")

    spring_model = tmp_path / (
        "source/RedRhex/RedRhex/tasks/direct/redrhex/torsion_spring.py"
    )
    spring_model.write_bytes(spring_model.read_bytes() + b"\n# spring law change\n")
    third = production_runtime_provenance(
        tmp_path, run_git=_git, toolchain_provider=toolchain
    )
    assert second["runtime_bundle_sha256"] != third["runtime_bundle_sha256"]
    assert second["torsion_spring_model_sha256"] != third["torsion_spring_model_sha256"]


def test_runtime_bundle_hash_tracks_profile_measurement_changes(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    profile_measurements = tmp_path / "tools/sim2real/profile_measurements.py"
    profile_measurements.parent.mkdir(parents=True, exist_ok=True)
    profile_measurements.write_text("calibration = 1\n", encoding="utf-8")
    toolchain = lambda: {
        "isaaclab_version": "0.54.2",
        "isaacsim_version": "5.1.0-rc.19+release.26219.9c81211b.gl",
    }

    first = production_runtime_provenance(
        tmp_path, run_git=_git, toolchain_provider=toolchain
    )
    profile_measurements.write_text("calibration = 2\n", encoding="utf-8")
    second = production_runtime_provenance(
        tmp_path, run_git=_git, toolchain_provider=toolchain
    )

    assert first["runtime_bundle_sha256"] != second["runtime_bundle_sha256"]
