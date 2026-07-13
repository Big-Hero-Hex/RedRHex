from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import CalibrationProfileV1, ContractError, ScenarioSpecV1
from .import_real import import_real_trace, resolve_latency_clock
from .scenarios import load_scenario
from .traces import _atomic_json, sha256_file, sha256_path


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class DatasetImport:
    dataset: Path
    episode: Path
    manifest: dict[str, Any]


def _id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase identifier")
    return value


def import_real_dataset(
    source_path: str | Path,
    output_root: str | Path,
    *,
    dataset_id: str,
    episode_id: str,
    scenario: ScenarioSpecV1 | str | Path,
    units: Mapping[str, str] | None = None,
    frames: Mapping[str, str] | None = None,
    latency_clock: str | None = None,
    time_bases: Mapping[str, str] | None = None,
    profile: CalibrationProfileV1 | None = None,
) -> DatasetImport:
    dataset_name = _id(dataset_id, "dataset_id")
    episode_name = _id(episode_id, "episode_id")
    source = Path(source_path).resolve()
    if not source.exists():
        raise ContractError(f"source path does not exist: {source}")
    clock = resolve_latency_clock(source, latency_clock)
    spec = scenario if isinstance(scenario, ScenarioSpecV1) else load_scenario(scenario)
    parent = Path(output_root) / "datasets" / "sim2real"
    final = parent / dataset_name
    parent.mkdir(parents=True, exist_ok=True)
    is_new = not final.exists()
    if is_new:
        final.mkdir()
        (final / "raw").mkdir()
        (final / "episodes").mkdir()
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "dataset_id": dataset_name,
            "raw": [],
            "episodes": [],
        }
    else:
        try:
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"existing dataset manifest is invalid: {exc}") from exc
        if manifest.get("schema_version") != 1 or manifest.get("dataset_id") != dataset_name:
            raise ContractError("existing dataset manifest has an incompatible identity/version")
        if not isinstance(manifest.get("raw"), list) or not isinstance(manifest.get("episodes"), list):
            raise ContractError("existing dataset manifest raw/episodes must be arrays")
    raw_final = final / "raw" / source.name
    episode_final = final / "episodes" / episode_name
    if raw_final.exists():
        raise ContractError(f"raw source already exists and is immutable: {raw_final}")
    if episode_final.exists() or any(
        item.get("episode_id") == episode_name for item in manifest["episodes"]
    ):
        raise ContractError(f"episode already exists and is immutable: {episode_name}")
    token = uuid.uuid4().hex
    raw_stage = final / "raw" / f".{source.stem}-{token}.tmp{source.suffix}"
    episode_stage = final / "episodes" / f".{episode_name}-{token}.tmp"
    raw_committed = False
    episode_committed = False
    try:
        if source.is_dir():
            shutil.copytree(source, raw_stage)
        else:
            shutil.copy2(source, raw_stage)
        trace_manifest = import_real_trace(
            raw_stage,
            episode_stage,
            scenario=spec,
            units=units,
            frames=frames,
            latency_clock=clock,
            time_bases=time_bases,
            profile=profile,
        )
        raw_hash = sha256_path(raw_stage)
        metadata_hash = sha256_file(episode_stage / "metadata.json")
        os.replace(raw_stage, raw_final)
        raw_committed = True
        os.replace(episode_stage, episode_final)
        episode_committed = True
        manifest["raw"].append(
            {"path": raw_final.relative_to(final).as_posix(), "sha256": raw_hash}
        )
        manifest["episodes"].append(
            {
                "episode_id": episode_name,
                "scenario_id": spec.scenario_id,
                "path": episode_final.relative_to(final).as_posix(),
                "trace_sha256": trace_manifest.provenance["trace_sha256"],
                "metadata_sha256": metadata_hash,
            }
        )
        _atomic_json(final / "manifest.json", manifest)
        return DatasetImport(
            dataset=final,
            episode=episode_final,
            manifest=manifest,
        )
    except BaseException:
        if raw_stage.is_dir():
            shutil.rmtree(raw_stage, ignore_errors=True)
        else:
            raw_stage.unlink(missing_ok=True)
        shutil.rmtree(episode_stage, ignore_errors=True)
        if episode_committed:
            shutil.rmtree(episode_final, ignore_errors=True)
        if raw_committed:
            if raw_final.is_dir():
                shutil.rmtree(raw_final, ignore_errors=True)
            else:
                raw_final.unlink(missing_ok=True)
        if is_new:
            shutil.rmtree(final, ignore_errors=True)
        raise
