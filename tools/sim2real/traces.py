from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .contracts import (
    CalibrationProfileV1,
    ContractError,
    ScenarioSpecV1,
    TraceManifestV1,
    load_manifest,
)
from .scenarios import load_scenario


@dataclass(frozen=True)
class LoadedTrace:
    manifest: TraceManifestV1
    arrays: dict[str, np.ndarray]
    directory: Path


@dataclass(frozen=True)
class _DatasetLink:
    root: Path
    episode: Mapping[str, Any]
    raw_entries: tuple[Mapping[str, Any], ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or a directory tree, including stable relative filenames."""

    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise ContractError(f"source path does not exist: {source}")
    digest = hashlib.sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_link(directory: Path) -> _DatasetLink | None:
    if directory.parent.name != "episodes":
        return None
    root = directory.parent.parent
    dataset_manifest = root / "manifest.json"
    if not dataset_manifest.is_file():
        return None
    try:
        payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"dataset manifest is invalid: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ContractError("dataset manifest has an invalid schema version")
    episodes = payload.get("episodes")
    raw_entries = payload.get("raw")
    if not isinstance(episodes, list) or not isinstance(raw_entries, list):
        raise ContractError("dataset manifest raw/episodes must be arrays")
    relative = directory.relative_to(root).as_posix()
    matches = [
        item
        for item in episodes
        if isinstance(item, Mapping) and item.get("path") == relative
    ]
    if len(matches) != 1:
        raise ContractError("dataset manifest must link the episode exactly once")
    if not all(isinstance(item, Mapping) for item in raw_entries):
        raise ContractError("dataset manifest raw entries must be objects")
    return _DatasetLink(root, matches[0], tuple(raw_entries))


def _dataset_sha(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContractError(f"dataset manifest {field} must be a lowercase SHA-256 digest")
    return value


def _dataset_path(root: Path, value: Any, *, prefix: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"dataset manifest {prefix} path must be a non-empty string")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or not relative.parts
        or relative.parts[0] != prefix
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ContractError(f"dataset manifest {prefix} path is unsafe")
    return root.joinpath(*relative.parts)


def _verify_dataset_metadata(link: _DatasetLink, metadata_path: Path) -> None:
    if metadata_path.name != "metadata.json" or not metadata_path.is_file():
        raise ContractError("dataset episode metadata.json is missing")
    expected = _dataset_sha(link.episode, "metadata_sha256")
    if sha256_file(metadata_path) != expected:
        raise ContractError("dataset metadata hash mismatch")


def _verify_dataset_sources(link: _DatasetLink, manifest: TraceManifestV1) -> None:
    episode = link.episode
    if episode.get("episode_id") != link.root.joinpath(episode["path"]).name:
        raise ContractError("dataset episode identity/path mismatch")
    if episode.get("scenario_id") != manifest.scenario_id:
        raise ContractError("dataset episode scenario linkage mismatch")
    trace_hash = _dataset_sha(episode, "trace_sha256")
    if trace_hash != manifest.provenance["trace_sha256"]:
        raise ContractError("dataset trace hash linkage mismatch")
    source_hash = manifest.provenance.get("source_sha256")
    raw_path = episode.get("raw_path")
    if raw_path is None:
        matches = [
            item
            for item in link.raw_entries
            if _dataset_sha(item, "sha256") == source_hash
        ]
        if len(matches) != 1:
            raise ContractError("dataset raw source linkage is missing or ambiguous")
        raw_entry = matches[0]
        raw_path = raw_entry.get("path")
    else:
        _dataset_path(link.root, raw_path, prefix="raw")
        matches = [item for item in link.raw_entries if item.get("path") == raw_path]
        if len(matches) != 1:
            raise ContractError("dataset raw source linkage must resolve exactly once")
        raw_entry = matches[0]
    raw_file = _dataset_path(link.root, raw_path, prefix="raw")
    expected_raw_hash = _dataset_sha(raw_entry, "sha256")
    if expected_raw_hash != source_hash:
        raise ContractError("dataset raw source linkage hash mismatch")
    try:
        actual_raw_hash = sha256_path(raw_file)
    except ContractError as exc:
        raise ContractError("dataset raw source is missing or invalid") from exc
    if actual_raw_hash != expected_raw_hash:
        raise ContractError("dataset raw source hash mismatch")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w+b") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _numeric_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ContractError("trace arrays must be a non-empty mapping")
    result: dict[str, np.ndarray] = {}
    for name, raw in arrays.items():
        if not isinstance(name, str) or not name:
            raise ContractError("trace channel names must be non-empty strings")
        array = np.asarray(raw)
        if array.dtype.kind not in "iuf":
            raise ContractError(f"channel {name} must contain numeric, non-object values")
        if array.ndim < 1 or array.shape[0] < 1:
            raise ContractError(f"channel {name} must contain at least one sample")
        if not np.isfinite(array).all():
            raise ContractError(f"channel {name} must contain only finite values")
        result[name] = np.array(array, copy=True)
    return result


def _validate_layout(
    arrays: Mapping[str, np.ndarray],
    time_bases: Mapping[str, str],
    *,
    required_channels: tuple[str, ...] = (),
) -> dict[str, int]:
    missing = (set(required_channels) | set(time_bases) | set(time_bases.values())) - set(arrays)
    if missing:
        raise ContractError(f"missing channels: {', '.join(sorted(missing))}")
    counts: dict[str, int] = {}
    for channel, time_name in time_bases.items():
        time = arrays[time_name]
        values = arrays[channel]
        if time.ndim != 1:
            raise ContractError(f"time base {time_name} must be one-dimensional")
        if time.shape[0] != values.shape[0]:
            raise ContractError(f"shape mismatch between {channel} and {time_name}")
        if time.shape[0] > 1 and not np.all(np.diff(time) > 0.0):
            raise ContractError(f"time base {time_name} must be strictly increasing")
        counts[channel] = int(values.shape[0])
    return counts


def _coerce_scenario(value: ScenarioSpecV1 | str | Path | None) -> ScenarioSpecV1 | None:
    if value is None or isinstance(value, ScenarioSpecV1):
        return value
    return load_scenario(value)


def _base_manifest_fields(
    manifest: TraceManifestV1 | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if manifest is None:
        return {}
    if isinstance(manifest, TraceManifestV1):
        return manifest.validate().to_dict()
    return TraceManifestV1.from_dict(manifest).to_dict()


def _metadata(
    *,
    arrays: Mapping[str, np.ndarray],
    time_bases: Mapping[str, str],
    scenario_hash: str | None,
    source_hash: str | None,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(supplied or {})
    data_channels = sorted(time_bases)
    metadata["units"] = {
        **{name: "unspecified" for name in data_channels},
        **dict(metadata.get("units", {})),
    }
    metadata["frames"] = {
        **{name: "unspecified" for name in data_channels},
        **dict(metadata.get("frames", {})),
    }
    metadata.setdefault("joint_order", [])
    metadata.setdefault(
        "clock",
        {
            "source": "independent",
            "timestamp_semantics": "relative_monotonic",
            "time_unit": "s",
        },
    )
    metadata.setdefault("scenario_schema_version", 1 if scenario_hash else None)
    metadata.setdefault("scenario_sha256", scenario_hash)
    metadata.setdefault("git_sha", None)
    metadata.setdefault("asset_sha256", None)
    metadata.setdefault("config_sha256", None)
    metadata.setdefault("calibration_constants", {})
    metadata.setdefault("raw_data_sha256", source_hash)
    return metadata


def write_trace(
    output_dir: str | Path,
    arrays: Mapping[str, Any],
    manifest: TraceManifestV1 | Mapping[str, Any] | None = None,
    *,
    scenario: ScenarioSpecV1 | str | Path | None = None,
    source: str | None = None,
    source_path: str | Path | None = None,
    profile: CalibrationProfileV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
    time_bases: Mapping[str, str] | None = None,
) -> TraceManifestV1:
    """Validate and atomically write ``trace.npz`` plus ``metadata.json``."""

    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ContractError(f"trace artifact already exists: {output}")
    created_output = not output.exists()
    clean_arrays = _numeric_arrays(arrays)
    spec = _coerce_scenario(scenario)
    base = _base_manifest_fields(manifest)
    scenario_id = spec.scenario_id if spec else base.get("scenario_id")
    if not scenario_id:
        raise ContractError("scenario or manifest.scenario_id is required")
    trace_source = source or base.get("source")
    if trace_source not in {"real", "sim", "derived"}:
        raise ContractError("source must be real, sim, or derived")
    resolved_time_bases = dict(spec.time_bases if spec else base.get("time_bases", {}))
    for channel, time_name in dict(time_bases or {}).items():
        if channel in resolved_time_bases and resolved_time_bases[channel] != time_name:
            raise ContractError(f"time-base conflict for {channel}")
        resolved_time_bases[channel] = time_name
    required = spec.required_channels if spec else tuple(resolved_time_bases)
    unmapped = set(clean_arrays) - set(resolved_time_bases) - set(resolved_time_bases.values())
    if unmapped:
        raise ContractError(
            "extra channels require explicit time_bases: " + ", ".join(sorted(unmapped))
        )
    sample_counts = _validate_layout(
        clean_arrays, resolved_time_bases, required_channels=required
    )
    trace_file = str(base.get("trace_file", "trace.npz"))
    if Path(trace_file).name != trace_file or not trace_file.endswith(".npz"):
        raise ContractError("trace_file must be a local .npz filename")

    scenario_hash = sha256_json(spec.to_dict()) if spec else None
    source_hash = sha256_path(source_path) if source_path is not None else None
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / trace_file
    metadata_path = output / "metadata.json"
    try:
        _atomic_npz(trace_path, clean_arrays)
        provenance = dict(base.get("provenance", {}))
        provenance["trace_sha256"] = sha256_file(trace_path)
        if scenario_hash is not None:
            provenance["scenario_sha256"] = scenario_hash
        if source_hash is not None:
            provenance["source_sha256"] = source_hash
        if profile is not None:
            provenance["profile_sha256"] = sha256_json(profile.to_dict())
        clean_metadata = _metadata(
            arrays=clean_arrays,
            time_bases=resolved_time_bases,
            scenario_hash=scenario_hash,
            source_hash=source_hash,
            supplied=metadata or base.get("metadata"),
        )
        payload = {
            "schema_version": 1,
            "scenario_id": scenario_id,
            "source": trace_source,
            "trace_file": trace_file,
            "channels": sorted(clean_arrays),
            "time_bases": resolved_time_bases,
            "sample_counts": sample_counts,
            "provenance": provenance,
            "metadata": clean_metadata,
        }
        result = TraceManifestV1.from_dict(payload)
        _atomic_json(metadata_path, result.to_dict())
        return result
    except BaseException:
        for artifact in (metadata_path, trace_path):
            try:
                artifact.unlink()
            except FileNotFoundError:
                pass
        if created_output:
            try:
                output.rmdir()
            except OSError:
                pass
        raise


def load_trace(
    value: str | Path,
    *,
    scenario: ScenarioSpecV1 | str | Path | None = None,
    profile: CalibrationProfileV1 | None = None,
    verify_hashes: bool = True,
    expected_units: Mapping[str, str] | None = None,
    expected_frames: Mapping[str, str] | None = None,
) -> LoadedTrace:
    path = Path(value)
    directory = path if path.is_dir() or path.suffix != ".npz" else path.parent
    dataset_link = _dataset_link(directory) if verify_hashes else None
    metadata_path = directory / "metadata.json"
    if dataset_link is not None:
        _verify_dataset_metadata(dataset_link, metadata_path)
    if not metadata_path.exists():
        metadata_path = directory / "manifest.json"
    manifest = load_manifest(metadata_path)
    if dataset_link is not None:
        _verify_dataset_sources(dataset_link, manifest)
    trace_path = directory / manifest.trace_file
    if verify_hashes and sha256_file(trace_path) != manifest.provenance["trace_sha256"]:
        raise ContractError("trace hash mismatch")
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            arrays = _numeric_arrays({name: archive[name] for name in archive.files})
    except (OSError, ValueError) as exc:
        raise ContractError(f"trace contains object or invalid arrays: {exc}") from exc
    if set(arrays) != set(manifest.channels):
        missing = set(manifest.channels) - set(arrays)
        extra = set(arrays) - set(manifest.channels)
        raise ContractError(f"manifest channel mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    counts = _validate_layout(arrays, manifest.time_bases)
    if counts != manifest.sample_counts:
        raise ContractError("manifest sample-count shape mismatch")
    spec = _coerce_scenario(scenario)
    if spec is not None:
        if manifest.scenario_id != spec.scenario_id:
            raise ContractError("scenario id mismatch")
        _validate_layout(arrays, spec.time_bases, required_channels=spec.required_channels)
        expected = sha256_json(spec.to_dict())
        if verify_hashes and manifest.provenance.get("scenario_sha256") != expected:
            raise ContractError("scenario hash mismatch")
    if profile is not None and verify_hashes:
        expected = sha256_json(profile.to_dict())
        if manifest.provenance.get("profile_sha256") != expected:
            raise ContractError("profile hash mismatch")
    for expected, field, label in (
        (expected_units, "units", "unit"),
        (expected_frames, "frames", "frame"),
    ):
        if expected is None:
            continue
        actual = manifest.metadata[field]
        for channel, value in expected.items():
            if channel not in actual or actual[channel] != value:
                raise ContractError(
                    f"expected {label} {value!r} for {channel}, got {actual.get(channel)!r}"
                )
    return LoadedTrace(manifest=manifest, arrays=arrays, directory=directory)
