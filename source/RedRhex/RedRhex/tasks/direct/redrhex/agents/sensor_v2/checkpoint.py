"""Strict, transition-aware Sensor V2 checkpoint format."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .models import (
    ACTION_DIM_V2,
    COMMAND_DIM_V2,
    LATENT_DIM_V2,
    SENSOR_FRAME_DIM_V2,
    SENSOR_HISTORY_LENGTH_V2,
)


CHECKPOINT_FORMAT_V2 = "redrhex.sensor-training.v2"

# These fields identify where/how a run was executed, but do not define its
# training behavior.  Keep this allowlist deliberately small: any unknown or
# newly introduced config field remains hash-bound by default.
CANONICAL_CONFIG_EXECUTION_IDENTITY_FIELDS_V2 = frozenset(
    {
        "device",
        "experiment_name",
        "load_checkpoint",
        "load_run",
        "logger",
        "neptune_project",
        "resume",
        "run_name",
        "seed",
        "wandb_project",
    }
)


class CheckpointKindV2(str, Enum):
    TEACHER = "teacher_v2"
    DISTILLED = "student_distilled_v2"
    PPO = "student_ppo_v2"


class CheckpointIntentV2(str, Enum):
    DISTILLATION_BOOTSTRAP = "distillation_bootstrap"
    DISTILLATION_RESUME = "distillation_resume"
    PPO_BOOTSTRAP = "ppo_bootstrap"
    ROBUSTNESS_BOOTSTRAP = "robustness_bootstrap"
    PPO_RESUME = "ppo_resume"
    TEACHER_RESUME = "teacher_resume"
    INFERENCE = "inference"


@dataclass(frozen=True)
class CheckpointManifestV2:
    """Hash-bound identity and provenance required by every V2 checkpoint."""

    kind: CheckpointKindV2 | str
    stage: str
    observation_contract_id: str
    contract_hash: str
    action_contract_id: str
    action_contract_hash: str
    calibration_hash: str
    architecture_hash: str
    config_hash: str
    canonical_config_hash: str
    training_seed: int
    action_order: tuple[str, ...]
    iteration: int = 0
    scheduler_state_present: bool = False
    source_checkpoint_hash: str | None = None
    source_checkpoint_kind: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    sensor_frame_dim: int = SENSOR_FRAME_DIM_V2
    history_length: int = SENSOR_HISTORY_LENGTH_V2
    command_dim: int = COMMAND_DIM_V2
    action_dim: int = ACTION_DIM_V2
    latent_dim: int = LATENT_DIM_V2
    schema_version: int = 2
    format: str = CHECKPOINT_FORMAT_V2

    def __post_init__(self) -> None:
        kind = CheckpointKindV2(self.kind)
        object.__setattr__(self, "kind", kind.value)
        if self.format != CHECKPOINT_FORMAT_V2:
            raise ValueError(f"unsupported V2 checkpoint format: {self.format!r}")
        if not self.stage:
            raise ValueError("checkpoint stage must not be empty")
        if self.schema_version != 2:
            raise ValueError("Sensor V2 checkpoint schema_version must be 2")
        if not self.observation_contract_id or not self.action_contract_id:
            raise ValueError("checkpoint observation/action contract IDs must not be empty")
        if self.iteration < 0:
            raise ValueError("checkpoint iteration must be non-negative")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed < 0
        ):
            raise ValueError("training_seed must be a non-negative integer")
        for name in (
            "contract_hash",
            "action_contract_hash",
            "calibration_hash",
            "architecture_hash",
            "config_hash",
            "canonical_config_hash",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        expected_dimensions = {
            "sensor_frame_dim": SENSOR_FRAME_DIM_V2,
            "history_length": SENSOR_HISTORY_LENGTH_V2,
            "command_dim": COMMAND_DIM_V2,
            "action_dim": ACTION_DIM_V2,
            "latent_dim": LATENT_DIM_V2,
        }
        for name, expected in expected_dimensions.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected} for Sensor V2")
        if len(self.action_order) != self.action_dim or len(set(self.action_order)) != self.action_dim:
            raise ValueError(f"action_order must contain {self.action_dim} unique names")
        if (self.source_checkpoint_hash is None) != (self.source_checkpoint_kind is None):
            raise ValueError("source checkpoint hash and kind must either both be set or both be absent")
        if self.source_checkpoint_hash is not None:
            if len(self.source_checkpoint_hash) != 64:
                raise ValueError("source_checkpoint_hash must be a SHA-256 hex digest")
            CheckpointKindV2(self.source_checkpoint_kind)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointManifestV2":
        if not isinstance(data, dict):
            raise TypeError("checkpoint manifest must be a dictionary")
        return cls(**data)


def canonical_hash_v2(value: Any) -> str:
    """SHA-256 of canonical JSON-compatible content."""

    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_training_config_v2(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return config content with only execution identity fields removed."""

    if not isinstance(config, Mapping):
        raise TypeError("Sensor V2 training config must be a mapping")

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("Sensor V2 training config keys must be strings")
                if key in CANONICAL_CONFIG_EXECUTION_IDENTITY_FIELDS_V2:
                    continue
                normalized[key] = normalize(item)
            return normalized
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return copy.deepcopy(value)

    return normalize(config)


def canonical_training_config_hash_v2(config: Mapping[str, Any]) -> str:
    """Hash training semantics independently of seed/run/log identity."""

    return canonical_hash_v2(normalize_training_config_v2(config))


def architecture_hash_v2(module: nn.Module) -> str:
    """Hash state keys, shapes, and dtypes without hashing learned values."""

    schema = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in module.state_dict().items()
    ]
    return canonical_hash_v2(schema)


def file_sha256_v2(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_intent_v2(kind: CheckpointKindV2 | str, intent: CheckpointIntentV2 | str) -> None:
    """Reject every transition outside the approved V2 lineage graph."""

    kind = CheckpointKindV2(kind)
    intent = CheckpointIntentV2(intent)
    legal = {
        (CheckpointKindV2.TEACHER, CheckpointIntentV2.DISTILLATION_BOOTSTRAP),
        (CheckpointKindV2.TEACHER, CheckpointIntentV2.TEACHER_RESUME),
        (CheckpointKindV2.DISTILLED, CheckpointIntentV2.DISTILLATION_RESUME),
        (CheckpointKindV2.DISTILLED, CheckpointIntentV2.PPO_BOOTSTRAP),
        (CheckpointKindV2.PPO, CheckpointIntentV2.ROBUSTNESS_BOOTSTRAP),
        (CheckpointKindV2.PPO, CheckpointIntentV2.PPO_RESUME),
        (CheckpointKindV2.TEACHER, CheckpointIntentV2.INFERENCE),
        (CheckpointKindV2.DISTILLED, CheckpointIntentV2.INFERENCE),
        (CheckpointKindV2.PPO, CheckpointIntentV2.INFERENCE),
    }
    if (kind, intent) not in legal:
        raise ValueError(f"illegal V2 checkpoint transition: {kind.value} -> {intent.value}")


def _assert_manifest_matches(
    actual: CheckpointManifestV2,
    expected: CheckpointManifestV2 | None,
) -> None:
    if expected is None:
        return
    fields = (
        "kind",
        "stage",
        "observation_contract_id",
        "contract_hash",
        "action_contract_id",
        "action_contract_hash",
        "calibration_hash",
        "architecture_hash",
        "canonical_config_hash",
        "training_seed",
        "sensor_frame_dim",
        "history_length",
        "command_dim",
        "action_dim",
        "latent_dim",
        "action_order",
        "schema_version",
    )
    mismatches = [
        f"{name}: expected {getattr(expected, name)!r}, got {getattr(actual, name)!r}"
        for name in fields
        if getattr(actual, name) != getattr(expected, name)
    ]
    if mismatches:
        raise ValueError("V2 checkpoint manifest mismatch: " + "; ".join(mismatches))


def save_checkpoint_v2(
    path: str | Path,
    *,
    manifest: CheckpointManifestV2,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    update: int = 0,
    extra_state: dict[str, Any] | None = None,
) -> None:
    if update < 0:
        raise ValueError("checkpoint update must be non-negative")
    if manifest.iteration != update:
        raise ValueError(
            f"manifest iteration {manifest.iteration} does not match checkpoint update {update}"
        )
    if manifest.scheduler_state_present != (scheduler is not None):
        raise ValueError("manifest scheduler_state_present disagrees with supplied scheduler")
    payload = {
        "format": CHECKPOINT_FORMAT_V2,
        "manifest": manifest.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "update": int(update),
        "extra_state": dict(extra_state or {}),
    }
    torch.save(payload, Path(path))


def load_checkpoint_v2(
    path: str | Path,
    *,
    model: nn.Module,
    intent: CheckpointIntentV2 | str,
    expected_manifest: CheckpointManifestV2 | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[CheckpointManifestV2, dict[str, Any]]:
    """Strictly load a V2 checkpoint; partial shape-compatible loads are forbidden."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT_V2:
        raise ValueError("not a Sensor V2 checkpoint")
    required = {"manifest", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "update"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"V2 checkpoint is missing fields: {sorted(missing)}")
    manifest = CheckpointManifestV2.from_dict(payload["manifest"])
    if payload["update"] != manifest.iteration:
        raise ValueError("V2 checkpoint update disagrees with manifest iteration")
    if (payload["scheduler_state_dict"] is not None) != manifest.scheduler_state_present:
        raise ValueError("V2 checkpoint scheduler state disagrees with manifest")
    validate_checkpoint_intent_v2(manifest.kind, intent)
    _assert_manifest_matches(manifest, expected_manifest)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        if payload["optimizer_state_dict"] is None:
            raise ValueError("V2 resume requested optimizer state, but checkpoint has none")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        if payload["scheduler_state_dict"] is None:
            raise ValueError("V2 resume requested scheduler state, but checkpoint has none")
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return manifest, payload
