from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS_ROOT = (
    REPO_ROOT
    / "source"
    / "RedRhex"
    / "RedRhex"
    / "tasks"
    / "direct"
    / "redrhex"
    / "agents"
)
if str(AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTS_ROOT))

from sensor_v2.checkpoint import (  # noqa: E402
    CANONICAL_CONFIG_EXECUTION_IDENTITY_FIELDS_V2,
    CheckpointManifestV2,
    CheckpointIntentV2,
    _assert_manifest_matches,
    canonical_hash_v2,
    canonical_training_config_hash_v2,
    load_checkpoint_v2,
    normalize_training_config_v2,
    save_checkpoint_v2,
)


EXECUTION_IDENTITY_FIELDS = {
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


def _config(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "device": f"cuda:{seed % 2}",
        "experiment_name": f"campaign-{seed}",
        "run_name": f"seed-{seed}",
        "logger": "tensorboard" if seed == 42 else "wandb",
        "wandb_project": f"wandb-{seed}",
        "neptune_project": f"neptune-{seed}",
        "resume": seed != 42,
        "load_run": f"run-{seed}",
        "load_checkpoint": f"model-{seed}.pt",
        "num_steps_per_env": 24,
        "max_iterations": 1500,
        "save_interval": 50,
        "policy": {"hidden_dims": [256, 128, 128]},
        "algorithm": {"learning_rate": 3.0e-4, "gamma": 0.99},
        "environment_provenance_v2": {
            "sensor_dr_profile_sha256": "a" * 64,
        },
        # Unknown fields remain bound; normalization is an allowlist, not a
        # heuristic based on path-like names.
        "output_dir": f"semantic-value-{seed}",
    }


def _manifest(**overrides: object) -> CheckpointManifestV2:
    values: dict[str, object] = {
        "kind": "student_ppo_v2",
        "stage": "ppo_f3",
        "observation_contract_id": "redrhex.student-observation.v2",
        "contract_hash": "1" * 64,
        "action_contract_id": "redrhex.forward-residual-action.v2",
        "action_contract_hash": "2" * 64,
        "calibration_hash": "3" * 64,
        "architecture_hash": "4" * 64,
        "config_hash": "5" * 64,
        "canonical_config_hash": "6" * 64,
        "training_seed": 42,
        "action_order": tuple(f"joint_{index}" for index in range(12)),
    }
    values.update(overrides)
    return CheckpointManifestV2(**values)


def test_canonical_training_config_strips_only_frozen_execution_identity_allowlist() -> None:
    assert CANONICAL_CONFIG_EXECUTION_IDENTITY_FIELDS_V2 == EXECUTION_IDENTITY_FIELDS
    first = _config(42)
    second = _config(43)
    # output_dir is deliberately not an execution-identity field, so first
    # prove an unknown changed field remains bound.
    second["output_dir"] = first["output_dir"]
    original = copy.deepcopy(first)

    assert canonical_hash_v2(first) != canonical_hash_v2(second)
    assert canonical_training_config_hash_v2(first) == canonical_training_config_hash_v2(
        second
    )
    assert first == original
    assert EXECUTION_IDENTITY_FIELDS.isdisjoint(normalize_training_config_v2(first))

    second["output_dir"] = "changed-unknown-field"
    assert canonical_training_config_hash_v2(first) != canonical_training_config_hash_v2(
        second
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        (None, "max_iterations", 1499),
        ("policy", "hidden_dims", [64, 64]),
        ("algorithm", "learning_rate", 1.0e-4),
        ("environment_provenance_v2", "sensor_dr_profile_sha256", "b" * 64),
    ),
)
def test_canonical_training_config_keeps_training_semantics(
    section: str | None,
    key: str,
    value: object,
) -> None:
    baseline = _config(42)
    changed = copy.deepcopy(baseline)
    if section is None:
        changed[key] = value
    else:
        nested = changed[section]
        assert isinstance(nested, dict)
        nested[key] = value

    assert canonical_training_config_hash_v2(baseline) != canonical_training_config_hash_v2(
        changed
    )


def test_current_v2_manifest_fails_closed_without_canonical_hash_or_training_seed() -> None:
    payload = _manifest().to_dict()
    payload.pop("canonical_config_hash")
    with pytest.raises(TypeError, match="canonical_config_hash"):
        CheckpointManifestV2.from_dict(payload)

    payload = _manifest().to_dict()
    payload.pop("training_seed")
    with pytest.raises(TypeError, match="training_seed"):
        CheckpointManifestV2.from_dict(payload)

    with pytest.raises(ValueError, match="training_seed"):
        _manifest(training_seed=-1)
    with pytest.raises(ValueError, match="training_seed"):
        _manifest(training_seed=True)


def test_resume_matches_semantics_and_seed_but_preserves_distinct_full_hashes() -> None:
    saved_config = _config(42)
    resume_config = copy.deepcopy(saved_config)
    resume_config.update(
        {
            "device": "cpu",
            "run_name": "resumed-run",
            "logger": "wandb",
            "resume": True,
            "load_run": "prior-run",
            "load_checkpoint": "model_100.pt",
        }
    )
    saved = _manifest(
        config_hash=canonical_hash_v2(saved_config),
        canonical_config_hash=canonical_training_config_hash_v2(saved_config),
    )
    execution_only_change = _manifest(
        config_hash=canonical_hash_v2(resume_config),
        canonical_config_hash=canonical_training_config_hash_v2(resume_config),
    )

    _assert_manifest_matches(saved, execution_only_change)
    assert saved.config_hash != execution_only_change.config_hash

    changed_hyperparameter = copy.deepcopy(resume_config)
    changed_hyperparameter["max_iterations"] = 1499
    with pytest.raises(ValueError, match="canonical_config_hash"):
        _assert_manifest_matches(
            saved,
            _manifest(
                config_hash=canonical_hash_v2(changed_hyperparameter),
                canonical_config_hash=canonical_training_config_hash_v2(
                    changed_hyperparameter
                ),
            ),
        )
    with pytest.raises(ValueError, match="training_seed"):
        _assert_manifest_matches(
            saved,
            _manifest(config_hash="b" * 64, training_seed=43),
        )


@pytest.mark.parametrize("missing", ("canonical_config_hash", "training_seed"))
def test_checkpoint_loader_rejects_legacy_v2_manifest_missing_provenance(
    tmp_path: Path,
    missing: str,
) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / f"missing-{missing}.pt"
    save_checkpoint_v2(path, manifest=_manifest(), model=model, update=0)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["manifest"].pop(missing)
    torch.save(payload, path)

    with pytest.raises(TypeError, match=missing):
        load_checkpoint_v2(
            path,
            model=torch.nn.Linear(2, 1),
            intent=CheckpointIntentV2.INFERENCE,
        )


def test_checkpoint_writer_binds_full_canonical_and_seed_provenance() -> None:
    source = (AGENTS_ROOT / "sensor_v2" / "backends.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    manifest_function = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_manifest"
    )
    rendered = ast.unparse(manifest_function)

    assert "config_hash=canonical_hash_v2(config)" in rendered
    assert "canonical_config_hash=canonical_training_config_hash_v2(config)" in rendered
    assert "training_seed=config['seed']" in rendered


def test_teacher_resume_cursor_points_to_next_update_without_duplication() -> None:
    source = (AGENTS_ROOT / "sensor_v2" / "backends.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    teacher = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "VersionedTeacherBackendV2"
    )
    methods = {
        item.name: ast.unparse(item)
        for item in teacher.body
        if isinstance(item, ast.FunctionDef)
    }

    assert "start = self.current_learning_iteration" in methods["learn"]
    assert "self.current_learning_iteration = start + count" in methods["learn"]
    assert "self._teacher_learn_active = False" in methods["learn"]
    assert "checkpoint_iteration = self.current_learning_iteration + int(" in methods["save"]
    assert "update=checkpoint_iteration" in methods["save"]
    assert "self.current_learning_iteration = manifest.iteration" in methods["load"]
