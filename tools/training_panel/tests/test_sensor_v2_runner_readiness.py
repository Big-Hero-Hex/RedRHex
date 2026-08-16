from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = (
    ROOT
    / "source"
    / "RedRhex"
    / "RedRhex"
    / "tasks"
    / "direct"
    / "redrhex"
    / "agents"
    / "sensor_v2"
)
PACKAGE = "redrhex_sensor_v2_readiness_under_test"


def _load_module(name: str):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(MODULE_ROOT)]
        sys.modules[PACKAGE] = package
    qualified_name = f"{PACKAGE}.{name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    spec = importlib.util.spec_from_file_location(
        qualified_name,
        MODULE_ROOT / f"{name}.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


_load_module("models")
storage = _load_module("storage")
distillation = _load_module("distillation")


def _backend_class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        item
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )


def test_next_history_readiness_is_a_causal_transition_boundary() -> None:
    valid = storage.causal_transition_mask_v2(
        torch.tensor([False, False, True, True]),
        torch.tensor([True, False, True, False]),
    )

    torch.testing.assert_close(
        valid,
        torch.tensor([True, False, False, False]),
    )
    with pytest.raises(ValueError, match="identical shapes"):
        storage.causal_transition_mask_v2(
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, 1, dtype=torch.bool),
        )


def test_4096_environment_dropout_priming_has_no_all_ready_barrier() -> None:
    readiness = torch.zeros(4096, dtype=torch.bool)
    readiness[2048] = True

    assert storage.has_trainable_history_v2(readiness)
    assert not bool(readiness.all())
    with pytest.raises(ValueError, match="shape"):
        storage.has_trainable_history_v2(readiness.unsqueeze(-1))


def test_invalid_next_history_cuts_gae_and_dynamics_target() -> None:
    returns, advantage = storage.causal_gae_step_v2(
        reward=torch.tensor([[0.0]]),
        value=torch.tensor([[10.0]]),
        following_value=torch.tensor([[20.0]]),
        following_advantage=torch.tensor([[1000.0]]),
        current_ready=torch.tensor([[True]]),
        transition_valid=torch.tensor([[False]]),
        gamma=1.0,
        lam=1.0,
    )

    torch.testing.assert_close(returns, torch.tensor([[0.0]]))
    torch.testing.assert_close(advantage, torch.tensor([[-10.0]]))
    dynamics_loss = distillation._masked_huber(
        torch.full((1, 36), 1000.0),
        torch.zeros(1, 36),
        torch.tensor([False]),
    )
    torch.testing.assert_close(dynamics_loss, torch.tensor(0.0))


def test_runner_disables_random_episode_age_and_uses_ready_rows_for_priming() -> None:
    tree = ast.parse((MODULE_ROOT / "backends.py").read_text(encoding="utf-8"))
    base = _backend_class(tree, "_CustomRunnerV2")
    warm_source = ast.unparse(_method(base, "_warm_sensor_history"))

    assert "has_trainable_history_v2(ready)" in warm_source
    assert "if bool(ready.all())" not in warm_source
    assert "ready_seen" not in warm_source

    for class_name in ("SensorDistillationBackendV2", "SensorOnPolicyBackendV2"):
        learn_source = ast.unparse(_method(_backend_class(tree, class_name), "learn"))
        assert "self._randomize_episode_lengths()" not in learn_source
        assert "episode_length_buf" not in learn_source
        assert "random episode-age initialization is disabled" in learn_source


def test_next_ready_masks_auxiliary_targets_and_gae_continuation() -> None:
    tree = ast.parse((MODULE_ROOT / "backends.py").read_text(encoding="utf-8"))
    distillation = ast.unparse(
        _method(_backend_class(tree, "SensorDistillationBackendV2"), "learn")
    )
    ppo = ast.unparse(
        _method(_backend_class(tree, "SensorOnPolicyBackendV2"), "learn")
    )

    assert "transition_valid = causal_transition_mask_v2" in distillation
    assert "terminal=~transition_valid[ready]" in distillation
    assert "next_history_ready_seen.append(next_ready)" in ppo
    assert "transition_valid_tensor = causal_transition_mask_v2" in ppo
    assert "causal_gae_step_v2" in ppo
    assert "transition_valid=transition_valid_tensor[step]" in ppo
    assert "'terminal': (~transition_valid_tensor).flatten" in ppo
    assert "time_outs.to(self.device).float() * next_ready" not in ppo


def test_custom_checkpoints_store_the_next_iteration_for_resume() -> None:
    tree = ast.parse((MODULE_ROOT / "backends.py").read_text(encoding="utf-8"))
    for class_name in ("SensorDistillationBackendV2", "SensorOnPolicyBackendV2"):
        learn = _method(_backend_class(tree, class_name), "learn")
        updates = [
            node
            for node in ast.walk(learn)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "current_learning_iteration"
                for target in node.targets
            )
        ]
        assert len(updates) == 1
        value = updates[0].value
        assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
        assert isinstance(value.left, ast.Name) and value.left.id == "iteration"
        assert isinstance(value.right, ast.Constant) and value.right.value == 1
        source = ast.unparse(learn)
        assert "self.current_learning_iteration % self.save_interval == 0" in source
        assert "model_{self.current_learning_iteration}.pt" in source
