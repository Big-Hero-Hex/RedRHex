from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).parents[3]
MODULE_ROOT = ROOT / "source/RedRhex/RedRhex/tasks/direct/redrhex/agents/sensor_v2"
PACKAGE = "redrhex_sensor_v2_training_under_test"


def _load_module(name: str):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(MODULE_ROOT)]
        sys.modules[PACKAGE] = package
    qualified_name = f"{PACKAGE}.{name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    spec = importlib.util.spec_from_file_location(qualified_name, MODULE_ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


models = _load_module("models")
bundle_contract = _load_module("bundle_contract")
storage_module = _load_module("storage")
_load_module("distillation")
ppo = _load_module("ppo")
checkpoint = _load_module("checkpoint")


def _actor_observations(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "sensor_history_v2": torch.randn(batch, 60, 36),
        "command_v2": torch.randn(batch, 3),
        "critic_privileged_v2": torch.randn(batch, 65),
    }


class _OutOfRangeTeacher(nn.Module):
    def forward(self, privileged_observation: torch.Tensor) -> torch.Tensor:
        row = torch.tensor(
            [2.0, -2.0, 0.5, -0.5, 1.0, -1.0, 0.2, -0.2, 3.0, -3.0, 0.7, -0.7],
            device=privileged_observation.device,
        )
        return row.expand(privileged_observation.shape[0], -1)


def test_bundle_contract_uses_builder_keys_and_simulator_rest_gravity(monkeypatch) -> None:
    parameters = dict(
        bundle_contract.causal_attitude_parameters_v2(
            correction_gain=0.02,
            accel_norm_gate_m_s2=(7.5, 12.0),
            gravity_vector_m_s2=(0.0, 0.0, -9.81),
        )
    )
    rest_gravity = bundle_contract.rest_projected_gravity_v2(
        torch.tensor([[0.0, -2.0, 0.0], [0.0, -1.0, 0.0]])
    )
    wrapped_rest_gravity = bundle_contract.environment_rest_projected_gravity_v2(
        types.SimpleNamespace(
            unwrapped=types.SimpleNamespace(
                reference_projected_gravity=torch.tensor([[0.0, -1.0, 0.0]])
            )
        )
    )

    assert set(parameters) == {
        "accel_correction_gain",
        "accel_magnitude_tolerance_ratio",
        "gravity_magnitude_m_s2",
    }
    assert parameters["gravity_magnitude_m_s2"] == pytest.approx(9.81)
    assert parameters["accel_magnitude_tolerance_ratio"] == pytest.approx((9.81 - 7.5) / 9.81)
    assert rest_gravity == pytest.approx((0.0, -1.0, 0.0))
    assert wrapped_rest_gravity == pytest.approx((0.0, -1.0, 0.0))
    monkeypatch.syspath_prepend(str(ROOT / "source/redrhex_policy_io"))
    from redrhex_policy_io import SensorFrameBuilderV2, StudentObservationContractV2

    contract = StudentObservationContractV2.causal_gyro_accel(
        rest_projected_gravity=rest_gravity,
        attitude_parameters=tuple(parameters.items()),
    )
    assert SensorFrameBuilderV2(contract).contract.rest_projected_gravity == pytest.approx(rest_gravity)
    with pytest.raises(ValueError, match="agree"):
        bundle_contract.rest_projected_gravity_v2(
            torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        )


def test_rollout_observation_snapshot_breaks_live_tensor_alias() -> None:
    live = {
        "sensor_history_v2": torch.zeros(2, 60, 36),
        "command_v2": torch.ones(2, 3),
    }

    snapshot = storage_module.clone_observations_v2(live, tuple(live))
    live["sensor_history_v2"].fill_(9.0)
    live["command_v2"].zero_()

    assert snapshot["sensor_history_v2"].data_ptr() != live["sensor_history_v2"].data_ptr()
    assert snapshot["command_v2"].data_ptr() != live["command_v2"].data_ptr()
    assert torch.count_nonzero(snapshot["sensor_history_v2"]) == 0
    assert torch.all(snapshot["command_v2"] == 1.0)


def test_teacher_labels_and_rollout_are_strict_forward_actions() -> None:
    model = models.SensorStudentTeacherV2(_OutOfRangeTeacher())
    privileged = torch.zeros(2, 65)

    labels = model.teacher_actions(privileged)

    torch.testing.assert_close(
        labels[:, :6],
        torch.tensor([[1.0, -1.0, 0.5, -0.5, 1.0, -1.0]]).expand(2, -1),
    )
    assert torch.count_nonzero(labels[:, 6:]) == 0

    rollout = model.rollout(
        torch.zeros(2, 60, 36),
        torch.zeros(2, 3),
        privileged,
        beta=1.0,
        noise_std=0.0,
    )
    torch.testing.assert_close(rollout.teacher, labels)
    torch.testing.assert_close(rollout.executed, labels)


def test_deployable_student_reset_is_a_noop() -> None:
    student = models.SensorStudentCoreV2()
    state_before = {name: value.clone() for name, value in student.state_dict().items()}

    assert student.reset(torch.tensor([True, False])) is None

    for name, value in student.state_dict().items():
        torch.testing.assert_close(value, state_before[name])


def test_ppo_distribution_models_only_main_actions_but_returns_twelve() -> None:
    torch.manual_seed(7)
    policy = ppo.SensorActorCriticV2(65, critic_hidden_dims=(8,))
    observations = _actor_observations()

    actions = policy.act(observations)

    assert actions.shape == (2, 12)
    assert policy.std.shape == (6,)
    assert policy.distribution is not None
    assert policy.distribution.mean.shape == (2, 6)
    assert torch.count_nonzero(actions[:, 6:]) == 0
    assert torch.count_nonzero(policy.action_mean[:, 6:]) == 0
    assert torch.count_nonzero(policy.action_std[:, 6:]) == 0

    changed_abad = actions.clone()
    changed_abad[:, 6:] = 1000.0
    torch.testing.assert_close(
        policy.get_actions_log_prob(changed_abad),
        policy.get_actions_log_prob(actions),
    )
    torch.testing.assert_close(
        policy.entropy,
        policy.distribution.entropy().sum(dim=-1),
    )


def test_f1_teacher_config_uses_six_stochastic_action_adapter() -> None:
    config_source = (MODULE_ROOT / "config.py").read_text(encoding="utf-8")
    backend_source = (MODULE_ROOT / "backends.py").read_text(encoding="utf-8")

    assert 'class_name="StrictForwardTeacherActorCriticV2"' in config_source
    assert "class StrictForwardTeacherActorCriticV2(ActorCritic)" in backend_source
    assert "pad_main_actions_v2(super().act" in backend_source
    assert "actions[..., : ACTION_DIM_V2 // 2]" in backend_source
    assert "return torch.cat((main, torch.ones_like(main)), dim=-1)" in backend_source


def test_ppo_initial_main_action_noise_is_small_and_bounded() -> None:
    policy = ppo.SensorActorCriticV2(65, critic_hidden_dims=(8,))

    torch.testing.assert_close(
        policy.std.detach(),
        torch.full((6,), ppo.INITIAL_MAIN_ACTION_NOISE_STD_V2),
    )
    assert ppo.INITIAL_MAIN_ACTION_NOISE_STD_V2 <= 0.05
    with pytest.raises(ValueError, match="at most"):
        ppo.SensorActorCriticV2(
            65,
            critic_hidden_dims=(8,),
            init_noise_std=math.nextafter(ppo.MAX_INITIAL_MAIN_ACTION_NOISE_STD_V2, math.inf),
        )


def test_distillation_storage_yields_bounded_minibatches() -> None:
    storage = storage_module.SensorDistillationStorageV2(8)
    batch = storage_module.SensorDistillationBatchV2(
        sensor_history=torch.arange(8, dtype=torch.float32).view(8, 1, 1).expand(8, 60, 36),
        command=torch.zeros(8, 3),
        teacher_actions=torch.zeros(8, 12),
        student_actions=torch.zeros(8, 12),
        executed_actions=torch.zeros(8, 12),
        base_velocity_target=torch.zeros(8, 3),
        next_sensor_frame_target=torch.zeros(8, 36),
        terminal=torch.zeros(8, dtype=torch.bool),
    )
    storage.add(batch)

    minibatches = list(storage.mini_batches(3, shuffle=False))

    assert [part.sensor_history.shape[0] for part in minibatches] == [3, 3, 2]
    recovered = torch.cat([part.sensor_history[:, 0, 0] for part in minibatches])
    torch.testing.assert_close(recovered, torch.arange(8, dtype=torch.float32))
    capped = list(storage.mini_batches(1, max_batch_size=3, shuffle=False))
    assert [part.sensor_history.shape[0] for part in capped] == [3, 3, 2]
    with pytest.raises(ValueError, match="positive"):
        list(storage.mini_batches(0))
    with pytest.raises(ValueError, match="positive"):
        list(storage.mini_batches(1, max_batch_size=0))


def test_f4_has_an_explicit_ppo_to_ppo_bootstrap_intent() -> None:
    checkpoint.validate_checkpoint_intent_v2(
        checkpoint.CheckpointKindV2.PPO,
        checkpoint.CheckpointIntentV2.ROBUSTNESS_BOOTSTRAP,
    )
    with pytest.raises(ValueError, match="illegal"):
        checkpoint.validate_checkpoint_intent_v2(
            checkpoint.CheckpointKindV2.DISTILLED,
            checkpoint.CheckpointIntentV2.ROBUSTNESS_BOOTSTRAP,
        )
    for kind in checkpoint.CheckpointKindV2:
        checkpoint.validate_checkpoint_intent_v2(
            kind,
            checkpoint.CheckpointIntentV2.INFERENCE,
        )
