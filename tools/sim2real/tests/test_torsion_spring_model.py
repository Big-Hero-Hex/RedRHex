from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[3]
MODULE_PATH = (
    ROOT
    / "source/RedRhex/RedRhex/tasks/direct/redrhex/torsion_spring.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("redrhex_torsion_spring", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restoring_torque_is_zero_at_neutral_and_opposes_unwrapped_deflection() -> None:
    spring = _load_module()
    rest = torch.tensor([[0.25, -0.75]])
    position = torch.tensor([[0.25, -0.75 + 2.0 * math.pi]])
    velocity = torch.zeros_like(position)
    stiffness = torch.tensor([[10.0, 3.0]])

    torque = spring.restoring_torque(
        position=position,
        velocity=velocity,
        rest_position=rest,
        stiffness=stiffness,
        damping=torch.zeros_like(stiffness),
    )

    assert torque[0, 0].item() == pytest.approx(0.0)
    assert torque[0, 1].item() == pytest.approx(-3.0 * 2.0 * math.pi)


def test_zero_damping_ignores_velocity_and_energy_matches_linear_spring() -> None:
    spring = _load_module()
    rest = torch.tensor([[0.5, -0.5]])
    position = torch.tensor([[0.7, -0.9]])
    velocity = torch.tensor([[100.0, -100.0]])
    stiffness = torch.tensor([[20.0, 5.0]])
    damping = torch.zeros_like(stiffness)

    torque = spring.restoring_torque(position, velocity, rest, stiffness, damping)
    energy = spring.potential_energy(position, rest, stiffness)
    power = spring.mechanical_power(torque, velocity)

    torch.testing.assert_close(torque, torch.tensor([[-4.0, 2.0]]))
    torch.testing.assert_close(energy, torch.tensor([[0.4, 0.4]]))
    torch.testing.assert_close(power, torque * velocity)


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("explicit", (0.0, 0.0)), ("native", (200.0, 0.0))],
)
def test_backend_actuator_gains_select_only_native_physics_drive(
    backend: str, expected: tuple[float, float]
) -> None:
    spring = _load_module()

    assert spring.actuator_gains(backend, stiffness=200.0, damping=0.0) == expected


def test_backend_validation_rejects_unknown_value() -> None:
    spring = _load_module()

    with pytest.raises(ValueError, match="spring backend"):
        spring.actuator_gains("servo", stiffness=200.0, damping=0.0)


def test_continuous_position_unwraps_across_both_pi_boundaries() -> None:
    spring = _load_module()
    previous_wrapped = torch.tensor([[3.1, -3.1]])
    previous_unwrapped = previous_wrapped.clone()
    current_wrapped = torch.tensor([[-3.1, 3.1]])

    unwrapped = spring.unwrap_continuous_position(
        current_wrapped, previous_wrapped, previous_unwrapped
    )

    torch.testing.assert_close(
        unwrapped,
        torch.tensor(
            [[3.1 + (2.0 * math.pi - 6.2), -3.1 - (2.0 * math.pi - 6.2)]]
        ),
    )


def test_continuous_position_uses_velocity_prediction_for_multiple_turn_steps() -> None:
    spring = _load_module()
    previous_wrapped = torch.tensor([[0.0]])
    previous_unwrapped = torch.tensor([[0.0]])
    current_wrapped = torch.tensor([[-2.5]])

    unwrapped = spring.unwrap_continuous_position(
        current_wrapped,
        previous_wrapped,
        previous_unwrapped,
        predicted_delta=torch.tensor([[3.8]]),
    )

    assert unwrapped.item() == pytest.approx(2.0 * math.pi - 2.5)
    assert spring.unwrap_ambiguity_mask(torch.tensor([[400.0]]), 1.0 / 120.0).item()
    assert not spring.unwrap_ambiguity_mask(torch.tensor([[10.0]]), 1.0 / 120.0).item()


def test_temporal_passivity_residual_balances_energy_work_and_dissipation() -> None:
    spring = _load_module()
    previous_power = torch.tensor([0.0, 2.0])
    current_power = torch.tensor([4.0, 6.0])

    work = spring.trapezoidal_energy_increment(
        previous_power, current_power, 0.5
    )
    residual = spring.energy_work_residual(
        current_energy=torch.tensor([8.0, 3.0]),
        reference_energy=torch.tensor([10.0, 10.0]),
        cumulative_work=work,
        cumulative_dissipation=torch.tensor([1.0, 3.0]),
    )

    torch.testing.assert_close(work, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(residual, torch.tensor([0.0, -2.0]))


def test_parameters_reorder_from_canonical_to_actuator_joint_order() -> None:
    spring = _load_module()
    canonical_names = ["j5", "j8", "j13", "j25", "j26", "j27"]
    actuator_names = ["j5", "j13", "j25", "j26", "j27", "j8"]
    values = torch.tensor([[5.0, 8.0, 13.0, 25.0, 26.0, 27.0]])

    reordered = spring.reorder_joint_parameters(
        values, canonical_names, actuator_names
    )

    torch.testing.assert_close(
        reordered, torch.tensor([[5.0, 13.0, 25.0, 26.0, 27.0, 8.0]])
    )
    with pytest.raises(ValueError, match="same unique joints"):
        spring.reorder_joint_parameters(values, canonical_names, actuator_names[:-1])
