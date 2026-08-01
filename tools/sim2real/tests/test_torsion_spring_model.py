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
