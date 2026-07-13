from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[3]
MODULE = (
    ROOT
    / "source/RedRhex/RedRhex/tasks/direct/redrhex/target_delay.py"
)


def _target_delay_module():
    spec = importlib.util.spec_from_file_location("redrhex_target_delay_under_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_delay_emits_exactly_n_initial_targets_before_first_request() -> None:
    module = _target_delay_module()
    drive_history = torch.zeros((1, 3, 2))
    abad_history = torch.full((1, 3, 2), 0.5)
    cursor = 0
    applied_drive = []
    applied_abad = []

    for value in (1.0, 2.0, 3.0, 4.0):
        drive = torch.full((1, 2), value)
        abad = torch.full((1, 2), value + 10.0)
        drive_out, abad_out, cursor = module.advance_target_delay(
            drive_history, abad_history, cursor, drive, abad
        )
        applied_drive.append(float(drive_out[0, 0]))
        applied_abad.append(float(abad_out[0, 0]))

    assert applied_drive == [0.0, 0.0, 0.0, 1.0]
    assert applied_abad == [0.5, 0.5, 0.5, 11.0]
    assert cursor == 1


def test_target_delay_copies_requests_instead_of_aliasing_caller_storage() -> None:
    module = _target_delay_module()
    drive_history = torch.zeros((1, 1, 1))
    abad_history = torch.zeros((1, 1, 1))
    drive = torch.tensor([[2.0]])
    abad = torch.tensor([[3.0]])

    module.advance_target_delay(drive_history, abad_history, 0, drive, abad)
    drive[:] = 99.0
    abad[:] = 99.0

    assert float(drive_history[0, 0, 0]) == 2.0
    assert float(abad_history[0, 0, 0]) == 3.0
