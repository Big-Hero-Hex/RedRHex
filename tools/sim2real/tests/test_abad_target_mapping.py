from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[3]
MODULE = ROOT / "source/RedRhex/RedRhex/tasks/direct/redrhex/abad_target_mapping.py"


def _mapping_module():
    spec = importlib.util.spec_from_file_location("redrhex_abad_mapping_under_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_abad_target_mapping_applies_measured_relation_and_identity_default() -> None:
    module = _mapping_module()
    requested = torch.tensor([[0.1, -0.2]])

    mapped = module.map_abad_targets(
        requested,
        scale=torch.tensor([[1.2, 0.8]]),
        offset=torch.tensor([[-0.03, 0.04]]),
        lower=torch.full_like(requested, -0.5),
        upper=torch.full_like(requested, 0.5),
    )

    torch.testing.assert_close(mapped, torch.tensor([[0.09, -0.12]]))
    torch.testing.assert_close(
        module.map_abad_targets(
            requested,
            scale=torch.ones_like(requested),
            offset=torch.zeros_like(requested),
            lower=torch.full_like(requested, -0.5),
            upper=torch.full_like(requested, 0.5),
        ),
        requested,
    )


def test_abad_target_mapping_clamps_after_measured_correction() -> None:
    module = _mapping_module()
    requested = torch.tensor([[0.45, -0.45]])

    mapped = module.map_abad_targets(
        requested,
        scale=torch.tensor([[1.5, 1.5]]),
        offset=torch.tensor([[0.2, -0.2]]),
        lower=torch.tensor([[-0.5, -0.5]]),
        upper=torch.tensor([[0.5, 0.5]]),
    )

    torch.testing.assert_close(mapped, torch.tensor([[0.5, -0.5]]))


def test_abad_target_mapping_rejects_shape_or_nonpositive_scale() -> None:
    module = _mapping_module()
    requested = torch.tensor([[0.1, -0.2]])

    with pytest.raises(ValueError, match="shape"):
        module.map_abad_targets(
            requested,
            scale=torch.ones((1, 1)),
            offset=torch.zeros_like(requested),
            lower=torch.full_like(requested, -0.5),
            upper=torch.full_like(requested, 0.5),
        )
    with pytest.raises(ValueError, match="positive"):
        module.map_abad_targets(
            requested,
            scale=torch.tensor([[1.0, 0.0]]),
            offset=torch.zeros_like(requested),
            lower=torch.full_like(requested, -0.5),
            upper=torch.full_like(requested, 0.5),
        )
    with pytest.raises(ValueError, match="bounds"):
        module.map_abad_targets(
            requested,
            scale=torch.ones_like(requested),
            offset=torch.zeros_like(requested),
            lower=torch.tensor([[0.0, 0.0]]),
            upper=torch.tensor([[0.0, 0.5]]),
        )
