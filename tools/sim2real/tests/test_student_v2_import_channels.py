from __future__ import annotations

from types import SimpleNamespace

from tools.sim2real.import_real import (
    _canonical_abad_encoder_counts,
    _twist_command,
)


def test_abad_encoder_import_uses_canonical_policy_order() -> None:
    message = SimpleNamespace(
        sl1=SimpleNamespace(position_encoder=11),
        sl2=SimpleNamespace(position_encoder=12),
        sl3=SimpleNamespace(position_encoder=13),
        sr1=SimpleNamespace(position_encoder=21),
        sr2=SimpleNamespace(position_encoder=22),
        sr3=SimpleNamespace(position_encoder=23),
    )

    assert _canonical_abad_encoder_counts(message) == [21.0, 22.0, 23.0, 11.0, 12.0, 13.0]


def test_cmd_vel_import_keeps_only_external_policy_command() -> None:
    message = SimpleNamespace(
        linear=SimpleNamespace(x=0.4, y=-0.2, z=99.0),
        angular=SimpleNamespace(x=88.0, y=77.0, z=0.3),
    )

    assert _twist_command(message) == [0.4, -0.2, 0.3]
