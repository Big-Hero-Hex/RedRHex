from __future__ import annotations

import numpy as np
import pytest

from redrhex_policy_io.contracts import ForwardResidualActionContractV2
from redrhex_rl_controller.action_decoder_v2 import ForwardResidualActionDecoderV2


def test_decoder_forces_policy_abad_actions_to_contract_neutral():
    contract = ForwardResidualActionContractV2()
    decoder = ForwardResidualActionDecoderV2(contract)
    action = np.concatenate((np.full(6, 0.25), np.full(6, 1.0)))
    result = decoder.decode(
        action,
        np.zeros(6),
        np.array([0.45, 0.0, 0.0]),
        1.0 / 60.0,
    )
    assert result.safe_action[6:12] == pytest.approx([0.0] * 6)
    assert result.target_abad_position == pytest.approx(contract.abad_neutral_position_rad)
    assert result.joint_names == list(contract.MAIN_JOINT_ORDER + contract.ABAD_JOINT_ORDER)


def test_hardware_limits_can_tighten_but_never_loosen_bundle_semantics():
    contract = ForwardResidualActionContractV2(main_velocity_limit_rad_s=5.0, action_clip=1.0)
    decoder = ForwardResidualActionDecoderV2(
        contract,
        {"main_drive_vel_limit_rad_s": 4.0, "action_clip": 0.8},
    )
    assert decoder.main_velocity_limit == 4.0
    assert decoder.action_clip == 0.8
    with pytest.raises(ValueError, match="only tighten"):
        ForwardResidualActionDecoderV2(contract, {"main_drive_vel_limit_rad_s": 6.0})
    with pytest.raises(ValueError, match="only tighten"):
        ForwardResidualActionDecoderV2(contract, {"action_clip": 1.1})


def test_decoder_rejects_non_forward_commands_and_slew_limits_output():
    contract = ForwardResidualActionContractV2(main_velocity_limit_rad_s=9.0)
    decoder = ForwardResidualActionDecoderV2(
        contract,
        {"main_drive_slew_rate_rad_s2": 6.0},
    )
    with pytest.raises(ValueError, match="lateral and yaw"):
        decoder.decode(np.zeros(12), np.zeros(6), np.array([0.2, 0.2, 0.0]), 1.0 / 60.0)
    result = decoder.decode(
        np.ones(12),
        np.zeros(6),
        np.array([0.45, 0.0, 0.0]),
        1.0 / 60.0,
    )
    assert np.max(np.abs(result.target_main_drive_velocity)) <= 0.100001


def test_decoder_hash_is_the_loaded_bundle_contract_hash():
    contract = ForwardResidualActionContractV2()
    assert ForwardResidualActionDecoderV2(contract).decoder_sha256 == contract.sha256
