from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _module(monkeypatch):
    class JointState:
        pass

    class RedRHexMotorState:
        pass

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = JointState
    sensor_msgs.msg = sensor_msgs_msg
    redrhex_msgs = types.ModuleType("redrhex_msgs")
    redrhex_msgs_msg = types.ModuleType("redrhex_msgs.msg")
    redrhex_msgs_msg.RedRhexMotorState = RedRHexMotorState
    redrhex_msgs.msg = redrhex_msgs_msg
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)
    monkeypatch.setitem(sys.modules, "redrhex_msgs", redrhex_msgs)
    monkeypatch.setitem(sys.modules, "redrhex_msgs.msg", redrhex_msgs_msg)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    for name in list(sys.modules):
        if name == "redrhex_lowlevel_bridge" or name.startswith("redrhex_lowlevel_bridge."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("redrhex_lowlevel_bridge.rinbo_ros_backend")


def _kwargs():
    return {
        "node": SimpleNamespace(),
        "command_topic": "/motor/command",
        "state_topic": "/motor/state",
        "joint_state_topic": "/joint_states",
        "preview_topic": "/preview",
        "publish_preview": True,
        "allow_enable": False,
        "publish_when_disabled": False,
        "disabled_servo_control_mode": 0,
        "probe_abad_disable_verified": False,
        "publish_shutdown_disable": True,
        "shutdown_disable_repeats": 2,
        "shutdown_disable_period_s": 0.0,
        "require_state": True,
        "block_if_duplicate_command_publishers": True,
        "state_timeout_s": 0.25,
        "main_position_counts_per_rev": 1000.0,
        "main_pwm_per_rad_s": 120.0,
        "main_max_pwm": 500.0,
        "main_encoder_zero_counts_rinbo_order": [0.0] * 6,
        "main_encoder_sign_rinbo_order": [1.0] * 6,
        "main_velocity_sign_policy_order": [1.0] * 6,
        "main_direction_positive_rinbo_order": [True] * 6,
        "main_velocity_filter_alpha": 1.0,
        "main_velocity_max_dt_s": 0.2,
        "main_velocity_clip_rad_s": 80.0,
        "abad_encoder_zero_rinbo_order": [100, 200, 300, 400, 500, 600],
        "abad_encoder_counts_per_rad": 100.0,
        "abad_encoder_min": 0,
        "abad_encoder_max": 1000,
        "abad_sign_rinbo_order": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        "servo_control_mode": 2,
        "main_joint_names_policy_order": [f"main_{i}" for i in range(6)],
        "abad_joint_names_policy_order": [f"abad_{i}" for i in range(6)],
        "publish_abad_joint_states": True,
        "main_encoder_calibration_verified": True,
        "abad_encoder_calibration_verified": True,
        "abad_velocity_filter_alpha": 1.0,
    }


def test_abad_conversion_uses_rf_rm_rr_lf_lm_lr_policy_order(monkeypatch):
    module = _module(monkeypatch)
    backend = module.RinboRosBackend(**_kwargs())
    # Rinbo order sl1/sl2/sl3/sr1/sr2/sr3 maps to policy LF/LM/LR/RF/RM/RR.
    raw = [110, 180, 330, 360, 550, 540]
    positions, valid = backend._convert_abad_feedback_to_policy(raw)
    assert positions == pytest.approx([0.4, 0.5, 0.6, 0.1, 0.2, 0.3])
    assert valid == [True] * 6


def test_abad_validity_stays_false_until_calibration_is_verified(monkeypatch):
    module = _module(monkeypatch)
    kwargs = _kwargs()
    kwargs["abad_encoder_calibration_verified"] = False
    backend = module.RinboRosBackend(**kwargs)
    positions, valid = backend._convert_abad_feedback_to_policy([100, 200, 300, 400, 500, 600])
    assert positions == pytest.approx([0.0] * 6)
    assert valid == [False] * 6


def test_abad_causal_velocity_rejects_repeated_or_large_dt(monkeypatch):
    module = _module(monkeypatch)
    backend = module.RinboRosBackend(**_kwargs())
    assert backend._estimate_abad_policy_velocities([0.0] * 6, 1.0) == [0.0] * 6
    assert backend._estimate_abad_policy_velocities([1.0] * 6, 1.0) == [0.0] * 6
    assert backend._estimate_abad_policy_velocities([2.0] * 6, 2.0) == [0.0] * 6


def test_v1_default_does_not_require_abad_joint_names(monkeypatch):
    module = _module(monkeypatch)
    kwargs = _kwargs()
    kwargs.pop("abad_joint_names_policy_order")
    kwargs.pop("publish_abad_joint_states")
    backend = module.RinboRosBackend(**kwargs)
    assert backend.publish_abad_joint_states is False
    assert backend.abad_joint_names_policy_order == []
