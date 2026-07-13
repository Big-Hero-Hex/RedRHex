from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
LOWLEVEL_ROOT = REPO_ROOT / "ros2_ws/src/redrhex_lowlevel_bridge"
LOWLEVEL_PACKAGE = LOWLEVEL_ROOT / "redrhex_lowlevel_bridge"
CONTROLLER_ROOT = REPO_ROOT / "ros2_ws/src/redrhex_rl_controller"
CONTROLLER_PACKAGE = CONTROLLER_ROOT / "redrhex_rl_controller"


def _load_standalone(path: Path, name: str):
    assert path.is_file(), f"missing ROS-independent safety module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_ros_message_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    class Header:
        def __init__(self) -> None:
            self.seq = 0
            self.stamp = None
            self.frame_id = ""

    class RedRhexMotorState:
        def __init__(self) -> None:
            self.header = Header()
            self.joint_names = []
            self.position_rad = []
            self.velocity_rad_s = []
            self.effort_nm = []
            self.current_a = []
            self.temperature_c = []
            self.fault = []

    class JointState:
        def __init__(self) -> None:
            self.header = Header()
            self.name = []
            self.position = []
            self.velocity = []

    redrhex_msgs = types.ModuleType("redrhex_msgs")
    redrhex_msgs_msg = types.ModuleType("redrhex_msgs.msg")
    redrhex_msgs_msg.RedRhexMotorState = RedRhexMotorState
    redrhex_msgs.msg = redrhex_msgs_msg
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = JointState
    sensor_msgs.msg = sensor_msgs_msg
    monkeypatch.setitem(sys.modules, "redrhex_msgs", redrhex_msgs)
    monkeypatch.setitem(sys.modules, "redrhex_msgs.msg", redrhex_msgs_msg)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)


def _import_lowlevel(monkeypatch: pytest.MonkeyPatch, module: str):
    _install_ros_message_stubs(monkeypatch)
    monkeypatch.syspath_prepend(str(LOWLEVEL_ROOT))
    for name in list(sys.modules):
        if name == "redrhex_lowlevel_bridge" or name.startswith("redrhex_lowlevel_bridge."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(f"redrhex_lowlevel_bridge.{module}")


def _command(
    *,
    enable: bool,
    main_drive_enable=(False, False, False, False, False, False),
    abad_output_enable: bool = False,
    sim2real_probe: bool = False,
):
    names = [f"joint_{index}" for index in range(12)]
    return SimpleNamespace(
        enable=enable,
        mode=2,
        joint_names=names,
        target_position_rad=[0.0] * 12,
        target_velocity_rad_s=[0.0] * 12,
        kp=[0.0] * 12,
        kd=[0.0] * 12,
        effort_limit_nm=[1.0] * 12,
        main_drive_enable=list(main_drive_enable),
        abad_output_enable=abad_output_enable,
        sim2real_probe=sim2real_probe,
    )


def _command_safety():
    return _load_standalone(LOWLEVEL_PACKAGE / "command_safety.py", "redrhex_command_safety_under_test")


def _manual_safety():
    return _load_standalone(
        CONTROLLER_PACKAGE / "manual_command_safety.py",
        "redrhex_manual_command_safety_under_test",
    )


def test_motor_command_message_declares_fixed_output_masks():
    text = (REPO_ROOT / "ros2_ws/src/redrhex_msgs/msg/RedRhexMotorCommand.msg").read_text(encoding="utf-8")
    assert "bool[6] main_drive_enable" in text
    assert "bool abad_output_enable" in text
    assert "bool sim2real_probe" in text


def test_disabled_command_overrides_even_malformed_masks():
    safety = _command_safety()
    selection = safety.resolve_output_selection(
        SimpleNamespace(enable=False, main_drive_enable=[True], abad_output_enable="invalid")
    )
    assert selection.main_drive_enable == (False,) * 6
    assert selection.abad_output_enable is False
    assert selection.any_enabled is False


def test_bridge_owned_probe_lease_blocks_concurrent_normal_output() -> None:
    safety = _command_safety()
    now = [0.0]
    gate = safety.ProbeSessionGate(timeout_s=0.25, clock=lambda: now[0])
    gate.authorize(_command(enable=False, sim2real_probe=True), output_active=False)

    assert gate.active is True
    with pytest.raises(safety.CommandRejectedError, match="probe session"):
        gate.authorize(
            _command(enable=True, main_drive_enable=[True] * 6),
            output_active=False,
        )
    assert gate.conflict_latched is True


def test_probe_lease_validates_single_drive_and_never_abad() -> None:
    safety = _command_safety()
    gate = safety.ProbeSessionGate(timeout_s=0.25)

    with pytest.raises(safety.CommandRejectedError, match="exactly one main drive"):
        gate.authorize(
            _command(
                enable=True,
                main_drive_enable=[True, True, False, False, False, False],
                sim2real_probe=True,
            ),
            output_active=False,
        )
    assert gate.conflict_latched is True

    gate = safety.ProbeSessionGate(timeout_s=0.25)
    with pytest.raises(safety.CommandRejectedError, match="ABAD"):
        gate.authorize(
            _command(
                enable=True,
                main_drive_enable=[True, False, False, False, False, False],
                abad_output_enable=True,
                sim2real_probe=True,
            ),
            output_active=False,
        )


def test_probe_lease_cleanly_expires_only_after_output_is_disabled() -> None:
    safety = _command_safety()
    now = [0.0]
    gate = safety.ProbeSessionGate(timeout_s=0.25, clock=lambda: now[0])
    gate.authorize(_command(enable=False, sim2real_probe=True), output_active=False)
    now[0] = 0.3

    assert gate.poll(output_active=False) is False
    assert gate.active is False
    gate.authorize(
        _command(enable=True, main_drive_enable=[True] * 6), output_active=False
    )


def test_probe_lease_expiry_with_active_output_requests_disable_and_latches() -> None:
    safety = _command_safety()
    now = [0.0]
    gate = safety.ProbeSessionGate(timeout_s=0.25, clock=lambda: now[0])
    gate.authorize(
        _command(
            enable=True,
            main_drive_enable=[True, False, False, False, False, False],
            sim2real_probe=True,
        ),
        output_active=False,
    )
    now[0] = 0.3

    assert gate.poll(output_active=True) is True
    assert gate.active is False
    assert gate.conflict_latched is True


def test_probe_backend_failure_latches_session_until_bridge_restart() -> None:
    safety = _command_safety()
    gate = safety.ProbeSessionGate(timeout_s=0.25)
    gate.latch_failure("hardware interlock rejected the probe")

    assert gate.conflict_latched is True
    with pytest.raises(safety.CommandRejectedError, match="hardware interlock"):
        gate.authorize(
            _command(
                enable=True,
                main_drive_enable=[True, False, False, False, False, False],
                sim2real_probe=True,
            ),
            output_active=False,
        )


def test_lowlevel_node_enforces_and_polls_bridge_owned_probe_lease() -> None:
    source = (LOWLEVEL_PACKAGE / "lowlevel_bridge_node.py").read_text(
        encoding="utf-8"
    )
    assert "self.probe_session_gate.authorize(" in source
    assert "self.probe_session_gate.poll(" in source
    assert "self.probe_session_gate.latch_failure(" in source
    assert "sim2real_probe.session_timeout_s" in source


@pytest.mark.parametrize(
    "mask",
    ([True] * 5, [True] * 7, [True, False, True, False, True, 1]),
)
def test_enabled_command_rejects_malformed_main_masks(mask):
    safety = _command_safety()
    with pytest.raises(safety.CommandSelectionError):
        safety.resolve_output_selection(
            SimpleNamespace(enable=True, main_drive_enable=mask, abad_output_enable=False)
        )


def test_enabled_command_resolves_explicit_outputs():
    safety = _command_safety()
    selection = safety.resolve_output_selection(
        _command(enable=True, main_drive_enable=[False, True, False, False, False, False], abad_output_enable=True)
    )
    assert selection.main_drive_enable == (False, True, False, False, False, False)
    assert selection.abad_output_enable is True
    assert selection.any_enabled is True


def test_enabled_command_accepts_ros_generated_numpy_boolean_array():
    safety = _command_safety()
    selection = safety.resolve_output_selection(
        SimpleNamespace(
            enable=True,
            main_drive_enable=np.array([False, True, False, False, False, False], dtype=np.bool_),
            abad_output_enable=np.bool_(True),
        )
    )
    assert selection.main_drive_enable == (False, True, False, False, False, False)
    assert selection.abad_output_enable is True


def test_enabled_command_rejects_non_vector_numpy_mask():
    safety = _command_safety()
    with pytest.raises(safety.CommandSelectionError, match="exactly 6"):
        safety.resolve_output_selection(
            SimpleNamespace(
                enable=True,
                main_drive_enable=np.zeros((6, 1), dtype=np.bool_),
                abad_output_enable=False,
            )
        )


def test_gate_starts_fail_closed_and_requires_explicit_estop_clear():
    safety = _command_safety()
    gate = safety.FailClosedOutputGate()
    assert gate.ready_for_output is False
    with pytest.raises(safety.CommandRejectedError, match="E-stop"):
        gate.accept_command(_command(enable=True, main_drive_enable=[True] + [False] * 5), state_fresh=True)
    assert gate.on_estop(False) is False
    assert gate.ready_for_output is True
    selection = gate.accept_command(
        _command(enable=True, main_drive_enable=[True] + [False] * 5), state_fresh=True
    )
    assert selection.any_enabled is True
    assert gate.output_active is False
    gate.mark_command_sent(selection)
    assert gate.output_active is True


def test_gate_requests_immediate_disable_for_estop_and_stale_state_once():
    safety = _command_safety()
    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)
    selection = gate.accept_command(_command(enable=True, abad_output_enable=True), state_fresh=True)
    gate.mark_command_sent(selection)
    assert gate.on_state_freshness(False) is True
    assert gate.output_active is True
    gate.mark_disabled()
    assert gate.output_active is False
    assert gate.on_state_freshness(False) is False
    selection = gate.accept_command(_command(enable=True, abad_output_enable=True), state_fresh=True)
    gate.mark_command_sent(selection)
    assert gate.on_estop(True) is True
    assert gate.ready_for_output is False
    assert gate.output_active is True
    gate.mark_disabled()
    assert gate.output_active is False


def test_failed_emergency_disable_stays_latched_after_trigger_recovers():
    safety = _command_safety()
    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)
    selection = gate.accept_command(
        _command(enable=True, main_drive_enable=[True] + [False] * 5),
        state_fresh=True,
    )
    gate.mark_command_sent(selection)

    assert gate.on_estop(True) is True
    assert gate.disable_pending is True
    gate.on_estop(False)
    assert gate.ready_for_output is False
    with pytest.raises(safety.CommandRejectedError, match="disable"):
        gate.accept_command(
            _command(enable=True, main_drive_enable=[True] + [False] * 5),
            state_fresh=True,
        )

    # Fresh feedback must not clear uncertainty about the failed disable.
    assert gate.on_state_freshness(True) is True
    gate.mark_disabled()
    assert gate.disable_pending is False
    assert gate.ready_for_output is True


def test_stale_state_disable_stays_latched_after_feedback_recovers():
    safety = _command_safety()
    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)
    selection = gate.accept_command(
        _command(enable=True, abad_output_enable=True), state_fresh=True
    )
    gate.mark_command_sent(selection)

    assert gate.on_state_freshness(False) is True
    assert gate.disable_pending is True
    assert gate.on_state_freshness(True) is True
    assert gate.ready_for_output is False
    with pytest.raises(safety.CommandRejectedError, match="disable"):
        gate.accept_command(
            _command(enable=True, abad_output_enable=True), state_fresh=True
        )


def test_enabled_command_rejects_nonfinite_numeric_payloads():
    safety = _command_safety()
    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)

    for field, value in (
        ("target_position_rad", float("nan")),
        ("target_velocity_rad_s", float("inf")),
        ("kp", float("-inf")),
        ("kd", float("nan")),
        ("effort_limit_nm", float("inf")),
    ):
        command = _command(enable=True, main_drive_enable=[True] + [False] * 5)
        getattr(command, field)[0] = value
        with pytest.raises(safety.CommandSelectionError, match=field):
            gate.accept_command(command, state_fresh=True)


def test_enabled_command_rejects_wrong_numeric_array_shapes():
    safety = _command_safety()
    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)
    command = _command(enable=True, main_drive_enable=[True] + [False] * 5)
    command.target_velocity_rad_s = command.target_velocity_rad_s[:6]

    with pytest.raises(safety.CommandSelectionError, match="target_velocity_rad_s"):
        gate.accept_command(command, state_fresh=True)


def test_failed_disabled_transition_preserves_active_output_for_emergency_disable():
    safety = _command_safety()
    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)
    enabled = gate.accept_command(
        _command(enable=True, main_drive_enable=[True] + [False] * 5),
        state_fresh=True,
    )
    gate.mark_command_sent(enabled)
    disabled = gate.accept_command(
        SimpleNamespace(enable=False, main_drive_enable=[True], abad_output_enable=True),
        state_fresh=False,
    )
    assert disabled.any_enabled is False
    assert gate.output_active is True
    assert gate.on_command_failure(command_enabled=False) is True


def test_dispatcher_emergency_disables_when_malformed_disabled_send_fails():
    safety = _command_safety()

    class Backend:
        def __init__(self) -> None:
            self.emergency_disable_calls = 0

        @staticmethod
        def output_state_is_fresh() -> bool:
            return True

        @staticmethod
        def send_motor_command(command) -> None:
            if not command.enable:
                raise ValueError("malformed disabled command arrays")

        def emergency_disable(self) -> None:
            self.emergency_disable_calls += 1

    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)
    backend = Backend()
    safety.dispatch_command_fail_closed(
        gate,
        backend,
        _command(enable=True, main_drive_enable=[True] + [False] * 5),
    )
    assert gate.output_active is True

    with pytest.raises(ValueError, match="malformed disabled"):
        safety.dispatch_command_fail_closed(
            gate,
            backend,
            _command(enable=False),
        )
    assert backend.emergency_disable_calls == 1
    assert gate.output_active is False


def test_dispatcher_latches_failed_emergency_disable_until_retry_succeeds():
    safety = _command_safety()

    class Backend:
        def __init__(self) -> None:
            self.disable_should_fail = True

        @staticmethod
        def output_state_is_fresh() -> bool:
            return True

        @staticmethod
        def send_motor_command(_command) -> None:
            raise RuntimeError("enabled send failed")

        def emergency_disable(self) -> None:
            if self.disable_should_fail:
                raise RuntimeError("disable failed")

    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)
    backend = Backend()
    with pytest.raises(RuntimeError, match="disable failed"):
        safety.dispatch_command_fail_closed(
            gate,
            backend,
            _command(enable=True, main_drive_enable=[True] + [False] * 5),
        )

    assert gate.disable_pending is True
    assert gate.output_active is True
    assert gate.ready_for_output is False
    backend.disable_should_fail = False
    backend.emergency_disable()
    gate.mark_disabled()
    assert gate.disable_pending is False
    assert gate.output_active is False


def test_backend_interface_requires_emergency_disable_and_state_freshness(monkeypatch):
    module = _import_lowlevel(monkeypatch, "bridge_base")
    abstract_methods = module.LowLevelBridgeBase.__abstractmethods__
    assert "emergency_disable" in abstract_methods
    assert "output_state_is_fresh" in abstract_methods


@pytest.mark.parametrize(
    ("module_name", "class_name", "kwargs"),
    [
        ("serial_bridge", "SerialLowLevelBridge", {"port": "/dev/null", "allow_enable": True}),
        ("sbrio_udp_bridge", "SbrioUdpBridge", {"allow_enable": True}),
    ],
)
def test_provisional_packet_backends_reject_every_enabled_command(
    monkeypatch, module_name, class_name, kwargs
):
    module = _import_lowlevel(monkeypatch, module_name)
    backend = getattr(module, class_name)(**kwargs)
    with pytest.raises(RuntimeError, match="cannot represent output masks"):
        backend._encode_command(
            _command(enable=True, main_drive_enable=[True, False, False, False, False, False])
        )


class _Header:
    def __init__(self) -> None:
        self.seq = 0
        self.stamp = None
        self.frame_id = ""


class _Leg:
    def __init__(self) -> None:
        self.enable = False
        self.direction = False
        self.voltage = 0.0
        self.state = 0
        self.reset_position = False


class _Servo:
    def __init__(self) -> None:
        self.position_encoder = 0


class _MotorCmd:
    def __init__(self) -> None:
        self.header = _Header()
        self.servo_control_mode = 0
        for field in ("l1", "l2", "l3", "r1", "r2", "r3"):
            setattr(self, field, _Leg())
        for field in ("sl1", "sl2", "sl3", "sr1", "sr2", "sr3"):
            setattr(self, field, _Servo())


def _rinbo_init_kwargs() -> dict:
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
        "shutdown_disable_repeats": 5,
        "shutdown_disable_period_s": 0.02,
        "require_state": True,
        "block_if_duplicate_command_publishers": True,
        "state_timeout_s": 0.25,
        "main_position_counts_per_rev": 54984.83,
        "main_pwm_per_rad_s": 120.0,
        "main_max_pwm": 500.0,
        "main_encoder_zero_counts_rinbo_order": [0.0] * 6,
        "main_encoder_sign_rinbo_order": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        "main_velocity_sign_policy_order": [1.0] * 6,
        "main_direction_positive_rinbo_order": [True, True, True, False, False, False],
        "main_velocity_filter_alpha": 0.35,
        "main_velocity_max_dt_s": 0.2,
        "main_velocity_clip_rad_s": 80.0,
        "abad_encoder_zero_rinbo_order": [740, 2565, 3283, 1944, 2071, 989],
        "abad_encoder_counts_per_rad": 1000.0,
        "abad_encoder_min": 0,
        "abad_encoder_max": 65535,
        "abad_sign_rinbo_order": [1.0] * 6,
        "servo_control_mode": 2,
        "main_joint_names_policy_order": [f"joint_{index}" for index in range(6)],
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("state_timeout_s", float("nan")),
        ("state_timeout_s", float("inf")),
        ("main_position_counts_per_rev", float("nan")),
        ("main_position_counts_per_rev", float("inf")),
        ("main_pwm_per_rad_s", float("nan")),
        ("main_pwm_per_rad_s", float("inf")),
        ("main_max_pwm", float("nan")),
        ("main_max_pwm", float("inf")),
        ("main_velocity_max_dt_s", float("nan")),
        ("main_velocity_clip_rad_s", float("inf")),
        ("abad_encoder_counts_per_rad", float("nan")),
        ("shutdown_disable_period_s", float("inf")),
        ("main_encoder_zero_counts_rinbo_order", [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]),
        ("main_encoder_sign_rinbo_order", [-1.0, -1.0, 0.0, 1.0, 1.0, 1.0]),
        ("main_velocity_sign_policy_order", [1.0, 1.0, float("inf"), 1.0, 1.0, 1.0]),
        ("main_direction_positive_rinbo_order", [True, True, 1, False, False, False]),
        ("abad_sign_rinbo_order", [1.0, 1.0, float("nan"), 1.0, 1.0, 1.0]),
    ],
)
def test_biorola_constructor_rejects_nonfinite_or_invalid_conversion_parameters(
    monkeypatch, field, bad_value
):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    kwargs = _rinbo_init_kwargs()
    kwargs[field] = bad_value

    with pytest.raises(ValueError, match=field):
        module.RinboRosBackend(**kwargs)


def test_biorola_constructor_accepts_finite_reviewed_conversion_parameters(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = module.RinboRosBackend(**_rinbo_init_kwargs())

    assert backend.main_pwm_per_rad_s == 120.0
    assert backend.main_max_pwm == 500.0
    assert backend.main_encoder_sign_rinbo_order == [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    assert backend.publisher_conflict_latched is False
    assert backend.probe_abad_disable_verified is False


def test_biorola_connect_rejects_remapped_preview_alias_before_creating_publishers(
    monkeypatch,
):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    rinbo_msgs = types.ModuleType("rinbo_msgs")
    rinbo_msgs_msg = types.ModuleType("rinbo_msgs.msg")
    rinbo_msgs_msg.MotorCmdStamped = _MotorCmd
    rinbo_msgs_msg.MotorStateStamped = object
    rinbo_msgs.msg = rinbo_msgs_msg
    monkeypatch.setitem(sys.modules, "rinbo_msgs", rinbo_msgs)
    monkeypatch.setitem(sys.modules, "rinbo_msgs.msg", rinbo_msgs_msg)
    publisher_topics = []

    def resolve_topic_name(topic):
        remaps = {
            "/motor/command": "/hardware/motor_command",
            "/preview": "/hardware/motor_command",
        }
        return remaps[topic]

    node = SimpleNamespace(
        resolve_topic_name=resolve_topic_name,
        create_publisher=lambda _msg_type, topic, _qos: publisher_topics.append(topic),
        create_subscription=lambda *_args: object(),
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
    )
    kwargs = _rinbo_init_kwargs()
    kwargs["node"] = node
    backend = module.RinboRosBackend(**kwargs)

    with pytest.raises(RuntimeError, match="resolve to the same ROS topic"):
        backend.connect()

    assert publisher_topics == []
    assert backend.connected is False


def _configured_rinbo(module):
    backend = module.RinboRosBackend.__new__(module.RinboRosBackend)
    backend.MotorCmdStamped = _MotorCmd
    backend.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp")),
        get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
    )
    backend.sequence = 0
    backend.servo_control_mode = 2
    backend.disabled_servo_control_mode = 0
    backend.probe_abad_disable_verified = False
    backend.main_velocity_sign_policy_order = [1.0] * 6
    backend.main_pwm_per_rad_s = 100.0
    backend.main_max_pwm = 500.0
    backend.main_direction_positive_rinbo_order = [True, True, True, False, False, False]
    backend.abad_encoder_zero_rinbo_order = [10, 20, 30, 40, 50, 60]
    backend.abad_sign_rinbo_order = [1.0] * 6
    backend.abad_encoder_counts_per_rad = 1000.0
    backend.abad_encoder_min = 0
    backend.abad_encoder_max = 65535
    backend.last_pwm_rinbo_order = [0.0] * 6
    backend.last_abad_encoder_targets_rinbo_order = [10, 20, 30, 40, 50, 60]
    backend.last_command_was_enabled = False
    backend.last_actual_publish_state = "never"
    backend.last_block_reason = ""
    backend._last_warned_block_reason = ""
    backend.publisher_conflict_latched = False
    backend.publisher_conflict_reason = ""
    return backend


def test_biorola_applies_individual_main_mask(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    cmd = _command(enable=True, main_drive_enable=[False, False, True, False, False, False])
    cmd.target_velocity_rad_s[2] = 0.25
    msg = backend._make_motor_cmd_msg(cmd, enabled=True, preview=False)
    enabled_fields = [
        field for field in backend.RINBO_LEG_ORDER if getattr(msg, field).enable
    ]
    assert enabled_fields == ["r3"]
    assert msg.r3.voltage == pytest.approx(25.0)


def test_biorola_probe_pwm_cap_is_immutable_after_configurable_mapping(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    backend.main_pwm_per_rad_s = 100_000.0
    backend.main_max_pwm = 100_000.0
    cmd = _command(
        enable=True,
        main_drive_enable=[True, False, False, False, False, False],
        sim2real_probe=True,
    )
    cmd.target_velocity_rad_s[0] = 0.25

    msg = backend._make_motor_cmd_msg(cmd, enabled=True, preview=False)

    assert msg.r1.voltage == pytest.approx(module.SIM2REAL_PROBE_PWM_CAP)


def test_biorola_enabled_probe_requires_verified_abad_disable_interlock(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    backend.connected = True
    cmd = _command(
        enable=True,
        main_drive_enable=[True, False, False, False, False, False],
        sim2real_probe=True,
    )

    with pytest.raises(module.CommandRejectedError, match="ABAD disable.*verified"):
        backend.send_motor_command(cmd)
    assert backend.last_actual_publish_state == "blocked_probe_abad_disable_unverified"
    assert module.SIM2REAL_PROBE_PWM_CAP == 30.0


def test_biorola_probe_cap_does_not_change_normal_controller_mapping(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    backend.main_pwm_per_rad_s = 100_000.0
    backend.main_max_pwm = 100_000.0
    cmd = _command(
        enable=True,
        main_drive_enable=[True, False, False, False, False, False],
        sim2real_probe=False,
    )
    cmd.target_velocity_rad_s[0] = 0.25

    msg = backend._make_motor_cmd_msg(cmd, enabled=True, preview=False)

    assert msg.r1.voltage == pytest.approx(25_000.0)


def test_biorola_disables_aggregate_abad_output_independently(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    main_only = _command(enable=True, main_drive_enable=[True] + [False] * 5, abad_output_enable=False)
    msg = backend._make_motor_cmd_msg(main_only, enabled=True, preview=False)
    assert msg.servo_control_mode == backend.disabled_servo_control_mode
    assert backend.last_abad_encoder_targets_rinbo_order == backend.abad_encoder_zero_rinbo_order

    abad_only = _command(enable=True, abad_output_enable=True)
    abad_only.target_position_rad[6] = 0.1
    msg = backend._make_motor_cmd_msg(abad_only, enabled=True, preview=False)
    assert msg.servo_control_mode == backend.servo_control_mode
    assert not any(getattr(msg, field).enable for field in backend.RINBO_LEG_ORDER)


def test_biorola_motor_state_is_consumed_only_once(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = module.RinboRosBackend.__new__(module.RinboRosBackend)
    state = object()
    backend.latest_motor_state = state
    assert backend.read_motor_state() is state
    assert backend.read_motor_state() is None


def test_biorola_emergency_disable_repeats_all_disabled_packets(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []
    backend.connected = True
    backend.cmd_pub = SimpleNamespace(publish=published.append)
    backend.shutdown_disable_repeats = 3
    backend.shutdown_disable_period_s = 0.0
    backend.emergency_disable()
    assert len(published) == 3
    assert all(msg.servo_control_mode == backend.disabled_servo_control_mode for msg in published)
    assert all(
        not getattr(msg, field).enable
        for msg in published
        for field in backend.RINBO_LEG_ORDER
    )
    assert [msg.header.seq for msg in published] == [1, 2, 3]
    assert backend.sequence == 3
    assert backend.last_command_was_enabled is False


def test_biorola_emergency_disable_has_a_two_packet_minimum(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []
    backend.connected = True
    backend.cmd_pub = SimpleNamespace(publish=published.append)
    backend.shutdown_disable_repeats = 0
    backend.shutdown_disable_period_s = 0.0
    backend.emergency_disable()
    assert len(published) >= 2


def test_biorola_invalid_context_attempts_every_disable_then_reports_watchdog_fallback(
    monkeypatch,
):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    attempts = 0

    def invalid_context(_message):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("publisher context is invalid")

    backend.connected = True
    backend.cmd_pub = SimpleNamespace(publish=invalid_context)
    backend.shutdown_disable_repeats = 3
    backend.shutdown_disable_period_s = 0.0

    with pytest.raises(module.EmergencyDisableError, match="watchdog"):
        backend.emergency_disable()

    assert attempts == 3


def test_biorola_disable_still_publishes_when_context_clock_is_invalid(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []

    def invalid_clock():
        raise RuntimeError("clock context is invalid")

    backend.node = SimpleNamespace(get_clock=invalid_clock)
    backend.connected = True
    backend.cmd_pub = SimpleNamespace(publish=published.append)
    backend.shutdown_disable_repeats = 3
    backend.shutdown_disable_period_s = 0.0

    backend.emergency_disable()

    assert len(published) == 3


def test_biorola_shutdown_disable_cannot_be_opted_out(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []
    backend.connected = True
    backend.cmd_pub = SimpleNamespace(publish=published.append)
    backend.publish_shutdown_disable = False
    backend.shutdown_disable_repeats = 2
    backend.shutdown_disable_period_s = 0.0
    backend.shutdown()
    assert len(published) == 2
    assert backend.connected is False


def test_biorola_repeats_disable_on_active_to_disabled_transition(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []
    backend.connected = True
    backend.publish_preview = False
    backend.command_topic = "/motor/command"
    backend.publish_when_disabled = False
    backend.last_command_was_enabled = True
    backend.cmd_pub = SimpleNamespace(
        publish=published.append,
        get_subscription_count=lambda: 1,
    )
    backend.shutdown_disable_repeats = 3
    backend.shutdown_disable_period_s = 0.0
    backend.send_motor_command(_command(enable=False))
    assert len(published) == 3
    assert backend.last_command_was_enabled is False


def test_biorola_repeats_disable_when_a_new_enabled_command_is_blocked(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []
    backend.node.get_logger = lambda: SimpleNamespace(warn=lambda _message: None)
    backend.connected = True
    backend.publish_preview = False
    backend.command_topic = "/motor/command"
    backend.allow_enable = True
    backend.block_if_duplicate_command_publishers = False
    backend.last_state_time = module.time.monotonic()
    backend.state_timeout_s = 1.0
    backend.last_command_was_enabled = True
    backend.cmd_pub = SimpleNamespace(
        publish=published.append,
        get_subscription_count=lambda: 0,
    )
    backend.shutdown_disable_repeats = 2
    backend.shutdown_disable_period_s = 0.0
    backend.last_actual_publish_state = "published_enabled"
    backend.last_block_reason = ""
    backend._last_warned_block_reason = ""
    with pytest.raises(module.CommandRejectedError, match="no subscriber"):
        backend.send_motor_command(
            _command(enable=True, main_drive_enable=[True, False, False, False, False, False])
        )
    assert len(published) == 2
    assert backend.last_command_was_enabled is False
    assert backend.last_actual_publish_state == "blocked_no_command_subscriber"


@pytest.mark.parametrize(
    "blocked_path",
    ["allow_enable", "recent_state", "duplicate_publishers", "command_subscriber"],
)
def test_biorola_enabled_block_paths_raise_without_publishing_enabled_packet(
    monkeypatch, blocked_path
):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []
    backend.node.get_logger = lambda: SimpleNamespace(warn=lambda _message: None)
    backend.connected = True
    backend.publish_preview = False
    backend.command_topic = "/motor/command"
    backend.allow_enable = True
    backend.block_if_duplicate_command_publishers = False
    backend.last_state_time = module.time.monotonic()
    backend.state_timeout_s = 1.0
    backend.last_actual_publish_state = "never"
    backend.last_block_reason = ""
    backend._last_warned_block_reason = ""
    subscriber_count = 1
    if blocked_path == "allow_enable":
        backend.allow_enable = False
    elif blocked_path == "recent_state":
        backend.last_state_time = None
    elif blocked_path == "duplicate_publishers":
        backend.block_if_duplicate_command_publishers = True
        backend._command_publisher_count = lambda: (2, "/a,/b")
    elif blocked_path == "command_subscriber":
        subscriber_count = 0
    backend.cmd_pub = SimpleNamespace(
        publish=published.append,
        get_subscription_count=lambda: subscriber_count,
    )
    backend.shutdown_disable_repeats = 2
    backend.shutdown_disable_period_s = 0.0

    with pytest.raises(module.CommandRejectedError):
        backend.send_motor_command(
            _command(enable=True, main_drive_enable=[True, False, False, False, False, False])
        )

    assert published == []
    assert backend.last_command_was_enabled is False
    assert backend.last_actual_publish_state.startswith("blocked_")


def _publisher_info(node_name: str, node_namespace: str = "/") -> SimpleNamespace:
    return SimpleNamespace(node_name=node_name, node_namespace=node_namespace)


def _rinbo_with_publisher_guard(module, publisher_query):
    backend = _configured_rinbo(module)
    published = []
    backend.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp")),
        get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
        get_name=lambda: "redrhex_lowlevel_bridge",
        get_namespace=lambda: "/robot",
        get_publishers_info_by_topic=publisher_query,
    )
    backend.connected = True
    backend.publish_preview = False
    backend.command_topic = "/motor/command"
    backend.allow_enable = True
    backend.probe_abad_disable_verified = True
    backend.block_if_duplicate_command_publishers = True
    backend.last_state_time = module.time.monotonic()
    backend.state_timeout_s = 1.0
    backend.last_actual_publish_state = "never"
    backend.last_block_reason = ""
    backend._last_warned_block_reason = ""
    backend.cmd_pub = SimpleNamespace(
        publish=published.append,
        get_subscription_count=lambda: 1,
    )
    backend.shutdown_disable_repeats = 2
    backend.shutdown_disable_period_s = 0.0
    return backend, published


def test_biorola_enabled_command_fails_closed_when_publisher_graph_query_raises(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")

    def fail_query(_topic):
        raise RuntimeError("graph unavailable")

    backend, published = _rinbo_with_publisher_guard(module, fail_query)

    with pytest.raises(module.CommandRejectedError, match="publisher graph query failed"):
        backend.send_motor_command(
            _command(enable=True, main_drive_enable=[True, False, False, False, False, False])
        )

    assert published == []
    assert backend.last_command_was_enabled is False
    assert backend.last_actual_publish_state == "blocked_publisher_graph_query"
    assert "RuntimeError: graph unavailable" in backend.last_block_reason


@pytest.mark.parametrize(
    ("publisher_infos", "reason_fragment"),
    [
        ([], "found 0 publishers"),
        ([_publisher_info("another_controller", "/robot")], "expected sole publisher /robot/redrhex_lowlevel_bridge"),
        (
            [
                _publisher_info("redrhex_lowlevel_bridge", "/robot"),
                _publisher_info("another_controller", "/robot"),
            ],
            "found 2 publishers",
        ),
    ],
)
def test_biorola_enabled_command_requires_exactly_one_self_publisher(
    monkeypatch, publisher_infos, reason_fragment
):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend, published = _rinbo_with_publisher_guard(
        module, lambda _topic: publisher_infos
    )

    with pytest.raises(module.CommandRejectedError, match=reason_fragment):
        backend.send_motor_command(
            _command(enable=True, main_drive_enable=[True, False, False, False, False, False])
        )

    assert published == []
    assert backend.last_actual_publish_state == "blocked_command_publisher_exclusivity"


def test_biorola_direct_command_publisher_conflict_latches_not_ready_until_restart(
    monkeypatch,
):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    publisher_infos = [
        _publisher_info("redrhex_lowlevel_bridge", "/robot"),
        _publisher_info("rogue_controller", "/robot"),
    ]
    backend, published = _rinbo_with_publisher_guard(
        module, lambda _topic: publisher_infos
    )
    command = _command(
        enable=True,
        main_drive_enable=[True, False, False, False, False, False],
        sim2real_probe=True,
    )

    with pytest.raises(module.CommandRejectedError, match="found 2 publishers"):
        backend.send_motor_command(command)

    assert backend.publisher_conflict_latched is True
    assert "rogue_controller" in backend.publisher_conflict_reason
    assert backend.is_alive() is False

    publisher_infos[:] = [_publisher_info("redrhex_lowlevel_bridge", "/robot")]
    with pytest.raises(module.CommandRejectedError, match="latched"):
        backend.send_motor_command(command)
    backend.emergency_disable()
    assert backend.publisher_conflict_latched is True
    assert not any(
        getattr(message, field).enable
        for message in published
        for field in backend.RINBO_LEG_ORDER
    )


def test_probe_cannot_disable_lowlevel_publisher_exclusivity_guard(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend, published = _rinbo_with_publisher_guard(
        module,
        lambda _topic: [
            _publisher_info("redrhex_lowlevel_bridge", "/robot"),
            _publisher_info("rogue_controller", "/robot"),
        ],
    )
    backend.block_if_duplicate_command_publishers = False

    with pytest.raises(module.CommandRejectedError, match="found 2 publishers"):
        backend.send_motor_command(
            _command(
                enable=True,
                main_drive_enable=[True, False, False, False, False, False],
                sim2real_probe=True,
            )
        )

    assert backend.publisher_conflict_latched is True
    assert published == []


def test_latched_lowlevel_publisher_conflict_forces_central_heartbeat_false(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend, _published = _rinbo_with_publisher_guard(
        module,
        lambda _topic: [
            _publisher_info("redrhex_lowlevel_bridge", "/robot"),
            _publisher_info("rogue_controller", "/robot"),
        ],
    )
    safety = _command_safety()
    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)

    with pytest.raises(module.CommandRejectedError):
        safety.dispatch_command_fail_closed(
            gate,
            backend,
            _command(
                enable=True,
                main_drive_enable=[True, False, False, False, False, False],
                sim2real_probe=True,
            ),
        )

    assert gate.ready_for_output is True
    assert backend.is_alive() is False
    assert bool(backend.is_alive() and backend.output_state_is_fresh() and gate.ready_for_output) is False


def test_biorola_enabled_command_accepts_exactly_one_self_publisher(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend, published = _rinbo_with_publisher_guard(
        module,
        lambda _topic: [_publisher_info("redrhex_lowlevel_bridge", "/robot")],
    )

    backend.send_motor_command(
        _command(enable=True, main_drive_enable=[True, False, False, False, False, False])
    )

    assert len(published) == 1
    assert published[0].r1.enable is True
    assert backend.last_actual_publish_state == "published_enabled"


def test_biorola_disabled_command_does_not_depend_on_publisher_graph_query(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")

    def fail_query(_topic):
        raise AssertionError("disabled output must not query the ROS graph")

    backend, published = _rinbo_with_publisher_guard(module, fail_query)
    backend.publish_when_disabled = True

    backend.send_motor_command(_command(enable=False))

    assert len(published) == 1
    assert not any(getattr(published[0], field).enable for field in backend.RINBO_LEG_ORDER)
    assert backend.last_actual_publish_state == "published_disabled"


def test_biorola_successful_enabled_send_is_observable_to_dispatcher(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []
    backend.connected = True
    backend.publish_preview = False
    backend.command_topic = "/motor/command"
    backend.allow_enable = True
    backend.block_if_duplicate_command_publishers = False
    backend.last_state_time = module.time.monotonic()
    backend.state_timeout_s = 1.0
    backend.cmd_pub = SimpleNamespace(
        publish=published.append,
        get_subscription_count=lambda: 1,
    )

    safety = _command_safety()
    gate = safety.FailClosedOutputGate()
    gate.on_estop(False)
    safety.dispatch_command_fail_closed(
        gate,
        backend,
        _command(enable=True, main_drive_enable=[True, False, False, False, False, False]),
    )

    assert len(published) == 1
    assert backend.last_actual_publish_state == "published_enabled"
    assert gate.output_active is True
    assert gate.disable_pending is False


def test_biorola_effective_active_to_inactive_transition_repeats_disable(monkeypatch):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []
    backend.connected = True
    backend.publish_preview = False
    backend.command_topic = "/motor/command"
    backend.allow_enable = True
    backend.block_if_duplicate_command_publishers = False
    backend.last_state_time = module.time.monotonic()
    backend.state_timeout_s = 1.0
    backend.last_command_was_enabled = True
    backend.cmd_pub = SimpleNamespace(
        publish=published.append,
        get_subscription_count=lambda: 1,
    )
    backend.shutdown_disable_repeats = 3
    backend.shutdown_disable_period_s = 0.0

    backend.send_motor_command(_command(enable=True))

    assert len(published) == 3
    assert backend.last_command_was_enabled is False
    assert backend.last_actual_publish_state == "published_disabled_repeated"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_biorola_rejects_nonfinite_command_before_preview_or_publish(monkeypatch, bad_value):
    module = _import_lowlevel(monkeypatch, "rinbo_ros_backend")
    backend = _configured_rinbo(module)
    published = []
    backend.connected = True
    backend.publish_preview = True
    backend.preview_pub = SimpleNamespace(publish=published.append)
    backend.allow_enable = True
    backend.block_if_duplicate_command_publishers = False
    backend.last_state_time = module.time.monotonic()
    backend.state_timeout_s = 1.0
    backend.cmd_pub = SimpleNamespace(
        publish=published.append,
        get_subscription_count=lambda: 1,
    )
    command = _command(enable=True, main_drive_enable=[True] + [False] * 5)
    command.target_velocity_rad_s[0] = bad_value

    with pytest.raises(module.CommandSelectionError, match="target_velocity_rad_s"):
        backend.send_motor_command(command)

    assert published == []


def test_mock_records_effective_selection_and_disable_override(monkeypatch):
    module = _import_lowlevel(monkeypatch, "mock_bridge")
    backend = module.MockLowLevelBridge(print_every_n=1000)
    backend.connect()
    backend.send_motor_command(
        _command(enable=True, main_drive_enable=[False, True, False, False, False, False])
    )
    assert backend.last_main_drive_enable == (False, True, False, False, False, False)
    assert backend.last_abad_output_enable is False
    backend.send_motor_command(
        SimpleNamespace(
            **{
                **vars(_command(enable=False)),
                "main_drive_enable": [True],
                "abad_output_enable": True,
            }
        )
    )
    assert backend.last_main_drive_enable == (False,) * 6
    assert backend.last_abad_output_enable is False


def test_mock_feedback_changes_only_effectively_selected_outputs(monkeypatch):
    module = _import_lowlevel(monkeypatch, "mock_bridge")
    backend = module.MockLowLevelBridge(print_every_n=1000)
    backend.connect()
    command = _command(
        enable=True,
        main_drive_enable=[False, True, False, False, False, False],
    )
    command.target_position_rad = [float(index + 1) for index in range(12)]
    command.target_velocity_rad_s = [float(index + 101) for index in range(12)]
    backend.send_motor_command(command)

    state = backend.read_motor_state()
    assert state.position_rad[:6] == [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    assert state.velocity_rad_s[:6] == [0.0, 102.0, 0.0, 0.0, 0.0, 0.0]
    assert state.position_rad[6:] == [0.0] * 6
    assert state.velocity_rad_s[6:] == [0.0] * 6

    backend.send_motor_command(_command(enable=False))
    disabled_state = backend.read_motor_state()
    assert disabled_state.position_rad[1] == 2.0
    assert disabled_state.velocity_rad_s == [0.0] * 12


def test_lowlevel_node_wires_estop_emergency_gate_and_preserves_state_timestamp():
    path = LOWLEVEL_PACKAGE / "lowlevel_bridge_node.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")
    assert '"/estop"' in source
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "emergency_disable"
        for node in ast.walk(tree)
    )
    tick = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_tick"
    )
    assert not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "stamp"
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "header"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "state"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        for node in ast.walk(tick)
    )
    assert "FailClosedOutputGate" in source
    assert "dispatch_command_fail_closed" in source
    assert "disable_pending" in ast.unparse(tick)


def test_rl_controller_explicitly_selects_all_outputs_only_when_enabled():
    path = CONTROLLER_PACKAGE / "rl_controller_node.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    publish_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_publish_motor_command"
    )
    assigned = {
        target.attr
        for node in ast.walk(publish_fn)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "msg"
    }
    assert {"main_drive_enable", "abad_output_enable", "sim2real_probe"} <= assigned
    source = ast.unparse(publish_fn)
    assert "msg.enable" in source


@pytest.mark.parametrize(
    ("mode", "index", "expected_main", "expected_abad"),
    [
        ("disable", 0, (False,) * 6, False),
        ("init-stand", 0, (False,) * 6, True),
        ("single-main-velocity", 2, (False, False, True, False, False, False), False),
        ("all-main-velocity", 0, (True,) * 6, False),
        ("single-abad", 3, (False,) * 6, True),
        ("all-abad", 0, (False,) * 6, True),
    ],
)
def test_manual_modes_select_only_requested_outputs(
    mode, index, expected_main, expected_abad
):
    safety = _manual_safety()
    selection = safety.selection_for_mode(mode, index=index, enabled=True)
    assert selection == (expected_main, expected_abad)
    assert safety.selection_for_mode(mode, index=index, enabled=False) == ((False,) * 6, False)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_fail_safe_runner_repeats_terminal_disable_on_interrupt_or_exception(failure):
    safety = _manual_safety()
    disables = []

    def operation():
        raise failure("stop")

    with pytest.raises(failure):
        safety.run_with_terminal_disable(operation, lambda: disables.append("disable"), repeats=5)
    assert disables == ["disable"] * 5


def test_fail_safe_runner_repeats_terminal_disable_on_normal_completion():
    safety = _manual_safety()
    disables = []
    result = safety.run_with_terminal_disable(lambda: 7, lambda: disables.append("disable"), repeats=4)
    assert result == 7
    assert disables == ["disable"] * 4


def test_fail_safe_runner_rejects_a_single_terminal_disable():
    safety = _manual_safety()
    with pytest.raises(ValueError, match="at least 2"):
        safety.run_with_terminal_disable(lambda: None, lambda: None, repeats=1)


def test_manual_tool_wires_estop_masks_and_fail_safe_finalization():
    source = (CONTROLLER_PACKAGE / "motor_command_tool.py").read_text(encoding="utf-8")
    assert "main_drive_enable" in source
    assert "abad_output_enable" in source
    assert "sim2real_probe" in source
    assert '"/estop"' in source or "estop_topic" in source
    assert "run_with_terminal_disable" in source
    assert "estop_state" in source


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_manual_command_cli_numeric_parser_rejects_nonfinite_values(value):
    safety = _manual_safety()
    with pytest.raises(ValueError, match="finite"):
        safety.finite_float(value)
