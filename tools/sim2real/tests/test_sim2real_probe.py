from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROLLER_ROOT = REPO_ROOT / "ros2_ws/src/redrhex_rl_controller"
CONTROLLER_PACKAGE = CONTROLLER_ROOT / "redrhex_rl_controller"


def _load_core():
    path = CONTROLLER_PACKAGE / "sim2real_probe_core.py"
    assert path.is_file(), f"missing ROS-independent probe core: {path}"
    name = "redrhex_sim2real_probe_core_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_probe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(CONTROLLER_ROOT))
    for name in list(sys.modules):
        if name == "redrhex_rl_controller" or name.startswith("redrhex_rl_controller."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("redrhex_rl_controller.sim2real_probe")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.deadlines: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def wait_until(self, deadline: float) -> None:
        assert deadline >= self.now
        self.deadlines.append(deadline)
        self.now = deadline


def _fresh_snapshot(core, clock: FakeClock):
    return core.SafetySnapshot(
        command_subscriber_count=1,
        command_publisher_count=1,
        command_publisher_is_self=True,
        heartbeat_value=True,
        heartbeat_received_at=clock.now,
        joint_state_received_at=clock.now,
        estop_value=False,
    )


def _runner(core, *, publisher, events, clock, safety=None, poll=lambda: None):
    return core.ProbeRunner(
        publish_command=publisher,
        publish_event=events.append,
        safety_snapshot=safety or (lambda: _fresh_snapshot(core, clock)),
        monotonic=clock.monotonic,
        wait_until=clock.wait_until,
        poll=poll,
        terminal_pause=lambda: None,
    )


def test_canonical_schedule_is_fixed_60_hz_three_repeat_990_tick_sequence() -> None:
    core = _load_core()
    schedule = core.build_schedule(2)

    assert core.RATE_HZ == 60.0
    assert core.REPEATS == 3
    assert core.COMMAND_SPEED_RAD_S == 0.25
    assert core.NEUTRAL_DURATION_S == 0.5
    assert core.DRIVE_DURATION_S == 1.0
    assert core.COAST_DURATION_S == 1.0
    assert len(schedule) == 990
    assert len(schedule) / core.RATE_HZ == pytest.approx(16.5)
    assert [segment["label"] for segment in core.scenario_spec(2)["command_segments"]] == [
        "neutral_before_positive",
        "drive_positive",
        "coast_positive",
        "neutral_between",
        "drive_negative",
        "coast_negative",
        "neutral_finish",
    ]
    for repetition in range(1, 4):
        segments = []
        for tick in schedule:
            if tick.repetition == repetition and (not segments or segments[-1] != tick.segment):
                segments.append(tick.segment)
        assert segments == [
            "neutral_before_positive",
            "drive_positive",
            "coast_positive",
            "neutral_between",
            "drive_negative",
            "coast_negative",
            "neutral_finish",
        ]


def test_drive_enables_exactly_selected_main_and_every_other_tick_is_global_disable() -> None:
    core = _load_core()
    schedule = core.build_schedule(4)

    for tick in schedule:
        command = tick.command
        assert command.abad_output_enable is False
        assert len(command.main_drive_enable) == 6
        assert len(command.target_velocity_rad_s) == 12
        if tick.segment in {"drive_positive", "drive_negative"}:
            assert command.enable is True
            assert command.main_drive_enable == (False, False, False, False, True, False)
            expected = 0.25 if tick.segment == "drive_positive" else -0.25
            assert command.target_velocity_rad_s[4] == expected
            assert all(
                value == 0.0
                for index, value in enumerate(command.target_velocity_rad_s)
                if index != 4
            )
        else:
            assert command.enable is False
            assert command.main_drive_enable == (False,) * 6
            assert command.target_velocity_rad_s == (0.0,) * 12


@pytest.mark.parametrize("index", [-1, 6, True, 1.5, "1"])
def test_schedule_accepts_only_integer_main_index_zero_through_five(index: object) -> None:
    core = _load_core()
    with pytest.raises(ValueError, match="main index must be 0..5"):
        core.build_schedule(index)


@pytest.mark.parametrize("main_index", range(6))
def test_probe_scenario_contract_and_hash_match_canonical_json(main_index: int) -> None:
    core = _load_core()
    from tools.sim2real.scenarios import load_scenario
    from tools.sim2real.traces import sha256_json

    scenario_id = f"suspended-main-{main_index}-step-coast"
    scenario = load_scenario(scenario_id)
    assert scenario.to_dict() == core.scenario_spec(main_index)
    assert scenario.joint == f"main_{main_index}"
    assert scenario.split == ("holdout" if main_index == 5 else "calibration")
    assert core.scenario_id(main_index) == scenario.scenario_id
    assert core.SCENARIO_SCHEMA_VERSION == scenario.schema_version
    assert core.scenario_sha256(main_index) == sha256_json(scenario.to_dict())


@pytest.mark.parametrize("main_index", range(6))
def test_isaac_schedule_disables_every_combined_neutral_and_coast_segment(main_index: int) -> None:
    core = _load_core()
    from tools.sim2real.characterization import scenario_schedule, scenario_step_count
    from tools.sim2real.scenarios import load_scenario

    scenario = load_scenario(core.scenario_id(main_index))
    schedule = scenario_schedule(scenario, scenario_step_count(scenario))
    assert len(schedule) == 1980
    for command in schedule:
        if command.label in {"drive_positive", "drive_negative"}:
            assert command.actuator_enabled is True
        else:
            assert command.actuator_enabled is False


def test_preview_is_machine_readable_and_reports_exact_caps() -> None:
    core = _load_core()
    preview = core.build_preview(3)
    encoded = json.dumps(preview, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["actuation"] is False
    assert decoded["scenario_id"] == "suspended-main-3-step-coast"
    assert decoded["scenario_schema_version"] == 1
    assert len(decoded["scenario_sha256"]) == 64
    assert decoded["main_index"] == 3
    assert decoded["rate_hz"] == 60.0
    assert decoded["repeats"] == 3
    assert decoded["ticks"] == 990
    assert decoded["duration_s"] == 16.5
    assert decoded["command_speed_cap_rad_s"] == 0.25
    assert decoded["max_tick_lateness_s"] == pytest.approx(1.0 / 60.0)
    assert decoded["terminal_disable_packets"] >= 5


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"command_subscriber_count": 0}, "subscriber"),
        ({"command_publisher_count": 0}, "publisher"),
        ({"command_publisher_count": 2}, "publisher"),
        ({"command_publisher_is_self": False}, "publisher"),
        ({"heartbeat_value": None}, "heartbeat"),
        ({"heartbeat_value": False}, "heartbeat"),
        ({"heartbeat_received_at": None}, "heartbeat"),
        ({"heartbeat_received_at": 9.0}, "heartbeat"),
        ({"joint_state_received_at": None}, "joint_states"),
        ({"joint_state_received_at": 9.0}, "joint_states"),
        ({"estop_value": None}, "E-stop"),
        ({"estop_value": True}, "E-stop"),
    ],
)
def test_actuation_safety_is_fail_closed_on_each_missing_or_stale_input(change, reason) -> None:
    core = _load_core()
    base = core.SafetySnapshot(
        command_subscriber_count=1,
        command_publisher_count=1,
        command_publisher_is_self=True,
        heartbeat_value=True,
        heartbeat_received_at=10.0,
        joint_state_received_at=10.0,
        estop_value=False,
    )
    snapshot = replace(base, **change)

    assert reason in core.safety_failure(snapshot, now=10.0)


def test_actuation_safety_uses_callback_receive_time_and_rejects_future_timestamps() -> None:
    core = _load_core()
    fresh = core.SafetySnapshot(1, 1, True, True, 4.8, 4.8, False)
    assert core.safety_failure(fresh, now=5.0) is None

    future_heartbeat = replace(fresh, heartbeat_received_at=5.01)
    future_joint = replace(fresh, joint_state_received_at=5.01)
    assert "heartbeat" in core.safety_failure(future_heartbeat, now=5.0)
    assert "joint_states" in core.safety_failure(future_joint, now=5.0)


def test_normal_completion_emits_markers_and_five_terminal_disable_packets() -> None:
    core = _load_core()
    clock = FakeClock()
    commands = []
    events = []
    runner = _runner(core, publisher=commands.append, events=events, clock=clock)

    runner.run(0)

    assert len(commands) == 990 + core.TERMINAL_DISABLE_PACKETS
    assert all(not command.enable for command in commands[-core.TERMINAL_DISABLE_PACKETS :])
    assert [event["event"] for event in events].count("scenario") == 1
    assert [event["event"] for event in events].count("repetition") == 3
    assert [event["event"] for event in events].count("segment") == 21
    assert [event["event"] for event in events].count("complete") == 1
    assert [event["event"] for event in events].count("abort") == 0
    segment_events = [event for event in events if event["event"] == "segment"]
    assert all("scheduled_elapsed_s" in event for event in segment_events)
    assert all("actual_elapsed_s" in event for event in segment_events)
    assert all("lateness_s" in event for event in segment_events)
    assert clock.now == pytest.approx(16.5)
    assert clock.deadlines[-1] == pytest.approx(16.5)


def test_safety_is_checked_before_every_60_hz_scenario_tick() -> None:
    core = _load_core()
    clock = FakeClock()
    safety_calls = 0

    def safety():
        nonlocal safety_calls
        safety_calls += 1
        return _fresh_snapshot(core, clock)

    runner = _runner(core, publisher=lambda _command: None, events=[], clock=clock, safety=safety)
    runner.run(1)

    assert safety_calls == 990


def test_state_loss_during_drive_aborts_and_disables_immediately_then_in_finally() -> None:
    core = _load_core()
    clock = FakeClock()
    commands = []
    events = []

    def safety():
        snapshot = _fresh_snapshot(core, clock)
        if clock.now >= core.NEUTRAL_DURATION_S + 1.0 / core.RATE_HZ:
            return replace(snapshot, joint_state_received_at=None)
        return snapshot

    runner = _runner(core, publisher=commands.append, events=events, clock=clock, safety=safety)
    with pytest.raises(core.ProbeAbort, match="joint_states"):
        runner.run(1)

    assert [event["event"] for event in events].count("abort") == 1
    assert len(commands) == 30 + 1 + 2 * core.TERMINAL_DISABLE_PACKETS
    assert all(not command.enable for command in commands[-2 * core.TERMINAL_DISABLE_PACKETS :])


def test_state_loss_during_disabled_coast_still_aborts_immediately() -> None:
    core = _load_core()
    clock = FakeClock()
    commands = []
    events = []

    def safety():
        snapshot = _fresh_snapshot(core, clock)
        if clock.now >= core.NEUTRAL_DURATION_S + core.DRIVE_DURATION_S:
            return replace(snapshot, heartbeat_value=False)
        return snapshot

    runner = _runner(core, publisher=commands.append, events=events, clock=clock, safety=safety)
    with pytest.raises(core.ProbeAbort, match="heartbeat"):
        runner.run(1)

    aborts = [event for event in events if event["event"] == "abort"]
    assert len(aborts) == 1
    assert commands[-2 * core.TERMINAL_DISABLE_PACKETS :] == [
        core.terminal_command()
    ] * (2 * core.TERMINAL_DISABLE_PACKETS)


def test_competing_command_publisher_appearing_mid_drive_aborts_immediately() -> None:
    core = _load_core()
    clock = FakeClock()
    commands = []
    events = []

    def safety():
        snapshot = _fresh_snapshot(core, clock)
        if clock.now >= core.NEUTRAL_DURATION_S + 1.0 / core.RATE_HZ:
            return replace(snapshot, command_publisher_count=2)
        return snapshot

    runner = _runner(core, publisher=commands.append, events=events, clock=clock, safety=safety)
    with pytest.raises(core.ProbeAbort, match="publisher"):
        runner.run(1)

    assert [event["event"] for event in events].count("abort") == 1
    assert all(not command.enable for command in commands[-2 * core.TERMINAL_DISABLE_PACKETS :])


def test_scheduler_overrun_aborts_before_an_overdue_enabled_tick_is_published() -> None:
    core = _load_core()
    clock = FakeClock()
    commands = []
    events = []
    delayed = False

    def wait_until(deadline: float) -> None:
        nonlocal delayed
        clock.wait_until(deadline)
        if not delayed and deadline >= core.NEUTRAL_DURATION_S:
            clock.now += 2.0 / core.RATE_HZ
            delayed = True

    runner = core.ProbeRunner(
        publish_command=commands.append,
        publish_event=events.append,
        safety_snapshot=lambda: _fresh_snapshot(core, clock),
        monotonic=clock.monotonic,
        wait_until=wait_until,
        poll=lambda: None,
        terminal_pause=lambda: None,
    )

    with pytest.raises(core.ProbeAbort, match="overrun"):
        runner.run(0)

    assert commands
    assert all(not command.enable for command in commands)
    assert [event["event"] for event in events].count("abort") == 1


def test_one_time_subperiod_jitter_returns_to_the_absolute_schedule() -> None:
    core = _load_core()
    clock = FakeClock()
    published = []
    delayed = False

    def wait_until(deadline: float) -> None:
        nonlocal delayed
        clock.wait_until(deadline)
        if not delayed and deadline >= core.NEUTRAL_DURATION_S:
            clock.now += 0.5 / core.RATE_HZ
            delayed = True

    runner = core.ProbeRunner(
        publish_command=lambda command: published.append((clock.now, command)),
        publish_event=lambda _event: None,
        safety_snapshot=lambda: _fresh_snapshot(core, clock),
        monotonic=clock.monotonic,
        wait_until=wait_until,
        poll=lambda: None,
        terminal_pause=lambda: None,
    )

    runner.run(0)

    scenario_times = [timestamp for timestamp, _command in published[:990]]
    intervals = [later - earlier for earlier, later in zip(scenario_times, scenario_times[1:])]
    assert min(intervals) == pytest.approx(0.5 / core.RATE_HZ)


def test_small_recurring_wake_jitter_does_not_accumulate_into_false_overrun() -> None:
    core = _load_core()
    clock = FakeClock()
    commands = []
    events = []
    wake_jitter_s = 0.0001

    def wait_until(deadline: float) -> None:
        assert deadline >= clock.now
        clock.deadlines.append(deadline)
        clock.now = deadline + wake_jitter_s

    runner = core.ProbeRunner(
        publish_command=lambda command: commands.append((clock.now, command)),
        publish_event=events.append,
        safety_snapshot=lambda: _fresh_snapshot(core, clock),
        monotonic=clock.monotonic,
        wait_until=wait_until,
        poll=lambda: None,
        terminal_pause=lambda: None,
    )

    runner.run(0)

    assert len(commands) == 990 + core.TERMINAL_DISABLE_PACKETS
    scenario_times = [timestamp for timestamp, _command in commands[:990]]
    intervals = [later - earlier for earlier, later in zip(scenario_times, scenario_times[1:])]
    assert min(intervals) == pytest.approx(1.0 / core.RATE_HZ)
    segment_events = [event for event in events if event["event"] == "segment"]
    assert max(event["lateness_s"] for event in segment_events) == pytest.approx(
        wake_jitter_s
    )


def test_immediate_estop_callback_path_publishes_disable_burst_before_loop_unwinds() -> None:
    core = _load_core()
    clock = FakeClock()
    commands = []
    events = []
    holder = {}
    fired = False

    def poll():
        nonlocal fired
        if not fired and clock.now >= core.NEUTRAL_DURATION_S:
            fired = True
            holder["runner"].request_abort("E-stop asserted", immediate=True)
            assert len(commands) >= core.TERMINAL_DISABLE_PACKETS
            assert all(not command.enable for command in commands[-core.TERMINAL_DISABLE_PACKETS :])

    runner = _runner(core, publisher=commands.append, events=events, clock=clock, poll=poll)
    holder["runner"] = runner
    with pytest.raises(core.ProbeAbort, match="E-stop"):
        runner.run(0)

    assert [event["event"] for event in events].count("abort") == 1
    assert all(not command.enable for command in commands[-2 * core.TERMINAL_DISABLE_PACKETS :])


def test_publish_exception_emits_abort_and_terminal_disable_packets() -> None:
    core = _load_core()
    clock = FakeClock()
    commands = []
    events = []

    def publish(command):
        if command.enable:
            raise RuntimeError("publish failed")
        commands.append(command)

    runner = _runner(core, publisher=publish, events=events, clock=clock)
    with pytest.raises(RuntimeError, match="publish failed"):
        runner.run(0)

    assert [event["event"] for event in events].count("abort") == 1
    assert all(not command.enable for command in commands[-core.TERMINAL_DISABLE_PACKETS :])


def test_sigint_emits_abort_and_terminal_disable_packets() -> None:
    core = _load_core()
    clock = FakeClock()
    commands = []
    events = []

    def publish(command):
        if command.enable:
            raise KeyboardInterrupt
        commands.append(command)

    runner = _runner(core, publisher=publish, events=events, clock=clock)
    with pytest.raises(KeyboardInterrupt):
        runner.run(0)

    aborts = [event for event in events if event["event"] == "abort"]
    assert len(aborts) == 1
    assert aborts[0]["reason"] == "SIGINT"
    assert all(not command.enable for command in commands[-core.TERMINAL_DISABLE_PACKETS :])


def test_terminal_disable_burst_is_best_effort_for_every_packet() -> None:
    core = _load_core()
    clock = FakeClock()
    attempts = 0

    def fail(_command):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("transport down")

    runner = _runner(core, publisher=fail, events=[], clock=clock)
    runner.request_abort("E-stop asserted", immediate=True)

    assert attempts == core.TERMINAL_DISABLE_PACKETS


def test_preflight_abort_marker_is_bound_to_selected_scenario_and_hash() -> None:
    core = _load_core()
    clock = FakeClock()
    events = []
    runner = _runner(core, publisher=lambda _command: None, events=events, clock=clock)

    runner.bind(2)
    runner.request_abort("preflight failed", immediate=True)

    assert events == [
        {
            "schema_version": 1,
            "event": "abort",
            "scenario_id": core.scenario_id(2),
            "scenario_schema_version": 1,
            "scenario_sha256": core.scenario_sha256(2),
            "main_index": 2,
            "reason": "preflight failed",
        }
    ]


def test_dry_run_and_default_preview_never_enter_ros(monkeypatch, capsys) -> None:
    probe = _import_probe(monkeypatch)
    entered_ros = False

    def fail_if_called(_args):
        nonlocal entered_ros
        entered_ros = True
        raise AssertionError("preview entered ROS")

    monkeypatch.setattr(probe, "_run_ros", fail_if_called)
    assert probe.main(["--main-index", "2", "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["actuation"] is False
    assert probe.main(["--main-index", "3"]) == 0
    default = json.loads(capsys.readouterr().out)
    assert default["actuation"] is False
    assert entered_ros is False


def test_actual_run_requires_enable_and_risk_confirmation_before_ros(monkeypatch) -> None:
    probe = _import_probe(monkeypatch)
    calls = []
    monkeypatch.setattr(probe, "_run_ros", lambda args: calls.append(args) or 0)

    with pytest.raises(SystemExit, match="--confirm-risk"):
        probe.main(["--main-index", "0", "--enable"])
    assert calls == []

    assert probe.main(["--main-index", "0", "--enable", "--confirm-risk"]) == 0
    assert len(calls) == 1


@pytest.mark.parametrize(
    "forbidden",
    ["--amplitude", "--velocity", "--duration", "--rate-hz", "--waveform", "--skip-safety"],
)
def test_cli_has_no_arbitrary_waveform_or_safety_override_surface(monkeypatch, forbidden) -> None:
    probe = _import_probe(monkeypatch)
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(["--main-index", "0", forbidden, "1"])


def test_ros_adapter_uses_callback_receive_time_topics_and_immediate_abort_hooks() -> None:
    path = CONTROLLER_PACKAGE / "sim2real_probe.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    option_strings = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value.startswith("--")
    }
    assert option_strings == {"--main-index", "--dry-run", "--enable", "--confirm-risk"}
    assert '"/redrhex/motor_commands"' in source
    assert '"/redrhex/lowlevel_heartbeat"' in source
    assert '"/joint_states"' in source
    assert '"/estop"' in source
    assert '"/redrhex/sim2real_probe/events"' in source
    assert "time.monotonic()" in source
    assert "msg.header.stamp" not in source
    assert 'request_abort("E-stop asserted", immediate=True)' in source
    assert 'request_abort("low-level heartbeat false", immediate=True)' in source
    assert "get_subscription_count()" in source
    assert "get_publishers_info_by_topic" in source
    assert "publisher.node_name" in source
    assert "publisher.node_namespace" in source
    assert "String()" in source and "json.dumps(" in source


def test_setup_registers_probe_and_operator_docs_are_fail_safe() -> None:
    setup = (CONTROLLER_ROOT / "setup.py").read_text(encoding="utf-8")
    readme = (CONTROLLER_ROOT / "README.md").read_text(encoding="utf-8")

    assert "sim2real_probe = redrhex_rl_controller.sim2real_probe:main" in setup
    assert "sim2real_probe --main-index 0 --dry-run" in readme
    assert "sim2real_probe --main-index 0 --enable --confirm-risk" in readme
    assert "/motor/command" in readme and "/motor/state" in readme
    assert "60 Hz" in readme
    assert "990" in readme and "16.5" in readme
    assert "16.7 ms" in readme and "overrun" in readme and "不會補送" in readme
    for mandatory in ("實體急停", "限流", "sbRIO watchdog"):
        assert mandatory in readme
