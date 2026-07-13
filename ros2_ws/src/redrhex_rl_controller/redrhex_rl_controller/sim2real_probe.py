"""Bounded, fail-closed suspended single-main-drive characterization probe."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .sim2real_probe_core import (
    INPUT_FRESHNESS_TIMEOUT_S,
    RATE_HZ,
    TERMINAL_DISABLE_PACKETS,
    TERMINAL_DISABLE_PERIOD_S,
    ProbeAbort,
    ProbeCommand,
    ProbeRunner,
    SafetySnapshot,
    build_preview,
    safety_failure,
    terminal_command,
)


def _main_index(value: str) -> int:
    try:
        selected = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("main index must be 0..5") from exc
    if not 0 <= selected < 6:
        raise argparse.ArgumentTypeError("main index must be 0..5")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview or run the fixed suspended single-main step/coast probe."
    )
    parser.add_argument("--main-index", type=_main_index, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print JSON preview; never start ROS."
    )
    parser.add_argument("--enable", action="store_true", help="Request the fixed hardware probe.")
    parser.add_argument(
        "--confirm-risk",
        action="store_true",
        help="Confirm physical E-stop, current limiting, suspension, and sbRIO watchdog.",
    )
    parser.add_argument(
        "--confirm-abad-disable",
        action="store_true",
        help=(
            "Confirm ABAD power is isolated or BioRoLa disabled-servo behavior was "
            "physically verified non-moving and the bridge interlock is enabled."
        ),
    )
    return parser


def _run_ros(args: argparse.Namespace) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, String

    from redrhex_msgs.msg import RedRhexMotorCommand

    from . import redrhex_contract as C
    from .shutdown_signals import install_controlled_signal_handlers, restore_signal_handlers

    try:
        from rclpy._rclpy_pybind11 import RCLError
    except Exception:  # pragma: no cover - rclpy-version dependent
        RCLError = RuntimeError

    class Sim2RealProbeNode(Node):
        def __init__(self) -> None:
            super().__init__("redrhex_sim2real_probe")
            self._heartbeat_value: bool | None = None
            self._heartbeat_received_at: float | None = None
            self._joint_state_received_at: float | None = None
            self._estop_value: bool | None = None
            self._probe_started = False
            self._runner_entered = False

            self._command_pub = self.create_publisher(
                RedRhexMotorCommand, "/redrhex/motor_commands", 10
            )
            self._event_pub = self.create_publisher(
                String, "/redrhex/sim2real_probe/events", 10
            )
            self.create_subscription(
                Bool, "/redrhex/lowlevel_heartbeat", self._on_heartbeat, 10
            )
            self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
            self.create_subscription(Bool, "/estop", self._on_estop, 10)

            self._runner = ProbeRunner(
                publish_command=self._publish_command,
                publish_event=self._publish_event,
                safety_snapshot=self._safety_snapshot,
                monotonic=time.monotonic,
                wait_until=self._wait_until,
                poll=self._poll,
                terminal_pause=lambda: time.sleep(TERMINAL_DISABLE_PERIOD_S),
                terminal_disable_failure=self._terminal_disable_unavailable,
            )
            self._runner.bind(args.main_index)

        def _on_heartbeat(self, message: Bool) -> None:
            self._heartbeat_received_at = time.monotonic()
            self._heartbeat_value = bool(message.data)
            if self._probe_started and self._heartbeat_value is not True:
                self._runner.request_abort("low-level heartbeat false", immediate=True)

        def _on_joint_states(self, _message: JointState) -> None:
            self._joint_state_received_at = time.monotonic()

        def _on_estop(self, message: Bool) -> None:
            self._estop_value = bool(message.data)
            if self._probe_started and self._estop_value:
                self._runner.request_abort("E-stop asserted", immediate=True)

        def _safety_snapshot(self) -> SafetySnapshot:
            try:
                publishers = self.get_publishers_info_by_topic(
                    self._command_pub.topic_name
                )
            except Exception as exc:
                self.get_logger().error(f"Motor command graph query failed: {exc}")
                publishers = []
            publisher = publishers[0] if len(publishers) == 1 else None
            publisher_is_self = (
                publisher is not None
                and publisher.node_name == self.get_name()
                and publisher.node_namespace == self.get_namespace()
            )
            return SafetySnapshot(
                command_subscriber_count=self._command_pub.get_subscription_count(),
                command_publisher_count=len(publishers),
                command_publisher_is_self=publisher_is_self,
                heartbeat_value=self._heartbeat_value,
                heartbeat_received_at=self._heartbeat_received_at,
                joint_state_received_at=self._joint_state_received_at,
                estop_value=self._estop_value,
            )

        def _message(self, command: ProbeCommand) -> RedRhexMotorCommand:
            command_message = RedRhexMotorCommand()
            command_message.header.frame_id = "redrhex_base"
            command_message.header.stamp = self.get_clock().now().to_msg()
            command_message.joint_names = C.MAIN_DRIVE_JOINT_NAMES + C.ABAD_JOINT_NAMES
            command_message.target_position_rad = list(C.INIT_MAIN_DRIVE_POS) + list(
                C.INIT_ABAD_POS
            )
            command_message.target_velocity_rad_s = list(command.target_velocity_rad_s)
            command_message.kp = [0.0] * 12
            command_message.kd = [0.0] * 12
            command_message.effort_limit_nm = [0.0] * 12
            command_message.enable = bool(command.enable)
            command_message.main_drive_enable = list(command.main_drive_enable)
            command_message.abad_output_enable = False
            command_message.sim2real_probe = True
            command_message.mode = 2
            return command_message

        def _publish_command(self, command: ProbeCommand) -> None:
            self._command_pub.publish(self._message(command))

        def _publish_event(self, payload: dict) -> None:
            event_message = String()
            event_message.data = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            self._event_pub.publish(event_message)

        def _poll(self) -> None:
            if not rclpy.ok():
                raise KeyboardInterrupt
            rclpy.spin_once(self, timeout_sec=0.0)

        def _wait_until(self, deadline: float) -> None:
            while rclpy.ok():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                rclpy.spin_once(self, timeout_sec=min(remaining, 1.0 / RATE_HZ))
            self._poll()

        def _collect_initial_callbacks(self) -> None:
            deadline = time.monotonic() + INPUT_FRESHNESS_TIMEOUT_S
            while rclpy.ok() and time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                rclpy.spin_once(self, timeout_sec=min(remaining, 1.0 / RATE_HZ))

        def _fallback_terminal_burst(self) -> None:
            published = 0
            for packet in range(TERMINAL_DISABLE_PACKETS):
                try:
                    self._publish_command(terminal_command())
                    published += 1
                except BaseException as exc:
                    self.get_logger().error(f"Terminal disable packet failed: {exc}")
                if packet + 1 < TERMINAL_DISABLE_PACKETS:
                    time.sleep(TERMINAL_DISABLE_PERIOD_S)
            if published == 0:
                self._terminal_disable_unavailable(TERMINAL_DISABLE_PACKETS)

        @staticmethod
        def _terminal_disable_unavailable(attempted: int) -> None:
            print(
                "CRITICAL: no terminal disable command could be published "
                f"after {attempted} attempts; assert the physical E-stop and rely on "
                "the verified sbRIO command watchdog.",
                file=sys.stderr,
                flush=True,
            )

        def execute(self) -> None:
            try:
                self._collect_initial_callbacks()
                failure = safety_failure(self._safety_snapshot(), now=time.monotonic())
                if failure is not None:
                    self._runner.request_abort(failure, immediate=True)
                    raise ProbeAbort(failure)
                self._probe_started = True
                self._runner_entered = True
                self.get_logger().warn(
                    f"Starting fixed 60 Hz suspended probe on main index {args.main_index}."
                )
                self._runner.run(args.main_index)
            finally:
                self._probe_started = False
                if not self._runner_entered:
                    self._fallback_terminal_burst()

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    previous_signal_handlers = install_controlled_signal_handlers()
    node = None
    try:
        node = Sim2RealProbeNode()
        node.execute()
        return 0
    except ProbeAbort as exc:
        if node is not None:
            node.get_logger().error(f"Probe aborted: {exc}")
        raise SystemExit(str(exc)) from exc
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        return 130
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        restore_signal_handlers(previous_signal_handlers)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run or not args.enable:
        print(json.dumps(build_preview(args.main_index), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if not args.confirm_risk:
        raise SystemExit(
            "Refusing --enable without --confirm-risk. Suspend the robot and verify "
            "all physical safeguards."
        )
    if not args.confirm_abad_disable:
        raise SystemExit(
            "Refusing --enable without --confirm-abad-disable. Isolate ABAD servo "
            "power or physically verify the disabled servo mode, then enable the "
            "low-level bridge ABAD-disable interlock."
        )
    return _run_ros(args)
