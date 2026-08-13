"""Adapter backend for JasonLiaoJCS/BioRoLaROS2 rinbo_msgs.

This backend is for the existing BioRoLaROS2/RhexROS2 stack:
  /motor/command  rinbo_msgs/msg/MotorCmdStamped
  /motor/state    rinbo_msgs/msg/MotorStateStamped

It also publishes /joint_states from the six main-drive encoders so the V1 RL
controller can build IsaacLab-compatible observations.  An explicitly enabled
V2 route appends the six measured ABAD encoders in canonical policy order and
publishes a versioned per-channel validity diagnostic.  The V1 default remains
the original six-joint message.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from sensor_msgs.msg import JointState

from redrhex_msgs.msg import RedRhexMotorState

from .bridge_base import LowLevelBridgeBase
from .command_safety import (
    CommandRejectedError,
    CommandSelectionError,
    OutputSelection,
    resolve_output_selection,
    validate_enabled_command_payload,
)


class EmergencyDisableError(RuntimeError):
    """Not every redundant hardware-disable packet could be published."""


SIM2REAL_PROBE_PWM_CAP = 30.0


@dataclass(frozen=True)
class RinboLegMapping:
    rinbo_field: str
    policy_index: int
    is_left: bool


class RinboRosBackend(LowLevelBridgeBase):
    """ROS adapter for the existing BioRoLaROS2/RhexROS2 bridge.

    The rinbo bridge expects PWM-like main-drive commands in LegCmd.voltage /
    LegCmd.direction. ABAD commands are sent as ServoCmd.position_encoder.
    """

    POLICY_TO_RINBO_LEGS = [
        RinboLegMapping("r1", 0, False),
        RinboLegMapping("r2", 1, False),
        RinboLegMapping("r3", 2, False),
        RinboLegMapping("l1", 3, True),
        RinboLegMapping("l2", 4, True),
        RinboLegMapping("l3", 5, True),
    ]
    RINBO_LEG_ORDER = ["l1", "l2", "l3", "r1", "r2", "r3"]
    RINBO_SERVO_ORDER = ["sl1", "sl2", "sl3", "sr1", "sr2", "sr3"]
    # Policy ABAD order is RF, RM, RR, LF, LM, LR.
    POLICY_ABAD_INDEX_BY_RINBO_SERVO = [3, 4, 5, 0, 1, 2]

    def __init__(
        self,
        node,
        command_topic: str,
        state_topic: str,
        joint_state_topic: str,
        preview_topic: str,
        publish_preview: bool,
        allow_enable: bool,
        publish_when_disabled: bool,
        disabled_servo_control_mode: int,
        probe_abad_disable_verified: bool,
        publish_shutdown_disable: bool,
        shutdown_disable_repeats: int,
        shutdown_disable_period_s: float,
        require_state: bool,
        block_if_duplicate_command_publishers: bool,
        state_timeout_s: float,
        main_position_counts_per_rev: float,
        main_pwm_per_rad_s: float,
        main_max_pwm: float,
        main_encoder_zero_counts_rinbo_order: list[float],
        main_encoder_sign_rinbo_order: list[float],
        main_velocity_sign_policy_order: list[float],
        main_direction_positive_rinbo_order: list[bool],
        main_velocity_filter_alpha: float,
        main_velocity_max_dt_s: float,
        main_velocity_clip_rad_s: float,
        abad_encoder_zero_rinbo_order: list[int],
        abad_encoder_counts_per_rad: float,
        abad_encoder_min: int,
        abad_encoder_max: int,
        abad_sign_rinbo_order: list[float],
        servo_control_mode: int,
        main_joint_names_policy_order: list[str],
        abad_joint_names_policy_order: list[str] | None = None,
        publish_abad_joint_states: bool = False,
        joint_feedback_status_topic_v2: str = "/redrhex/joint_feedback_status_v2",
        main_encoder_calibration_verified: bool = False,
        abad_encoder_calibration_verified: bool = False,
        abad_velocity_filter_alpha: float = 0.35,
        abad_velocity_max_dt_s: float = 0.20,
        abad_velocity_clip_rad_s: float = 20.0,
    ) -> None:
        self.node = node
        self.command_topic = command_topic
        self.state_topic = state_topic
        self.joint_state_topic = joint_state_topic
        self.preview_topic = preview_topic
        self.publish_preview = bool(publish_preview)
        self.allow_enable = bool(allow_enable)
        self.publish_when_disabled = bool(publish_when_disabled)
        self.disabled_servo_control_mode = int(disabled_servo_control_mode)
        self.probe_abad_disable_verified = bool(probe_abad_disable_verified)
        self.publish_shutdown_disable = bool(publish_shutdown_disable)
        self.shutdown_disable_repeats = int(shutdown_disable_repeats)
        self.shutdown_disable_period_s = float(shutdown_disable_period_s)
        self.require_state = bool(require_state)
        self.block_if_duplicate_command_publishers = bool(block_if_duplicate_command_publishers)
        self.state_timeout_s = float(state_timeout_s)
        self.main_position_counts_per_rev = float(main_position_counts_per_rev)
        self.main_pwm_per_rad_s = float(main_pwm_per_rad_s)
        self.main_max_pwm = float(main_max_pwm)
        self.main_encoder_zero_counts_rinbo_order = [float(x) for x in main_encoder_zero_counts_rinbo_order]
        self.main_encoder_sign_rinbo_order = [float(x) for x in main_encoder_sign_rinbo_order]
        self.main_velocity_sign_policy_order = [float(x) for x in main_velocity_sign_policy_order]
        self.main_direction_positive_rinbo_order = list(main_direction_positive_rinbo_order)
        self.main_velocity_filter_alpha = float(main_velocity_filter_alpha)
        self.main_velocity_max_dt_s = float(main_velocity_max_dt_s)
        self.main_velocity_clip_rad_s = float(main_velocity_clip_rad_s)
        self.abad_encoder_zero_rinbo_order = [int(x) for x in abad_encoder_zero_rinbo_order]
        self.abad_encoder_counts_per_rad = float(abad_encoder_counts_per_rad)
        self.abad_encoder_min = int(abad_encoder_min)
        self.abad_encoder_max = int(abad_encoder_max)
        self.abad_sign_rinbo_order = [float(x) for x in abad_sign_rinbo_order]
        self.servo_control_mode = int(servo_control_mode)
        self.main_joint_names_policy_order = list(main_joint_names_policy_order)
        self.abad_joint_names_policy_order = list(abad_joint_names_policy_order or [])
        self.publish_abad_joint_states = bool(publish_abad_joint_states)
        self.joint_feedback_status_topic_v2 = str(joint_feedback_status_topic_v2)
        self.main_encoder_calibration_verified = bool(main_encoder_calibration_verified)
        self.abad_encoder_calibration_verified = bool(abad_encoder_calibration_verified)
        self.abad_velocity_filter_alpha = float(abad_velocity_filter_alpha)
        self.abad_velocity_max_dt_s = float(abad_velocity_max_dt_s)
        self.abad_velocity_clip_rad_s = float(abad_velocity_clip_rad_s)

        if len(self.main_encoder_zero_counts_rinbo_order) != 6:
            raise ValueError("main_encoder_zero_counts_rinbo_order must have length 6")
        if len(self.main_encoder_sign_rinbo_order) != 6:
            raise ValueError("main_encoder_sign_rinbo_order must have length 6")
        if len(self.main_velocity_sign_policy_order) != 6:
            raise ValueError("main_velocity_sign_policy_order must have length 6")
        if len(self.main_direction_positive_rinbo_order) != 6:
            raise ValueError("main_direction_positive_rinbo_order must have length 6")
        if len(self.abad_encoder_zero_rinbo_order) != 6:
            raise ValueError("abad_encoder_zero_rinbo_order must have length 6")
        if len(self.abad_sign_rinbo_order) != 6:
            raise ValueError("abad_sign_rinbo_order must have length 6")
        if len(self.main_joint_names_policy_order) != 6:
            raise ValueError("main_joint_names_policy_order must have length 6")
        if self.publish_abad_joint_states and len(self.abad_joint_names_policy_order) != 6:
            raise ValueError("abad_joint_names_policy_order must have length 6 when ABAD publication is enabled")
        if self.abad_joint_names_policy_order and len(self.abad_joint_names_policy_order) != 6:
            raise ValueError("abad_joint_names_policy_order must be empty or have length 6")
        if set(self.main_joint_names_policy_order).intersection(self.abad_joint_names_policy_order):
            raise ValueError("main and ABAD joint names must not overlap")
        if not math.isfinite(self.state_timeout_s) or self.state_timeout_s <= 0.0:
            raise ValueError("state_timeout_s must be positive and finite")
        if (
            not math.isfinite(self.main_position_counts_per_rev)
            or self.main_position_counts_per_rev <= 0.0
        ):
            raise ValueError("main_position_counts_per_rev must be positive and finite")
        if not math.isfinite(self.main_pwm_per_rad_s) or self.main_pwm_per_rad_s <= 0.0:
            raise ValueError("main_pwm_per_rad_s must be positive and finite")
        if not math.isfinite(self.main_max_pwm) or self.main_max_pwm <= 0.0:
            raise ValueError("main_max_pwm must be positive and finite")
        if not all(math.isfinite(value) for value in self.main_encoder_zero_counts_rinbo_order):
            raise ValueError("main_encoder_zero_counts_rinbo_order must contain finite values")
        if any(value not in (-1.0, 1.0) for value in self.main_encoder_sign_rinbo_order):
            raise ValueError("main_encoder_sign_rinbo_order must contain only -1 or 1")
        if any(value not in (-1.0, 1.0) for value in self.main_velocity_sign_policy_order):
            raise ValueError("main_velocity_sign_policy_order must contain only -1 or 1")
        if any(type(value) is not bool for value in self.main_direction_positive_rinbo_order):
            raise ValueError("main_direction_positive_rinbo_order must contain only booleans")
        if (
            not math.isfinite(self.abad_encoder_counts_per_rad)
            or self.abad_encoder_counts_per_rad <= 0.0
        ):
            raise ValueError("abad_encoder_counts_per_rad must be positive and finite")
        if any(value not in (-1.0, 1.0) for value in self.abad_sign_rinbo_order):
            raise ValueError("abad_sign_rinbo_order must contain only -1 or 1")
        if self.abad_encoder_min >= self.abad_encoder_max:
            raise ValueError("abad_encoder_min must be smaller than abad_encoder_max")
        if (
            not math.isfinite(self.main_velocity_filter_alpha)
            or not 0.0 <= self.main_velocity_filter_alpha <= 1.0
        ):
            raise ValueError("main_velocity_filter_alpha must be finite and in [0, 1]")
        if not math.isfinite(self.main_velocity_max_dt_s) or self.main_velocity_max_dt_s <= 0.0:
            raise ValueError("main_velocity_max_dt_s must be positive and finite")
        if not math.isfinite(self.main_velocity_clip_rad_s) or self.main_velocity_clip_rad_s <= 0.0:
            raise ValueError("main_velocity_clip_rad_s must be positive and finite")
        if (
            not math.isfinite(self.abad_velocity_filter_alpha)
            or not 0.0 <= self.abad_velocity_filter_alpha <= 1.0
        ):
            raise ValueError("abad_velocity_filter_alpha must be finite and in [0, 1]")
        if not math.isfinite(self.abad_velocity_max_dt_s) or self.abad_velocity_max_dt_s <= 0.0:
            raise ValueError("abad_velocity_max_dt_s must be positive and finite")
        if not math.isfinite(self.abad_velocity_clip_rad_s) or self.abad_velocity_clip_rad_s <= 0.0:
            raise ValueError("abad_velocity_clip_rad_s must be positive and finite")
        if self.shutdown_disable_repeats < 2:
            raise ValueError("shutdown_disable_repeats must be at least 2")
        if not math.isfinite(self.shutdown_disable_period_s) or self.shutdown_disable_period_s < 0.0:
            raise ValueError("shutdown_disable_period_s must be non-negative and finite")

        self.main_rad_per_count = 2.0 * math.pi / self.main_position_counts_per_rev

        self.connected = False
        self.sequence = 0
        self.last_state_time: float | None = None
        self.latest_motor_state: RedRhexMotorState | None = None
        self.latest_positions_policy = [0.0] * 6
        self.latest_velocities_policy = [0.0] * 6
        self._prev_positions_policy: list[float] | None = None
        self._prev_state_time: float | None = None
        self.latest_abad_positions_policy = [0.0] * 6
        self.latest_abad_velocities_policy = [0.0] * 6
        self.latest_joint_validity_policy = [False] * 12
        self._prev_abad_positions_policy: list[float] | None = None
        self._prev_abad_state_time: float | None = None
        self.latest_raw_positions_rinbo = [0.0] * 6
        self.latest_servo_positions_rinbo = list(self.abad_encoder_zero_rinbo_order)
        self.latest_servo_raw_validity_rinbo = [False] * 6
        self.last_command_was_enabled = False
        self.last_pwm_rinbo_order = [0.0] * 6
        self.last_abad_encoder_targets_rinbo_order = list(self.abad_encoder_zero_rinbo_order)
        self.last_actual_publish_state = "never"
        self.last_block_reason = ""
        self._last_warned_block_reason = ""
        self.publisher_conflict_latched = False
        self.publisher_conflict_reason = ""

    def connect(self) -> None:
        resolved_command_topic = self.node.resolve_topic_name(self.command_topic)
        resolved_preview_topic = self.node.resolve_topic_name(self.preview_topic)
        if resolved_command_topic == resolved_preview_topic:
            raise RuntimeError(
                "rinbo command and preview topics resolve to the same ROS topic "
                f"({resolved_command_topic}); choose distinct topics/remaps"
            )

        try:
            from rinbo_msgs.msg import MotorCmdStamped, MotorStateStamped
            from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
        except Exception as exc:  # pragma: no cover - requires external BioRoLaROS2 overlay
            raise RuntimeError(
                "rinbo_msgs is required for backend='biorola_ros'/'rinbo_ros'. Build/source BioRoLaROS2 first."
            ) from exc

        self.MotorCmdStamped = MotorCmdStamped
        self.MotorStateStamped = MotorStateStamped
        self.DiagnosticArray = DiagnosticArray
        self.DiagnosticStatus = DiagnosticStatus
        self.KeyValue = KeyValue
        self.cmd_pub = self.node.create_publisher(MotorCmdStamped, self.command_topic, 10)
        self.preview_pub = self.node.create_publisher(MotorCmdStamped, self.preview_topic, 10)
        self.joint_pub = self.node.create_publisher(JointState, self.joint_state_topic, 10)
        self.joint_feedback_status_pub_v2 = None
        if self.publish_abad_joint_states:
            self.joint_feedback_status_pub_v2 = self.node.create_publisher(
                DiagnosticArray, self.joint_feedback_status_topic_v2, 10
            )
        self.state_sub = self.node.create_subscription(MotorStateStamped, self.state_topic, self._on_rinbo_state, 10)
        self.connected = True
        self.node.get_logger().info(
            f"Rinbo ROS backend connected: command={self.command_topic}, state={self.state_topic}"
        )

    def send_motor_command(self, cmd) -> None:
        if not self.connected:
            raise RuntimeError("Rinbo ROS backend is not connected")
        enabled = bool(cmd.enable)
        selection = resolve_output_selection(cmd)
        if enabled:
            validate_enabled_command_payload(cmd)
        if (
            enabled
            and bool(getattr(cmd, "sim2real_probe", False))
            and not self.probe_abad_disable_verified
        ):
            self._block_enabled_command(
                "blocked_probe_abad_disable_unverified",
                "BioRoLa ABAD disable has not been physically verified; keep servo "
                "power isolated or set rinbo.probe_abad_disable_verified only after "
                "confirming the disabled servo mode cannot move",
            )
            return

        preview_msg = self._make_motor_cmd_msg(cmd, enabled=enabled, preview=True)
        if self.publish_preview:
            self.preview_pub.publish(preview_msg)

        if enabled and self.publisher_conflict_latched:
            self._block_enabled_command(
                "blocked_command_publisher_exclusivity_latched",
                f"latched publisher exclusivity failure requires a safe bridge restart: "
                f"{self.publisher_conflict_reason}",
            )
        if enabled and not self.allow_enable:
            self._block_enabled_command("blocked_allow_enable", "rinbo.allow_enable is false")
            return
        if enabled and not self.output_state_is_fresh():
            self._block_enabled_command("blocked_no_recent_state", "no recent /motor/state")
            return
        if enabled and (
            self.block_if_duplicate_command_publishers
            or bool(getattr(cmd, "sim2real_probe", False))
        ):
            publisher_count, endpoint_names = self._command_publisher_count()
            if publisher_count < 0:
                if bool(getattr(cmd, "sim2real_probe", False)):
                    self._latch_publisher_conflict(
                        f"publisher graph query failed for {self.command_topic}: {endpoint_names}"
                    )
                self._block_enabled_command(
                    "blocked_publisher_graph_query",
                    f"publisher graph query failed for {self.command_topic}: {endpoint_names}",
                )
                return
            if publisher_count != 1:
                reason = f"found {publisher_count} publishers on {self.command_topic}: {endpoint_names}"
                self._latch_publisher_conflict(reason)
                self._block_enabled_command(
                    "blocked_command_publisher_exclusivity",
                    reason,
                )
                return
            own_node_name = self._own_node_fully_qualified_name()
            if own_node_name is None or endpoint_names != own_node_name:
                expected = own_node_name or "an identifiable backend node"
                reason = (
                    f"expected sole publisher {expected} on {self.command_topic}; "
                    f"found {endpoint_names}"
                )
                self._latch_publisher_conflict(reason)
                self._block_enabled_command(
                    "blocked_command_publisher_exclusivity",
                    reason,
                )
                return
        if enabled and self.cmd_pub.get_subscription_count() == 0:
            self._block_enabled_command(
                "blocked_no_command_subscriber",
                f"no subscriber on {self.command_topic}",
            )
            return

        if not selection.any_enabled and self.last_command_was_enabled:
            self.emergency_disable()
            self.last_actual_publish_state = "published_disabled_repeated"
            return

        # During dry-run, avoid publishing disabled preview packets because
        # BioRoLaROS2 servo commands have no per-servo enable. If motors were
        # previously enabled, the transition above sends repeated disable packets.
        if not enabled and not self.publish_when_disabled and not self.last_command_was_enabled:
            self.last_command_was_enabled = False
            self.last_pwm_rinbo_order = [0.0] * 6
            self.last_actual_publish_state = "preview_only_disabled"
            return

        msg = self._make_motor_cmd_msg(cmd, enabled=enabled, preview=False)
        self.cmd_pub.publish(msg)
        self.last_command_was_enabled = selection.any_enabled
        self.last_actual_publish_state = "published_enabled" if selection.any_enabled else "published_disabled"

    def _make_motor_cmd_msg(self, cmd, enabled: bool, preview: bool):
        selection = resolve_output_selection(cmd)
        msg = self.MotorCmdStamped()
        now = self.node.get_clock().now().to_msg()
        if preview:
            seq = self.sequence
        else:
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            seq = self.sequence
        msg.header.seq = seq
        msg.header.stamp = now
        msg.header.frame_id = "redrhex_preview" if preview else "redrhex_base"
        abad_enabled = enabled and selection.abad_output_enable
        msg.servo_control_mode = self.servo_control_mode if abad_enabled else self.disabled_servo_control_mode

        self._disable_all_legs(msg)
        self._set_main_drive_pwm(msg, cmd, enabled, selection)
        if abad_enabled:
            self._set_abad_servo_targets(msg, cmd)
        else:
            self._set_abad_neutral_targets(msg)
        return msg

    def read_motor_state(self):
        state = self.latest_motor_state
        self.latest_motor_state = None
        return state

    def is_alive(self) -> bool:
        if not self.connected:
            return False
        if self.publisher_conflict_latched:
            return False
        if not self.require_state:
            return True
        if self.last_state_time is None:
            return False
        return time.monotonic() - self.last_state_time <= self.state_timeout_s

    def output_state_is_fresh(self) -> bool:
        if not self.connected or self.last_state_time is None:
            return False
        return time.monotonic() - self.last_state_time <= self.state_timeout_s

    def emergency_disable(self) -> None:
        if self.connected:
            self._publish_shutdown_disable()
        self.last_command_was_enabled = False
        self.last_pwm_rinbo_order = [0.0] * 6

    def shutdown(self) -> None:
        if self.connected:
            self.emergency_disable()
        self.connected = False

    def diagnostic_values(self) -> dict[str, str]:
        state_age = "none" if self.last_state_time is None else f"{time.monotonic() - self.last_state_time:.4f}"
        publisher_count, publisher_names = self._command_publisher_count()
        return {
            "rinbo_command_topic": self.command_topic,
            "rinbo_state_topic": self.state_topic,
            "rinbo_preview_topic": self.preview_topic,
            "rinbo_publish_preview": str(self.publish_preview),
            "rinbo_require_state": str(self.require_state),
            "rinbo_allow_enable": str(self.allow_enable),
            "rinbo_publish_when_disabled": str(self.publish_when_disabled),
            "rinbo_probe_abad_disable_verified": str(
                self.probe_abad_disable_verified
            ),
            "rinbo_block_if_duplicate_command_publishers": str(self.block_if_duplicate_command_publishers),
            "rinbo_command_subscribers": str(self.cmd_pub.get_subscription_count() if hasattr(self, "cmd_pub") else 0),
            "rinbo_command_publishers": f"{publisher_count}: {publisher_names}",
            "rinbo_main_velocity_filter_alpha": f"{self.main_velocity_filter_alpha:.3f}",
            "rinbo_main_velocity_clip_rad_s": f"{self.main_velocity_clip_rad_s:.3f}",
            "rinbo_last_state_age_s": state_age,
            "rinbo_last_command_enabled": str(self.last_command_was_enabled),
            "rinbo_actual_publish_state": self.last_actual_publish_state,
            "rinbo_last_block_reason": self.last_block_reason,
            "rinbo_publisher_conflict_latched": str(self.publisher_conflict_latched),
            "rinbo_publisher_conflict_reason": self.publisher_conflict_reason or "none",
            "rinbo_last_pwm_l1_l2_l3_r1_r2_r3": ",".join(f"{x:.2f}" for x in self.last_pwm_rinbo_order),
            "rinbo_last_abad_sl1_sl2_sl3_sr1_sr2_sr3": ",".join(str(x) for x in self.last_abad_encoder_targets_rinbo_order),
            "rinbo_servo_state_sl1_sl2_sl3_sr1_sr2_sr3": ",".join(str(x) for x in self.latest_servo_positions_rinbo),
            "rinbo_main_vel_policy_order_rad_s": ",".join(f"{x:.3f}" for x in self.latest_velocities_policy),
        }

    def _warn_once(self, reason: str) -> None:
        if reason != self._last_warned_block_reason:
            self.node.get_logger().warn(f"Blocking enabled BioRoLaROS2 command: {reason}")
            self._last_warned_block_reason = reason

    def _block_enabled_command(self, publish_state: str, reason: str) -> None:
        output_was_active = self.last_command_was_enabled
        self.last_block_reason = reason
        self.last_actual_publish_state = publish_state
        self._warn_once(reason)
        if output_was_active:
            self.emergency_disable()
        self.last_command_was_enabled = False
        raise CommandRejectedError(reason)

    def _latch_publisher_conflict(self, reason: str) -> None:
        if not self.publisher_conflict_latched:
            self.publisher_conflict_reason = str(reason)
        self.publisher_conflict_latched = True

    def _command_publisher_count(self) -> tuple[int, str]:
        try:
            infos = self.node.get_publishers_info_by_topic(self.command_topic)
        except Exception as exc:
            return -1, f"{type(exc).__name__}: {exc}"
        names = []
        for info in infos:
            node_name = getattr(info, "node_name", "")
            node_namespace = getattr(info, "node_namespace", "")
            names.append(self._fully_qualified_node_name(node_name, node_namespace) or "<unknown>")
        return len(infos), ",".join(names) if names else "none"

    def _own_node_fully_qualified_name(self) -> str | None:
        try:
            node_name = self.node.get_name()
            node_namespace = self.node.get_namespace()
        except Exception:
            return None
        return self._fully_qualified_node_name(node_name, node_namespace) or None

    @staticmethod
    def _fully_qualified_node_name(node_name: object, node_namespace: object) -> str:
        name = str(node_name).strip("/")
        if not name:
            return ""
        namespace = str(node_namespace).strip("/")
        return f"/{namespace}/{name}" if namespace else f"/{name}"

    def _publish_shutdown_disable(self) -> None:
        if not hasattr(self, "cmd_pub") or not hasattr(self, "MotorCmdStamped"):
            return
        attempts = max(2, self.shutdown_disable_repeats)
        failures: list[BaseException] = []
        published = 0
        for _ in range(attempts):
            msg = self.MotorCmdStamped()
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            msg.header.seq = self.sequence
            msg.header.frame_id = "redrhex_shutdown_disable"
            msg.servo_control_mode = self.disabled_servo_control_mode
            self._disable_all_legs(msg)
            self._set_abad_neutral_targets(msg)
            try:
                msg.header.stamp = self.node.get_clock().now().to_msg()
            except BaseException:
                # A disable packet is safer with a stale/default stamp than not sent.
                pass
            try:
                self.cmd_pub.publish(msg)
                published += 1
            except BaseException as exc:
                failures.append(exc)
            if self.shutdown_disable_period_s > 0.0:
                time.sleep(self.shutdown_disable_period_s)
        if failures:
            raise EmergencyDisableError(
                f"published {published}/{attempts} emergency-disable packets; "
                "assert the physical E-stop and rely on the verified hardware watchdog"
            ) from failures[-1]
        self.last_command_was_enabled = False
        self.last_pwm_rinbo_order = [0.0] * 6

    def _disable_all_legs(self, msg) -> None:
        for field in self.RINBO_LEG_ORDER:
            leg = getattr(msg, field)
            leg.enable = False
            leg.direction = False
            leg.voltage = 0.0
            leg.state = 0
            leg.reset_position = False

    def _set_main_drive_pwm(
        self,
        msg,
        cmd,
        enabled: bool,
        selection: OutputSelection,
    ) -> None:
        self.last_pwm_rinbo_order = [0.0] * 6
        for mapping in self.POLICY_TO_RINBO_LEGS:
            leg = getattr(msg, mapping.rinbo_field)
            rinbo_idx = self.RINBO_LEG_ORDER.index(mapping.rinbo_field)
            target_velocity = (
                float(cmd.target_velocity_rad_s[mapping.policy_index])
                * self.main_velocity_sign_policy_order[mapping.policy_index]
            )
            pwm_cap = self.main_max_pwm
            if bool(getattr(cmd, "sim2real_probe", False)):
                pwm_cap = min(pwm_cap, SIM2REAL_PROBE_PWM_CAP)
            pwm = max(-pwm_cap, min(pwm_cap, target_velocity * self.main_pwm_per_rad_s))
            selected = enabled and selection.main_drive_enable[mapping.policy_index]
            self.last_pwm_rinbo_order[rinbo_idx] = float(pwm) if selected else 0.0
            if selected:
                leg.enable = True
                leg.state = 1
                leg.reset_position = False
                direction_positive = self.main_direction_positive_rinbo_order[rinbo_idx]
                leg.direction = direction_positive if pwm >= 0.0 else not direction_positive
                leg.voltage = abs(float(pwm))

    def _set_abad_servo_targets(self, msg, cmd) -> None:
        targets: list[int] = []
        abad_targets_policy = list(cmd.target_position_rad[6:12])
        for servo_idx, field in enumerate(self.RINBO_SERVO_ORDER):
            policy_idx = self.POLICY_ABAD_INDEX_BY_RINBO_SERVO[servo_idx]
            target_rad = float(abad_targets_policy[policy_idx])
            raw = (
                self.abad_encoder_zero_rinbo_order[servo_idx]
                + self.abad_sign_rinbo_order[servo_idx] * target_rad * self.abad_encoder_counts_per_rad
            )
            target = int(max(self.abad_encoder_min, min(self.abad_encoder_max, round(raw))))
            getattr(msg, field).position_encoder = target
            targets.append(target)
        self.last_abad_encoder_targets_rinbo_order = targets

    def _set_abad_neutral_targets(self, msg) -> None:
        targets: list[int] = []
        for servo_idx, field in enumerate(self.RINBO_SERVO_ORDER):
            target = int(max(self.abad_encoder_min, min(self.abad_encoder_max, self.abad_encoder_zero_rinbo_order[servo_idx])))
            getattr(msg, field).position_encoder = target
            targets.append(target)
        self.last_abad_encoder_targets_rinbo_order = targets

    def _on_rinbo_state(self, msg) -> None:
        state_time = time.monotonic()
        self.last_state_time = state_time
        rinbo_positions = [
            float(msg.l1.position),
            float(msg.l2.position),
            float(msg.l3.position),
            float(msg.r1.position),
            float(msg.r2.position),
            float(msg.r3.position),
        ]
        self.latest_raw_positions_rinbo = rinbo_positions
        servo_messages = [msg.sl1, msg.sl2, msg.sl3, msg.sr1, msg.sr2, msg.sr3]
        self.latest_servo_raw_validity_rinbo = [
            hasattr(servo, "position_encoder")
            and isinstance(getattr(servo, "position_encoder", None), (int, float))
            and math.isfinite(float(getattr(servo, "position_encoder", 0)))
            for servo in servo_messages
        ]
        self.latest_servo_positions_rinbo = [
            int(getattr(servo, "position_encoder", 0)) for servo in servo_messages
        ]

        rinbo_rad = [
            (rinbo_positions[i] - self.main_encoder_zero_counts_rinbo_order[i])
            * self.main_encoder_sign_rinbo_order[i]
            * self.main_rad_per_count
            for i in range(6)
        ]
        # Policy order: RF, RM, RR, LF, LM, LR.
        policy_positions = [0.0] * 6
        for mapping in self.POLICY_TO_RINBO_LEGS:
            rinbo_idx = self.RINBO_LEG_ORDER.index(mapping.rinbo_field)
            policy_positions[mapping.policy_index] = rinbo_rad[rinbo_idx]
        policy_velocities = self._estimate_policy_velocities(policy_positions, state_time)
        self.latest_positions_policy = policy_positions
        self.latest_velocities_policy = policy_velocities

        abad_positions_policy, abad_validity_policy = self._convert_abad_feedback_to_policy(
            self.latest_servo_positions_rinbo,
            self.latest_servo_raw_validity_rinbo,
        )
        abad_velocities_policy = self._estimate_abad_policy_velocities(abad_positions_policy, state_time)
        self.latest_abad_positions_policy = abad_positions_policy
        self.latest_abad_velocities_policy = abad_velocities_policy
        main_valid = bool(
            self.main_encoder_calibration_verified
            and all(math.isfinite(value) for value in policy_positions + policy_velocities)
        )
        self.latest_joint_validity_policy = [main_valid] * 6 + abad_validity_policy

        js = JointState()
        js.header.stamp = self._joint_state_source_stamp(msg)
        js.header.frame_id = "redrhex_base"
        js.name = list(self.main_joint_names_policy_order)
        js.position = [float(x) for x in policy_positions]
        js.velocity = [float(x) for x in policy_velocities]
        if self.publish_abad_joint_states:
            js.name.extend(self.abad_joint_names_policy_order)
            js.position.extend(float(x) for x in abad_positions_policy)
            js.velocity.extend(float(x) for x in abad_velocities_policy)
        self.joint_pub.publish(js)
        self._publish_joint_feedback_status_v2(js.header.stamp)

        state = RedRhexMotorState()
        state.header.stamp = js.header.stamp
        state.header.frame_id = "redrhex_base"
        state.joint_names = list(self.main_joint_names_policy_order)
        state.position_rad = [float(x) for x in policy_positions]
        state.velocity_rad_s = [float(x) for x in policy_velocities]
        state.effort_nm = [0.0] * 6
        state.current_a = []
        state.temperature_c = []
        state.fault = [False] * 6
        self.latest_motor_state = state

    def _convert_abad_feedback_to_policy(
        self,
        raw_rinbo: list[int],
        raw_validity_rinbo: list[bool] | None = None,
    ) -> tuple[list[float], list[bool]]:
        """Convert six raw servo encoders to neutral-relative radians in policy order.

        Calibration validity is deliberately independent from numeric conversion:
        provisional values can be inspected offline, but V2 readiness stays false
        until the reviewed calibration gate is explicitly enabled.
        """
        if len(raw_rinbo) != 6:
            raise ValueError("raw Rinbo ABAD feedback must have length 6")
        if raw_validity_rinbo is None:
            raw_validity_rinbo = [True] * 6
        if len(raw_validity_rinbo) != 6:
            raise ValueError("raw Rinbo ABAD validity must have length 6")
        policy_positions = [0.0] * 6
        policy_validity = [False] * 6
        for servo_idx, raw in enumerate(raw_rinbo):
            numeric_valid = isinstance(raw, (int, float)) and math.isfinite(float(raw))
            in_range = (
                bool(raw_validity_rinbo[servo_idx])
                and numeric_valid
                and self.abad_encoder_min <= float(raw) <= self.abad_encoder_max
            )
            policy_idx = self.POLICY_ABAD_INDEX_BY_RINBO_SERVO[servo_idx]
            if numeric_valid:
                policy_positions[policy_idx] = (
                    (float(raw) - self.abad_encoder_zero_rinbo_order[servo_idx])
                    * self.abad_sign_rinbo_order[servo_idx]
                    / self.abad_encoder_counts_per_rad
                )
            policy_validity[policy_idx] = bool(in_range and self.abad_encoder_calibration_verified)
        return policy_positions, policy_validity

    def _joint_state_source_stamp(self, rinbo_state_msg):
        """Preserve the hardware source stamp on the explicit V2 route."""
        if self.publish_abad_joint_states:
            stamp = getattr(getattr(rinbo_state_msg, "header", None), "stamp", None)
            if stamp is not None:
                seconds = float(getattr(stamp, "sec", 0)) + 1.0e-9 * float(
                    getattr(stamp, "nanosec", 0)
                )
                if math.isfinite(seconds) and seconds > 0.0:
                    return stamp
        return self.node.get_clock().now().to_msg()

    def _estimate_abad_policy_velocities(
        self, policy_positions: list[float], state_time: float
    ) -> list[float]:
        if self._prev_abad_positions_policy is None or self._prev_abad_state_time is None:
            self._prev_abad_positions_policy = list(policy_positions)
            self._prev_abad_state_time = state_time
            return [0.0] * 6
        previous = list(self._prev_abad_positions_policy)
        dt = state_time - self._prev_abad_state_time
        self._prev_abad_positions_policy = list(policy_positions)
        self._prev_abad_state_time = state_time
        if dt <= 1.0e-6 or dt > self.abad_velocity_max_dt_s:
            return [0.0] * 6
        limit = self.abad_velocity_clip_rad_s
        raw_velocity = [
            max(-limit, min(limit, (float(policy_positions[i]) - float(previous[i])) / dt))
            for i in range(6)
        ]
        alpha = self.abad_velocity_filter_alpha
        return [
            alpha * raw_velocity[i] + (1.0 - alpha) * self.latest_abad_velocities_policy[i]
            for i in range(6)
        ]

    def _publish_joint_feedback_status_v2(self, source_stamp) -> None:
        publisher = getattr(self, "joint_feedback_status_pub_v2", None)
        if not self.publish_abad_joint_states or publisher is None:
            return
        array = self.DiagnosticArray()
        array.header.stamp = source_stamp
        names = self.main_joint_names_policy_order + self.abad_joint_names_policy_order
        statuses = []
        for name, valid in zip(names, self.latest_joint_validity_policy, strict=True):
            status = self.DiagnosticStatus()
            status.name = f"redrhex_joint_feedback_v2/{name}"
            status.hardware_id = "rinbo"
            status.level = self.DiagnosticStatus.OK if valid else self.DiagnosticStatus.ERROR
            status.message = "valid" if valid else "calibration or raw feedback unverified"
            status.values = [
                self.KeyValue(key="joint_name", value=name),
                self.KeyValue(key="valid", value=str(bool(valid)).lower()),
            ]
            statuses.append(status)
        array.status = statuses
        publisher.publish(array)

    def _estimate_policy_velocities(self, policy_positions: list[float], state_time: float) -> list[float]:
        if self._prev_positions_policy is None or self._prev_state_time is None:
            self._prev_positions_policy = list(policy_positions)
            self._prev_state_time = state_time
            return list(self.latest_velocities_policy)

        prev_positions = list(self._prev_positions_policy)
        dt = state_time - self._prev_state_time
        self._prev_positions_policy = list(policy_positions)
        self._prev_state_time = state_time
        if dt <= 1.0e-6 or dt > self.main_velocity_max_dt_s:
            return [0.0] * 6

        raw_vel = [
            max(
                -self.main_velocity_clip_rad_s,
                min(self.main_velocity_clip_rad_s, (float(policy_positions[i]) - float(prev_positions[i])) / dt),
            )
            for i in range(6)
        ]
        alpha = self.main_velocity_filter_alpha
        return [
            alpha * raw_vel[i] + (1.0 - alpha) * self.latest_velocities_policy[i]
            for i in range(6)
        ]
