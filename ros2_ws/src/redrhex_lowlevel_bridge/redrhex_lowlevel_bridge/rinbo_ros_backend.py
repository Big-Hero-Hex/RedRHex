from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np
from sensor_msgs.msg import JointState

from redrhex_msgs.msg import RedRhexMotorState

from .bridge_base import LowLevelBridgeBase

try:
    from rinbo_msgs.msg import LegCmd, MotorCmdStamped, MotorStateStamped, PowerStateStamped
except Exception:  # pragma: no cover - rinbo_msgs exists on Jetson, not always on dev machine.
    LegCmd = None
    MotorCmdStamped = None
    MotorStateStamped = None
    PowerStateStamped = None


POLICY_TO_RINBO_LEG = [3, 4, 5, 0, 1, 2]  # LF, LM, LR, RF, RM, RR
RINBO_LEG_FIELDS = ["l1", "l2", "l3", "r1", "r2", "r3"]
RINBO_SERVO_FIELDS = ["sl1", "sl2", "sl3", "sr1", "sr2", "sr3"]
MAIN_POLICY_NAMES = ["Revolute_15", "Revolute_7", "Revolute_12", "Revolute_18", "Revolute_23", "Revolute_24"]
ABAD_POLICY_NAMES = ["Revolute_14", "Revolute_6", "Revolute_11", "Revolute_17", "Revolute_22", "Revolute_21"]
RINBO_TO_POLICY = [3, 4, 5, 0, 1, 2]


class RinboRosBackend(LowLevelBridgeBase):
    def __init__(
        self,
        node,
        allow_enable: bool = False,
        command_topic: str = "/motor/command",
        preview_topic: str = "/redrhex/rinbo_motor_command_preview",
        motor_state_topic: str = "/motor/state",
        power_state_topic: str = "/power/state",
        main_max_pwm: float = 150.0,
        main_pwm_per_rad_s: float = 120.0,
        main_pwm_slew_per_s: float = 600.0,
        main_encoder_counts_per_rev: float = 54984.83,
        main_direction_positive_rinbo_order: Sequence[bool] | None = None,
        servo_zero_encoder_rinbo_order: Sequence[int] | None = None,
        servo_counts_per_rad: float = 1000.0,
        servo_min_encoder: int = 0,
        servo_max_encoder: int = 4095,
        servo_control_mode: int = 2,
        require_power_state_when_enabled: bool = True,
        max_main_channel_current_a: float = 3.0,
        max_bus_current_a: float = 12.0,
        min_bus_voltage_v: float | None = None,
        max_bus_voltage_v: float | None = None,
        current_trip_latch: bool = True,
        command_timeout_s: float = 0.10,
    ) -> None:
        if MotorCmdStamped is None:
            raise RuntimeError("rinbo_msgs is not available. Source ~/rinbo_ros_ws/install/setup.bash first.")
        self.node = node
        self.allow_enable = bool(allow_enable)
        self.command_topic = command_topic
        self.preview_topic = preview_topic
        self.main_max_pwm = float(main_max_pwm)
        self.main_pwm_per_rad_s = float(main_pwm_per_rad_s)
        self.main_pwm_slew_per_s = float(main_pwm_slew_per_s)
        self.counts_per_rad = float(main_encoder_counts_per_rev) / (2.0 * math.pi)
        self.direction_positive = list(main_direction_positive_rinbo_order or [True, True, True, False, False, False])
        self.servo_zero = list(servo_zero_encoder_rinbo_order or [740, 2565, 3283, 1944, 2071, 989])
        self.servo_counts_per_rad = float(servo_counts_per_rad)
        self.servo_min = int(servo_min_encoder)
        self.servo_max = int(servo_max_encoder)
        self.servo_control_mode = int(servo_control_mode)
        self.require_power_state_when_enabled = bool(require_power_state_when_enabled)
        self.max_main_current = float(max_main_channel_current_a)
        self.max_bus_current = float(max_bus_current_a)
        self.min_bus_voltage = min_bus_voltage_v
        self.max_bus_voltage = max_bus_voltage_v
        self.current_trip_latch = bool(current_trip_latch)
        self.command_timeout_s = float(command_timeout_s)

        self.cmd_pub = node.create_publisher(MotorCmdStamped, command_topic, 10)
        self.preview_pub = node.create_publisher(MotorCmdStamped, preview_topic, 10)
        self.joint_state_pub = node.create_publisher(JointState, "/joint_states", 10)
        self.feedback_pub = node.create_publisher(RedRhexMotorState, "/motor_feedback", 10)
        self.motor_state_sub = node.create_subscription(MotorStateStamped, motor_state_topic, self._on_motor_state, 20)
        self.power_state_sub = node.create_subscription(PowerStateStamped, power_state_topic, self._on_power_state, 20)

        self.connected = False
        self.seq = 0
        self.last_command_time = 0.0
        self.last_enabled = False
        self.last_pwm = np.zeros(6, dtype=np.float64)
        self.last_pwm_time = time.monotonic()
        self.power_state = None
        self.last_power_time = 0.0
        self.trip_latched = False
        self.trip_reason = ""
        self.prev_positions = None
        self.prev_motor_time = 0.0

    def connect(self) -> None:
        self.connected = True
        self.node.get_logger().warn(
            f"BioRoLa/Rinbo backend connected. allow_enable={self.allow_enable}; command_topic={self.command_topic}"
        )

    def send_motor_command(self, cmd) -> None:
        now = time.monotonic()
        rinbo_msg = self._convert_command(cmd, now)
        self.last_command_time = now
        self.last_enabled = bool(cmd.enable and not self.trip_latched and self.allow_enable)
        self.preview_pub.publish(rinbo_msg)
        if self.allow_enable and not self.trip_latched:
            self.cmd_pub.publish(rinbo_msg)
        elif cmd.enable:
            self.node.get_logger().warn(
                "Preview-only: command requested enable but rinbo.allow_enable=false or power trip active.",
                throttle_duration_sec=1.0,
            )

    def _convert_command(self, cmd, now: float):
        msg = MotorCmdStamped()
        msg.header.seq = self.seq
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.servo_control_mode = self.servo_control_mode

        requested_enable = bool(cmd.enable)
        safe_enable = requested_enable and self.allow_enable and not self.trip_latched
        if requested_enable:
            self._check_power_for_enable(now)
        if self.trip_latched:
            safe_enable = False

        target_vel_policy = np.asarray(cmd.target_velocity_rad_s[:6], dtype=np.float64)
        target_pwm_policy = np.clip(target_vel_policy * self.main_pwm_per_rad_s, -self.main_max_pwm, self.main_max_pwm)
        dt = max(now - self.last_pwm_time, 1e-4)
        pwm_step = self.main_pwm_slew_per_s * dt
        target_pwm_policy = self.last_pwm + np.clip(target_pwm_policy - self.last_pwm, -pwm_step, pwm_step)
        self.last_pwm = target_pwm_policy.copy()
        self.last_pwm_time = now

        for rinbo_i, field in enumerate(RINBO_LEG_FIELDS):
            policy_i = RINBO_TO_POLICY[rinbo_i]
            pwm = float(target_pwm_policy[policy_i])
            leg = getattr(msg, field)
            leg.enable = bool(safe_enable and abs(pwm) > 1e-6)
            positive_dir = self.direction_positive[rinbo_i]
            leg.direction = bool(positive_dir if pwm >= 0.0 else not positive_dir)
            leg.voltage = float(abs(pwm))
            leg.state = 0
            leg.reset_position = False

        abad_policy = np.asarray(cmd.target_position_rad[6:12], dtype=np.float64)
        abad_rinbo = abad_policy[POLICY_TO_RINBO_LEG]
        for rinbo_i, field in enumerate(RINBO_SERVO_FIELDS):
            servo = getattr(msg, field)
            encoder = int(round(self.servo_zero[rinbo_i] + abad_rinbo[rinbo_i] * self.servo_counts_per_rad))
            servo.position_encoder = max(self.servo_min, min(self.servo_max, encoder))
        return msg

    def _check_power_for_enable(self, now: float) -> None:
        if self.require_power_state_when_enabled and (self.power_state is None or now - self.last_power_time > 0.25):
            self._trip("missing /power/state while command is enabled")
            return
        if self.power_state is None:
            return
        currents = self._main_currents()
        bus_current = abs(float(getattr(self.power_state, "i_7", 0.0)))
        bus_voltage = float(getattr(self.power_state, "v_7", 0.0))
        if currents and max(abs(v) for v in currents) > self.max_main_current:
            self._trip(f"main channel current too high: {currents}")
        if bus_current > self.max_bus_current:
            self._trip(f"bus current too high: {bus_current:.3f}A")
        if self.min_bus_voltage is not None and bus_voltage < float(self.min_bus_voltage):
            self._trip(f"bus voltage too low: {bus_voltage:.3f}V")
        if self.max_bus_voltage is not None and bus_voltage > float(self.max_bus_voltage):
            self._trip(f"bus voltage too high: {bus_voltage:.3f}V")

    def _trip(self, reason: str) -> None:
        self.trip_reason = reason
        self.trip_latched = bool(self.current_trip_latch)
        self.node.get_logger().error(f"POWER SAFETY TRIP: {reason}")
        self._publish_disabled()

    def _publish_disabled(self) -> None:
        msg = MotorCmdStamped()
        msg.header.seq = self.seq
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.servo_control_mode = self.servo_control_mode
        for field in RINBO_LEG_FIELDS:
            leg = getattr(msg, field)
            leg.enable = False
            leg.voltage = 0.0
            leg.direction = True
            leg.state = 0
            leg.reset_position = False
        for rinbo_i, field in enumerate(RINBO_SERVO_FIELDS):
            getattr(msg, field).position_encoder = int(self.servo_zero[rinbo_i])
        self.preview_pub.publish(msg)
        if self.allow_enable:
            self.cmd_pub.publish(msg)
        self.last_enabled = False
        self.last_pwm[:] = 0.0

    def _on_power_state(self, msg) -> None:
        self.power_state = msg
        self.last_power_time = time.monotonic()
        if self.allow_enable and self.last_enabled:
            self._check_power_for_enable(self.last_power_time)

    def _on_motor_state(self, msg) -> None:
        now = time.monotonic()
        rinbo_positions = np.array(
            [msg.l1.position, msg.l2.position, msg.l3.position, msg.r1.position, msg.r2.position, msg.r3.position],
            dtype=np.float64,
        )
        policy_positions = rinbo_positions[POLICY_TO_RINBO_LEG] / self.counts_per_rad
        if self.prev_positions is None:
            policy_vel = np.zeros(6, dtype=np.float64)
        else:
            dt = max(now - self.prev_motor_time, 1e-4)
            policy_vel = (policy_positions - self.prev_positions) / dt
        self.prev_positions = policy_positions.copy()
        self.prev_motor_time = now

        js = JointState()
        js.header.stamp = self.node.get_clock().now().to_msg()
        js.name = MAIN_POLICY_NAMES
        js.position = [float(v) for v in policy_positions]
        js.velocity = [float(v) for v in policy_vel]
        self.joint_state_pub.publish(js)

        fb = RedRhexMotorState()
        fb.header.stamp = js.header.stamp
        fb.joint_names = MAIN_POLICY_NAMES + ABAD_POLICY_NAMES
        fb.position_rad = [float(v) for v in policy_positions] + [0.0] * 6
        fb.velocity_rad_s = [float(v) for v in policy_vel] + [0.0] * 6
        fb.effort_nm = [0.0] * 12
        currents = self._main_currents()
        fb.current_a = [float(v) for v in currents] + [0.0] * 6
        fb.temperature_c = [0.0] * 12
        fb.fault = [False] * 12
        self.feedback_pub.publish(fb)

    def _main_currents(self) -> list[float]:
        if self.power_state is None:
            return []
        return [
            float(getattr(self.power_state, "i_4", 0.0)),
            float(getattr(self.power_state, "i_5", 0.0)),
            float(getattr(self.power_state, "i_6", 0.0)),
            float(getattr(self.power_state, "i_1", 0.0)),
            float(getattr(self.power_state, "i_2", 0.0)),
            float(getattr(self.power_state, "i_3", 0.0)),
        ]

    def tick(self) -> None:
        now = time.monotonic()
        if self.last_enabled and now - self.last_command_time > self.command_timeout_s:
            self.node.get_logger().error("Command watchdog timeout. Sending disabled command.")
            self._publish_disabled()

    def power_trip_active(self) -> bool:
        return self.trip_latched

    def clear_power_trip(self) -> tuple[bool, str]:
        if self.allow_enable:
            return False, "refuse to clear while rinbo.allow_enable=true"
        if self.power_state is not None and self._main_currents():
            if max(abs(v) for v in self._main_currents()) > self.max_main_current * 0.5:
                return False, "current still not back to a quiet level"
        self.trip_latched = False
        self.trip_reason = ""
        return True, "power trip latch cleared"

    def is_alive(self) -> bool:
        return self.connected

    def diagnostics(self) -> dict[str, str]:
        values = {
            "backend": "biorola_ros",
            "allow_enable": str(self.allow_enable),
            "power_trip": str(self.trip_latched),
            "trip_reason": self.trip_reason,
            "last_pwm": np.array2string(self.last_pwm, precision=2, suppress_small=True),
        }
        if self.power_state is not None:
            values["main_currents_a"] = str([round(v, 3) for v in self._main_currents()])
            values["bus_current_a"] = f"{float(getattr(self.power_state, 'i_7', 0.0)):.3f}"
            values["bus_voltage_v"] = f"{float(getattr(self.power_state, 'v_7', 0.0)):.3f}"
            values["power_state_age_s"] = f"{time.monotonic() - self.last_power_time:.3f}"
        else:
            values["power_state"] = "missing"
        return values

    def shutdown(self) -> None:
        self._publish_disabled()
        self.connected = False
