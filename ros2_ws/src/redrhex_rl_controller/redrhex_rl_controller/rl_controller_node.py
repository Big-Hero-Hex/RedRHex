from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32MultiArray, String

from redrhex_msgs.msg import RedRhexMotorCommand, RedRhexMotorState

from .action_decoder import ActionDecoder
from .observation_builder import ObservationBuilder
from .policy_onnx_runner import PolicyONNXRunner
from .redrhex_contract import CONTROL_DT, EXPECTED_ACTION_DIM, EXPECTED_OBS_DIM, POLICY_HZ
from .safety_filter import SafetyFilter
from .state_machine import RedRhexState, RedRhexStateMachine


def _stamp_to_monotonic(_msg_stamp) -> float:
    # Hardware clocks may not be synchronized. Use local receive time for timeout checks.
    return time.monotonic()


class RedRhexRLControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("redrhex_rl_controller")
        self._declare_parameters()

        self.policy_hz = float(self.get_parameter("policy.policy_hz").value or POLICY_HZ)
        self.control_dt = 1.0 / max(self.policy_hz, 1.0)
        self.require_lowlevel_heartbeat = bool(self.get_parameter("safety.require_lowlevel_heartbeat").value)
        self.estop_default = bool(self.get_parameter("safety.estop_on_start").value)
        self.enable_policy = bool(self.get_parameter("policy.enable_policy_on_start").value)
        self.enable_motors = bool(self.get_parameter("safety.enable_motor_output_on_start").value)
        self.estop = self.estop_default

        onnx_path = str(self.get_parameter("policy.onnx_path").value)
        self.runner = PolicyONNXRunner(
            onnx_path=onnx_path,
            expected_obs_dim=int(self.get_parameter("policy.expected_obs_dim").value),
            expected_action_dim=int(self.get_parameter("policy.expected_action_dim").value),
            use_cuda=bool(self.get_parameter("policy.use_cuda").value),
            use_tensorrt=bool(self.get_parameter("policy.use_tensorrt").value),
        )
        self.get_logger().info("Loaded ONNX policy:\n" + self.runner.describe())

        self.builder = ObservationBuilder(
            cmd_timeout_s=float(self.get_parameter("safety.cmd_timeout_s").value),
            base_lin_vel_source=str(self.get_parameter("observation.base_lin_vel_source").value),
            abad_feedback_source=str(self.get_parameter("observation.abad_feedback_source").value),
        )
        self.decoder = ActionDecoder(
            action_clip=float(self.get_parameter("safety.action_clip").value),
            main_drive_vel_limit_rad_s=float(self.get_parameter("safety.main_drive_vel_limit_rad_s").value),
            main_drive_slew_rate_rad_s2=float(self.get_parameter("safety.main_drive_slew_rate_rad_s2").value),
            abad_pos_limit_rad=float(self.get_parameter("safety.abad_pos_limit_rad").value),
            abad_slew_rate_rad_s=float(self.get_parameter("safety.abad_slew_rate_rad_s").value),
            init_stand_max_main_drive_vel_rad_s=float(
                self.get_parameter("safety.init_stand_max_main_drive_vel_rad_s").value
            ),
            enable_forward_bias=bool(self.get_parameter("decoder.enable_forward_bias").value),
            main_sign_correction=self.get_parameter("decoder.main_sign_correction").value,
            abad_sign_correction=self.get_parameter("decoder.abad_sign_correction").value,
            main_zero_offset_rad=self.get_parameter("decoder.main_zero_offset_rad").value,
            abad_zero_offset_rad=self.get_parameter("decoder.abad_zero_offset_rad").value,
        )
        self.safety = SafetyFilter(
            max_abs_roll_rad=float(self.get_parameter("safety.max_abs_roll_rad").value),
            max_abs_pitch_rad=float(self.get_parameter("safety.max_abs_pitch_rad").value),
            sensor_timeout_s=float(self.get_parameter("safety.sensor_timeout_s").value),
            heartbeat_timeout_s=float(self.get_parameter("safety.heartbeat_timeout_s").value),
            action_clip=float(self.get_parameter("safety.action_clip").value),
            main_drive_vel_limit_rad_s=float(self.get_parameter("safety.main_drive_vel_limit_rad_s").value),
            abad_pos_limit_rad=float(self.get_parameter("safety.abad_pos_limit_rad").value),
            max_motor_temperature_c=float(self.get_parameter("safety.max_motor_temperature_c").value),
            max_motor_current_a=float(self.get_parameter("safety.max_motor_current_a").value),
            control_deadline_factor=float(self.get_parameter("safety.control_deadline_factor").value),
            control_deadline_miss_limit=int(self.get_parameter("safety.control_deadline_miss_limit").value),
        )
        self.safety.update_estop(self.estop)
        self.sm = RedRhexStateMachine()

        self.last_heartbeat_time = 0.0
        self.last_loop_time = time.monotonic()
        self.last_raw_action: np.ndarray | None = None
        self.last_safe_action = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)
        self.init_stand_started = 0.0

        self._make_ros_io()
        self.timer = self.create_timer(self.control_dt, self._control_tick)

    def _declare_parameters(self) -> None:
        self.declare_parameter("policy.onnx_path", "/home/jetson/RedRHex/policy.onnx")
        self.declare_parameter("policy.use_cuda", False)
        self.declare_parameter("policy.use_tensorrt", False)
        self.declare_parameter("policy.expected_obs_dim", EXPECTED_OBS_DIM)
        self.declare_parameter("policy.expected_action_dim", EXPECTED_ACTION_DIM)
        self.declare_parameter("policy.policy_hz", 0.0)
        self.declare_parameter("policy.enable_policy_on_start", False)
        self.declare_parameter("safety.estop_on_start", True)
        self.declare_parameter("safety.enable_motor_output_on_start", False)
        self.declare_parameter("safety.require_lowlevel_heartbeat", False)
        self.declare_parameter("safety.max_abs_roll_rad", 0.7)
        self.declare_parameter("safety.max_abs_pitch_rad", 0.7)
        self.declare_parameter("safety.cmd_timeout_s", 0.25)
        self.declare_parameter("safety.sensor_timeout_s", 0.10)
        self.declare_parameter("safety.heartbeat_timeout_s", 0.10)
        self.declare_parameter("safety.action_clip", 1.0)
        self.declare_parameter("safety.main_drive_vel_limit_rad_s", 1.0)
        self.declare_parameter("safety.main_drive_slew_rate_rad_s2", 4.0)
        self.declare_parameter("safety.abad_pos_limit_rad", 0.15)
        self.declare_parameter("safety.abad_slew_rate_rad_s", 1.0)
        self.declare_parameter("safety.init_stand_max_main_drive_vel_rad_s", 0.5)
        self.declare_parameter("safety.max_motor_temperature_c", 70.0)
        self.declare_parameter("safety.max_motor_current_a", 3.0)
        self.declare_parameter("safety.control_deadline_factor", 2.0)
        self.declare_parameter("safety.control_deadline_miss_limit", 5)
        self.declare_parameter("safety.init_stand_min_time_s", 2.0)
        self.declare_parameter("safety.init_stand_pose_tol_rad", 0.25)
        self.declare_parameter("observation.base_lin_vel_source", "zero")
        self.declare_parameter("observation.abad_feedback_source", "commanded")
        self.declare_parameter("decoder.enable_forward_bias", True)
        self.declare_parameter("decoder.main_sign_correction", [1.0] * 6)
        self.declare_parameter("decoder.abad_sign_correction", [1.0] * 6)
        self.declare_parameter("decoder.main_zero_offset_rad", [0.0] * 6)
        self.declare_parameter("decoder.abad_zero_offset_rad", [0.0] * 6)

    def _make_ros_io(self) -> None:
        self.create_subscription(Imu, "/imu/data", self._on_imu, 20)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 20)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Bool, "/estop", self._on_estop, 10)
        self.create_subscription(Bool, "/redrhex/enable_policy", self._on_enable_policy, 10)
        self.create_subscription(Bool, "/redrhex/enable_motors", self._on_enable_motors, 10)
        self.create_subscription(Bool, "/redrhex/lowlevel_heartbeat", self._on_lowlevel_heartbeat, 10)
        self.create_subscription(RedRhexMotorState, "/motor_feedback", self._on_motor_feedback, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)

        self.obs_pub = self.create_publisher(Float32MultiArray, "/redrhex/observation", 10)
        self.raw_action_pub = self.create_publisher(Float32MultiArray, "/redrhex/policy_action_raw", 10)
        self.safe_action_pub = self.create_publisher(Float32MultiArray, "/redrhex/policy_action_safe", 10)
        self.motor_cmd_pub = self.create_publisher(RedRhexMotorCommand, "/redrhex/motor_commands", 10)
        self.state_pub = self.create_publisher(String, "/redrhex/state_machine_state", 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/redrhex/diagnostics", 10)

    def _on_imu(self, msg: Imu) -> None:
        self.builder.update_imu(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            _stamp_to_monotonic(msg.header.stamp),
        )

    def _on_joint_states(self, msg: JointState) -> None:
        self.builder.update_joint_states(msg.name, msg.position, msg.velocity, _stamp_to_monotonic(msg.header.stamp))

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.builder.update_command(msg.linear.x, msg.linear.y, msg.angular.z, time.monotonic())

    def _on_estop(self, msg: Bool) -> None:
        self.estop = bool(msg.data)
        self.safety.update_estop(self.estop)

    def _on_enable_policy(self, msg: Bool) -> None:
        self.enable_policy = bool(msg.data)

    def _on_enable_motors(self, msg: Bool) -> None:
        self.enable_motors = bool(msg.data)

    def _on_lowlevel_heartbeat(self, _msg: Bool) -> None:
        self.last_heartbeat_time = time.monotonic()

    def _on_motor_feedback(self, msg: RedRhexMotorState) -> None:
        self.safety.update_motor_state(msg.current_a, msg.temperature_c, msg.fault)

    def _on_odom(self, msg: Odometry) -> None:
        self.builder.update_odom_body_velocity(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        )

    def _control_tick(self) -> None:
        now = time.monotonic()
        dt = now - self.last_loop_time
        self.last_loop_time = now
        self.safety.last_loop_dt = dt
        self.builder.advance_phase(dt)

        status = self.builder.status(now, float(self.get_parameter("safety.sensor_timeout_s").value))
        obs = None
        raw_action = None
        decoded = self.decoder.disabled_command()
        safety_ok = False
        safety_reasons: list[str] = []

        try:
            if status.ready:
                obs = self.builder.build(now)
                raw_action = self.runner.run(obs)
                decoded = self.decoder.decode(raw_action, self.builder.cmd, dt)
                self.builder.update_commanded_abad(decoded.target_position_rad[6:], dt)
                safety_status = self.safety.check(
                    obs,
                    raw_action,
                    decoded,
                    self.builder.imu_quat_xyzw,
                    self.builder.last_imu_time,
                    self.builder.last_joint_time,
                    self.last_heartbeat_time,
                    now,
                    self.control_dt,
                    self.require_lowlevel_heartbeat,
                )
                safety_ok = safety_status.ok
                safety_reasons = safety_status.reasons
            else:
                safety_reasons = list(status.reasons)
                safety_ok = False
        except Exception as exc:
            safety_ok = False
            safety_reasons = [str(exc)]
            self.get_logger().warn(f"Control tick failed: {exc}", throttle_duration_sec=1.0)

        init_done = self._init_stand_done(now)
        state = self.sm.update(
            sensors_ready=status.ready,
            safety_ok=safety_ok,
            enable_motors=self.enable_motors,
            enable_policy=self.enable_policy,
            init_stand_complete=init_done,
            safety_reason="; ".join(safety_reasons),
        )

        enable_output = False
        if state == RedRhexState.INIT_STAND:
            if self.init_stand_started <= 0.0:
                self.init_stand_started = now
            decoded = self.decoder.init_stand_command(self.builder.main_pos, dt)
            enable_output = self.enable_motors
        elif state == RedRhexState.POLICY_RUN and decoded is not None:
            enable_output = self.enable_motors and self.enable_policy and safety_ok
        else:
            decoded = self.decoder.disabled_command()
            enable_output = False

        self.last_raw_action = raw_action
        if decoded is not None:
            self.last_safe_action = decoded.safe_action
            if state == RedRhexState.POLICY_RUN:
                self.builder.update_last_actions(decoded.safe_action)

        self._publish_debug(obs, raw_action, decoded)
        self._publish_command(decoded, enable_output)
        self._publish_state_and_diag(state, status.reasons, safety_reasons, decoded)

    def _init_stand_done(self, now: float) -> bool:
        if self.init_stand_started <= 0.0:
            return False
        min_time = float(self.get_parameter("safety.init_stand_min_time_s").value)
        tol = float(self.get_parameter("safety.init_stand_pose_tol_rad").value)
        elapsed_ok = now - self.init_stand_started >= min_time
        pose_err = np.max(np.abs(np.arctan2(np.sin(self.builder.main_pos - np.array([0.785398] * 3 + [-0.785398] * 3)), np.cos(self.builder.main_pos - np.array([0.785398] * 3 + [-0.785398] * 3)))))
        return bool(elapsed_ok and pose_err < tol)

    def _publish_debug(self, obs, raw_action, decoded) -> None:
        if obs is not None:
            msg = Float32MultiArray()
            msg.data = [float(v) for v in obs]
            self.obs_pub.publish(msg)
        if raw_action is not None:
            msg = Float32MultiArray()
            msg.data = [float(v) for v in raw_action]
            self.raw_action_pub.publish(msg)
        if decoded is not None:
            msg = Float32MultiArray()
            msg.data = [float(v) for v in decoded.safe_action]
            self.safe_action_pub.publish(msg)

    def _publish_command(self, decoded, enable: bool) -> None:
        msg = RedRhexMotorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = decoded.joint_names
        msg.target_position_rad = [float(v) for v in decoded.target_position_rad]
        msg.target_velocity_rad_s = [float(v) for v in decoded.target_velocity_rad_s]
        msg.kp = [float(v) for v in decoded.kp]
        msg.kd = [float(v) for v in decoded.kd]
        msg.effort_limit_nm = [float(v) for v in decoded.effort_limit_nm]
        msg.enable = bool(enable)
        msg.mode = int(decoded.mode)
        self.motor_cmd_pub.publish(msg)

    def _publish_state_and_diag(self, state, sensor_reasons, safety_reasons, decoded) -> None:
        state_msg = String()
        state_msg.data = f"{state.value}: {self.sm.reason}"
        self.state_pub.publish(state_msg)

        diag = DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "redrhex_rl_controller"
        status.hardware_id = "redrhex_jetson"
        status.level = DiagnosticStatus.OK if state in (RedRhexState.POLICY_READY, RedRhexState.POLICY_RUN, RedRhexState.INIT_STAND, RedRhexState.MOTOR_IDLE) else DiagnosticStatus.WARN
        status.message = self.sm.reason
        status.values = [
            KeyValue(key="state", value=state.value),
            KeyValue(key="estop", value=str(self.estop)),
            KeyValue(key="enable_motors", value=str(self.enable_motors)),
            KeyValue(key="enable_policy", value=str(self.enable_policy)),
            KeyValue(key="policy_hz", value=f"{self.policy_hz:.2f}"),
            KeyValue(key="last_loop_dt_s", value=f"{self.safety.last_loop_dt:.5f}"),
            KeyValue(key="sensor_reasons", value="; ".join(sensor_reasons)),
            KeyValue(key="safety_reasons", value="; ".join(safety_reasons)),
            KeyValue(key="decoder_mode", value=str(decoded.debug.get("mode_name", "")) if decoded else ""),
        ]
        diag.status.append(status)
        self.diag_pub.publish(diag)


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RedRhexRLControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
