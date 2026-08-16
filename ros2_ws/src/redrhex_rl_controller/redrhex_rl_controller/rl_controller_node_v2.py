"""Explicit Sensor-Only Distillation V2 ROS2 deployment controller."""

from __future__ import annotations

import math
import traceback

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32MultiArray, String

from redrhex_msgs.msg import RedRhexMotorCommand, RedRhexMotorState

try:
    from rclpy._rclpy_pybind11 import RCLError
except Exception:  # pragma: no cover - depends on rclpy version
    RCLError = RuntimeError

from .action_decoder import DecodedMotorCommand
from .action_decoder_v2 import ForwardResidualActionDecoderV2
from .deployment_guard_v2 import (
    DeploymentGuardV2,
    action_target_envelope_matches_v2,
    warmup_complete_v2,
)
from .deployment_route import SENSOR_ONLY_CONTRACT_ID_V2
from .observation_builder_v2 import (
    ABAD_JOINT_NAMES_V2,
    MAIN_JOINT_NAMES_V2,
    ObservationStatusV2,
    SensorObservationBuilderV2,
)
from .policy_onnx_runner_v2 import SensorPolicyONNXRunnerV2
from .safety_filter import SafetyFilter, SafetyState
from .state_machine import RedRhexState, RedRhexStateMachine, StateMachineInputs


def _declare_get(node: Node, name: str, default):
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


def _source_stamp_s(msg: object) -> float:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    value = float(getattr(stamp, "sec", 0)) + 1.0e-9 * float(
        getattr(stamp, "nanosec", 0)
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("V2 deployment requires a positive finite source timestamp")
    return value


def _roll_pitch_from_projected_gravity(gravity: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(gravity, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise ValueError("projected gravity is invalid")
    x, y, z = vector / norm
    return math.atan2(-y, -z), math.atan2(x, math.sqrt(y * y + z * z))


class RedRhexRLControllerNodeV2(Node):
    """Run only the hash-bound 60x36 physical-sensor policy route."""

    def __init__(self) -> None:
        super().__init__("redrhex_rl_controller_v2")

        contract_id = str(
            _declare_get(self, "policy.contract_id", SENSOR_ONLY_CONTRACT_ID_V2)
        )
        if contract_id != SENSOR_ONLY_CONTRACT_ID_V2:
            raise ValueError(
                f"V2 node requires contract_id={SENSOR_ONLY_CONTRACT_ID_V2!r}, "
                f"got {contract_id!r}"
            )
        self.onnx_path = str(
            _declare_get(
                self,
                "policy.onnx_path",
                "/home/jetson/redrhex_models/sensor_policy_v2.onnx",
            )
        )
        self.sidecar_path = str(
            _declare_get(
                self,
                "policy.sidecar_path",
                "/home/jetson/redrhex_models/sensor_policy_v2.onnx.json",
            )
        )
        self.expected_contract_hash = str(
            _declare_get(self, "policy.expected_contract_hash", "UNVERIFIED")
        )
        self.expected_action_contract_hash = str(
            _declare_get(self, "policy.expected_action_contract_hash", "UNVERIFIED")
        )
        self.expected_calibration_hash = str(
            _declare_get(self, "policy.expected_calibration_hash", "UNVERIFIED")
        )
        self.expected_checkpoint_hash = str(
            _declare_get(self, "policy.expected_checkpoint_hash", "UNVERIFIED")
        )
        self.use_cuda = bool(_declare_get(self, "policy.use_cuda", False))
        self.use_tensorrt = bool(_declare_get(self, "policy.use_tensorrt", False))
        self.policy_hz = float(_declare_get(self, "policy.policy_hz", 60.0))
        if self.policy_hz != 60.0:
            raise ValueError("Sensor V2 policy rate is fixed at 60 Hz")

        enable_policy_on_start = bool(
            _declare_get(self, "state_machine.enable_policy_on_start", False)
        )
        enable_motor_on_start = bool(
            _declare_get(self, "state_machine.enable_motor_output_on_start", False)
        )
        if enable_policy_on_start or enable_motor_on_start:
            raise ValueError("Sensor V2 policy and motor output must start disabled")
        self.init_stand_duration_s = float(
            _declare_get(self, "state_machine.init_stand_duration_s", 2.0)
        )
        self.warmup_duration_s = float(
            _declare_get(self, "state_machine.warmup_duration_s", 1.0)
        )
        self.require_history_ready = bool(
            _declare_get(self, "state_machine.require_history_ready", True)
        )
        if not self.require_history_ready:
            raise ValueError("Sensor V2 deployment requires history readiness")

        self.allow_motor_enable = bool(
            _declare_get(self, "hardware_gate.allow_motor_enable", False)
        )
        self.recorded_imu_evidence = bool(
            _declare_get(self, "hardware_gate.recorded_imu_evidence", False)
        )
        self.recorded_encoder_evidence = bool(
            _declare_get(self, "hardware_gate.recorded_encoder_evidence", False)
        )

        self.sensor_timeout_s = float(
            _declare_get(self, "safety.sensor_timeout_s", 0.10)
        )
        self.cmd_timeout_s = float(_declare_get(self, "safety.cmd_timeout_s", 0.25))
        self.require_motor_feedback = bool(
            _declare_get(self, "safety.require_motor_feedback", True)
        )
        self.require_lowlevel_heartbeat = bool(
            _declare_get(self, "safety.require_lowlevel_heartbeat", True)
        )
        command_limits = {
            "vx_min": float(_declare_get(self, "commands.vx_min", 0.0)),
            "vx_max": float(_declare_get(self, "commands.vx_max", 0.56)),
            "vy_min": float(_declare_get(self, "commands.vy_min", -0.08)),
            "vy_max": float(_declare_get(self, "commands.vy_max", 0.08)),
            "wz_min": float(_declare_get(self, "commands.wz_min", -0.10)),
            "wz_max": float(_declare_get(self, "commands.wz_max", 0.10)),
        }
        safety_cfg = {
            "sensor_timeout_s": self.sensor_timeout_s,
            "cmd_timeout_s": self.cmd_timeout_s,
            "motor_feedback_timeout_s": float(
                _declare_get(self, "safety.motor_feedback_timeout_s", 0.25)
            ),
            "heartbeat_timeout_s": float(
                _declare_get(self, "safety.heartbeat_timeout_s", 0.10)
            ),
            "max_abs_roll_rad": float(
                _declare_get(self, "safety.max_abs_roll_rad", 0.7)
            ),
            "max_abs_pitch_rad": float(
                _declare_get(self, "safety.max_abs_pitch_rad", 0.7)
            ),
            "action_clip": float(_declare_get(self, "safety.action_clip", 1.0)),
            "main_drive_vel_limit_rad_s": float(
                _declare_get(self, "safety.main_drive_vel_limit_rad_s", 9.0)
            ),
            "abad_pos_limit_rad": float(
                _declare_get(self, "safety.abad_pos_limit_rad", 0.7)
            ),
            "max_motor_temperature_c": float(
                _declare_get(self, "safety.max_motor_temperature_c", 70.0)
            ),
            "max_motor_current_a": float(
                _declare_get(self, "safety.max_motor_current_a", 20.0)
            ),
            "max_control_loop_dt_s": float(
                _declare_get(self, "safety.max_control_loop_dt_s", 0.03)
            ),
            "require_motor_feedback": self.require_motor_feedback,
            "require_lowlevel_heartbeat": self.require_lowlevel_heartbeat,
            "command_limits": command_limits,
        }
        self.safety_filter = SafetyFilter(safety_cfg)
        self.state_machine = RedRhexStateMachine(
            require_motor_feedback=self.require_motor_feedback,
            require_lowlevel_heartbeat=self.require_lowlevel_heartbeat,
        )

        self.policy_runner: SensorPolicyONNXRunnerV2 | None = None
        self.observation_builder: SensorObservationBuilderV2 | None = None
        self.action_decoder: ForwardResidualActionDecoderV2 | None = None
        self.policy_loaded = False
        self.configured_action_clip = float(safety_cfg["action_clip"])
        self.configured_main_velocity_limit_rad_s = float(
            safety_cfg["main_drive_vel_limit_rad_s"]
        )
        self.contract_action_clip: float | None = None
        self.contract_main_velocity_limit_rad_s: float | None = None
        self.action_target_envelope_compatible = False
        self.runtime_action_target_compatible = True
        self.hardware_tightening_event_count = 0
        self.last_hardware_action_clip_applied = False
        self.last_hardware_slew_applied = False
        self.last_hardware_velocity_limit_applied = False
        calibration_ready = False
        self.calibration_blockers: tuple[str, ...] = ("policy bundle not loaded",)
        try:
            self.policy_runner = SensorPolicyONNXRunnerV2(
                self.onnx_path,
                sidecar_path=self.sidecar_path,
                expected_contract_sha256=self.expected_contract_hash,
                expected_action_contract_sha256=self.expected_action_contract_hash,
                expected_calibration_sha256=self.expected_calibration_hash,
                expected_checkpoint_sha256=self.expected_checkpoint_hash,
                require_hardware_ready=self.allow_motor_enable,
                use_cuda=self.use_cuda,
                use_tensorrt=self.use_tensorrt,
            )
            observation_cfg = self._observation_config()
            observation_cfg["contract_sha256"] = self.expected_contract_hash
            observation_cfg["abad_neutral_position_rad"] = list(
                self.policy_runner.action_contract.abad_neutral_position_rad
            )
            self.observation_builder = SensorObservationBuilderV2(
                observation_cfg,
                contract=self.policy_runner.observation_contract,
            )
            expected_decoder_hash = str(
                _declare_get(self, "action.expected_decoder_hash", "UNVERIFIED")
            )
            if expected_decoder_hash != self.policy_runner.action_contract.decoder_sha256:
                raise ValueError("configured decoder hash does not match bundle action contract")
            decoder_limit = float(safety_cfg["main_drive_vel_limit_rad_s"])
            self.contract_action_clip = float(
                self.policy_runner.action_contract.action_clip
            )
            self.contract_main_velocity_limit_rad_s = float(
                self.policy_runner.action_contract.main_velocity_limit_rad_s
            )
            self.action_target_envelope_compatible = (
                action_target_envelope_matches_v2(
                    configured_action_clip=self.configured_action_clip,
                    configured_main_velocity_limit_rad_s=(
                        self.configured_main_velocity_limit_rad_s
                    ),
                    contract_action_clip=self.contract_action_clip,
                    contract_main_velocity_limit_rad_s=(
                        self.contract_main_velocity_limit_rad_s
                    ),
                )
            )
            self.action_decoder = ForwardResidualActionDecoderV2(
                self.policy_runner.action_contract,
                {
                    "action_clip": float(safety_cfg["action_clip"]),
                    "main_drive_vel_limit_rad_s": decoder_limit,
                    "main_drive_slew_rate_rad_s2": float(
                        _declare_get(
                            self,
                            "safety.main_drive_slew_rate_rad_s2",
                            120.0,
                        )
                    ),
                    "init_stand_main_drive_position_gain": float(
                        _declare_get(
                            self,
                            "action.init_stand_main_drive_position_gain",
                            3.0,
                        )
                    ),
                    "init_stand_max_main_drive_vel_rad_s": float(
                        _declare_get(
                            self,
                            "action.init_stand_max_main_drive_vel_rad_s",
                            1.5,
                        )
                    ),
                    "main_drive_kp": list(
                        _declare_get(self, "action.main_drive_kp", [0.0] * 6)
                    ),
                    "main_drive_kd": list(
                        _declare_get(self, "action.main_drive_kd", [50.0] * 6)
                    ),
                    "abad_kp": list(_declare_get(self, "action.abad_kp", [40.0] * 6)),
                    "abad_kd": list(_declare_get(self, "action.abad_kd", [4.0] * 6)),
                },
            )
            calibration_ready = self.policy_runner.calibration_profile.hardware_ready
            self.calibration_blockers = (
                self.policy_runner.calibration_profile.readiness_blockers
            )
            self.policy_loaded = True
            self.get_logger().info(
                f"Loaded Sensor V2 policy: {self.policy_runner.io_info}"
            )
        except Exception as exc:
            self.policy_runner = None
            self.observation_builder = None
            self.action_decoder = None
            self.get_logger().error(f"Failed to load Sensor V2 policy bundle: {exc}")

        self.deployment_guard = DeploymentGuardV2(
            allow_motor_enable=self.allow_motor_enable,
            calibration_hardware_ready=calibration_ready,
            action_target_envelope_compatible=(
                self.action_target_envelope_compatible
            ),
        )
        if not self.deployment_guard.hardware_authorized:
            self.get_logger().warn(
                "Motor output is blocked: explicit authorization, hardware-ready "
                "calibration, and an exact bundle/hardware action envelope are required"
            )

        self.estop = False
        self.enable_policy = False
        self.enable_motor_output = False
        self.recover_requested = False
        self.last_imu_source_time: float | None = None
        self.last_joint_source_time: float | None = None
        self.last_validity_source_time: float | None = None
        self.last_motor_feedback_source_time: float | None = None
        self.last_lowlevel_heartbeat_time: float | None = None
        self.motor_temperatures: list[float] = []
        self.motor_currents: list[float] = []
        self.motor_faults: list[bool] = []
        self.last_loop_time: float | None = None
        self.state_enter_time = self._now_s()

        self.create_subscription(Imu, "/imu/data", self._on_imu, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        self.create_subscription(
            DiagnosticArray,
            "/redrhex/joint_feedback_status_v2",
            self._on_joint_validity,
            10,
        )
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Bool, "/estop", self._on_estop, 10)
        self.create_subscription(Bool, "/redrhex/enable_policy", self._on_enable_policy, 10)
        self.create_subscription(Bool, "/redrhex/enable_motors", self._on_enable_motors, 10)
        self.create_subscription(Bool, "/redrhex/recover", self._on_recover, 10)
        self.create_subscription(
            Bool,
            "/redrhex/lowlevel_heartbeat",
            self._on_lowlevel_heartbeat,
            10,
        )
        self.create_subscription(
            RedRhexMotorState,
            "/motor_feedback",
            self._on_motor_feedback,
            10,
        )

        self.obs_pub = self.create_publisher(
            Float32MultiArray, "/redrhex/observation_v2", 10
        )
        self.raw_action_pub = self.create_publisher(
            Float32MultiArray, "/redrhex/policy_action_raw_v2", 10
        )
        self.safe_action_pub = self.create_publisher(
            Float32MultiArray, "/redrhex/policy_action_safe_v2", 10
        )
        self.motor_cmd_pub = self.create_publisher(
            RedRhexMotorCommand, "/redrhex/motor_commands", 10
        )
        self.state_pub = self.create_publisher(
            String, "/redrhex/state_machine_state", 10
        )
        self.diag_pub = self.create_publisher(
            DiagnosticArray, "/redrhex/diagnostics_v2", 10
        )
        self.timer = self.create_timer(1.0 / self.policy_hz, self._control_tick)

    def _observation_config(self) -> dict[str, object]:
        return {
            "sensor_frame_dim": int(
                _declare_get(self, "observation.sensor_frame_dim", 36)
            ),
            "history_length": int(
                _declare_get(self, "observation.history_length", 60)
            ),
            "sample_rate_hz": float(
                _declare_get(self, "observation.sample_rate_hz", 60.0)
            ),
            "main_drive_joint_names": list(
                _declare_get(
                    self,
                    "observation.main_drive_joint_names",
                    list(MAIN_JOINT_NAMES_V2),
                )
            ),
            "abad_joint_names": list(
                _declare_get(
                    self,
                    "observation.abad_joint_names",
                    list(ABAD_JOINT_NAMES_V2),
                )
            ),
            "attitude_mode": str(
                _declare_get(self, "observation.attitude_mode", "")
            ),
            "imu_frame_id": str(
                _declare_get(self, "observation.imu_frame_id", "")
            ),
            "policy_body_frame_id": str(
                _declare_get(
                    self,
                    "observation.policy_body_frame_id",
                    "redrhex_policy_body",
                )
            ),
            "imu_mount_rpy_deg": list(
                _declare_get(self, "observation.imu_mount_rpy_deg", [0.0, 0.0, 0.0])
            ),
            "imu_mount_calibration_verified": bool(
                _declare_get(
                    self,
                    "observation.imu_mount_calibration_verified",
                    False,
                )
            ),
            "rest_gravity_verified": bool(
                _declare_get(self, "observation.rest_gravity_verified", False)
            ),
            "expected_rest_projected_gravity": list(
                _declare_get(
                    self,
                    "observation.expected_rest_projected_gravity",
                    [0.0, 0.0, -1.0],
                )
            ),
            "quaternion_norm_tolerance": float(
                _declare_get(
                    self,
                    "observation.quaternion_norm_tolerance",
                    0.02,
                )
            ),
            "causal_accel_correction_gain": float(
                _declare_get(
                    self,
                    "observation.causal_accel_correction_gain",
                    0.02,
                )
            ),
            "accel_magnitude_tolerance_ratio": float(
                _declare_get(
                    self,
                    "observation.accel_magnitude_tolerance_ratio",
                    0.25,
                )
            ),
            "gravity_magnitude_m_s2": float(
                _declare_get(
                    self,
                    "observation.gravity_magnitude_m_s2",
                    9.80665,
                )
            ),
            "sensor_timeout_s": self.sensor_timeout_s,
            "command_timeout_s": self.cmd_timeout_s,
            "min_channel_period_s": float(
                _declare_get(self, "observation.min_channel_period_s", 0.001)
            ),
            "max_sensor_source_skew_s": float(
                _declare_get(
                    self,
                    "observation.max_sensor_source_skew_s",
                    0.5 / self.policy_hz,
                )
            ),
            "max_history_period_error_ratio": float(
                _declare_get(
                    self,
                    "observation.max_history_period_error_ratio",
                    0.25,
                )
            ),
            "require_joint_validity": True,
        }

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_imu(self, msg: Imu) -> None:
        if self.observation_builder is None:
            return
        try:
            stamp = _source_stamp_s(msg)
            if self.observation_builder.update_imu(msg):
                self.last_imu_source_time = stamp
        except Exception as exc:
            self.get_logger().warn(f"Rejected V2 IMU event: {exc}")

    def _on_joint_states(self, msg: JointState) -> None:
        if self.observation_builder is None:
            return
        try:
            stamp = _source_stamp_s(msg)
            if self.observation_builder.update_joint_state(msg):
                self.last_joint_source_time = stamp
        except Exception as exc:
            self.get_logger().warn(f"Rejected V2 joint event: {exc}")

    def _on_joint_validity(self, msg: DiagnosticArray) -> None:
        if self.observation_builder is None:
            return
        try:
            stamp = _source_stamp_s(msg)
            if self.observation_builder.update_joint_validity_diagnostic(msg):
                self.last_validity_source_time = stamp
        except Exception as exc:
            self.get_logger().warn(f"Rejected V2 joint-validity event: {exc}")

    def _on_cmd_vel(self, msg: Twist) -> None:
        if self.observation_builder is None:
            return
        try:
            # geometry_msgs/Twist is unstamped; arrival time is used only for the
            # external command, never for IMU/encoder finite differences.
            self.observation_builder.update_command(msg, self._now_s())
        except Exception as exc:
            self.get_logger().warn(f"Rejected V2 command: {exc}")

    def _drop_enable_latches(self, reason: str) -> None:
        if self.enable_policy or self.enable_motor_output:
            self.get_logger().warn(f"Dropping enable latches: {reason}")
        self.enable_policy = False
        self.enable_motor_output = False

    def _policy_enable_allowed(self) -> bool:
        return bool(
            not self.estop
            and self.observation_builder is not None
            and self.observation_builder.history_ready
            and self.state_machine.state
            in (RedRhexState.POLICY_READY, RedRhexState.POLICY_RUN)
        )

    def _motor_state_allowed(self) -> bool:
        return self.state_machine.state in (
            RedRhexState.INIT_STAND,
            RedRhexState.WARMUP,
            RedRhexState.POLICY_READY,
            RedRhexState.POLICY_RUN,
        )

    def _on_estop(self, msg: Bool) -> None:
        self.estop = bool(msg.data)
        if self.estop:
            self._drop_enable_latches("E-stop asserted")

    def _on_enable_policy(self, msg: Bool) -> None:
        if not bool(msg.data):
            self.enable_policy = False
            return
        if not self._policy_enable_allowed():
            self.enable_policy = False
            self.get_logger().warn("Rejecting V2 policy enable before history/state readiness")
            return
        self.enable_policy = True

    def _on_enable_motors(self, msg: Bool) -> None:
        if not bool(msg.data):
            self.enable_motor_output = False
            return
        allowed = self.deployment_guard.motor_output_allowed(
            requested=True,
            state_allowed=self._motor_state_allowed(),
            estop=self.estop,
            runtime_action_target_compatible=(
                self.runtime_action_target_compatible
            ),
        )
        if not allowed:
            self.enable_motor_output = False
            self.get_logger().error(
                "Rejecting V2 motor enable: state, E-stop, explicit hardware gate, "
                "and bundle calibration must all be ready"
            )
            return
        self.enable_motor_output = True

    def _on_recover(self, msg: Bool) -> None:
        self.recover_requested = bool(msg.data)

    def _on_lowlevel_heartbeat(self, msg: Bool) -> None:
        if msg.data:
            self.last_lowlevel_heartbeat_time = self._now_s()

    def _on_motor_feedback(self, msg: RedRhexMotorState) -> None:
        try:
            self.last_motor_feedback_source_time = _source_stamp_s(msg)
        except ValueError as exc:
            self.get_logger().warn(f"Rejected unstamped V2 motor feedback: {exc}")
            return
        self.motor_temperatures = [float(value) for value in msg.temperature_c]
        self.motor_currents = [float(value) for value in msg.current_a]
        self.motor_faults = [bool(value) for value in msg.fault]

    def _advance_history(self, now_s: float) -> None:
        builder = self.observation_builder
        if builder is None or self.state_machine.state not in (
            RedRhexState.WARMUP,
            RedRhexState.POLICY_READY,
            RedRhexState.POLICY_RUN,
        ):
            return
        status = builder.status(now_s)
        if not status.ok or not builder.has_complete_new_sensor_generation:
            return
        if builder.velocity_baseline_required:
            builder.prime_velocity_baseline()
            return
        builder.append_sensor_frame(builder.latest_sensor_source_time_s)

    def _observation_status(self, now_s: float) -> ObservationStatusV2:
        if self.observation_builder is None:
            return ObservationStatusV2(
                ok=False,
                history_ready=False,
                reasons=("Sensor V2 policy bundle is not loaded",),
            )
        return self.observation_builder.status(now_s)

    def _disabled_command(self) -> DecodedMotorCommand:
        names = list(MAIN_JOINT_NAMES_V2 + ABAD_JOINT_NAMES_V2)
        zeros_6 = np.zeros(6, dtype=np.float64)
        return DecodedMotorCommand(
            joint_names=names,
            target_position_rad=[0.0] * 12,
            target_velocity_rad_s=[0.0] * 12,
            kp=[0.0] * 12,
            kd=[0.0] * 12,
            effort_limit_nm=[0.0] * 12,
            enable=False,
            mode=ForwardResidualActionDecoderV2.MODE_DISABLED,
            safe_action=np.zeros(12, dtype=np.float32),
            target_main_drive_velocity=zeros_6.copy(),
            target_abad_position=zeros_6.copy(),
        )

    def _control_tick(self) -> None:
        now_s = self._now_s()
        dt = (
            1.0 / self.policy_hz
            if self.last_loop_time is None
            else max(0.0, now_s - self.last_loop_time)
        )
        self.last_loop_time = now_s

        try:
            self._advance_history(now_s)
        except Exception as exc:
            if self.observation_builder is not None:
                self.observation_builder.reset_history(str(exc))
            self.get_logger().error(f"V2 history update failed: {exc}")
        obs_status = self._observation_status(now_s)

        roll = 0.0
        pitch = 0.0
        if self.observation_builder is not None:
            try:
                roll, pitch = _roll_pitch_from_projected_gravity(
                    self.observation_builder.projected_gravity_body()
                )
            except RuntimeError:
                pass
        imu_age = (
            None
            if self.last_imu_source_time is None
            else now_s - self.last_imu_source_time
        )
        joint_age = (
            None
            if self.last_joint_source_time is None
            else now_s - self.last_joint_source_time
        )
        motor_age = (
            None
            if self.last_motor_feedback_source_time is None
            else now_s - self.last_motor_feedback_source_time
        )
        heartbeat_age = (
            None
            if self.last_lowlevel_heartbeat_time is None
            else now_s - self.last_lowlevel_heartbeat_time
        )
        command = (
            np.zeros(3, dtype=np.float32)
            if self.observation_builder is None
            else self.observation_builder.command
        )
        pre_safety = SafetyState(
            estop=self.estop,
            imu_age_s=imu_age,
            joint_state_age_s=joint_age,
            motor_feedback_age_s=motor_age,
            heartbeat_age_s=heartbeat_age,
            roll_rad=roll,
            pitch_rad=pitch,
            command=command,
            motor_temperatures_c=self.motor_temperatures,
            motor_currents_a=self.motor_currents,
            motor_faults=self.motor_faults,
            control_loop_dt_s=dt,
        )
        safety_result = self.safety_filter.check(pre_safety)
        waiting = self.state_machine.state in (
            RedRhexState.BOOT,
            RedRhexState.SENSOR_CHECK,
        )
        safety_ok = safety_result.ok and obs_status.ok
        if waiting:
            safety_ok = True
        reasons = list(safety_result.reasons) + list(obs_status.reasons)
        if not safety_ok and not waiting:
            self._drop_enable_latches("; ".join(reasons[:3]))

        elapsed = now_s - self.state_enter_time
        motor_ready = (
            motor_age is not None
            and 0.0 <= motor_age <= self.safety_filter.motor_feedback_timeout_s
        )
        bridge_ready = (
            heartbeat_age is not None
            and 0.0 <= heartbeat_age <= self.safety_filter.heartbeat_timeout_s
        )
        history_ready = bool(
            self.observation_builder is not None
            and self.observation_builder.history_ready
        )
        inputs = StateMachineInputs(
            policy_loaded=self.policy_loaded,
            sensors_ready=obs_status.ok,
            motor_feedback_ready=motor_ready,
            lowlevel_alive=bridge_ready,
            estop=self.estop,
            safety_ok=safety_ok,
            fall_detected=(
                abs(roll) > self.safety_filter.max_abs_roll_rad
                or abs(pitch) > self.safety_filter.max_abs_pitch_rad
            ),
            init_stand_done=elapsed >= self.init_stand_duration_s,
            warmup_done=warmup_complete_v2(
                elapsed_s=elapsed,
                minimum_duration_s=self.warmup_duration_s,
                history_ready=history_ready,
                require_history_ready=self.require_history_ready,
            ),
            enable_policy=self.enable_policy,
            recover_requested=self.recover_requested,
            reasons=reasons,
        )
        old_state = self.state_machine.state
        state = self.state_machine.update(inputs)
        if state != old_state:
            self.state_enter_time = now_s
            self.get_logger().info(
                f"V2 state transition: {old_state.value} -> {state.value}: "
                f"{self.state_machine.last_transition_reason}"
            )
            if state == RedRhexState.INIT_STAND:
                if self.observation_builder is not None:
                    self.observation_builder.reset_history("entering INIT_STAND")
                if self.action_decoder is not None:
                    self.action_decoder.reset()
            if state in (
                RedRhexState.PROTECTIVE_STOP,
                RedRhexState.FALL_DETECTED,
                RedRhexState.RECOVER,
            ):
                self._drop_enable_latches(self.state_machine.last_transition_reason)

        decoded = self._disabled_command()
        try:
            if self.action_decoder is None or self.observation_builder is None:
                decoded = self._disabled_command()
            elif state in (
                RedRhexState.INIT_STAND,
                RedRhexState.WARMUP,
                RedRhexState.POLICY_READY,
            ):
                decoded = self.action_decoder.init_stand_command(
                    self.observation_builder.get_main_drive_positions(),
                    enable=True,
                )
            elif state == RedRhexState.POLICY_RUN:
                if self.policy_runner is None:
                    raise RuntimeError("Sensor V2 policy runner is unavailable")
                policy_inputs = self.observation_builder.policy_inputs(now_s)
                outputs = self.policy_runner.run(
                    policy_inputs.sensor_history,
                    policy_inputs.command,
                )
                decoded = self.action_decoder.decode(
                    outputs.actions,
                    self.observation_builder.get_main_drive_positions(),
                    policy_inputs.command,
                    dt,
                )
                target_status = self.action_decoder.last_target_status
                if target_status is None:
                    raise RuntimeError("V2 action decoder did not report target parity")
                self.last_hardware_action_clip_applied = bool(
                    target_status.hardware_action_clip_applied
                )
                self.last_hardware_slew_applied = bool(
                    target_status.hardware_slew_applied
                )
                self.last_hardware_velocity_limit_applied = bool(
                    target_status.hardware_velocity_limit_applied
                )
                if target_status.hardware_tightening_applied:
                    self.hardware_tightening_event_count += 1
                    self.runtime_action_target_compatible = False
                    raise RuntimeError(
                        "hardware action tightening changed the V2 bundle target; "
                        "motor authorization is latched off"
                    )
                post_safety = self.safety_filter.check(
                    pre_safety,
                    policy_inputs.sensor_history[-1],
                    outputs.actions,
                    decoded,
                )
                if not post_safety.ok:
                    reasons.extend(post_safety.reasons)
                    raise RuntimeError("; ".join(post_safety.reasons))
                self.obs_pub.publish(
                    Float32MultiArray(
                        data=[
                            float(value)
                            for value in policy_inputs.sensor_history[-1]
                        ]
                    )
                )
                self.raw_action_pub.publish(
                    Float32MultiArray(data=[float(value) for value in outputs.actions])
                )
                self.safe_action_pub.publish(
                    Float32MultiArray(
                        data=[float(value) for value in decoded.safe_action]
                    )
                )
            elif state in (
                RedRhexState.PROTECTIVE_STOP,
                RedRhexState.FALL_DETECTED,
                RedRhexState.RECOVER,
            ):
                decoded = self.action_decoder.protective_stop_command(
                    self.observation_builder.get_main_drive_positions(),
                    self.observation_builder.get_abad_positions(),
                )
        except Exception as exc:
            reasons.append(str(exc))
            safety_ok = False
            self.get_logger().error(
                f"V2 control tick failed: {exc}\n{traceback.format_exc()}"
            )
            self._drop_enable_latches(str(exc))
            self.state_machine.transition(RedRhexState.PROTECTIVE_STOP, str(exc))
            if self.action_decoder is not None and self.observation_builder is not None:
                decoded = self.action_decoder.protective_stop_command(
                    self.observation_builder.get_main_drive_positions(),
                    self.observation_builder.get_abad_positions(),
                )
            else:
                decoded = self._disabled_command()

        self._publish_motor_command(decoded)
        self._publish_state_and_diagnostics(reasons, safety_ok)
        self.recover_requested = False

    def _publish_motor_command(self, decoded: DecodedMotorCommand) -> None:
        requested = bool(decoded.enable and self.enable_motor_output)
        actual_enable = self.deployment_guard.motor_output_allowed(
            requested=requested,
            state_allowed=self._motor_state_allowed(),
            estop=self.estop,
            runtime_action_target_compatible=(
                self.runtime_action_target_compatible
            ),
        )
        msg = RedRhexMotorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "redrhex_policy_body"
        msg.joint_names = decoded.joint_names
        msg.target_position_rad = decoded.target_position_rad
        msg.target_velocity_rad_s = decoded.target_velocity_rad_s
        msg.kp = decoded.kp
        msg.kd = decoded.kd
        msg.effort_limit_nm = decoded.effort_limit_nm
        msg.enable = actual_enable
        msg.main_drive_enable = [actual_enable] * 6
        msg.abad_output_enable = actual_enable
        msg.sim2real_probe = False
        msg.mode = decoded.mode
        self.motor_cmd_pub.publish(msg)

    def _publish_state_and_diagnostics(self, reasons: list[str], ok: bool) -> None:
        self.state_pub.publish(String(data=self.state_machine.state.value))
        history_size = (
            0 if self.observation_builder is None else self.observation_builder.history_size
        )
        values = {
            "contract_id": SENSOR_ONLY_CONTRACT_ID_V2,
            "history_size": str(history_size),
            "history_ready": str(
                bool(
                    self.observation_builder is not None
                    and self.observation_builder.history_ready
                )
            ).lower(),
            "hardware_gate_allow_motor_enable": str(self.allow_motor_enable).lower(),
            "bundle_calibration_hardware_ready": str(
                self.deployment_guard.calibration_hardware_ready
            ).lower(),
            "hardware_authorized": str(
                self.deployment_guard.hardware_authorized
            ).lower(),
            "action_target_envelope_compatible": str(
                self.action_target_envelope_compatible
            ).lower(),
            "runtime_action_target_compatible": str(
                self.runtime_action_target_compatible
            ).lower(),
            "configured_action_clip": str(self.configured_action_clip),
            "contract_action_clip": str(self.contract_action_clip),
            "configured_main_velocity_limit_rad_s": str(
                self.configured_main_velocity_limit_rad_s
            ),
            "contract_main_velocity_limit_rad_s": str(
                self.contract_main_velocity_limit_rad_s
            ),
            "hardware_tightening_event_count": str(
                self.hardware_tightening_event_count
            ),
            "last_hardware_action_clip_applied": str(
                self.last_hardware_action_clip_applied
            ).lower(),
            "last_hardware_slew_applied": str(
                self.last_hardware_slew_applied
            ).lower(),
            "last_hardware_velocity_limit_applied": str(
                self.last_hardware_velocity_limit_applied
            ).lower(),
            "recorded_imu_evidence": str(self.recorded_imu_evidence).lower(),
            "recorded_encoder_evidence": str(self.recorded_encoder_evidence).lower(),
        }
        status = DiagnosticStatus()
        status.name = "redrhex_rl_controller_v2"
        status.hardware_id = "redrhex"
        status.level = DiagnosticStatus.OK if ok else DiagnosticStatus.ERROR
        status.message = "ready" if ok else "; ".join(reasons[:4]) or "not ready"
        status.values = [KeyValue(key=key, value=value) for key, value in values.items()]
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [status]
        self.diag_pub.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RedRhexRLControllerNodeV2()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
