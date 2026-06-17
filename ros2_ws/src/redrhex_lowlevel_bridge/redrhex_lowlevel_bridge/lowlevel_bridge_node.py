from __future__ import annotations

import time
from typing import Sequence

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from redrhex_msgs.msg import RedRhexMotorCommand

from .mock_bridge import MockLowLevelBridge
from .rinbo_ros_backend import RinboRosBackend
from .serial_bridge import SerialLowLevelBridge


class RedRhexLowLevelBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("redrhex_lowlevel_bridge")
        self._declare_parameters()
        self.backend = self._make_backend()
        self.backend.connect()
        self.create_subscription(RedRhexMotorCommand, "/redrhex/motor_commands", self._on_command, 10)
        self.heartbeat_pub = self.create_publisher(Bool, "/redrhex/lowlevel_heartbeat", 10)
        self.trip_pub = self.create_publisher(Bool, "/redrhex/power_safety_trip", 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/redrhex/lowlevel_diagnostics", 10)
        self.clear_srv = self.create_service(Trigger, "/redrhex/clear_power_safety_trip", self._clear_trip)
        hz = float(self.get_parameter("bridge.publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(hz, 1.0), self._tick)

    def _declare_parameters(self) -> None:
        self.declare_parameter("bridge.backend", "mock")
        self.declare_parameter("bridge.publish_rate_hz", 50.0)
        self.declare_parameter("serial.port", "/dev/ttyUSB0")
        self.declare_parameter("serial.baudrate", 115200)
        self.declare_parameter("serial.timeout_s", 0.01)

        self.declare_parameter("rinbo.allow_enable", False)
        self.declare_parameter("rinbo.command_topic", "/motor/command")
        self.declare_parameter("rinbo.preview_topic", "/redrhex/rinbo_motor_command_preview")
        self.declare_parameter("rinbo.motor_state_topic", "/motor/state")
        self.declare_parameter("rinbo.power_state_topic", "/power/state")
        self.declare_parameter("rinbo.main_max_pwm", 150.0)
        self.declare_parameter("rinbo.main_pwm_per_rad_s", 120.0)
        self.declare_parameter("rinbo.main_pwm_slew_per_s", 600.0)
        self.declare_parameter("rinbo.main_encoder_counts_per_rev", 54984.83)
        self.declare_parameter("rinbo.main_direction_positive_rinbo_order", [True, True, True, False, False, False])
        self.declare_parameter("rinbo.servo_zero_encoder_rinbo_order", [740, 2565, 3283, 1944, 2071, 989])
        self.declare_parameter("rinbo.servo_counts_per_rad", 1000.0)
        self.declare_parameter("rinbo.servo_min_encoder", 0)
        self.declare_parameter("rinbo.servo_max_encoder", 4095)
        self.declare_parameter("rinbo.servo_control_mode", 2)
        self.declare_parameter("rinbo.require_power_state_when_enabled", True)
        self.declare_parameter("rinbo.max_main_channel_current_a", 3.0)
        self.declare_parameter("rinbo.max_bus_current_a", 12.0)
        self.declare_parameter("rinbo.min_bus_voltage_v", 0.0)
        self.declare_parameter("rinbo.max_bus_voltage_v", 0.0)
        self.declare_parameter("rinbo.current_trip_latch", True)
        self.declare_parameter("rinbo.command_timeout_s", 0.10)

    def _nullable_float_param(self, name: str):
        value = float(self.get_parameter(name).value)
        return None if value == 0.0 else value

    def _make_backend(self):
        backend = str(self.get_parameter("bridge.backend").value)
        if backend == "mock":
            return MockLowLevelBridge(self)
        if backend == "serial":
            return SerialLowLevelBridge(
                self,
                str(self.get_parameter("serial.port").value),
                int(self.get_parameter("serial.baudrate").value),
                float(self.get_parameter("serial.timeout_s").value),
            )
        if backend in ("biorola_ros", "rinbo_ros"):
            return RinboRosBackend(
                self,
                allow_enable=bool(self.get_parameter("rinbo.allow_enable").value),
                command_topic=str(self.get_parameter("rinbo.command_topic").value),
                preview_topic=str(self.get_parameter("rinbo.preview_topic").value),
                motor_state_topic=str(self.get_parameter("rinbo.motor_state_topic").value),
                power_state_topic=str(self.get_parameter("rinbo.power_state_topic").value),
                main_max_pwm=float(self.get_parameter("rinbo.main_max_pwm").value),
                main_pwm_per_rad_s=float(self.get_parameter("rinbo.main_pwm_per_rad_s").value),
                main_pwm_slew_per_s=float(self.get_parameter("rinbo.main_pwm_slew_per_s").value),
                main_encoder_counts_per_rev=float(self.get_parameter("rinbo.main_encoder_counts_per_rev").value),
                main_direction_positive_rinbo_order=self.get_parameter("rinbo.main_direction_positive_rinbo_order").value,
                servo_zero_encoder_rinbo_order=self.get_parameter("rinbo.servo_zero_encoder_rinbo_order").value,
                servo_counts_per_rad=float(self.get_parameter("rinbo.servo_counts_per_rad").value),
                servo_min_encoder=int(self.get_parameter("rinbo.servo_min_encoder").value),
                servo_max_encoder=int(self.get_parameter("rinbo.servo_max_encoder").value),
                servo_control_mode=int(self.get_parameter("rinbo.servo_control_mode").value),
                require_power_state_when_enabled=bool(self.get_parameter("rinbo.require_power_state_when_enabled").value),
                max_main_channel_current_a=float(self.get_parameter("rinbo.max_main_channel_current_a").value),
                max_bus_current_a=float(self.get_parameter("rinbo.max_bus_current_a").value),
                min_bus_voltage_v=self._nullable_float_param("rinbo.min_bus_voltage_v"),
                max_bus_voltage_v=self._nullable_float_param("rinbo.max_bus_voltage_v"),
                current_trip_latch=bool(self.get_parameter("rinbo.current_trip_latch").value),
                command_timeout_s=float(self.get_parameter("rinbo.command_timeout_s").value),
            )
        raise RuntimeError(f"Unknown bridge.backend={backend}")

    def _on_command(self, msg: RedRhexMotorCommand) -> None:
        self.backend.send_motor_command(msg)

    def _tick(self) -> None:
        self.backend.tick()
        hb = Bool()
        hb.data = bool(self.backend.is_alive())
        self.heartbeat_pub.publish(hb)
        trip = Bool()
        trip.data = bool(self.backend.power_trip_active())
        self.trip_pub.publish(trip)
        self._publish_diagnostics()

    def _clear_trip(self, _request, response):
        ok, message = self.backend.clear_power_trip()
        response.success = bool(ok)
        response.message = message
        return response

    def _publish_diagnostics(self) -> None:
        diag = DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "redrhex_lowlevel_bridge"
        status.hardware_id = "redrhex_sbrio"
        status.level = DiagnosticStatus.ERROR if self.backend.power_trip_active() else DiagnosticStatus.OK
        status.message = "power trip" if self.backend.power_trip_active() else "ok"
        values = self.backend.diagnostics()
        status.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
        diag.status.append(status)
        self.diag_pub.publish(diag)

    def destroy_node(self) -> bool:
        try:
            self.backend.shutdown()
        finally:
            return super().destroy_node()


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RedRhexLowLevelBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
