from __future__ import annotations

import time

from .bridge_base import LowLevelBridgeBase


class MockLowLevelBridge(LowLevelBridgeBase):
    def __init__(self, node) -> None:
        self.node = node
        self.connected = False
        self.last_command = None
        self.last_command_time = 0.0

    def connect(self) -> None:
        self.connected = True
        self.node.get_logger().info("Mock low-level bridge connected. No hardware command will be sent.")

    def send_motor_command(self, cmd) -> None:
        self.last_command = cmd
        self.last_command_time = time.monotonic()
        max_vel = max([abs(v) for v in cmd.target_velocity_rad_s[:6]] or [0.0])
        max_abad = max([abs(v) for v in cmd.target_position_rad[6:]] or [0.0])
        self.node.get_logger().info(
            f"mock cmd enable={cmd.enable} max_main_vel={max_vel:.3f} rad/s max_abad={max_abad:.3f} rad",
            throttle_duration_sec=1.0,
        )

    def is_alive(self) -> bool:
        return self.connected

    def diagnostics(self) -> dict[str, str]:
        return {
            "backend": "mock",
            "connected": str(self.connected),
            "last_command_age_s": f"{time.monotonic() - self.last_command_time:.3f}" if self.last_command else "never",
        }

    def shutdown(self) -> None:
        self.connected = False
