"""Mock bridge used for bench and ROS graph tests."""

from __future__ import annotations

import time

from redrhex_msgs.msg import RedRhexMotorState

from .bridge_base import LowLevelBridgeBase
from .command_safety import resolve_output_selection


class MockLowLevelBridge(LowLevelBridgeBase):
    def __init__(self, print_every_n: int = 50) -> None:
        self.print_every_n = max(1, int(print_every_n))
        self.connected = False
        self.sequence = 0
        self.last_command = None
        self.state_pending = False
        self.last_main_drive_enable = (False,) * 6
        self.last_abad_output_enable = False
        self.last_rx_time = time.monotonic()

    def connect(self) -> None:
        self.connected = True
        self.last_rx_time = time.monotonic()
        print("[MockLowLevelBridge] connected")

    def send_motor_command(self, cmd) -> None:
        selection = resolve_output_selection(cmd)
        self.sequence += 1
        self.last_command = cmd
        self.state_pending = True
        self.last_main_drive_enable = selection.main_drive_enable
        self.last_abad_output_enable = selection.abad_output_enable
        self.last_rx_time = time.monotonic()
        if self.sequence % self.print_every_n == 1:
            first = cmd.joint_names[0] if cmd.joint_names else "none"
            print(
                "[MockLowLevelBridge] command "
                f"seq={self.sequence} enable={cmd.enable} mode={cmd.mode} "
                f"joints={len(cmd.joint_names)} first={first}"
            )

    def read_motor_state(self):
        if self.last_command is None or not self.state_pending:
            return None
        self.state_pending = False
        state = RedRhexMotorState()
        state.joint_names = list(self.last_command.joint_names)
        state.position_rad = list(self.last_command.target_position_rad)
        state.velocity_rad_s = list(self.last_command.target_velocity_rad_s)
        state.effort_nm = [0.0] * len(state.joint_names)
        state.current_a = [0.0] * len(state.joint_names)
        state.temperature_c = [30.0] * len(state.joint_names)
        state.fault = [False] * len(state.joint_names)
        return state

    def is_alive(self) -> bool:
        return self.connected

    def output_state_is_fresh(self) -> bool:
        return self.connected

    def emergency_disable(self) -> None:
        self.last_main_drive_enable = (False,) * 6
        self.last_abad_output_enable = False
        self.state_pending = False

    def shutdown(self) -> None:
        self.emergency_disable()
        self.connected = False
        print("[MockLowLevelBridge] shutdown")
