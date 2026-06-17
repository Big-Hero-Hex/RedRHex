from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import rclpy

from redrhex_rl_controller.redrhex_contract import INIT_MAIN_DRIVE_POS_RAD, MAIN_DRIVE_JOINT_NAMES
from redrhex_rl_controller.rl_controller_node import RedRhexRLControllerNode


class FiveLegRLControllerNode(RedRhexRLControllerNode):
    """RedRhex policy controller variant that ignores Rinbo L3 during init-stand.

    Rinbo L3 maps to policy main-drive index 5, joint Revolute_24 / LR. The
    policy still runs with the normal 56-D observation and 12-D action contract;
    the lowlevel bridge is responsible for dropping the disabled leg command.
    """

    def _declare_parameters(self) -> None:
        super()._declare_parameters()
        self.declare_parameter("five_leg.disabled_main_indices", [5])
        self.declare_parameter("five_leg.init_stand_ready_after_s", 6.0)
        self.declare_parameter("five_leg.require_init_stand_pose", False)

    def _disabled_main_indices(self) -> list[int]:
        raw = self.get_parameter("five_leg.disabled_main_indices").value
        indices = [int(v) for v in raw]
        return sorted(i for i in set(indices) if 0 <= i < 6)

    def _active_main_indices(self) -> list[int]:
        disabled = set(self._disabled_main_indices())
        return [i for i in range(6) if i not in disabled]

    def _init_stand_done(self, now: float) -> bool:
        active = self._active_main_indices()
        if not active or self.init_stand_started <= 0.0:
            return False

        min_time = float(self.get_parameter("safety.init_stand_min_time_s").value)
        ready_after = float(self.get_parameter("five_leg.init_stand_ready_after_s").value)
        require_pose = bool(self.get_parameter("five_leg.require_init_stand_pose").value)
        tol = float(self.get_parameter("safety.init_stand_pose_tol_rad").value)
        elapsed = now - self.init_stand_started
        elapsed_ok = elapsed >= min_time

        current = self.builder.main_pos[active]
        target = np.asarray(INIT_MAIN_DRIVE_POS_RAD, dtype=np.float64)[active]
        pose_err = np.max(np.abs(np.arctan2(np.sin(current - target), np.cos(current - target))))
        if require_pose:
            return bool(elapsed_ok and pose_err < tol)
        return bool(elapsed >= max(min_time, ready_after))

    def _publish_state_and_diag(self, state, sensor_reasons, safety_reasons, decoded) -> None:
        super()._publish_state_and_diag(state, sensor_reasons, safety_reasons, decoded)
        if state.value == "INIT_STAND":
            active = self._active_main_indices()
            disabled_names = [MAIN_DRIVE_JOINT_NAMES[i] for i in self._disabled_main_indices()]
            current = self.builder.main_pos[active]
            target = np.asarray(INIT_MAIN_DRIVE_POS_RAD, dtype=np.float64)[active]
            err = np.abs(np.arctan2(np.sin(current - target), np.cos(current - target)))
            self.get_logger().info(
                "five-leg INIT_STAND active max_err="
                f"{float(np.max(err)):.3f} rad; disabled={disabled_names}",
                throttle_duration_sec=2.0,
            )


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FiveLegRLControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
