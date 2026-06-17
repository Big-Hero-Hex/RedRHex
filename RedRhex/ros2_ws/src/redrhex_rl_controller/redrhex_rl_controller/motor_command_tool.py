from __future__ import annotations

import argparse
from typing import Sequence

import rclpy
from rclpy.node import Node

from redrhex_msgs.msg import RedRhexMotorCommand

from .redrhex_contract import INIT_ABAD_POS_RAD, INIT_MAIN_DRIVE_POS_RAD, MOTOR_JOINT_NAMES


class OneShotPublisher(Node):
    def __init__(self) -> None:
        super().__init__("redrhex_motor_command_tool")
        self.pub = self.create_publisher(RedRhexMotorCommand, "/redrhex/motor_commands", 10)

    def publish_command(self, cmd: RedRhexMotorCommand) -> None:
        cmd.header.stamp = self.get_clock().now().to_msg()
        for _ in range(5):
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)


def _base_msg(enable: bool) -> RedRhexMotorCommand:
    msg = RedRhexMotorCommand()
    msg.joint_names = list(MOTOR_JOINT_NAMES)
    msg.target_position_rad = [float(v) for v in INIT_MAIN_DRIVE_POS_RAD + INIT_ABAD_POS_RAD]
    msg.target_velocity_rad_s = [0.0] * 12
    msg.kp = [0.0] * 6 + [2.0] * 6
    msg.kd = [0.0] * 6 + [0.1] * 6
    msg.effort_limit_nm = [0.0] * 12
    msg.enable = enable
    msg.mode = 9
    return msg


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bench-safe one-shot RedRhex motor command tool.")
    parser.add_argument("kind", choices=["disable", "main", "abad"])
    parser.add_argument("--index", type=int, default=0, help="Policy-order joint index 0..5")
    parser.add_argument("--velocity", type=float, default=0.15, help="Main drive rad/s, default intentionally tiny")
    parser.add_argument("--position", type=float, default=0.05, help="ABAD target rad, default intentionally tiny")
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--confirm-risk", action="store_true")
    args = parser.parse_args(argv)

    if args.enable and not args.confirm_risk:
        raise SystemExit("--enable requires --confirm-risk")
    if args.kind == "main" and abs(args.velocity) > 0.5:
        raise SystemExit("bench-safe tool refuses main velocity > 0.5 rad/s")
    if args.kind == "abad" and abs(args.position) > 0.15:
        raise SystemExit("bench-safe tool refuses ABAD position > 0.15 rad")
    if not 0 <= args.index < 6:
        raise SystemExit("--index must be 0..5")

    rclpy.init()
    node = OneShotPublisher()
    try:
        msg = _base_msg(enable=args.enable and args.kind != "disable")
        if args.kind == "main":
            msg.target_velocity_rad_s[args.index] = float(args.velocity)
        elif args.kind == "abad":
            msg.target_position_rad[6 + args.index] = float(args.position)
        else:
            msg.enable = False
        node.publish_command(msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
