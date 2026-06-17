from __future__ import annotations

import argparse
from typing import Sequence

import rclpy
from rclpy.node import Node

try:
    from rinbo_msgs.msg import PowerCmdStamped
except Exception:
    PowerCmdStamped = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish BioRoLa/Rinbo /power/command safely.")
    parser.add_argument("mode", choices=["digital", "signal", "relay", "off"])
    args = parser.parse_args(argv)
    if PowerCmdStamped is None:
        raise SystemExit("rinbo_msgs not found. Run: source ~/rinbo_ros_ws/install/setup.bash")

    states = {
        "digital": (True, False, False),
        "signal": (True, True, False),
        "relay": (True, True, True),
        "off": (False, False, False),
    }
    digital, signal, power = states[args.mode]
    rclpy.init()
    node = Node("rinbo_power_tool")
    pub = node.create_publisher(PowerCmdStamped, "/power/command", 10)
    msg = PowerCmdStamped()
    msg.header.seq = 1
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.digital = digital
    msg.signal = signal
    msg.power = power
    for _ in range(5):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    print(f"published /power/command digital={digital} signal={signal} power={power}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
