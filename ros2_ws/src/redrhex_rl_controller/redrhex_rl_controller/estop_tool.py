from __future__ import annotations

import argparse
from typing import Sequence

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish RedRhex E-stop state.")
    parser.add_argument("state", choices=["on", "off"])
    args = parser.parse_args(argv)
    rclpy.init()
    node = Node("redrhex_estop_tool")
    pub = node.create_publisher(Bool, "/estop", 10)
    msg = Bool()
    msg.data = args.state == "on"
    for _ in range(5):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
