from __future__ import annotations

import argparse
import time
from typing import Sequence

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish RedRhex E-stop state.")
    parser.add_argument("state", choices=["on", "off"])
    parser.add_argument("--wait-s", type=float, default=2.0, help="Seconds to wait for a matching subscriber.")
    parser.add_argument("--duration-s", type=float, default=1.0, help="Seconds to keep publishing the state.")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="Publish rate while asserting the state.")
    args = parser.parse_args(argv)
    rclpy.init()
    node = Node("redrhex_estop_tool")
    pub = node.create_publisher(Bool, "/estop", 10)
    msg = Bool()
    msg.data = args.state == "on"

    wait_deadline = time.monotonic() + max(args.wait_s, 0.0)
    while pub.get_subscription_count() == 0 and time.monotonic() < wait_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    period = 1.0 / max(args.rate_hz, 1.0)
    publish_deadline = time.monotonic() + max(args.duration_s, period)
    while time.monotonic() < publish_deadline:
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=min(0.05, period))
        time.sleep(max(period - 0.05, 0.0))

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
