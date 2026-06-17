from __future__ import annotations

from typing import Sequence

import rclpy
from rclpy.node import Node


def main(argv: Sequence[str] | None = None) -> int:
    rclpy.init(args=list(argv) if argv is not None else None)
    node = Node("rinbo_bringup_check")
    topics = dict(node.get_topic_names_and_types())
    required = ["/motor/state", "/power/state"]
    missing = [t for t in required if t not in topics]
    if missing:
        print(f"Missing topics: {missing}")
        rc = 2
    else:
        print("PASS: /motor/state and /power/state are visible.")
        rc = 0
    node.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
