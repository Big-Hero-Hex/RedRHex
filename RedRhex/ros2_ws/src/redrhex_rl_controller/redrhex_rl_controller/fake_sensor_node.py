from __future__ import annotations

import math
from typing import Sequence

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool

from .redrhex_contract import INIT_MAIN_DRIVE_POS_RAD, MAIN_DRIVE_JOINT_NAMES


class FakeRedRhexSensors(Node):
    def __init__(self) -> None:
        super().__init__("fake_redrhex_sensors")
        self.phase = 0.0
        self.imu_pub = self.create_publisher(Imu, "/imu/data", 10)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.hb_pub = self.create_publisher(Bool, "/redrhex/lowlevel_heartbeat", 10)
        self.timer = self.create_timer(0.01, self._tick)

    def _tick(self) -> None:
        self.phase += 0.01
        now = self.get_clock().now().to_msg()
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "base_link"
        imu.orientation.w = 1.0
        imu.angular_velocity.x = 0.0
        imu.angular_velocity.y = 0.0
        imu.angular_velocity.z = 0.0
        self.imu_pub.publish(imu)

        js = JointState()
        js.header.stamp = now
        js.name = list(MAIN_DRIVE_JOINT_NAMES)
        js.position = [float(p + 0.02 * math.sin(self.phase)) for p in INIT_MAIN_DRIVE_POS_RAD]
        js.velocity = [0.02 * math.cos(self.phase)] * 6
        self.joint_pub.publish(js)

        hb = Bool()
        hb.data = True
        self.hb_pub.publish(hb)


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FakeRedRhexSensors()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
