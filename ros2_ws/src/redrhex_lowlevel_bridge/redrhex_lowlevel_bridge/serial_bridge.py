from __future__ import annotations

import struct
import time
import zlib

from .bridge_base import LowLevelBridgeBase


class SerialLowLevelBridge(LowLevelBridgeBase):
    """Provisional serial backend skeleton.

    This is intentionally not marked as a final hardware protocol. The sbRIO
    path should use the BioRoLa/Rinbo ROS backend unless a real UART packet
    contract is agreed with firmware.
    """

    MAGIC = b"RHX1"

    def __init__(self, node, port: str = "/dev/ttyUSB0", baudrate: int = 115200, timeout_s: float = 0.01) -> None:
        self.node = node
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self.serial = None
        self.seq = 0
        self.last_tx = 0.0

    def connect(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for SerialLowLevelBridge") from exc
        self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        self.node.get_logger().warn("Serial backend is a provisional skeleton, not a validated motor protocol.")

    def send_motor_command(self, cmd) -> None:
        if self.serial is None:
            return
        payload = struct.pack("<I d B H", self.seq, time.time(), int(cmd.enable), len(cmd.joint_names))
        arrays = (
            list(cmd.target_position_rad)
            + list(cmd.target_velocity_rad_s)
            + list(cmd.kp)
            + list(cmd.kd)
            + list(cmd.effort_limit_nm)
        )
        payload += struct.pack("<" + "f" * len(arrays), *[float(v) for v in arrays])
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        packet = self.MAGIC + struct.pack("<I", len(payload)) + payload + struct.pack("<I", crc)
        self.serial.write(packet)
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        self.last_tx = time.monotonic()

    def is_alive(self) -> bool:
        return self.serial is not None and getattr(self.serial, "is_open", False)

    def diagnostics(self) -> dict[str, str]:
        return {"backend": "serial", "port": self.port, "last_tx_s": f"{time.monotonic() - self.last_tx:.3f}"}

    def shutdown(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None
