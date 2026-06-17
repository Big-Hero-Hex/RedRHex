from __future__ import annotations

from abc import ABC, abstractmethod


class LowLevelBridgeBase(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_motor_command(self, cmd) -> None:
        raise NotImplementedError

    def read_motor_state(self):
        return None

    @abstractmethod
    def is_alive(self) -> bool:
        raise NotImplementedError

    def tick(self) -> None:
        pass

    def power_trip_active(self) -> bool:
        return False

    def clear_power_trip(self) -> tuple[bool, str]:
        return False, "backend has no power trip latch"

    def diagnostics(self) -> dict[str, str]:
        return {}

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError
