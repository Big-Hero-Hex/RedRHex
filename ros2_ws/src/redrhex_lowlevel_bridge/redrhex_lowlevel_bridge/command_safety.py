"""ROS-independent output-selection and E-stop gate semantics."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass


class CommandSelectionError(ValueError):
    """An enabled command does not contain a valid fixed output selection."""


class CommandRejectedError(RuntimeError):
    """An enabled command was rejected by the fail-closed gate."""


@dataclass(frozen=True)
class OutputSelection:
    main_drive_enable: tuple[bool, bool, bool, bool, bool, bool]
    abad_output_enable: bool

    @property
    def any_enabled(self) -> bool:
        return any(self.main_drive_enable) or self.abad_output_enable


_ALL_MAIN_DISABLED = (False, False, False, False, False, False)
_NUMERIC_COMMAND_FIELDS = (
    "target_position_rad",
    "target_velocity_rad_s",
    "kp",
    "kd",
    "effort_limit_nm",
)
_COMMAND_JOINT_COUNT = 12


def _coerce_boolean(value, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception as exc:
            raise CommandSelectionError(f"enabled command {field} must contain only booleans") from exc
        if isinstance(scalar, bool):
            return scalar
    raise CommandSelectionError(f"enabled command {field} must contain only booleans")


def resolve_output_selection(command) -> OutputSelection:
    """Return effective outputs, with global disable taking unconditional priority."""
    if not bool(command.enable):
        return OutputSelection(_ALL_MAIN_DISABLED, False)

    mask = getattr(command, "main_drive_enable", None)
    if isinstance(mask, (str, bytes, bytearray, Mapping)):
        raise CommandSelectionError("enabled command main_drive_enable must contain exactly 6 booleans")
    if getattr(mask, "ndim", 1) != 1:
        raise CommandSelectionError("enabled command main_drive_enable must contain exactly 6 booleans")
    try:
        mask_values = tuple(mask)
    except TypeError as exc:
        raise CommandSelectionError("enabled command main_drive_enable must contain exactly 6 booleans") from exc
    if len(mask_values) != 6:
        raise CommandSelectionError("enabled command main_drive_enable must contain exactly 6 booleans")
    normalized_mask = tuple(
        _coerce_boolean(value, field="main_drive_enable") for value in mask_values
    )
    abad = getattr(command, "abad_output_enable", None)
    normalized_abad = _coerce_boolean(abad, field="abad_output_enable")
    return OutputSelection(normalized_mask, normalized_abad)


def _coerce_finite_number(value, *, field: str, index: int) -> float:
    scalar = value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception as exc:
            raise CommandSelectionError(
                f"enabled command {field}[{index}] must be a finite number"
            ) from exc
    if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
        raise CommandSelectionError(
            f"enabled command {field}[{index}] must be a finite number"
        )
    result = float(scalar)
    if not math.isfinite(result):
        raise CommandSelectionError(
            f"enabled command {field}[{index}] must be a finite number"
        )
    return result


def validate_enabled_command_payload(command) -> None:
    """Reject malformed or non-finite numeric arrays before hardware sees them."""

    for field in _NUMERIC_COMMAND_FIELDS:
        values = getattr(command, field, None)
        if isinstance(values, (str, bytes, bytearray, Mapping)):
            raise CommandSelectionError(
                f"enabled command {field} must contain exactly {_COMMAND_JOINT_COUNT} finite numbers"
            )
        if getattr(values, "ndim", 1) != 1:
            raise CommandSelectionError(
                f"enabled command {field} must contain exactly {_COMMAND_JOINT_COUNT} finite numbers"
            )
        try:
            items = tuple(values)
        except TypeError as exc:
            raise CommandSelectionError(
                f"enabled command {field} must contain exactly {_COMMAND_JOINT_COUNT} finite numbers"
            ) from exc
        if len(items) != _COMMAND_JOINT_COUNT:
            raise CommandSelectionError(
                f"enabled command {field} must contain exactly {_COMMAND_JOINT_COUNT} finite numbers"
            )
        for index, value in enumerate(items):
            _coerce_finite_number(value, field=field, index=index)


class FailClosedOutputGate:
    """Track central E-stop clearance and whether hardware output may be active."""

    def __init__(self) -> None:
        self.estop_state: bool | None = None
        self.output_active = False
        self.disable_pending = False

    @property
    def ready_for_output(self) -> bool:
        return self.estop_state is False and not self.disable_pending

    def require_disable(self) -> bool:
        """Latch an uncertain hardware state until emergency disable succeeds."""

        self.disable_pending = True
        return True

    def on_estop(self, asserted: bool) -> bool:
        self.estop_state = bool(asserted)
        if asserted:
            return self.require_disable()
        return False

    def accept_command(self, command, *, state_fresh: bool) -> OutputSelection:
        selection = resolve_output_selection(command)
        if not bool(command.enable):
            return selection
        validate_enabled_command_payload(command)
        if self.disable_pending:
            raise CommandRejectedError(
                "enabled command rejected while an emergency disable is pending"
            )
        if self.estop_state is not False:
            raise CommandRejectedError("enabled command rejected until E-stop (/estop) is explicitly false")
        if not state_fresh:
            raise CommandRejectedError("enabled command rejected because raw motor state is stale")
        return selection

    def mark_command_sent(self, selection: OutputSelection) -> None:
        if selection.any_enabled:
            self.output_active = True
        elif not self.disable_pending:
            self.output_active = False

    def on_command_failure(self, *, command_enabled: bool) -> bool:
        if command_enabled:
            self.output_active = True
        return self.require_disable()

    def on_state_freshness(self, state_fresh: bool) -> bool:
        if self.output_active and not state_fresh:
            self.require_disable()
        return self.disable_pending

    def mark_disabled(self) -> None:
        self.output_active = False
        self.disable_pending = False


class ProbeSessionGate:
    """Bridge-owned lease that makes low-energy probe output exclusive."""

    def __init__(self, timeout_s: float, *, clock=time.monotonic) -> None:
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("probe session timeout_s must be positive and finite")
        if not callable(clock):
            raise TypeError("probe session clock must be callable")
        self._clock = clock
        self.active = False
        self.last_probe_command_time: float | None = None
        self.conflict_latched = False
        self.conflict_reason = ""

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise CommandRejectedError("probe session clock is not finite")
        return now

    def _latch_conflict(self, reason: str) -> None:
        if not self.conflict_latched:
            self.conflict_reason = str(reason)
        self.conflict_latched = True
        self.active = False
        self.last_probe_command_time = None

    def latch_failure(self, reason: str) -> None:
        """Keep all enabled output blocked after a probe-path backend failure."""

        self._latch_conflict(str(reason))

    def poll(self, *, output_active: bool, now: float | None = None) -> bool:
        """Expire a silent probe lease and request disable if output may be live."""

        if not self.active or self.last_probe_command_time is None:
            return False
        current = self._now() if now is None else float(now)
        if not math.isfinite(current):
            self._latch_conflict("probe session clock became invalid")
            return bool(output_active)
        if current - self.last_probe_command_time < self.timeout_s:
            return False
        self.active = False
        self.last_probe_command_time = None
        if output_active:
            self._latch_conflict(
                "probe session heartbeat expired while hardware output was active"
            )
            return True
        return False

    def authorize(self, command, *, output_active: bool) -> None:
        """Authorize a command without relying on producer-side graph checks."""

        now = self._now()
        timed_out_active = self.poll(output_active=output_active, now=now)
        enabled = bool(command.enable)
        is_probe = bool(getattr(command, "sim2real_probe", False))

        # Global disable is always accepted. A valid probe heartbeat also
        # acquires/renews the lease during its initial neutral segment.
        if not enabled:
            if is_probe and not self.conflict_latched:
                self.active = True
                self.last_probe_command_time = now
            return

        if timed_out_active or self.conflict_latched:
            reason = self.conflict_reason or "probe session safety conflict"
            raise CommandRejectedError(
                f"enabled command rejected after latched probe session conflict: {reason}"
            )

        if is_probe:
            try:
                selection = resolve_output_selection(command)
            except CommandSelectionError as exc:
                self._latch_conflict(str(exc))
                raise CommandRejectedError(
                    f"invalid sim2real probe output selection: {exc}"
                ) from exc
            if selection.abad_output_enable:
                self._latch_conflict("sim2real probe attempted to enable ABAD output")
                raise CommandRejectedError("sim2real probe must keep ABAD output disabled")
            if sum(selection.main_drive_enable) != 1:
                self._latch_conflict(
                    "sim2real probe did not select exactly one main drive"
                )
                raise CommandRejectedError(
                    "sim2real probe must enable exactly one main drive"
                )
            self.active = True
            self.last_probe_command_time = now
            return

        if self.active:
            self._latch_conflict(
                "non-probe enabled command arrived during an exclusive probe session"
            )
            raise CommandRejectedError(
                "non-probe enabled command rejected during exclusive probe session"
            )


def dispatch_command_fail_closed(gate: FailClosedOutputGate, backend, command) -> OutputSelection:
    """Authorize, send, and atomically commit one command or emergency-disable."""
    selection = gate.accept_command(
        command,
        state_fresh=backend.output_state_is_fresh(),
    )
    try:
        backend.send_motor_command(command)
    except Exception:
        if gate.on_command_failure(command_enabled=bool(command.enable)):
            backend.emergency_disable()
            gate.mark_disabled()
        raise
    gate.mark_command_sent(selection)
    return selection
