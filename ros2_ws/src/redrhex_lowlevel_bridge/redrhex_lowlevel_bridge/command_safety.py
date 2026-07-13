"""ROS-independent output-selection and E-stop gate semantics."""

from __future__ import annotations

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


class FailClosedOutputGate:
    """Track central E-stop clearance and whether hardware output may be active."""

    def __init__(self) -> None:
        self.estop_state: bool | None = None
        self.output_active = False

    @property
    def ready_for_output(self) -> bool:
        return self.estop_state is False

    def on_estop(self, asserted: bool) -> bool:
        self.estop_state = bool(asserted)
        if asserted:
            return True
        return False

    def accept_command(self, command, *, state_fresh: bool) -> OutputSelection:
        selection = resolve_output_selection(command)
        if not bool(command.enable):
            return selection
        if self.estop_state is not False:
            raise CommandRejectedError("enabled command rejected until E-stop (/estop) is explicitly false")
        if not state_fresh:
            raise CommandRejectedError("enabled command rejected because raw motor state is stale")
        return selection

    def mark_command_sent(self, selection: OutputSelection) -> None:
        self.output_active = selection.any_enabled

    def on_command_failure(self, *, command_enabled: bool) -> bool:
        if command_enabled:
            self.output_active = True
        return self.output_active

    def on_state_freshness(self, state_fresh: bool) -> bool:
        if self.output_active and not state_fresh:
            return True
        return False

    def mark_disabled(self) -> None:
        self.output_active = False


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
