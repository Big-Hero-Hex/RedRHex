from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import ContractError, ScenarioSpecV1
from .traces import LoadedTrace, sha256_json


_PROFILE_SOURCE_PREFIX = "profile:"
_PROBE_SOURCE_PREFIX = "authenticated_probe_events:"


def _measured_profile_source(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith(_PROFILE_SOURCE_PREFIX):
        return None
    profile_id = value.removeprefix(_PROFILE_SOURCE_PREFIX)
    if not profile_id or ":" in profile_id:
        return None
    return profile_id


def _validate_main_drive_mapping(
    real: LoadedTrace,
    scenario: ScenarioSpecV1,
) -> None:
    constants = real.manifest.metadata.get("calibration_constants", {})
    if not isinstance(constants, Mapping):
        raise ContractError("real main-drive trace has invalid hardware mapping provenance")

    position_profile_id = _measured_profile_source(
        constants.get("position_mapping_source")
    )
    if position_profile_id is None or "profile_sha256" not in real.manifest.provenance:
        raise ContractError(
            "real main-drive trace uses provisional or missing hardware mapping provenance "
            "for encoder position"
        )

    command_source = constants.get("requested_command_source")
    command_profile_id = _measured_profile_source(command_source)
    if command_profile_id is not None:
        if command_profile_id != position_profile_id:
            raise ContractError(
                "real main-drive trace hardware mapping provenance names different profiles"
            )
        return

    scenario_hash = sha256_json(scenario.to_dict())
    if command_source != f"{_PROBE_SOURCE_PREFIX}{scenario_hash}":
        raise ContractError(
            "real main-drive trace uses provisional or missing hardware mapping provenance "
            "for requested command"
        )
    if scenario.experiment_kind != "step_coast":
        raise ContractError(
            "authenticated probe-event command provenance is only valid for step_coast scenarios"
        )
    evidence = constants.get("probe_event_evidence")
    if not isinstance(evidence, Mapping):
        raise ContractError("authenticated probe-event mapping provenance is missing evidence")

    expected_duration_s = sum(
        float(segment["duration_s"]) for segment in scenario.command_segments
    ) * scenario.repeats
    expected_ticks = int(round(expected_duration_s * 60.0))
    expected_segment_count = scenario.repeats * len(scenario.command_segments)
    try:
        valid_evidence = (
            evidence["scenario_sha256"] == scenario_hash
            and evidence["abad_output_disabled_verified"] is True
            and int(evidence["repetition_count"]) == scenario.repeats
            and int(evidence["segment_count"]) == expected_segment_count
            and int(evidence["complete_ticks"]) == expected_ticks
            and math.isclose(
                float(evidence["receive_duration_s"]),
                expected_duration_s,
                rel_tol=0.0,
                abs_tol=float(evidence["receive_jitter_bound_s"]),
            )
            and 0.0 < float(evidence["receive_jitter_bound_s"]) <= 1.0 / 60.0
        )
    except (KeyError, TypeError, ValueError):
        valid_evidence = False
    if not valid_evidence:
        raise ContractError("authenticated probe-event mapping provenance evidence is invalid")


def validate_real_trace_provenance(
    real: LoadedTrace,
    scenario: ScenarioSpecV1,
) -> None:
    """Fail closed unless a real trace has the provenance its subsystem requires."""

    if real.manifest.source != "real":
        raise ContractError('real trace must have source "real"')
    if real.manifest.scenario_id != scenario.scenario_id:
        raise ContractError("scenario id mismatch")
    if real.manifest.provenance.get("scenario_sha256") != sha256_json(
        scenario.to_dict()
    ):
        raise ContractError("scenario hash mismatch")
    if scenario.subsystem == "main_drive":
        _validate_main_drive_mapping(real, scenario)
