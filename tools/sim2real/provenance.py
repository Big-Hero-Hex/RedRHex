from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import CalibrationProfileV1, ContractError, ScenarioSpecV1
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


def _verified_mapping_snapshot(
    real: LoadedTrace,
) -> tuple[str, Mapping[str, Any]]:
    constants = real.manifest.metadata.get("calibration_constants", {})
    snapshot = (
        constants.get("hardware_mapping_snapshot")
        if isinstance(constants, Mapping)
        else None
    )
    if not isinstance(snapshot, Mapping):
        raise ContractError(
            "real main-drive trace uses provisional hardware mapping provenance or is "
            "missing its hardware mapping snapshot"
        )
    expected_fields = {
        "schema_version",
        "profile_id",
        "profile_sha256",
        "hardware_mapping",
    }
    if set(snapshot) != expected_fields:
        raise ContractError("hardware mapping snapshot has missing or unknown fields")
    expected_hash = real.manifest.provenance.get("hardware_mapping_sha256")
    if expected_hash is None or sha256_json(snapshot) != expected_hash:
        raise ContractError("hardware mapping snapshot hash mismatch")
    profile_hash = real.manifest.provenance.get("profile_sha256")
    if snapshot.get("profile_sha256") != profile_hash:
        raise ContractError("hardware mapping snapshot profile hash mismatch")
    try:
        profile = CalibrationProfileV1.from_dict(
            {
                "schema_version": snapshot["schema_version"],
                "profile_id": snapshot["profile_id"],
                "hardware_mapping": snapshot["hardware_mapping"],
                "sensor_timing": {},
                "simulation_physics": {},
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"hardware mapping snapshot is invalid: {exc}") from exc
    return profile.profile_id, profile.hardware_mapping


def _require_joint_mapping(
    mapping: Mapping[str, Any],
    *,
    joints: tuple[str, ...],
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        values = mapping.get(field)
        for joint in joints:
            if not isinstance(values, Mapping) or joint not in values:
                raise ContractError(
                    f"hardware mapping snapshot is missing {field}.{joint}"
                )


def _validate_main_drive_mapping(
    real: LoadedTrace,
    scenario: ScenarioSpecV1,
) -> tuple[str, Mapping[str, Any]]:
    constants = real.manifest.metadata.get("calibration_constants", {})
    if not isinstance(constants, Mapping):
        raise ContractError("real main-drive trace has invalid hardware mapping provenance")

    snapshot_profile_id, mapping = _verified_mapping_snapshot(real)
    _require_joint_mapping(
        mapping,
        joints=(scenario.joint,),
        fields=("encoder_counts_per_rev", "encoder_zero_count", "encoder_sign"),
    )

    position_profile_id = _measured_profile_source(
        constants.get("position_mapping_source")
    )
    if position_profile_id is None or position_profile_id != snapshot_profile_id:
        raise ContractError(
            "real main-drive trace uses provisional or missing hardware mapping provenance "
            "for encoder position"
        )

    command_source = constants.get("requested_command_source")
    command_profile_id = _measured_profile_source(command_source)
    if command_profile_id is not None:
        if command_profile_id != snapshot_profile_id:
            raise ContractError(
                "real main-drive trace hardware mapping provenance names different profiles"
            )
        _require_joint_mapping(
            mapping,
            joints=(scenario.joint,),
            fields=("joint_direction", "pwm_scale", "pwm_cap"),
        )
        return snapshot_profile_id, mapping

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
    return snapshot_profile_id, mapping


def validate_real_trace_provenance(
    real: LoadedTrace,
    scenario: ScenarioSpecV1,
    *,
    require_all_main_positions: bool = False,
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
    # Encoder mapping provenance is meaningful only for experiments that
    # actually contain encoder position.  Manual known-load traces contain
    # force/geometry/PWM observations and must not fabricate a position source.
    if scenario.subsystem == "main_drive" and scenario.experiment_kind in {
        "step",
        "coast",
        "step_coast",
    }:
        position_profile, mapping = _validate_main_drive_mapping(real, scenario)
        if require_all_main_positions:
            constants = real.manifest.metadata.get("calibration_constants", {})
            all_main_profile = _measured_profile_source(
                constants.get("all_main_position_mapping_source")
            )
            if all_main_profile is None or all_main_profile != position_profile:
                raise ContractError(
                    "replay requires measured encoder mapping provenance for all six main joints"
                )
            _require_joint_mapping(
                mapping,
                joints=tuple(f"main_{index}" for index in range(6)),
                fields=(
                    "encoder_counts_per_rev",
                    "encoder_zero_count",
                    "encoder_sign",
                ),
            )
