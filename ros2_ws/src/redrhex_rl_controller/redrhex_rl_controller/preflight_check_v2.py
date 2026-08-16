"""Fail-closed offline preflight for Sensor-Only Distillation V2."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from .deployment_guard_v2 import (
    DeploymentGuardV2,
    action_target_envelope_matches_v2,
)
from .deployment_route import SENSOR_ONLY_CONTRACT_ID_V2, resolve_deployment_route
from .policy_onnx_runner_v2 import SensorPolicyONNXRunnerV2


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _nested(params: dict, *keys: str, default=None):
    current = params
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _append(result: dict[str, object], name: str, ok: bool, **fields: object) -> None:
    item = {"name": name, "ok": bool(ok)}
    item.update(fields)
    result["checks"].append(item)


def _load_params(config_path: str) -> dict:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"V2 ROS config not found: {path}")
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - packaging dependency
        raise RuntimeError("PyYAML is required for V2 preflight") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = data.get("redrhex_rl_controller_v2", {}).get("ros__parameters")
    if not isinstance(params, dict):
        raise ValueError("expected redrhex_rl_controller_v2.ros__parameters in V2 config")
    return params


def validate_v2_config(params: dict) -> tuple[list[dict[str, object]], list[str]]:
    result: dict[str, object] = {"checks": []}
    blockers: list[str] = []
    contract_id = str(_nested(params, "policy", "contract_id", default=""))
    try:
        route = resolve_deployment_route(contract_id)
        route_ok = route.contract_id == SENSOR_ONLY_CONTRACT_ID_V2
    except ValueError:
        route_ok = False
    _append(result, "exact_v2_route", route_ok, contract_id=contract_id)

    fixed_layout = (
        int(_nested(params, "observation", "sensor_frame_dim", default=-1)) == 36
        and int(_nested(params, "observation", "history_length", default=-1)) == 60
        and str(_nested(params, "observation", "history_order", default="")) == "oldest_to_newest"
        and float(_nested(params, "observation", "sample_rate_hz", default=-1.0)) == 60.0
    )
    _append(result, "fixed_sensor_layout", fixed_layout)

    sample_period_s = 1.0 / 60.0
    max_source_skew_s = float(
        _nested(params, "observation", "max_sensor_source_skew_s", default=-1.0)
    )
    max_period_error_ratio = float(
        _nested(
            params,
            "observation",
            "max_history_period_error_ratio",
            default=-1.0,
        )
    )
    source_timing_ok = (
        0.0 < max_source_skew_s <= sample_period_s
        and 0.0 <= max_period_error_ratio < 1.0
    )
    _append(
        result,
        "source_skew_and_cadence_bounds",
        source_timing_ok,
        max_sensor_source_skew_s=max_source_skew_s,
        max_history_period_error_ratio=max_period_error_ratio,
    )

    main_names = list(_nested(params, "observation", "main_drive_joint_names", default=[]))
    abad_names = list(_nested(params, "observation", "abad_joint_names", default=[]))
    ordering_ok = (
        len(main_names) == 6
        and len(abad_names) == 6
        and len(set(main_names + abad_names)) == 12
    )
    _append(result, "twelve_measured_joint_order", ordering_ok, joint_names=main_names + abad_names)

    attitude_mode = str(_nested(params, "observation", "attitude_mode", default=""))
    attitude_ok = attitude_mode in ("validated_quaternion", "causal_gyro_accel")
    _append(result, "explicit_attitude_mode", attitude_ok, attitude_mode=attitude_mode)
    if attitude_mode == "validated_quaternion":
        evidence_ok = bool(_nested(params, "observation", "imu_mount_calibration_verified", default=False)) and bool(
            _nested(params, "observation", "rest_gravity_verified", default=False)
        )
    elif attitude_mode == "causal_gyro_accel":
        evidence_ok = bool(_nested(params, "observation", "imu_mount_calibration_verified", default=False))
    else:
        evidence_ok = False
    _append(result, "attitude_evidence", evidence_ok)
    if not evidence_ok:
        blockers.append("recorded IMU/frame/rest-gravity evidence is not verified")

    encoder_evidence = bool(_nested(params, "calibration", "twelve_encoders_verified", default=False))
    _append(result, "twelve_encoder_calibration_evidence", encoder_evidence)
    if not encoder_evidence:
        blockers.append("all twelve encoder signs and zeros are not verified")

    recorded_hardware_evidence = bool(
        _nested(params, "hardware_gate", "recorded_imu_evidence", default=False)
    ) and bool(
        _nested(params, "hardware_gate", "recorded_encoder_evidence", default=False)
    )
    _append(result, "recorded_hardware_evidence", recorded_hardware_evidence)
    if not recorded_hardware_evidence:
        blockers.append("hardware-gate IMU and encoder evidence is not recorded")

    allow_motor_enable = bool(
        _nested(params, "hardware_gate", "allow_motor_enable", default=False)
    )
    _append(result, "explicit_motor_enable_authorization", allow_motor_enable)
    if not allow_motor_enable:
        blockers.append("hardware_gate.allow_motor_enable is false")

    hashes: dict[str, str] = {}
    for name in ("contract", "action_contract", "calibration", "checkpoint"):
        value = str(_nested(params, "policy", f"expected_{name}_hash", default=""))
        hashes[name] = value
        ok = bool(_SHA256_RE.fullmatch(value))
        _append(result, f"{name}_hash", ok, value=value)
        if not ok:
            blockers.append(f"{name} hash is missing or unverified")

    action_ok = (
        str(_nested(params, "action", "contract_id", default=""))
        == "redrhex.forward-residual-action.v2"
        and not bool(_nested(params, "action", "learned_abad", default=True))
        and bool(_nested(params, "action", "independent_limits_may_only_tighten", default=False))
    )
    _append(result, "strict_forward_residual_action", action_ok)
    decoder_hash = str(
        _nested(params, "action", "expected_decoder_hash", default="")
    )
    decoder_hash_ok = bool(_SHA256_RE.fullmatch(decoder_hash)) and (
        decoder_hash == hashes["action_contract"]
    )
    _append(
        result,
        "decoder_hash_matches_action_contract",
        decoder_hash_ok,
        value=decoder_hash,
    )
    if not decoder_hash_ok:
        blockers.append("decoder hash is missing or differs from the action contract")

    start_disabled = (
        not bool(_nested(params, "state_machine", "enable_policy_on_start", default=True))
        and not bool(_nested(params, "state_machine", "enable_motor_output_on_start", default=True))
    )
    _append(result, "disabled_on_start", start_disabled)
    return list(result["checks"]), blockers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check a strict RedRHex sensor-policy V2 bundle.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--sidecar", default=None)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--use-tensorrt", action="store_true")
    args = parser.parse_args(argv)

    report: dict[str, object] = {
        "python": sys.executable,
        "contract_id": SENSOR_ONLY_CONTRACT_ID_V2,
        "checks": [],
        "hardware_blockers": [],
        "motor_enable_attempted": False,
    }
    try:
        params = _load_params(args.config)
        checks, blockers = validate_v2_config(params)
        report["checks"].extend(checks)
        report["hardware_blockers"] = blockers

        onnx_path = args.onnx or str(_nested(params, "policy", "onnx_path", default=""))
        sidecar_path = args.sidecar or str(_nested(params, "policy", "sidecar_path", default=""))
        runner = SensorPolicyONNXRunnerV2(
            onnx_path,
            sidecar_path=sidecar_path,
            expected_contract_sha256=str(_nested(params, "policy", "expected_contract_hash", default="")),
            expected_action_contract_sha256=str(
                _nested(params, "policy", "expected_action_contract_hash", default="")
            ),
            expected_calibration_sha256=str(
                _nested(params, "policy", "expected_calibration_hash", default="")
            ),
            expected_checkpoint_sha256=str(
                _nested(params, "policy", "expected_checkpoint_hash", default="")
            ),
            require_hardware_ready=False,
            use_cuda=args.use_cuda,
            use_tensorrt=args.use_tensorrt,
        )
        output = runner.run(
            np.zeros((1, 60, 36), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
        _append(
            report,
            "bundle_load_and_zero_inference",
            bool(output.actions.shape == (12,) and output.base_velocity_estimate.shape == (3,)),
        )
        calibration_ready = runner.calibration_profile.hardware_ready
        _append(
            report,
            "bundle_calibration_hardware_ready",
            calibration_ready,
            readiness_blockers=list(
                runner.calibration_profile.readiness_blockers
            ),
        )
        if not calibration_ready:
            blockers.extend(
                f"bundle calibration: {name}"
                for name in runner.calibration_profile.readiness_blockers
            )
        configured_action_clip = float(
            _nested(params, "safety", "action_clip", default=-1.0)
        )
        configured_velocity_limit = float(
            _nested(
                params,
                "safety",
                "main_drive_vel_limit_rad_s",
                default=-1.0,
            )
        )
        contract_action_clip = float(runner.action_contract.action_clip)
        contract_velocity_limit = float(
            runner.action_contract.main_velocity_limit_rad_s
        )
        action_envelope_ok = action_target_envelope_matches_v2(
            configured_action_clip=configured_action_clip,
            configured_main_velocity_limit_rad_s=configured_velocity_limit,
            contract_action_clip=contract_action_clip,
            contract_main_velocity_limit_rad_s=contract_velocity_limit,
        )
        _append(
            report,
            "action_target_envelope_matches_bundle",
            action_envelope_ok,
            configured_action_clip=configured_action_clip,
            contract_action_clip=contract_action_clip,
            configured_main_velocity_limit_rad_s=configured_velocity_limit,
            contract_main_velocity_limit_rad_s=contract_velocity_limit,
        )
        if not action_envelope_ok:
            blockers.append(
                "configured action clip/main velocity limit changes bundle targets"
            )
        guard = DeploymentGuardV2(
            allow_motor_enable=bool(
                _nested(
                    params,
                    "hardware_gate",
                    "allow_motor_enable",
                    default=False,
                )
            ),
            calibration_hardware_ready=calibration_ready,
            action_target_envelope_compatible=action_envelope_ok,
        )
        _append(
            report,
            "runtime_motor_enable_gate",
            guard.hardware_authorized,
        )
    except Exception as exc:
        _append(report, "preflight_exception", False, error=str(exc))

    checks_ok = all(bool(check.get("ok")) for check in report["checks"])
    report["offline_ready"] = checks_ok
    report["hardware_ready"] = checks_ok and not report["hardware_blockers"]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["hardware_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
