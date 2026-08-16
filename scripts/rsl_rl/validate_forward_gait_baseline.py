#!/usr/bin/env python3
"""Validate the deterministic RedRHex forward-controller baseline before RL.

This gate is intentionally dependency-light: it parses the Isaac environment
configuration without importing Isaac Lab, then checks the shared NumPy action
decoder against the controller equations used by the environment.  The optional
``--isaac`` mode adds a real zero-residual rollout; the default makes no simulator
displacement or stability claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG_PATH = Path("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py")
SENSOR_CFG_PATH = Path(
    "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_sensor_v2_env_cfg.py"
)
EVAL_SWEEP_PATH = Path("scripts/rsl_rl/eval_command_sweep.py")
F0_PROVENANCE_PATHS = (
    Path("scripts/rsl_rl/validate_forward_gait_baseline.py"),
    ENV_CFG_PATH,
    SENSOR_CFG_PATH,
    EVAL_SWEEP_PATH,
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_sensor_v2_env.py"),
    Path("source/RedRhex/RedRhex/tasks/direct/redrhex/sensor_v2_action.py"),
    Path("source/redrhex_policy_io/redrhex_policy_io/_action_core.py"),
    Path("source/redrhex_policy_io/redrhex_policy_io/contracts.py"),
)
ISAAC_TASK = "Template-Redrhex-ForwardSensorV2-Direct-v0"
F0_FORWARD_COMMANDS_M_S = (0.22, 0.35, 0.42)
CANONICAL_MAIN_JOINT_ORDER = (
    "Revolute_15",
    "Revolute_7",
    "Revolute_12",
    "Revolute_18",
    "Revolute_23",
    "Revolute_24",
)
CANONICAL_ABAD_JOINT_ORDER = (
    "Revolute_14",
    "Revolute_6",
    "Revolute_11",
    "Revolute_17",
    "Revolute_22",
    "Revolute_21",
)
CANONICAL_DIRECTION_MULTIPLIER = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
EXPECTED_RESET_EFFECTIVE_TRIPOD_A_PHASE_RAD = -math.pi / 4.0
EXPECTED_RESET_TRIPOD_SEPARATION_RAD = 0.0
EXPECTED_GAIT_FREQUENCY_HZ = 0.9
EXPECTED_STANCE_DUTY_CYCLE = 0.65
RESET_PHASE_TOLERANCE_RAD = 0.05
NUMERIC_TOLERANCE = 1.0e-8


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _safe_eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "math"
    ):
        return getattr(math, node.attr)
    if isinstance(node, ast.List):
        return [_safe_eval(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _safe_eval(key): _safe_eval(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    raise ValueError(f"unsupported configuration expression: {ast.dump(node)}")


def _assignment_nodes(tree: ast.Module, class_name: str) -> dict[str, ast.AST]:
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    result: dict[str, ast.AST] = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                result[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                result[node.target.id] = node.value
    return result


def _module_assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise ValueError(f"module assignment {name!r} was not found")


def _call_keyword(node: ast.AST, name: str) -> ast.AST:
    if not isinstance(node, ast.Call):
        raise ValueError(f"expected a call containing keyword {name!r}")
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    raise ValueError(f"call keyword {name!r} was not found")


def _dict_value_node(node: ast.AST, name: str) -> ast.AST:
    if not isinstance(node, ast.Dict):
        raise ValueError(f"expected a dictionary containing key {name!r}")
    for key, value in zip(node.keys, node.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == name:
            return value
    raise ValueError(f"dictionary key {name!r} was not found")


def _argument_default(tree: ast.Module, option: str) -> Any:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
            continue
        option_names = {
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        }
        if option not in option_names:
            continue
        return _safe_eval(_call_keyword(node, "default"))
    raise ValueError(f"evaluation argument {option!r} was not found")


def _load_forward_acceptance_thresholds(repo_root: Path = REPO_ROOT) -> dict[str, float]:
    """Load the authoritative forward defaults without importing the Isaac evaluator."""

    tree = ast.parse((repo_root / EVAL_SWEEP_PATH).read_text(encoding="utf-8"))
    options = {
        "forward_abs_m_s": "--accept_vx_abs",
        "forward_command_ratio": "--accept_lin_ratio",
        "lateral_leak_m_s": "--accept_forward_lateral_leak",
        "yaw_leak_rad_s": "--accept_forward_yaw_leak",
        "forward_tilt_bound_rad": "--accept-forward-tilt-bound",
        "forward_min_base_height_m": "--accept-forward-min-base-height",
        "max_fall_rate": "--accept_max_fall_rate",
        "accept_duration_s": "--accept_duration_s",
        "contiguous_env_ratio": "--accept-contiguous-env-ratio",
    }
    thresholds = {
        name: float(_argument_default(tree, option)) for name, option in options.items()
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in thresholds.values()):
        raise ValueError("forward acceptance thresholds must be finite and non-negative")
    for name in ("forward_command_ratio", "max_fall_rate", "contiguous_env_ratio"):
        if thresholds[name] > 1.0:
            raise ValueError(f"forward acceptance threshold {name!r} must be at most one")
    return thresholds


def _load_snapshot(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    env_tree = ast.parse((repo_root / ENV_CFG_PATH).read_text(encoding="utf-8"))
    sensor_tree = ast.parse((repo_root / SENSOR_CFG_PATH).read_text(encoding="utf-8"))
    assignments: dict[str, ast.AST] = {}
    assignments.update(_assignment_nodes(env_tree, "RedrhexEnvCfg"))
    assignments.update(_assignment_nodes(env_tree, "RedrhexForwardFastEnvCfg"))
    assignments.update(_assignment_nodes(sensor_tree, "RedrhexForwardSensorV2EnvCfg"))

    def value(name: str) -> Any:
        if name not in assignments:
            raise ValueError(f"required configuration value {name!r} was not found")
        return _safe_eval(assignments[name])

    sim_dt = _safe_eval(_call_keyword(assignments["sim"], "dt"))
    robot_cfg = _module_assignment(env_tree, "REDRHEX_CFG")
    init_state = _call_keyword(robot_cfg, "init_state")
    joint_positions = _safe_eval(_call_keyword(init_state, "joint_pos"))
    main_drive_actuator = _dict_value_node(
        _call_keyword(robot_cfg, "actuators"), "main_drive"
    )
    main_joint_order = tuple(value("main_drive_joint_names"))

    reset_main_position = tuple(float(item) for item in value("v2_reset_main_position_rad"))
    reset_gait_phase = float(value("v2_reset_gait_phase_rad"))

    return {
        "action_space": int(value("action_space")),
        "main_joint_order": main_joint_order,
        "abad_joint_order": tuple(value("abad_joint_names")),
        "direction_multiplier": tuple(float(item) for item in value("leg_direction_multiplier")),
        "tripod_a": tuple(int(item) for item in value("tripod_a_leg_indices")),
        "tripod_b": tuple(int(item) for item in value("tripod_b_leg_indices")),
        "tripod_phase_offset_rad": float(value("tripod_phase_offset")),
        "legacy_reset_main_position_rad": tuple(
            float(joint_positions[name]) for name in main_joint_order
        ),
        "reset_main_position_rad": reset_main_position,
        "reset_abad_position_rad": tuple(
            float(joint_positions[name]) for name in tuple(value("abad_joint_names"))
        ),
        "reset_gait_phase_rad": reset_gait_phase,
        "sim_dt_s": float(sim_dt),
        "decimation": int(value("decimation")),
        "sensor_sample_hz": float(value("sensor_sample_hz")),
        "gait_frequency_hz": float(value("base_gait_frequency")),
        "main_drive_physics_velocity_limit_rad_s": float(
            _safe_eval(_call_keyword(main_drive_actuator, "velocity_limit_sim"))
        ),
        "main_drive_contract_velocity_limit_rad_s": float(
            value("main_drive_contract_velocity_limit_rad_s")
        ),
        "stance_phase_start_rad": float(value("stance_phase_start")),
        "stance_phase_end_rad": float(value("stance_phase_end")),
        "stance_duty_cycle": float(value("stance_duty_cycle")),
        "stance_velocity_ratio": float(value("stance_velocity_ratio")),
        "swing_velocity_ratio": float(value("swing_velocity_ratio")),
        "phase_lock_gain": float(value("forward_phase_lock_gain")),
        "forward_command_reference_m_s": float(value("drive_bias_vx_ref")),
        "drive_velocity_scale": float(value("stage_drive_vel_scale")[0]),
        "forward_bias_scale": float(value("stage_forward_bias_scale")[0]),
        "forward_residual_scale": float(value("stage_forward_policy_drive_residual_scale")[0]),
        "forward_residual_cap_ratio": float(value("stage_forward_residual_cap_ratio")[0]),
        "action_warmup_steps": int(value("stage_action_warmup_steps")[0]),
        "mode_forward_min_vx": float(value("mode_forward_min_vx")),
        "mode_lin_zero_thresh": float(value("mode_lin_zero_thresh")),
        "mode_yaw_zero_thresh": float(value("mode_yaw_zero_thresh")),
        "strict_forward_residual_actions": bool(value("strict_forward_residual_actions")),
    }


def _wrapped_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _circular_mean(values: Sequence[float]) -> float:
    return math.atan2(
        sum(math.sin(value) for value in values),
        sum(math.cos(value) for value in values),
    )


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "details": details}


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return bool(math.isclose(float(actual), expected, abs_tol=NUMERIC_TOLERANCE))
    if isinstance(expected, tuple) and all(
        isinstance(item, (int, float)) for item in expected
    ):
        return bool(np.allclose(actual, expected, atol=NUMERIC_TOLERANCE, rtol=0.0))
    return actual == expected


def _time_warped_reference(
    gait_phase: np.ndarray | float,
    snapshot: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent reference for the serialized 65/35 time-to-angle map."""

    phase = np.asarray(gait_phase, dtype=np.float64)
    offsets = np.zeros(6, dtype=np.float64)
    offsets[list(snapshot["tripod_b"])] = snapshot["tripod_phase_offset_rad"]
    time_phase = np.mod(phase[..., None] + offsets, 2.0 * math.pi)
    time_fraction = time_phase / (2.0 * math.pi)
    duty = snapshot["stance_duty_cycle"]
    in_stance = time_fraction < duty
    stance_start = snapshot["stance_phase_start_rad"] % (2.0 * math.pi)
    stance_arc = (
        snapshot["stance_phase_end_rad"] - snapshot["stance_phase_start_rad"]
    ) % (2.0 * math.pi)
    swing_arc = 2.0 * math.pi - stance_arc
    desired = np.where(
        in_stance,
        stance_start + stance_arc * time_fraction / duty,
        stance_start
        + stance_arc
        + swing_arc * (time_fraction - duty) / (1.0 - duty),
    )
    base_velocity = 2.0 * math.pi * snapshot["gait_frequency_hz"]
    profile = np.where(
        in_stance,
        base_velocity * snapshot["stance_velocity_ratio"],
        base_velocity * snapshot["swing_velocity_ratio"],
    )
    return np.mod(desired, 2.0 * math.pi), profile, in_stance


def _reference_decode(snapshot: dict[str, Any]) -> tuple[np.ndarray, ...]:
    action = np.asarray(
        [0.25, -0.50, 0.75, -1.20, 0.10, 0.90, 0.50, -0.50, 1.20, 0.0, 0.2, -0.2],
        dtype=np.float64,
    )
    command_vx = 0.32
    gait_phase = 0.37
    raw_position = np.asarray([0.20, -0.70, 1.10, -0.40, 0.80, -1.30], dtype=np.float64)
    direction = np.asarray(snapshot["direction_multiplier"], dtype=np.float64)
    safe_action = np.clip(action, -1.0, 1.0)
    safe_action[6:] = 0.0
    desired_phase, profile, _ = _time_warped_reference(gait_phase, snapshot)
    effective_position = raw_position * direction
    phase_error = np.arctan2(
        np.sin(effective_position - desired_phase),
        np.cos(effective_position - desired_phase),
    )
    correction = np.clip(-snapshot["phase_lock_gain"] * phase_error, -2.0, 2.0)
    vx_scale = np.clip(command_vx / snapshot["forward_command_reference_m_s"], 0.0, 1.0)
    nominal = (
        (profile + correction)
        * direction
        * vx_scale
        * snapshot["forward_bias_scale"]
    )
    residual = (
        safe_action[:6]
        * snapshot["drive_velocity_scale"]
        * snapshot["forward_residual_scale"]
    )
    residual_cap = np.maximum(
        np.abs(nominal) * snapshot["forward_residual_cap_ratio"], 0.08
    )
    residual = np.clip(residual, -residual_cap, residual_cap)
    velocity_limit = snapshot["main_drive_contract_velocity_limit_rad_s"]
    target = np.clip(nominal + residual, -velocity_limit, velocity_limit)
    return action, safe_action, nominal, residual, target, raw_position, np.asarray(
        [command_vx, 0.0, 0.0], dtype=np.float64
    ), np.asarray(gait_phase)


def _shared_action_contract(repo_root: Path, snapshot: dict[str, Any]) -> Any:
    """Build the exact config-derived contract without importing Isaac Lab."""

    package_root = repo_root / "source" / "redrhex_policy_io"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    factory_path = (
        repo_root
        / "source"
        / "RedRhex"
        / "RedRhex"
        / "tasks"
        / "direct"
        / "redrhex"
        / "sensor_v2_action.py"
    )
    spec = importlib.util.spec_from_file_location("redrhex_f0_sensor_v2_action", factory_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Sensor V2 action factory from {factory_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    joint_positions = {
        # Deliberately leave robot_cfg on the legacy pose.  Contract parity must
        # therefore prove that the explicit V2 override wins, rather than
        # accidentally passing because both inputs contain the same values.
        **dict(zip(snapshot["main_joint_order"], snapshot["legacy_reset_main_position_rad"])),
        **dict(zip(snapshot["abad_joint_order"], snapshot["reset_abad_position_rad"])),
    }
    cfg = SimpleNamespace(
        action_space=snapshot["action_space"],
        main_drive_joint_names=list(snapshot["main_joint_order"]),
        abad_joint_names=list(snapshot["abad_joint_order"]),
        tripod_a_leg_indices=list(snapshot["tripod_a"]),
        tripod_b_leg_indices=list(snapshot["tripod_b"]),
        leg_direction_multiplier=list(snapshot["direction_multiplier"]),
        base_gait_frequency=snapshot["gait_frequency_hz"],
        stance_phase_start=snapshot["stance_phase_start_rad"],
        stance_phase_end=snapshot["stance_phase_end_rad"],
        stance_duty_cycle=snapshot["stance_duty_cycle"],
        stance_velocity_ratio=snapshot["stance_velocity_ratio"],
        swing_velocity_ratio=snapshot["swing_velocity_ratio"],
        tripod_phase_offset=snapshot["tripod_phase_offset_rad"],
        strict_forward_residual_actions=snapshot["strict_forward_residual_actions"],
        stage_drive_vel_scale=[snapshot["drive_velocity_scale"]],
        stage_forward_policy_drive_residual_scale=[snapshot["forward_residual_scale"]],
        drive_bias_vx_ref=snapshot["forward_command_reference_m_s"],
        main_drive_contract_velocity_limit_rad_s=snapshot[
            "main_drive_contract_velocity_limit_rad_s"
        ],
        stage_forward_bias_scale=[snapshot["forward_bias_scale"]],
        forward_phase_lock_gain=snapshot["phase_lock_gain"],
        stage_forward_residual_cap_ratio=[snapshot["forward_residual_cap_ratio"]],
        stage_action_warmup_steps=[snapshot["action_warmup_steps"]],
        mode_forward_min_vx=snapshot["mode_forward_min_vx"],
        mode_lin_zero_thresh=snapshot["mode_lin_zero_thresh"],
        mode_yaw_zero_thresh=snapshot["mode_yaw_zero_thresh"],
        v2_reset_main_position_rad=snapshot["reset_main_position_rad"],
        v2_reset_gait_phase_rad=snapshot["reset_gait_phase_rad"],
        robot_cfg=SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=joint_positions),
        ),
    )
    return module.forward_residual_action_contract_v2_from_config(cfg)


def _configured_reset_contract_check(contract: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    configured = np.asarray(snapshot["reset_main_position_rad"], dtype=np.float64)
    bound = np.asarray(contract.initial_main_position_rad, dtype=np.float64)
    matches = configured.shape == (6,) and np.allclose(
        configured,
        bound,
        atol=NUMERIC_TOLERANCE,
        rtol=0.0,
    )
    return _check(
        "configured_reset_contract_parity",
        bool(matches),
        configured_main_position_rad=[float(value) for value in configured],
        contract_main_position_rad=[float(value) for value in bound],
        contract_sha256=str(contract.sha256),
        source="RedrhexForwardSensorV2EnvCfg.v2_reset_main_position_rad",
    )


def _shared_decoder_check(
    repo_root: Path,
    snapshot: dict[str, Any],
    contract: Any,
) -> dict[str, Any]:
    package_root = repo_root / "source" / "redrhex_policy_io"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from redrhex_policy_io.action import decode_forward_residual_action_v2

    expected_constants = {
        "main_joint_order": tuple(contract.MAIN_JOINT_ORDER),
        "abad_joint_order": tuple(contract.ABAD_JOINT_ORDER),
        "direction_multiplier": tuple(contract.LEG_DIRECTION_MULTIPLIER),
        "tripod_a": tuple(contract.TRIPOD_A),
        "tripod_b": tuple(contract.TRIPOD_B),
        "tripod_phase_offset_rad": float(contract.TRIPOD_PHASE_OFFSET_RAD),
        "policy_rate_hz": float(contract.POLICY_RATE_HZ),
        "gait_frequency_hz": float(contract.NOMINAL_GAIT_FREQUENCY_HZ),
        "main_drive_contract_velocity_limit_rad_s": float(
            contract.main_velocity_limit_rad_s
        ),
        "reset_main_position_rad": tuple(contract.initial_main_position_rad),
    }
    env_constants = {
        key: snapshot[key]
        for key in (
            "main_joint_order",
            "abad_joint_order",
            "direction_multiplier",
            "tripod_a",
            "tripod_b",
            "tripod_phase_offset_rad",
            "gait_frequency_hz",
            "main_drive_contract_velocity_limit_rad_s",
            "reset_main_position_rad",
        )
    }
    env_constants["policy_rate_hz"] = 1.0 / (
        snapshot["sim_dt_s"] * snapshot["decimation"]
    )
    constants_match = all(
        _values_match(env_constants[key], expected)
        for key, expected in expected_constants.items()
    )

    raw_action, safe_action, nominal, residual, target, raw_position, command, gait_phase = (
        _reference_decode(snapshot)
    )
    decoded = decode_forward_residual_action_v2(
        raw_action,
        command,
        float(gait_phase),
        raw_position,
        contract=contract,
    )
    comparisons = {
        "safe_action": np.allclose(decoded.safe_action, safe_action, atol=NUMERIC_TOLERANCE),
        "nominal_velocity": np.allclose(
            decoded.nominal_main_velocity_rad_s, nominal, atol=NUMERIC_TOLERANCE
        ),
        "residual_velocity": np.allclose(
            decoded.residual_main_velocity_rad_s, residual, atol=NUMERIC_TOLERANCE
        ),
        "target_velocity": np.allclose(
            decoded.target_main_velocity_rad_s, target, atol=NUMERIC_TOLERANCE
        ),
        "neutral_abad": np.allclose(
            decoded.target_abad_position_rad,
            np.zeros(6),
            atol=NUMERIC_TOLERANCE,
        ),
    }
    saturating_phase = 0.8 * 2.0 * math.pi
    saturating_desired, saturating_profile, _ = _time_warped_reference(
        saturating_phase, snapshot
    )
    saturating_effective_position = saturating_desired - math.pi
    saturating_raw_position = saturating_effective_position / np.asarray(
        snapshot["direction_multiplier"], dtype=np.float64
    )
    saturating_action = np.ones(12, dtype=np.float64)
    saturating_action[6:] = 0.0
    saturating_command = np.asarray(
        [snapshot["forward_command_reference_m_s"], 0.0, 0.0],
        dtype=np.float64,
    )
    saturating_phase_error = np.full(6, -math.pi, dtype=np.float64)
    saturating_correction = np.clip(
        -snapshot["phase_lock_gain"] * saturating_phase_error,
        -2.0,
        2.0,
    )
    saturating_nominal = (
        (saturating_profile + saturating_correction)
        * np.asarray(snapshot["direction_multiplier"], dtype=np.float64)
        * snapshot["forward_bias_scale"]
    )
    saturating_residual = (
        saturating_action[:6]
        * snapshot["drive_velocity_scale"]
        * snapshot["forward_residual_scale"]
    )
    saturating_cap = np.maximum(
        np.abs(saturating_nominal) * snapshot["forward_residual_cap_ratio"],
        0.08,
    )
    saturating_unclipped = saturating_nominal + np.clip(
        saturating_residual, -saturating_cap, saturating_cap
    )
    saturating_expected = np.clip(
        saturating_unclipped,
        -snapshot["main_drive_contract_velocity_limit_rad_s"],
        snapshot["main_drive_contract_velocity_limit_rad_s"],
    )
    saturating_decoded = decode_forward_residual_action_v2(
        saturating_action,
        saturating_command,
        saturating_phase,
        saturating_raw_position,
        contract=contract,
    )
    saturating_actual = np.asarray(
        saturating_decoded.target_main_velocity_rad_s, dtype=np.float64
    )
    saturation_exercised = bool(
        np.max(np.abs(saturating_unclipped))
        > snapshot["main_drive_contract_velocity_limit_rad_s"]
        and np.isclose(
            np.max(np.abs(saturating_actual)),
            snapshot["main_drive_contract_velocity_limit_rad_s"],
            atol=NUMERIC_TOLERANCE,
        )
        and np.allclose(
            saturating_actual,
            saturating_expected,
            atol=NUMERIC_TOLERANCE,
            rtol=0.0,
        )
    )
    return _check(
        "shared_decoder_parity",
        constants_match and all(comparisons.values()) and saturation_exercised,
        constants_match=bool(constants_match),
        equation_comparisons={key: bool(value) for key, value in comparisons.items()},
        max_target_error_rad_s=float(
            np.max(np.abs(np.asarray(decoded.target_main_velocity_rad_s) - target))
        ),
        velocity_limit_saturation_exercised=saturation_exercised,
        saturating_unclipped_abs_max_rad_s=float(
            np.max(np.abs(saturating_unclipped))
        ),
        saturating_target_abs_max_rad_s=float(np.max(np.abs(saturating_actual))),
    )


def build_report(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        snapshot = _load_snapshot(repo_root)
    except (OSError, SyntaxError, StopIteration, TypeError, ValueError) as exc:
        checks.append(_check("configuration_snapshot", False, error=str(exc)))
        return {
            "schema_version": "redrhex.forward-gait-f0.v2",
            "overall_status": "FAIL",
            "checks": checks,
            "simulator_rollout": {
                "status": "NOT_RUN",
                "reason": "structural configuration could not be loaded",
            },
        }

    checks.append(
        _check(
            "canonical_six_joint_order",
            snapshot["main_joint_order"] == CANONICAL_MAIN_JOINT_ORDER
            and snapshot["abad_joint_order"] == CANONICAL_ABAD_JOINT_ORDER,
            main_joint_order=list(snapshot["main_joint_order"]),
            abad_joint_order=list(snapshot["abad_joint_order"]),
        )
    )
    checks.append(
        _check(
            "direction_multipliers",
            snapshot["direction_multiplier"] == CANONICAL_DIRECTION_MULTIPLIER,
            actual=list(snapshot["direction_multiplier"]),
            expected=list(CANONICAL_DIRECTION_MULTIPLIER),
        )
    )

    tripod_a = set(snapshot["tripod_a"])
    tripod_b = set(snapshot["tripod_b"])
    tripods_valid = (
        len(snapshot["tripod_a"]) == 3
        and len(snapshot["tripod_b"]) == 3
        and len(tripod_a) == 3
        and len(tripod_b) == 3
        and tripod_a.isdisjoint(tripod_b)
        and tripod_a | tripod_b == set(range(6))
        and math.isclose(
            snapshot["tripod_phase_offset_rad"], math.pi, abs_tol=NUMERIC_TOLERANCE
        )
    )
    checks.append(
        _check(
            "tripod_partition",
            tripods_valid,
            tripod_a=list(snapshot["tripod_a"]),
            tripod_b=list(snapshot["tripod_b"]),
            phase_offset_rad=snapshot["tripod_phase_offset_rad"],
        )
    )

    reset_position = np.asarray(snapshot["reset_main_position_rad"], dtype=np.float64)
    direction = np.asarray(snapshot["direction_multiplier"], dtype=np.float64)
    effective_phase = np.mod(reset_position * direction, 2.0 * math.pi)
    phase_a = effective_phase[list(snapshot["tripod_a"])]
    phase_b = effective_phase[list(snapshot["tripod_b"])]
    mean_a = _circular_mean(phase_a)
    mean_b = _circular_mean(phase_b)
    spread_a = max(abs(_wrapped_angle(float(value) - mean_a)) for value in phase_a)
    spread_b = max(abs(_wrapped_angle(float(value) - mean_b)) for value in phase_b)
    separation = abs(_wrapped_angle(mean_b - mean_a))
    separation_error = abs(separation - EXPECTED_RESET_TRIPOD_SEPARATION_RAD)
    phase_a_anchor_error = abs(
        _wrapped_angle(mean_a - EXPECTED_RESET_EFFECTIVE_TRIPOD_A_PHASE_RAD)
    )
    phase_b_anchor_error = abs(
        _wrapped_angle(mean_b - EXPECTED_RESET_EFFECTIVE_TRIPOD_A_PHASE_RAD)
    )
    reset_coherent = (
        spread_a <= RESET_PHASE_TOLERANCE_RAD
        and spread_b <= RESET_PHASE_TOLERANCE_RAD
        and separation_error <= RESET_PHASE_TOLERANCE_RAD
        and phase_a_anchor_error <= RESET_PHASE_TOLERANCE_RAD
        and phase_b_anchor_error <= RESET_PHASE_TOLERANCE_RAD
    )
    checks.append(
        _check(
            "reset_effective_phase_coherence",
            reset_coherent,
            effective_phase_rad=[float(value) for value in effective_phase],
            tripod_a_spread_rad=float(spread_a),
            tripod_b_spread_rad=float(spread_b),
            tripod_a_mean_rad=float(mean_a),
            expected_tripod_a_mean_rad=EXPECTED_RESET_EFFECTIVE_TRIPOD_A_PHASE_RAD,
            tripod_a_anchor_error_rad=float(phase_a_anchor_error),
            tripod_b_anchor_error_rad=float(phase_b_anchor_error),
            tripod_separation_rad=float(separation),
            expected_separation_rad=EXPECTED_RESET_TRIPOD_SEPARATION_RAD,
            separation_error_rad=float(separation_error),
            tolerance_rad=RESET_PHASE_TOLERANCE_RAD,
            reset_semantics=(
                "supported neutral pose; the pi tripod offset belongs to the "
                "time-warped CPG reference, not the physical reset pose"
            ),
        )
    )

    desired_reset_phase, _, _ = _time_warped_reference(
        snapshot["reset_gait_phase_rad"], snapshot
    )
    initial_phase_error = np.arctan2(
        np.sin(effective_phase - desired_reset_phase),
        np.cos(effective_phase - desired_reset_phase),
    )
    maximum_phase_error = float(np.max(np.abs(initial_phase_error)))
    phase_correction_limit = 2.0
    unsaturated_error_bound = phase_correction_limit / snapshot["phase_lock_gain"]
    maximum_phase_correction = float(
        np.max(np.abs(snapshot["phase_lock_gain"] * initial_phase_error))
    )
    phase_lock_bounded = (
        math.isfinite(snapshot["reset_gait_phase_rad"])
        and maximum_phase_error <= unsaturated_error_bound + NUMERIC_TOLERANCE
        and maximum_phase_correction <= phase_correction_limit + NUMERIC_TOLERANCE
    )
    checks.append(
        _check(
            "initial_phase_lock_error_bound",
            phase_lock_bounded,
            reset_gait_phase_rad=snapshot["reset_gait_phase_rad"],
            desired_effective_phase_rad=[float(value) for value in desired_reset_phase],
            initial_phase_error_rad=[float(value) for value in initial_phase_error],
            maximum_abs_error_rad=maximum_phase_error,
            unsaturated_error_bound_rad=float(unsaturated_error_bound),
            maximum_abs_correction_rad_s=maximum_phase_correction,
            phase_correction_limit_rad_s=phase_correction_limit,
            correction_saturated=bool(
                maximum_phase_correction > phase_correction_limit + NUMERIC_TOLERANCE
            ),
        )
    )

    sample_phases = np.linspace(0.0, 2.0 * math.pi, 6000, endpoint=False)
    _, sampled_profile, sampled_stance = _time_warped_reference(
        sample_phases, snapshot
    )
    tripod_a_stance = np.all(
        sampled_stance[:, list(snapshot["tripod_a"])], axis=1
    )
    tripod_b_stance = np.all(
        sampled_stance[:, list(snapshot["tripod_b"])], axis=1
    )
    tripod_a_duty = float(np.mean(tripod_a_stance))
    tripod_b_duty = float(np.mean(tripod_b_stance))
    overlap_fraction = float(np.mean(tripod_a_stance & tripod_b_stance))
    support_fraction = float(np.mean(tripod_a_stance | tripod_b_stance))
    stance_arc_fraction = (
        (
            snapshot["stance_phase_end_rad"]
            - snapshot["stance_phase_start_rad"]
        )
        % (2.0 * math.pi)
    ) / (2.0 * math.pi)
    expected_overlap = max(0.0, 2.0 * snapshot["stance_duty_cycle"] - 1.0)
    expected_stance_ratio = stance_arc_fraction / snapshot["stance_duty_cycle"]
    expected_swing_ratio = (1.0 - stance_arc_fraction) / (
        1.0 - snapshot["stance_duty_cycle"]
    )
    duty_tolerance = 1.0 / len(sample_phases) + NUMERIC_TOLERANCE
    time_warp_valid = (
        math.isclose(
            snapshot["stance_duty_cycle"],
            EXPECTED_STANCE_DUTY_CYCLE,
            abs_tol=NUMERIC_TOLERANCE,
        )
        and abs(tripod_a_duty - snapshot["stance_duty_cycle"]) <= duty_tolerance
        and abs(tripod_b_duty - snapshot["stance_duty_cycle"]) <= duty_tolerance
        and abs(overlap_fraction - expected_overlap) <= duty_tolerance
        and abs(support_fraction - 1.0) <= duty_tolerance
        and math.isclose(
            snapshot["stance_velocity_ratio"],
            expected_stance_ratio,
            abs_tol=NUMERIC_TOLERANCE,
        )
        and math.isclose(
            snapshot["swing_velocity_ratio"],
            expected_swing_ratio,
            abs_tol=NUMERIC_TOLERANCE,
        )
        and np.isfinite(sampled_profile).all()
    )
    checks.append(
        _check(
            "time_warped_duty_cycle",
            time_warp_valid,
            stance_duty_cycle=snapshot["stance_duty_cycle"],
            tripod_a_observed_duty=tripod_a_duty,
            tripod_b_observed_duty=tripod_b_duty,
            tripod_overlap_fraction=overlap_fraction,
            expected_overlap_fraction=expected_overlap,
            continuous_tripod_support_fraction=support_fraction,
            stance_velocity_ratio=snapshot["stance_velocity_ratio"],
            expected_stance_velocity_ratio=expected_stance_ratio,
            swing_velocity_ratio=snapshot["swing_velocity_ratio"],
            expected_swing_velocity_ratio=expected_swing_ratio,
            sample_count=len(sample_phases),
        )
    )

    velocity_limit_bound = math.isclose(
        snapshot["main_drive_contract_velocity_limit_rad_s"],
        snapshot["main_drive_physics_velocity_limit_rad_s"],
        abs_tol=NUMERIC_TOLERANCE,
    )
    checks.append(
        _check(
            "main_drive_velocity_limit_binding",
            velocity_limit_bound,
            contract_limit_rad_s=snapshot[
                "main_drive_contract_velocity_limit_rad_s"
            ],
            physics_limit_rad_s=snapshot[
                "main_drive_physics_velocity_limit_rad_s"
            ],
            semantics=(
                "raw simulator parity target is capped before the PhysX actuator; "
                "ROS may only tighten this limit"
            ),
        )
    )

    policy_rate_hz = 1.0 / (snapshot["sim_dt_s"] * snapshot["decimation"])
    timing_valid = (
        math.isclose(policy_rate_hz, 60.0, abs_tol=NUMERIC_TOLERANCE)
        and math.isclose(
            snapshot["sensor_sample_hz"], policy_rate_hz, abs_tol=NUMERIC_TOLERANCE
        )
        and math.isclose(
            snapshot["gait_frequency_hz"],
            EXPECTED_GAIT_FREQUENCY_HZ,
            abs_tol=NUMERIC_TOLERANCE,
        )
    )
    checks.append(
        _check(
            "timing_and_rates",
            timing_valid,
            sim_dt_s=snapshot["sim_dt_s"],
            decimation=snapshot["decimation"],
            policy_rate_hz=policy_rate_hz,
            sensor_sample_hz=snapshot["sensor_sample_hz"],
            gait_frequency_hz=snapshot["gait_frequency_hz"],
        )
    )

    try:
        contract = _shared_action_contract(repo_root, snapshot)
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
        checks.append(_check("configured_reset_contract_parity", False, error=str(exc)))
        checks.append(_check("shared_decoder_parity", False, error=str(exc)))
    else:
        checks.append(_configured_reset_contract_check(contract, snapshot))
        try:
            checks.append(_shared_decoder_check(repo_root, snapshot, contract))
        except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
            checks.append(_check("shared_decoder_parity", False, error=str(exc)))

    overall_status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    return {
        "schema_version": "redrhex.forward-gait-f0.v2",
        "overall_status": overall_status,
        "checks": checks,
        "simulator_rollout": {
            "status": "NOT_RUN",
            "reason": (
                "This dependency-light F0 gate provides structural evidence only; "
                "no Isaac displacement, drift, contact, or fall claim was made."
            ),
        },
    }


def build_evidence_provenance(
    repo_root: Path = REPO_ROOT,
    *,
    structural_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind F0 evidence to the current decoder, task config, and thresholds."""

    report = structural_report or build_report(repo_root)
    contract_hash = None
    for check in report.get("checks", []):
        if (
            isinstance(check, dict)
            and check.get("name") == "configured_reset_contract_parity"
            and isinstance(check.get("details"), dict)
        ):
            contract_hash = check["details"].get("contract_sha256")
            break
    if (
        not isinstance(contract_hash, str)
        or len(contract_hash) != 64
        or any(character not in "0123456789abcdef" for character in contract_hash)
    ):
        raise ValueError("F0 provenance cannot resolve the action-contract hash")
    thresholds = _load_forward_acceptance_thresholds(repo_root)
    file_hashes = {
        str(relative): _sha256_file(repo_root / relative)
        for relative in F0_PROVENANCE_PATHS
    }
    return {
        "schema": "redrhex.forward-gait-f0-provenance.v2",
        "task": ISAAC_TASK,
        "action_contract_sha256": contract_hash,
        "commands_sha256": _canonical_sha256(list(F0_FORWARD_COMMANDS_M_S)),
        "thresholds_sha256": _canonical_sha256(thresholds),
        "source_files": file_hashes,
    }


def _reference_relative_tilt(projected_gravity: Any, reference_gravity: Any) -> Any:
    """Match the reference-relative tilt definition used by eval_command_sweep."""

    import torch

    gravity = projected_gravity / torch.linalg.vector_norm(
        projected_gravity, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    reference = reference_gravity / torch.linalg.vector_norm(
        reference_gravity, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    if reference.shape == (3,):
        reference = reference.unsqueeze(0).expand_as(gravity)
    if reference.shape != gravity.shape:
        raise ValueError("reference projected gravity shape does not match F0 batch")
    return torch.acos(torch.sum(gravity * reference, dim=-1).clamp(-1.0, 1.0))


def _finite_metric(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Isaac F0 metric {name!r} is NaN or Inf")
    return result


def _run_isaac_rollout(
    args: argparse.Namespace, thresholds: dict[str, float]
) -> dict[str, Any]:
    """Run fixed forward commands with zero residuals through the real task."""

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import RedRhex.tasks as redrhex_tasks
    from tools.sim2real.repo_binding import assert_redrhex_module_source

    assert_redrhex_module_source(redrhex_tasks, REPO_ROOT)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = int(args.seed)
    if hasattr(env_cfg, "spring_backend"):
        env_cfg.spring_backend = args.spring_backend
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    rows: list[dict[str, Any]] = []
    try:
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "commands"):
            raise RuntimeError("F0 task does not expose an external command buffer")
        unwrapped.external_control = True
        if hasattr(unwrapped.cfg, "play_forward_compat_enable"):
            unwrapped.cfg.play_forward_compat_enable = False

        num_envs = int(unwrapped.num_envs)
        device = unwrapped.device
        zero_actions = torch.zeros(num_envs, 12, device=device)
        step_dt = float(
            getattr(
                unwrapped,
                "step_dt",
                float(unwrapped.cfg.sim.dt) * int(unwrapped.cfg.decimation),
            )
        )
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise RuntimeError("Isaac F0 control timestep must be finite and positive")

        def step_with_command(vx: float) -> tuple[Any, Any]:
            command = unwrapped.commands.new_tensor((vx, 0.0, 0.0))
            unwrapped.commands[:] = command
            _, _, terminated, truncated, _ = env.step(zero_actions)
            # A reset during step may resample a command. Restore the fixed F0 command.
            unwrapped.commands[:] = command
            return terminated.bool(), truncated.bool()

        for command_vx in F0_FORWARD_COMMANDS_M_S:
            unwrapped.commands[:] = unwrapped.commands.new_tensor((0.0, 0.0, 0.0))
            env.reset()
            for _ in range(args.settle_steps):
                step_with_command(0.0)
            for _ in range(args.warmup_steps):
                step_with_command(command_vx)

            previous_x = unwrapped.robot.data.root_pos_w[:, 0].clone()
            displacement = torch.zeros(num_envs, device=device)
            contiguous_success = torch.zeros(num_envs, dtype=torch.long, device=device)
            max_contiguous_success = torch.zeros_like(contiguous_success)
            action_contract = getattr(unwrapped, "_action_contract_v2", None)
            if action_contract is None or not hasattr(
                action_contract, "command_scaled_cycle_steps"
            ):
                raise RuntimeError(
                    "Sensor V2 F0 requires the contract-bound gait-cycle window"
                )
            gait_cycle_window_steps = int(
                action_contract.command_scaled_cycle_steps(command_vx)
            )
            rolling_forward = torch.zeros(
                gait_cycle_window_steps, num_envs, device=device
            )
            rolling_lateral = torch.zeros_like(rolling_forward)
            rolling_yaw = torch.zeros_like(rolling_forward)
            rolling_forward_sum = torch.zeros(num_envs, device=device)
            rolling_lateral_sum = torch.zeros_like(rolling_forward_sum)
            rolling_yaw_sum = torch.zeros_like(rolling_forward_sum)
            rolling_valid_steps = torch.zeros(
                num_envs, dtype=torch.long, device=device
            )
            rolling_index = 0
            vx_sum = 0.0
            abs_vy_sum = 0.0
            abs_wz_sum = 0.0
            vx_error_sum = 0.0
            tilt_sum = 0.0
            height_sum = 0.0
            tilt_max = -math.inf
            height_min = math.inf
            success_samples = 0
            fall_events = 0
            episode_ends = 0
            abad_target_observed = False
            max_abad_target_offset = 0.0

            required_forward = max(
                thresholds["forward_abs_m_s"],
                thresholds["forward_command_ratio"] * command_vx,
            )
            for _ in range(args.sweep_steps):
                terminated, truncated = step_with_command(command_vx)
                boundary = terminated | truncated
                actual_vx = unwrapped.base_lin_vel[:, 0]
                actual_vy = torch.abs(unwrapped.base_lin_vel[:, 1])
                actual_wz = torch.abs(unwrapped.base_ang_vel[:, 2])
                height = unwrapped.robot.data.root_pos_w[:, 2]
                reference = getattr(
                    unwrapped,
                    "reference_projected_gravity",
                    unwrapped.projected_gravity.new_tensor((0.0, 0.0, -1.0)),
                )
                tilt = _reference_relative_tilt(unwrapped.projected_gravity, reference)

                current_x = unwrapped.robot.data.root_pos_w[:, 0]
                displacement += torch.where(
                    boundary, torch.zeros_like(current_x), current_x - previous_x
                )
                previous_x = current_x.clone()

                vx_sum += float(actual_vx.sum().item())
                abs_vy_sum += float(actual_vy.sum().item())
                abs_wz_sum += float(actual_wz.sum().item())
                vx_error_sum += float(torch.abs(actual_vx - command_vx).sum().item())
                tilt_sum += float(tilt.sum().item())
                height_sum += float(height.sum().item())
                tilt_max = max(tilt_max, float(tilt.max().item()))
                height_min = min(height_min, float(height.min().item()))
                fall_events += int(torch.count_nonzero(terminated).item())
                episode_ends += int(torch.count_nonzero(boundary).item())

                stable_sample = (
                    (tilt <= thresholds["forward_tilt_bound_rad"])
                    & (height >= thresholds["forward_min_base_height_m"])
                    & ~boundary
                )
                rolling_forward_sum += actual_vx - rolling_forward[rolling_index]
                rolling_lateral_sum += actual_vy - rolling_lateral[rolling_index]
                rolling_yaw_sum += actual_wz - rolling_yaw[rolling_index]
                rolling_forward[rolling_index] = actual_vx
                rolling_lateral[rolling_index] = actual_vy
                rolling_yaw[rolling_index] = actual_wz
                rolling_index = (rolling_index + 1) % gait_cycle_window_steps
                rolling_valid_steps = torch.where(
                    stable_sample,
                    rolling_valid_steps + 1,
                    torch.zeros_like(rolling_valid_steps),
                )
                cycle_ready = rolling_valid_steps >= gait_cycle_window_steps
                window_denominator = float(gait_cycle_window_steps)
                success = (
                    cycle_ready
                    & (rolling_forward_sum / window_denominator >= required_forward)
                    & (
                        rolling_lateral_sum / window_denominator
                        <= thresholds["lateral_leak_m_s"]
                    )
                    & (
                        rolling_yaw_sum / window_denominator
                        <= thresholds["yaw_leak_rad_s"]
                    )
                )
                success_samples += int(torch.count_nonzero(success).item())
                contiguous_success = torch.where(
                    success, contiguous_success + 1, torch.zeros_like(contiguous_success)
                )
                max_contiguous_success = torch.maximum(
                    max_contiguous_success, contiguous_success
                )

                if hasattr(unwrapped, "_target_abad_pos") and hasattr(
                    unwrapped, "_abad_rest_pos"
                ):
                    abad_target_observed = True
                    offset = torch.abs(
                        unwrapped._target_abad_pos
                        - unwrapped._abad_rest_pos.expand(num_envs, -1)
                    )
                    max_abad_target_offset = max(
                        max_abad_target_offset, float(offset.max().item())
                    )

            sample_count = args.sweep_steps * num_envs
            denominator = float(sample_count)
            mean_vx = _finite_metric("mean_vx", vx_sum / denominator)
            mean_abs_vy = _finite_metric("mean_abs_vy", abs_vy_sum / denominator)
            mean_abs_wz = _finite_metric("mean_abs_wz", abs_wz_sum / denominator)
            mean_tilt = _finite_metric("mean_tilt", tilt_sum / denominator)
            maximum_tilt = _finite_metric("maximum_tilt", tilt_max)
            mean_height = _finite_metric("mean_height", height_sum / denominator)
            minimum_height = _finite_metric("minimum_height", height_min)
            mean_displacement = _finite_metric(
                "mean_displacement", displacement.mean().item()
            )
            mae_vx = _finite_metric("mae_vx", vx_error_sum / denominator)
            fall_rate = _finite_metric(
                "fall_rate", float(fall_events) / float(max(1, episode_ends))
            )
            contiguous_duration = max_contiguous_success.float() * step_dt
            contiguous_ratio = _finite_metric(
                "contiguous_success_env_ratio",
                torch.mean(
                    (contiguous_duration >= thresholds["accept_duration_s"]).float()
                ).item(),
            )
            forward_mae_limit = max(
                thresholds["forward_abs_m_s"],
                (1.0 - thresholds["forward_command_ratio"]) * command_vx,
            )
            checks = [
                _check("finite_metrics", True),
                _check(
                    "forward_speed",
                    mean_vx >= required_forward,
                    actual_m_s=mean_vx,
                    minimum_m_s=required_forward,
                ),
                _check(
                    "lateral_leak",
                    mean_abs_vy <= thresholds["lateral_leak_m_s"],
                    actual_m_s=mean_abs_vy,
                    maximum_m_s=thresholds["lateral_leak_m_s"],
                ),
                _check(
                    "yaw_leak",
                    mean_abs_wz <= thresholds["yaw_leak_rad_s"],
                    actual_rad_s=mean_abs_wz,
                    maximum_rad_s=thresholds["yaw_leak_rad_s"],
                ),
                _check(
                    "reference_relative_tilt",
                    maximum_tilt <= thresholds["forward_tilt_bound_rad"],
                    actual_max_rad=maximum_tilt,
                    maximum_rad=thresholds["forward_tilt_bound_rad"],
                ),
                _check(
                    "base_height",
                    minimum_height >= thresholds["forward_min_base_height_m"],
                    actual_min_m=minimum_height,
                    minimum_m=thresholds["forward_min_base_height_m"],
                ),
                _check(
                    "fall_rate",
                    fall_rate <= thresholds["max_fall_rate"],
                    actual=fall_rate,
                    maximum=thresholds["max_fall_rate"],
                ),
                _check(
                    "contiguous_success",
                    contiguous_ratio >= thresholds["contiguous_env_ratio"],
                    actual_env_ratio=contiguous_ratio,
                    minimum_env_ratio=thresholds["contiguous_env_ratio"],
                    required_duration_s=thresholds["accept_duration_s"],
                ),
                _check(
                    "forward_mae",
                    mae_vx <= forward_mae_limit,
                    actual_m_s=mae_vx,
                    maximum_m_s=forward_mae_limit,
                ),
                _check(
                    "forward_displacement",
                    mean_displacement > 0.0,
                    actual_mean_m=mean_displacement,
                    minimum_exclusive_m=0.0,
                ),
                _check(
                    "neutral_abad_target",
                    abad_target_observed and max_abad_target_offset <= NUMERIC_TOLERANCE,
                    observed=abad_target_observed,
                    maximum_offset_rad=max_abad_target_offset,
                    tolerance_rad=NUMERIC_TOLERANCE,
                ),
            ]
            row_status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
            rows.append(
                {
                    "command_vx_m_s": command_vx,
                    "status": row_status,
                    "sample_count": sample_count,
                    "mean_forward_displacement_m": mean_displacement,
                    "actual_forward_speed_mean_m_s": mean_vx,
                    "actual_lateral_leak_mean_m_s": mean_abs_vy,
                    "actual_yaw_leak_mean_rad_s": mean_abs_wz,
                    "reference_relative_tilt_mean_rad": mean_tilt,
                    "reference_relative_tilt_max_rad": maximum_tilt,
                    "base_height_mean_m": mean_height,
                    "base_height_min_m": minimum_height,
                    "fall_events": fall_events,
                    "episode_ends": episode_ends,
                    "fall_rate": fall_rate,
                    "forward_mae_m_s": mae_vx,
                    "success_sample_ratio": float(success_samples) / denominator,
                    "contiguous_success_env_ratio": contiguous_ratio,
                    "gait_cycle_window_steps": gait_cycle_window_steps,
                    "gait_cycle_window_duration_s": (
                        gait_cycle_window_steps * step_dt
                    ),
                    "contiguous_success_semantics": (
                        "one_command_scaled_gait_cycle_velocity_means_with_"
                        "pointwise_tilt_height_and_episode_boundary_safety"
                    ),
                    "max_abad_target_offset_rad": max_abad_target_offset,
                    "checks": checks,
                }
            )

        return {
            "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
            "requested": True,
            "mode": "isaac_zero_residual",
            "task": args.task,
            "spring_backend": args.spring_backend,
            "seed": int(args.seed),
            "num_envs": num_envs,
            "policy_step_dt_s": step_dt,
            "settle_steps": int(args.settle_steps),
            "warmup_steps": int(args.warmup_steps),
            "measurement_steps": int(args.sweep_steps),
            "zero_residual_actions": True,
            "neutral_abad_actions": True,
            "commands_vx_m_s": list(F0_FORWARD_COMMANDS_M_S),
            "threshold_source": str(EVAL_SWEEP_PATH),
            "thresholds": thresholds,
            "commands": rows,
        }
    finally:
        env.close()


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True, help="F0 JSON report path.")
    parser.add_argument(
        "--isaac", action="store_true", help="Run the real Isaac zero-residual rollout."
    )
    parser.add_argument(
        "--task",
        choices=(ISAAC_TASK,),
        default=ISAAC_TASK,
        help="Registered Sensor V2 Isaac task for F0.",
    )
    parser.add_argument("--num-envs", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle-steps", type=_positive_int, default=120)
    parser.add_argument("--warmup-steps", type=_positive_int, default=120)
    parser.add_argument("--sweep-steps", type=_positive_int, default=240)
    parser.add_argument(
        "--spring-backend",
        choices=("explicit", "native"),
        default="native",
        help="Passive spring implementation for the Isaac F0 rollout.",
    )
    return parser


def _gate_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "isaac_requested": bool(args.isaac),
        "task": args.task,
        "num_envs": int(args.num_envs),
        "seed": int(args.seed),
        "settle_steps": int(args.settle_steps),
        "warmup_steps": int(args.warmup_steps),
        "measurement_steps": int(args.sweep_steps),
        "commands_vx_m_s": list(F0_FORWARD_COMMANDS_M_S),
        "spring_backend": args.spring_backend,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args, unknown = parser.parse_known_args(arguments)
    report = build_report()
    structural_status = report["overall_status"]
    try:
        report["provenance"] = build_evidence_provenance(
            structural_report=report
        )
    except (OSError, TypeError, ValueError) as exc:
        report["provenance"] = {
            "schema": "redrhex.forward-gait-f0-provenance.v2",
            "status": "UNAVAILABLE",
            "reason": str(exc),
        }
    report["structural_status"] = structural_status
    report["gate_configuration"] = _gate_configuration(args)
    report["simulator_rollout"]["requested"] = bool(args.isaac)

    simulation_app = None
    if args.isaac and structural_status != "PASS":
        report["simulator_rollout"] = {
            "status": "SKIPPED",
            "requested": True,
            "reason": (
                "Isaac rollout was not launched because one or more structural F0 checks failed."
            ),
            "failed_structural_checks": [
                check["name"] for check in report["checks"] if check["status"] == "FAIL"
            ],
        }
    elif args.isaac:
        try:
            thresholds = _load_forward_acceptance_thresholds()
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from tools.sim2real.repo_binding import bind_redrhex_source

            bind_redrhex_source(REPO_ROOT)
            from isaaclab.app import AppLauncher

            AppLauncher.add_app_launcher_args(parser)
            args = parser.parse_args(arguments)
            report["gate_configuration"] = _gate_configuration(args)
            simulation_app = AppLauncher(args).app
            report["simulator_rollout"] = _run_isaac_rollout(args, thresholds)
        except Exception as exc:
            report["simulator_rollout"] = {
                "status": "ERROR",
                "requested": True,
                "reason": "Isaac F0 rollout raised an exception.",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    elif unknown:
        parser.error("unrecognized arguments: " + " ".join(unknown))

    if args.isaac:
        report["overall_status"] = (
            "PASS"
            if structural_status == "PASS"
            and report["simulator_rollout"]["status"] == "PASS"
            else "FAIL"
        )
    write_report(report, args.json)
    print(f"[F0] status={report['overall_status']} report={args.json}")
    for check in report["checks"]:
        print(f"[F0] {check['name']}: {check['status']}")
    print(f"[F0] simulator_rollout: {report['simulator_rollout']['status']}")
    exit_code = 0 if report["overall_status"] == "PASS" else 2
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code and simulation_app is not None:
        # Isaac Kit normalizes the process status to zero during close().  The
        # failing JSON is already durable; terminate the gate process with its
        # required non-zero status and let the OS release simulator resources.
        import os

        os._exit(exit_code)
    if simulation_app is not None:
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    status = main()
    if status:
        # Kit shutdown normalizes its application status to zero.  The F0 CLI is
        # a gate, so preserve the JSON-derived failure status after clean close.
        import os

        os._exit(status)
    raise SystemExit(status)
