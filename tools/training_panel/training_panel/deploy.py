from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .commands import DEFAULT_TASK, export_onnx_argv, resolve_spring_backend, shell_for_isaaclab
from .config import PanelPaths, timestamp_id
from .history import latest_onnx
from .mujoco_rollout import (
    DEFAULT_PLAYBACK_FPS,
    DEFAULT_PLAYBACK_HEIGHT,
    DEFAULT_PLAYBACK_STEPS,
    DEFAULT_PLAYBACK_WIDTH,
    default_scenarios,
    load_calibration_config,
    run_mujoco_rollouts,
)


REPORT_VERSION = 1
DEPLOY_TARGET = "Jetson ROS2"
TARGET_RUNTIME = "ONNX Runtime via redrhex_rl_controller"
DEFAULT_RTOL = 1.0e-4
DEFAULT_ATOL = 1.0e-4
DEFAULT_MUJOCO_STEPS = 250


FALLBACK_CONTRACT = SimpleNamespace(
    OBS_DIM_SINGLE=56,
    ACTION_DIM=12,
    POLICY_HISTORY_LENGTH=5,
    POLICY_HZ=60.0,
    CONTROL_DT=1.0 / 60.0,
    MAIN_DRIVE_JOINT_NAMES=[
        "Revolute_15",
        "Revolute_7",
        "Revolute_12",
        "Revolute_18",
        "Revolute_23",
        "Revolute_24",
    ],
    ABAD_JOINT_NAMES=[
        "Revolute_14",
        "Revolute_6",
        "Revolute_11",
        "Revolute_17",
        "Revolute_22",
        "Revolute_21",
    ],
    DAMPER_JOINT_NAMES=[
        "Revolute_5",
        "Revolute_8",
        "Revolute_13",
        "Revolute_25",
        "Revolute_26",
        "Revolute_27",
    ],
    COMMAND_LIMITS={
        "vx_min": 0.0,
        "vx_max": 0.56,
        "vy_min": -0.60,
        "vy_max": 0.60,
        "wz_min": -0.70,
        "wz_max": 0.70,
    },
    STAGE_ABAD_POS_LIMIT=0.62,
)
FALLBACK_CONTRACT.ALL_CONTROLLED_JOINT_NAMES = (
    FALLBACK_CONTRACT.MAIN_DRIVE_JOINT_NAMES
    + FALLBACK_CONTRACT.ABAD_JOINT_NAMES
    + FALLBACK_CONTRACT.DAMPER_JOINT_NAMES
)
FALLBACK_CONTRACT.OBSERVATION_SLICES = {
    "base_lin_vel": (0, 3),
    "base_ang_vel": (3, 6),
    "projected_gravity": (6, 9),
    "main_drive_pos_sin": (9, 15),
    "main_drive_pos_cos": (15, 21),
    "main_drive_vel_scaled": (21, 27),
    "abad_pos_scaled": (27, 33),
    "abad_vel": (33, 39),
    "velocity_command": (39, 42),
    "gait_phase_sin_cos": (42, 44),
    "last_actions": (44, 56),
}


@dataclass
class PolicyManifest:
    run_id: str
    display_name: str
    target: str
    target_runtime: str
    created_at: str
    log_dir: str
    checkpoint_path: str
    policy_onnx_path: str
    policy_torchscript_path: str
    env_yaml_path: str
    agent_yaml_path: str
    torsion_spring_path: str
    physics_profile_path: str
    physics_profile_metadata_path: str
    deploy_config_path: str
    expected_obs_dim: int
    history_obs_dim: int
    expected_action_dim: int
    policy_history_length: int
    policy_hz: float
    command_limits: dict[str, float]
    main_drive_joint_names: list[str]
    abad_joint_names: list[str]
    damper_joint_names: list[str]
    observation_slices: dict[str, tuple[int, int]]
    hashes: dict[str, str] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationStageResult:
    name: str
    title: str
    status: str
    duration_s: float
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatorSpec:
    name: str
    title: str
    run: Callable[[], ValidationStageResult]
    dependencies: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    skip_next_steps: list[str] = field(default_factory=list)


@dataclass
class DeploymentReport:
    version: int
    pipeline_id: str
    created_at: str
    completed_at: str
    overall_status: str
    readiness_level: str
    manifest: PolicyManifest
    stages: list[ValidationStageResult]
    artifacts: dict[str, str] = field(default_factory=dict)
    operator_checklist: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pipeline_id": self.pipeline_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "overall_status": self.overall_status,
            "readiness_level": self.readiness_level,
            "manifest": self.manifest.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "artifacts": dict(self.artifacts),
            "operator_checklist": list(self.operator_checklist),
            "assumptions": list(self.assumptions),
            "runtime": _json_safe(self.runtime),
        }


def deploy_defaults(paths: PanelPaths) -> dict[str, Any]:
    onnx_status = _module_status("onnx")
    mujoco_status = _module_status("mujoco")
    viewer_status = _module_status("mujoco.viewer")
    glfw_status = _module_status("glfw")
    ort_status = _module_status("onnxruntime")
    torch_status = _module_status("torch")
    imageio_status = _module_status("imageio")
    imageio_ffmpeg_status = _module_status("imageio_ffmpeg")
    model_path = default_mujoco_model_path(paths)
    config = load_calibration_config(repo_root=paths.repo_root, model_path=model_path, contract=FALLBACK_CONTRACT)
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or ""
    return {
        "target": DEPLOY_TARGET,
        "target_runtime": TARGET_RUNTIME,
        "include_ros_mock_default": False,
        "include_mujoco_default": True,
        "export_first_default": False,
        "expected_obs_dim": FALLBACK_CONTRACT.OBS_DIM_SINGLE,
        "history_obs_dim": FALLBACK_CONTRACT.OBS_DIM_SINGLE * FALLBACK_CONTRACT.POLICY_HISTORY_LENGTH,
        "expected_action_dim": FALLBACK_CONTRACT.ACTION_DIM,
        "rtol": DEFAULT_RTOL,
        "atol": DEFAULT_ATOL,
        "deploy_runtime_python": sys.executable,
        "onnx_installed": onnx_status["installed"],
        "onnx_version": onnx_status["version"],
        "mujoco_model_path": str(model_path),
        "mujoco_installed": mujoco_status["installed"],
        "mujoco_version": mujoco_status["version"],
        "onnxruntime_installed": ort_status["installed"],
        "onnxruntime_version": ort_status["version"],
        "torch_installed": torch_status["installed"],
        "torch_version": torch_status["version"],
        "deploy_runtime_dependencies": {
            "onnx": onnx_status,
            "onnxruntime": ort_status,
            "mujoco": mujoco_status,
            "torch": torch_status,
        },
        "mujoco_viewer_available": bool(mujoco_status["installed"] and viewer_status["installed"] and glfw_status["installed"] and display),
        "mujoco_renderer_available": bool(mujoco_status["installed"]),
        "mujoco_encoder_available": bool(imageio_status["installed"] and imageio_ffmpeg_status["installed"]),
        "mujoco_display": display,
        "mujoco_scenarios": [scenario.to_dict() for scenario in default_scenarios(steps=DEFAULT_PLAYBACK_STEPS)],
        "mujoco_playback_defaults": {
            "steps": DEFAULT_PLAYBACK_STEPS,
            "width": DEFAULT_PLAYBACK_WIDTH,
            "height": DEFAULT_PLAYBACK_HEIGHT,
            "fps": DEFAULT_PLAYBACK_FPS,
        },
        "mujoco_calibrated": bool(config.calibrated),
        "report_version": REPORT_VERSION,
    }


def default_mujoco_model_path(paths: PanelPaths) -> Path:
    return paths.repo_root / "test_7_description" / "test_7_description" / "urdf" / "test_7.urdf"


def report_dir_for_run(run: dict[str, Any]) -> Path | None:
    log_dir = run.get("log_dir")
    if not log_dir:
        return None
    return Path(str(log_dir)) / "deploy"


def list_deploy_reports(run: dict[str, Any]) -> list[dict[str, Any]]:
    report_dir = report_dir_for_run(run)
    if not report_dir or not report_dir.is_dir():
        return []
    reports = []
    for path in sorted(report_dir.glob("readiness_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reports.append(
            {
                "path": str(path),
                "pipeline_id": payload.get("pipeline_id"),
                "created_at": payload.get("created_at"),
                "completed_at": payload.get("completed_at"),
                "overall_status": payload.get("overall_status"),
                "readiness_level": payload.get("readiness_level"),
                "stage_counts": stage_counts(payload.get("stages") or []),
                "report": payload,
            }
        )
    return reports


def latest_deploy_report(run: dict[str, Any]) -> dict[str, Any] | None:
    reports = list_deploy_reports(run)
    return reports[0] if reports else None


def stage_counts(stages: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"pass": 0, "warn": 0, "fail": 0, "skipped": 0}
    for stage in stages:
        status = str(stage.get("status") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _module_status(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError) as exc:
        return {"installed": False, "version": "", "error": str(exc)}
    if spec is None:
        return {"installed": False, "version": ""}
    try:
        module = importlib.import_module(name)
        return {"installed": True, "version": str(getattr(module, "__version__", ""))}
    except Exception as exc:
        return {"installed": False, "version": "", "error": str(exc)}


def _deploy_runtime_dependencies() -> dict[str, dict[str, Any]]:
    return {
        "onnx": _module_status("onnx"),
        "onnxruntime": _module_status("onnxruntime"),
        "mujoco": _module_status("mujoco"),
        "torch": _module_status("torch"),
    }


def _deploy_runtime_info() -> dict[str, Any]:
    return {"python": sys.executable, "dependencies": _deploy_runtime_dependencies()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional dependency
        return {}, f"PyYAML unavailable while reading {path}: {exc}"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {}, f"Could not parse {path}: {exc}"
    return data if isinstance(data, dict) else {}, None


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _add_ros_controller_to_path(paths: PanelPaths) -> None:
    src = paths.repo_root / "ros2_ws" / "src" / "redrhex_rl_controller"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _contract(paths: PanelPaths):
    _add_ros_controller_to_path(paths)
    try:
        from redrhex_rl_controller import redrhex_contract as contract

        return contract
    except Exception:
        return FALLBACK_CONTRACT


def _stage(
    spec: ValidatorSpec,
) -> ValidationStageResult:
    started = time.monotonic()
    if spec.skip_reason:
        return _result(
            spec.name,
            spec.title,
            "skipped",
            spec.skip_reason,
            details={"dependencies": spec.dependencies},
            next_steps=spec.skip_next_steps,
        )
    try:
        result = spec.run()
        result.duration_s = round(time.monotonic() - started, 4)
        result.details.setdefault("declared_dependencies", spec.dependencies)
        return result
    except Exception as exc:
        return ValidationStageResult(
            name=spec.name,
            title=spec.title,
            status="fail",
            duration_s=round(time.monotonic() - started, 4),
            summary=f"{spec.title} raised {type(exc).__name__}: {exc}",
            details={"declared_dependencies": spec.dependencies},
            next_steps=["Open the deploy process log and fix the failing validator before hardware bring-up."],
        )


def _result(
    name: str,
    title: str,
    status: str,
    summary: str,
    *,
    details: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
    next_steps: list[str] | None = None,
) -> ValidationStageResult:
    return ValidationStageResult(
        name=name,
        title=title,
        status=status,
        duration_s=0.0,
        summary=summary,
        details=_json_safe(details or {}),
        artifacts={str(k): str(v) for k, v in (artifacts or {}).items()},
        next_steps=list(next_steps or []),
    )


def _file_metadata(paths: list[tuple[str, Path]]) -> tuple[dict[str, str], dict[str, int]]:
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for key, path in paths:
        if path.is_file():
            sizes[key] = path.stat().st_size
            hashes[key] = _sha256(path)
    return hashes, sizes


def build_policy_manifest(paths: PanelPaths, run: dict[str, Any]) -> PolicyManifest:
    contract = _contract(paths)
    log_dir = Path(str(run.get("log_dir") or ""))
    checkpoint = Path(str(run.get("latest_checkpoint") or ""))
    onnx_path = Path(str(run.get("onnx_path") or ""))
    if not onnx_path or str(onnx_path) == ".":
        onnx = latest_onnx(log_dir) if log_dir.is_dir() else None
        onnx_path = Path(onnx) if onnx else log_dir / "exported" / "policy.onnx"
    policy_pt_path = onnx_path.with_name("policy.pt") if onnx_path.name else log_dir / "exported" / "policy.pt"
    env_yaml = log_dir / "params" / "env.yaml"
    agent_yaml = log_dir / "params" / "agent.yaml"
    torsion_spring = log_dir / "params" / "torsion_spring.yaml"
    physics_profile = log_dir / "params" / "physics_profile.json"
    physics_profile_metadata = log_dir / "params" / "physics_profile_metadata.json"
    deploy_config = paths.repo_root / "ros2_ws" / "src" / "redrhex_rl_controller" / "config" / "redrhex_policy.yaml"
    deploy_yaml, yaml_warning = _load_yaml(deploy_config)
    params = _nested(deploy_yaml, "redrhex_rl_controller", "ros__parameters", default={})
    command_limits = dict(getattr(contract, "COMMAND_LIMITS", FALLBACK_CONTRACT.COMMAND_LIMITS))
    command_limits.update(_nested(params, "commands", default={}) or {})
    history_length = int(_nested(params, "observation", "policy_history_length", default=getattr(contract, "POLICY_HISTORY_LENGTH", 5)))
    warnings: list[str] = []
    if yaml_warning:
        warnings.append(yaml_warning)
    if str(_nested(params, "observation", "base_lin_vel_source", default="zero")) == "zero":
        warnings.append("base_lin_vel_source is zero; this is acceptable for bench checks but not final locomotion.")
    if str(_nested(params, "observation", "abad_feedback_source", default="commanded")) == "commanded":
        warnings.append("ABAD observation uses commanded position because ABAD encoder feedback is not available.")
    hashes, sizes = _file_metadata(
        [
            ("checkpoint", checkpoint),
            ("policy_onnx", onnx_path),
            ("policy_torchscript", policy_pt_path),
            ("env_yaml", env_yaml),
            ("agent_yaml", agent_yaml),
            ("torsion_spring", torsion_spring),
            ("physics_profile", physics_profile),
            ("physics_profile_metadata", physics_profile_metadata),
            ("deploy_config", deploy_config),
        ]
    )
    return PolicyManifest(
        run_id=str(run.get("id") or ""),
        display_name=str(run.get("display_name") or run.get("id") or ""),
        target=DEPLOY_TARGET,
        target_runtime=TARGET_RUNTIME,
        created_at=_iso_now(),
        log_dir=str(log_dir) if run.get("log_dir") else "",
        checkpoint_path=str(checkpoint) if run.get("latest_checkpoint") else "",
        policy_onnx_path=str(onnx_path),
        policy_torchscript_path=str(policy_pt_path),
        env_yaml_path=str(env_yaml),
        agent_yaml_path=str(agent_yaml),
        torsion_spring_path=str(torsion_spring),
        physics_profile_path=str(physics_profile),
        physics_profile_metadata_path=str(physics_profile_metadata),
        deploy_config_path=str(deploy_config),
        expected_obs_dim=int(getattr(contract, "OBS_DIM_SINGLE", 56)),
        history_obs_dim=int(getattr(contract, "OBS_DIM_SINGLE", 56)) * history_length,
        expected_action_dim=int(getattr(contract, "ACTION_DIM", 12)),
        policy_history_length=history_length,
        policy_hz=float(getattr(contract, "POLICY_HZ", 60.0)),
        command_limits={str(k): float(v) for k, v in command_limits.items()},
        main_drive_joint_names=list(getattr(contract, "MAIN_DRIVE_JOINT_NAMES", FALLBACK_CONTRACT.MAIN_DRIVE_JOINT_NAMES)),
        abad_joint_names=list(getattr(contract, "ABAD_JOINT_NAMES", FALLBACK_CONTRACT.ABAD_JOINT_NAMES)),
        damper_joint_names=list(getattr(contract, "DAMPER_JOINT_NAMES", FALLBACK_CONTRACT.DAMPER_JOINT_NAMES)),
        observation_slices=dict(getattr(contract, "OBSERVATION_SLICES", FALLBACK_CONTRACT.OBSERVATION_SLICES)),
        hashes=hashes,
        sizes=sizes,
        warnings=warnings,
    )


def validate_export_integrity(manifest: PolicyManifest) -> ValidationStageResult:
    checks = {
        "checkpoint": Path(manifest.checkpoint_path),
        "policy_onnx": Path(manifest.policy_onnx_path),
        "policy_torchscript": Path(manifest.policy_torchscript_path),
        "env_yaml": Path(manifest.env_yaml_path),
        "agent_yaml": Path(manifest.agent_yaml_path),
    }
    missing = [name for name, path in checks.items() if not path.is_file()]
    empty = [name for name, path in checks.items() if path.is_file() and path.stat().st_size <= 0]
    details = {
        "missing": missing,
        "empty": empty,
        "sizes": manifest.sizes,
        "hashes": manifest.hashes,
    }
    if missing or empty:
        return _result(
            "export_integrity",
            "Export Integrity",
            "fail",
            "Required policy export artifacts are missing or empty.",
            details=details,
            next_steps=[
                "Run Export ONNX from History or use Export ONNX + Validate from Deploy.",
                "Confirm the run has params/env.yaml, params/agent.yaml, policy.pt, and policy.onnx.",
            ],
        )
    return _result(
        "export_integrity",
        "Export Integrity",
        "pass",
        "Checkpoint, TorchScript, ONNX, and training parameter files are present and hashed.",
        details=details,
        artifacts={
            "policy_onnx": manifest.policy_onnx_path,
            "policy_torchscript": manifest.policy_torchscript_path,
            "env_yaml": manifest.env_yaml_path,
            "agent_yaml": manifest.agent_yaml_path,
        },
    )


def validate_spring_calibration(manifest: PolicyManifest) -> ValidationStageResult:
    """Block deployment unless the run's physical spring evidence still verifies."""

    try:
        from tools.sim2real.checkpoint_spring import (
            validate_checkpoint_spring_deployment,
        )

        details = validate_checkpoint_spring_deployment(manifest.log_dir)
    except (ImportError, ValueError) as exc:
        return _result(
            "spring_calibration",
            "Torsion-Spring Calibration",
            "fail",
            f"Deployment requires a calibrated torsion-spring checkpoint: {exc}",
            details={
                "torsion_spring_path": manifest.torsion_spring_path,
                "physics_profile_path": manifest.physics_profile_path,
                "physics_profile_metadata_path": manifest.physics_profile_metadata_path,
            },
            next_steps=[
                "Complete the approved physical calibration/holdout workflow and retrain from its authenticated profile."
            ],
        )
    return _result(
        "spring_calibration",
        "Torsion-Spring Calibration",
        "pass",
        "Checkpoint spring parameters are bound to authenticated calibration and holdout evidence.",
        details=details,
        artifacts={
            "torsion_spring": manifest.torsion_spring_path,
            "physics_profile": manifest.physics_profile_path,
            "physics_profile_metadata": manifest.physics_profile_metadata_path,
        },
    )


def _onnx_tensor_info(value_info: Any) -> dict[str, Any]:
    tensor = value_info.type.tensor_type
    dims = []
    for dim in tensor.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return {"name": value_info.name, "elem_type": int(tensor.elem_type), "shape": dims}


def validate_static_onnx(manifest: PolicyManifest) -> ValidationStageResult:
    path = Path(manifest.policy_onnx_path)
    if not path.is_file():
        return _result(
            "static_onnx",
            "Static ONNX",
            "skipped",
            "policy.onnx is missing, so static ONNX checks were skipped.",
            next_steps=["Export policy.onnx before running static model validation."],
        )
    try:
        import onnx
        from onnx import checker, shape_inference
    except Exception as exc:  # pragma: no cover - optional dependency
        return _result(
            "static_onnx",
            "Static ONNX",
            "skipped",
            f"onnx package is not installed: {exc}",
            next_steps=["Install onnx in the panel environment to enable checker and shape inference."],
        )
    try:
        model = onnx.load(str(path))
        checker.check_model(model)
        inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
    except Exception as exc:
        return _result(
            "static_onnx",
            "Static ONNX",
            "fail",
            f"ONNX checker or shape inference failed: {exc}",
            next_steps=["Re-export the policy and confirm the exporter produced a valid inference graph."],
        )
    initializer_names = {init.name for init in model.graph.initializer}
    inputs = [_onnx_tensor_info(item) for item in model.graph.input if item.name not in initializer_names]
    outputs = [_onnx_tensor_info(item) for item in model.graph.output]
    domains = sorted({node.domain for node in model.graph.node if node.domain})
    opsets = {op.domain or "": int(op.version) for op in model.opset_import}
    status = "warn" if domains else "pass"
    summary = "ONNX graph is structurally valid."
    if domains:
        summary = "ONNX graph is valid but contains non-default operator domains."
    return _result(
        "static_onnx",
        "Static ONNX",
        status,
        summary,
        details={
            "ir_version": int(model.ir_version),
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "opsets": opsets,
            "inputs": inputs,
            "outputs": outputs,
            "custom_domains": domains,
            "inferred_value_info_count": len(inferred.graph.value_info),
        },
        next_steps=[
            "Verify custom operator domains are supported on Jetson before deployment."
        ]
        if domains
        else [],
    )


def _last_static_dim(shape: list[Any]) -> int | None:
    if not shape:
        return None
    dim = shape[-1]
    return int(dim) if isinstance(dim, int) and dim > 0 else None


def _ort_session(path: Path, providers: list[str]):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def validate_onnx_runtime(manifest: PolicyManifest, *, use_cuda: bool = False, use_tensorrt: bool = False) -> ValidationStageResult:
    path = Path(manifest.policy_onnx_path)
    if not path.is_file():
        return _result(
            "onnx_runtime",
            "ONNX Runtime",
            "skipped",
            "policy.onnx is missing, so runtime inference checks were skipped.",
            next_steps=["Export policy.onnx before running ONNX Runtime checks."],
        )
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - optional dependency
        return _result(
            "onnx_runtime",
            "ONNX Runtime",
            "skipped",
            f"onnxruntime is not installed: {exc}",
            next_steps=["Install onnxruntime or onnxruntime-gpu in the deployment environment."],
        )
    available = ort.get_available_providers()
    providers: list[str] = []
    if use_tensorrt and "TensorrtExecutionProvider" in available:
        providers.append("TensorrtExecutionProvider")
    if use_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    try:
        session = _ort_session(path, providers)
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        obs_dim = _last_static_dim(list(input_meta.shape)) or manifest.expected_obs_dim
        allowed_obs_dims = {manifest.expected_obs_dim, manifest.history_obs_dim}
        if obs_dim not in allowed_obs_dims:
            return _result(
                "onnx_runtime",
                "ONNX Runtime",
                "fail",
                f"ONNX input dim {obs_dim} is outside allowed dims {sorted(allowed_obs_dims)}.",
                details={"input_shape": list(input_meta.shape), "output_shape": list(output_meta.shape)},
                next_steps=["Check observation history export and the ROS controller policy.expected_obs_dim setting."],
            )
        rng = np.random.default_rng(7)
        samples = [
            np.zeros((1, obs_dim), dtype=np.float32),
            rng.normal(0.0, 0.2, size=(1, obs_dim)).astype(np.float32),
        ]
        actions = [np.asarray(session.run([output_meta.name], {input_meta.name: obs})[0], dtype=np.float32) for obs in samples]
        flat_actions = [action.reshape(-1) for action in actions]
        if any(action.shape != (manifest.expected_action_dim,) for action in flat_actions):
            return _result(
                "onnx_runtime",
                "ONNX Runtime",
                "fail",
                "ONNX output shape does not match RedRHex action dimension.",
                details={"action_shapes": [list(action.shape) for action in actions]},
                next_steps=["Re-export the actor and confirm action_space remains 12."],
            )
        if any(not np.isfinite(action).all() for action in flat_actions):
            return _result(
                "onnx_runtime",
                "ONNX Runtime",
                "fail",
                "ONNX produced NaN or Inf action values.",
                details={"action_min_max": [[float(np.min(a)), float(np.max(a))] for a in flat_actions]},
                next_steps=["Inspect policy weights, normalizer state, and observation clipping."],
            )
        repeat_obs = samples[1]
        repeat_outputs = [
            np.asarray(session.run([output_meta.name], {input_meta.name: repeat_obs})[0], dtype=np.float32).reshape(-1)
            for _ in range(5)
        ]
        repeat_max_diff = max(float(np.max(np.abs(repeat_outputs[0] - item))) for item in repeat_outputs[1:])
        latencies_ms = []
        for _ in range(20):
            start = time.perf_counter()
            session.run([output_meta.name], {input_meta.name: repeat_obs})
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        status = "pass" if repeat_max_diff <= 1.0e-7 else "warn"
        summary = "ONNX Runtime loaded the policy and produced finite deterministic actions."
        if status == "warn":
            summary = "ONNX Runtime loaded the policy, but repeated inference was not bitwise-stable."
        return _result(
            "onnx_runtime",
            "ONNX Runtime",
            status,
            summary,
            details={
                "available_providers": available,
                "session_providers": session.get_providers(),
                "input_name": input_meta.name,
                "input_shape": list(input_meta.shape),
                "input_type": input_meta.type,
                "output_name": output_meta.name,
                "output_shape": list(output_meta.shape),
                "output_type": output_meta.type,
                "obs_dim": obs_dim,
                "action_dim": manifest.expected_action_dim,
                "action_min": float(min(np.min(a) for a in flat_actions)),
                "action_max": float(max(np.max(a) for a in flat_actions)),
                "repeat_max_abs_diff": repeat_max_diff,
                "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
                "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
                "latency_ms_max": float(np.max(latencies_ms)),
            },
            next_steps=["Benchmark again on Jetson with CUDA/TensorRT enabled before final deployment."]
            if not use_cuda and not use_tensorrt
            else [],
        )
    except Exception as exc:
        return _result(
            "onnx_runtime",
            "ONNX Runtime",
            "fail",
            f"ONNX Runtime could not load or run the policy: {exc}",
            next_steps=["Install compatible ONNX Runtime providers and verify policy.onnx is not corrupted."],
        )


def _input_dim_from_onnxruntime(onnx_path: Path, default: int) -> tuple[int, str, str]:
    session = _ort_session(onnx_path, ["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    return _last_static_dim(list(input_meta.shape)) or default, input_meta.name, output_meta.name


def _golden_observations(path: Path, obs_dim: int) -> list[np.ndarray]:
    if not path.is_dir():
        return []
    observations: list[np.ndarray] = []
    for item in sorted(path.iterdir()):
        if item.suffix == ".npy":
            obs = np.load(item).astype(np.float32)
        elif item.suffix == ".json":
            obs = np.asarray(json.loads(item.read_text(encoding="utf-8")), dtype=np.float32)
        else:
            continue
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.shape[-1] == obs_dim:
            observations.append(obs)
    return observations


def validate_torch_onnx_parity(
    manifest: PolicyManifest,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    golden_dir: Path | None = None,
) -> ValidationStageResult:
    onnx_path = Path(manifest.policy_onnx_path)
    torch_path = Path(manifest.policy_torchscript_path)
    if not onnx_path.is_file() or not torch_path.is_file():
        return _result(
            "torch_onnx_parity",
            "Torch/ONNX Parity",
            "skipped",
            "TorchScript or ONNX policy artifact is missing.",
            next_steps=["Export both policy.pt and policy.onnx before parity validation."],
        )
    try:
        import onnxruntime as ort  # noqa: F401
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        return _result(
            "torch_onnx_parity",
            "Torch/ONNX Parity",
            "skipped",
            f"torch or onnxruntime is not installed: {exc}",
            next_steps=["Install torch and onnxruntime in the validation environment."],
        )
    try:
        obs_dim, input_name, output_name = _input_dim_from_onnxruntime(onnx_path, manifest.expected_obs_dim)
        session = _ort_session(onnx_path, ["CPUExecutionProvider"])
        module = torch.jit.load(str(torch_path), map_location="cpu")
        module.eval()
        rng = np.random.default_rng(11)
        observations = [
            np.zeros((1, obs_dim), dtype=np.float32),
            rng.normal(0.0, 0.2, size=(1, obs_dim)).astype(np.float32),
            np.clip(rng.normal(0.0, 1.0, size=(1, obs_dim)), -1.0, 1.0).astype(np.float32),
        ]
        observations += _golden_observations(golden_dir or Path(manifest.log_dir) / "deploy" / "golden_observations", obs_dim)
        max_abs = 0.0
        mean_abs = 0.0
        sample_count = 0
        failures = []
        with torch.no_grad():
            for index, obs in enumerate(observations):
                onnx_action = np.asarray(session.run([output_name], {input_name: obs})[0], dtype=np.float32)
                torch_action = module(torch.from_numpy(obs)).detach().cpu().numpy().astype(np.float32)
                diff = onnx_action - torch_action
                sample_max = float(np.max(np.abs(diff)))
                sample_mean = float(np.mean(np.abs(diff)))
                max_abs = max(max_abs, sample_max)
                mean_abs += sample_mean
                sample_count += 1
                if not np.allclose(onnx_action, torch_action, rtol=rtol, atol=atol):
                    failures.append({"sample": index, "max_abs_diff": sample_max, "mean_abs_diff": sample_mean})
        mean_abs = mean_abs / max(sample_count, 1)
        if failures:
            return _result(
                "torch_onnx_parity",
                "Torch/ONNX Parity",
                "fail",
                "TorchScript and ONNX outputs differ beyond tolerance.",
                details={
                    "rtol": rtol,
                    "atol": atol,
                    "sample_count": sample_count,
                    "max_abs_diff": max_abs,
                    "mean_abs_diff": mean_abs,
                    "failures": failures,
                },
                next_steps=["Check exporter normalizer state, eval mode, and policy input order."],
            )
        return _result(
            "torch_onnx_parity",
            "Torch/ONNX Parity",
            "pass",
            "TorchScript and ONNX outputs match within tolerance.",
            details={"rtol": rtol, "atol": atol, "sample_count": sample_count, "max_abs_diff": max_abs, "mean_abs_diff": mean_abs},
        )
    except Exception as exc:
        return _result(
            "torch_onnx_parity",
            "Torch/ONNX Parity",
            "fail",
            f"Parity validation failed: {exc}",
            next_steps=["Open the deploy process log and confirm policy.pt is a TorchScript export."],
        )


def validate_contract(paths: PanelPaths, manifest: PolicyManifest) -> ValidationStageResult:
    contract = _contract(paths)
    errors: list[str] = []
    warnings = list(manifest.warnings)
    all_joints = manifest.main_drive_joint_names + manifest.abad_joint_names + manifest.damper_joint_names
    if manifest.expected_obs_dim != int(getattr(contract, "OBS_DIM_SINGLE", 56)):
        errors.append("manifest expected_obs_dim does not match ROS contract")
    if manifest.expected_action_dim != int(getattr(contract, "ACTION_DIM", 12)):
        errors.append("manifest expected_action_dim does not match ROS contract")
    if manifest.history_obs_dim != manifest.expected_obs_dim * manifest.policy_history_length:
        errors.append("history_obs_dim must equal expected_obs_dim * policy_history_length")
    if len(manifest.main_drive_joint_names) != 6 or len(manifest.abad_joint_names) != 6 or len(manifest.damper_joint_names) != 6:
        errors.append("joint groups must each contain 6 joints")
    if len(set(all_joints)) != len(all_joints):
        errors.append("controlled joint names must be unique")
    if set(manifest.observation_slices.keys()) != set(FALLBACK_CONTRACT.OBSERVATION_SLICES.keys()):
        errors.append("observation slice names do not match RedRHex contract")
    deploy_yaml, yaml_warning = _load_yaml(Path(manifest.deploy_config_path))
    if yaml_warning:
        warnings.append(yaml_warning)
    params = _nested(deploy_yaml, "redrhex_rl_controller", "ros__parameters", default={})
    policy_expected_obs = int(_nested(params, "policy", "expected_obs_dim", default=manifest.expected_obs_dim) or manifest.expected_obs_dim)
    policy_expected_action = int(_nested(params, "policy", "expected_action_dim", default=manifest.expected_action_dim) or manifest.expected_action_dim)
    if policy_expected_obs != manifest.expected_obs_dim:
        errors.append("ROS config policy.expected_obs_dim differs from manifest expected_obs_dim")
    if policy_expected_action != manifest.expected_action_dim:
        errors.append("ROS config policy.expected_action_dim differs from manifest expected_action_dim")
    if bool(_nested(params, "state_machine", "enable_policy_on_start", default=False)):
        errors.append("enable_policy_on_start must remain false for readiness checks")
    if bool(_nested(params, "state_machine", "enable_motor_output_on_start", default=False)):
        errors.append("enable_motor_output_on_start must remain false for readiness checks")
    if bool(_nested(params, "action", "include_damper_command", default=False)):
        errors.append("include_damper_command must remain false because real dampers are passive")
    if str(_nested(params, "observation", "base_lin_vel_source", default="zero")) == "zero":
        warnings.append("base linear velocity is zero-filled in deploy config; final locomotion needs a real estimator.")
    status = "fail" if errors else ("warn" if warnings else "pass")
    summary = "Observation/action contract matches the RedRHex deployment constants."
    if errors:
        summary = "Observation/action contract has blocking mismatches."
    elif warnings:
        summary = "Observation/action contract is usable but has sim-to-real warnings."
    return _result(
        "contract",
        "Observation/Action Contract",
        status,
        summary,
        details={
            "errors": errors,
            "warnings": warnings,
            "expected_obs_dim": manifest.expected_obs_dim,
            "history_obs_dim": manifest.history_obs_dim,
            "expected_action_dim": manifest.expected_action_dim,
            "policy_history_length": manifest.policy_history_length,
            "command_limits": manifest.command_limits,
            "joint_order": {
                "main_drive": manifest.main_drive_joint_names,
                "abad": manifest.abad_joint_names,
                "damper": manifest.damper_joint_names,
            },
            "observation_slices": manifest.observation_slices,
        },
        next_steps=[
            "Fix ROS deployment YAML and re-run readiness before enabling the policy."
        ]
        if errors
        else warnings,
    )


def validate_safety_faults(paths: PanelPaths, manifest: PolicyManifest) -> ValidationStageResult:
    _add_ros_controller_to_path(paths)
    try:
        from redrhex_rl_controller.action_decoder import DecodedMotorCommand
        from redrhex_rl_controller.safety_filter import SafetyFilter, SafetyState
    except Exception as exc:
        return _result(
            "safety_faults",
            "Safety Fault Injection",
            "skipped",
            f"Could not import RedRHex safety modules: {exc}",
            next_steps=["Build/source the ROS workspace or make redrhex_rl_controller importable."],
        )
    cfg = {
        "sensor_timeout_s": 0.10,
        "cmd_timeout_s": 0.25,
        "motor_feedback_timeout_s": 0.25,
        "heartbeat_timeout_s": 0.10,
        "max_abs_roll_rad": 0.7,
        "max_abs_pitch_rad": 0.7,
        "action_clip": 1.0,
        "main_drive_vel_limit_rad_s": 30.0,
        "abad_pos_limit_rad": 0.7,
        "max_motor_temperature_c": 70.0,
        "max_motor_current_a": 20.0,
        "max_control_loop_dt_s": 0.03,
        "command_limits": manifest.command_limits,
    }
    safety = SafetyFilter(cfg)
    nominal = SafetyState(
        estop=False,
        imu_age_s=0.01,
        joint_state_age_s=0.01,
        heartbeat_age_s=0.01,
        roll_rad=0.0,
        pitch_rad=0.0,
        command=np.asarray([0.1, 0.0, 0.0], dtype=np.float64),
        control_loop_dt_s=0.008,
    )
    safe_command = DecodedMotorCommand(
        joint_names=manifest.main_drive_joint_names + manifest.abad_joint_names,
        target_position_rad=[0.0] * 12,
        target_velocity_rad_s=[0.0] * 12,
        kp=[0.0] * 12,
        kd=[0.0] * 12,
        effort_limit_nm=[0.0] * 12,
        enable=False,
        mode=0,
        safe_action=np.zeros(12, dtype=np.float32),
        target_main_drive_velocity=np.zeros(6, dtype=np.float64),
        target_abad_position=np.zeros(6, dtype=np.float64),
    )
    cases: list[tuple[str, SafetyState, np.ndarray | None, np.ndarray | None, DecodedMotorCommand | None, bool]] = [
        ("nominal", nominal, np.zeros(manifest.expected_obs_dim), np.zeros(manifest.expected_action_dim), safe_command, True),
        ("estop", SafetyState(**{**asdict(nominal), "estop": True}), None, None, None, False),
        ("imu_timeout", SafetyState(**{**asdict(nominal), "imu_age_s": 1.0}), None, None, None, False),
        ("joint_timeout", SafetyState(**{**asdict(nominal), "joint_state_age_s": 1.0}), None, None, None, False),
        ("roll_limit", SafetyState(**{**asdict(nominal), "roll_rad": 1.0}), None, None, None, False),
        ("bad_command", SafetyState(**{**asdict(nominal), "command": np.asarray([9.0, 0.0, 0.0])}), None, None, None, False),
        ("deadline_miss", SafetyState(**{**asdict(nominal), "control_loop_dt_s": 0.2}), None, None, None, False),
        ("nan_observation", nominal, np.asarray([np.nan] + [0.0] * (manifest.expected_obs_dim - 1)), None, None, False),
        ("large_raw_action", nominal, None, np.ones(manifest.expected_action_dim) * 2.0, None, False),
        (
            "decoded_limit",
            nominal,
            None,
            None,
            DecodedMotorCommand(
                joint_names=safe_command.joint_names,
                target_position_rad=safe_command.target_position_rad,
                target_velocity_rad_s=safe_command.target_velocity_rad_s,
                kp=safe_command.kp,
                kd=safe_command.kd,
                effort_limit_nm=safe_command.effort_limit_nm,
                enable=False,
                mode=0,
                safe_action=np.zeros(12, dtype=np.float32),
                target_main_drive_velocity=np.ones(6, dtype=np.float64) * 99.0,
                target_abad_position=np.zeros(6, dtype=np.float64),
            ),
            False,
        ),
    ]
    results = []
    failures = []
    for name, state, obs, raw_action, command, expected_ok in cases:
        result = safety.check(state, obs, raw_action, command)
        case = {"name": name, "ok": result.ok, "reasons": result.reasons, "expected_ok": expected_ok}
        results.append(case)
        if result.ok != expected_ok:
            failures.append(case)
    if failures:
        return _result(
            "safety_faults",
            "Safety Fault Injection",
            "fail",
            "One or more synthetic safety faults were not handled as expected.",
            details={"cases": results, "failures": failures},
            next_steps=["Fix SafetyFilter before running any ROS policy takeover tests."],
        )
    return _result(
        "safety_faults",
        "Safety Fault Injection",
        "pass",
        "Safety filter accepts nominal input and rejects representative unsafe conditions.",
        details={
            "cases": results,
            "note": "Duplicate ROS command publisher protection is covered by the low-level bridge ROS mock stage.",
        },
    )


def validate_ros_mock(paths: PanelPaths, manifest: PolicyManifest, *, run_ros_mock: bool = False) -> ValidationStageResult:
    command = [
        "ros2",
        "launch",
        "redrhex_rl_controller",
        "redrhex_policy_bringup.launch.py",
        "use_fake_sensors:=true",
        "start_bridge:=true",
        "bridge_backend:=mock",
        f"onnx_path:={manifest.policy_onnx_path}",
        "enable_policy_on_start:=false",
        "enable_motor_output_on_start:=false",
        "require_lowlevel_heartbeat:=false",
        "require_motor_feedback:=false",
    ]
    install_setup = paths.repo_root / "ros2_ws" / "install" / "setup.bash"
    details = {"command": " ".join(command), "install_setup": str(install_setup)}
    if not run_ros_mock:
        return _result(
            "ros_mock",
            "ROS Mock/Fake Sensor",
            "skipped",
            "Bounded ROS mock launch was not requested for this run.",
            details=details,
            next_steps=["Enable ROS mock validation from the Deploy tab on a machine with the ROS workspace built."],
        )
    if not shutil.which("ros2"):
        return _result(
            "ros_mock",
            "ROS Mock/Fake Sensor",
            "skipped",
            "ros2 was not found on PATH.",
            details=details,
            next_steps=["Install/source ROS2 and rebuild ros2_ws before running the ROS mock stage."],
        )
    if not install_setup.is_file():
        return _result(
            "ros_mock",
            "ROS Mock/Fake Sensor",
            "skipped",
            "ROS workspace install/setup.bash is missing.",
            details=details,
            next_steps=["Run colcon build in ros2_ws, then source install/setup.bash."],
        )
    shell = (
        "set -e; "
        "source /opt/ros/humble/setup.bash; "
        f"source {install_setup}; "
        "timeout 12s "
        + " ".join(command)
    )
    completed = subprocess.run(
        ["bash", "-lc", shell],
        cwd=str(paths.repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )
    output = completed.stdout or ""
    safe_output = "enable_motor_output_on_start:=false" in shell and "enable_policy_on_start:=false" in shell
    if completed.returncode in (0, 124) and safe_output and "Traceback" not in output:
        return _result(
            "ros_mock",
            "ROS Mock/Fake Sensor",
            "pass",
            "Bounded ROS mock launch completed without enabling motor output.",
            details={**details, "returncode": completed.returncode, "output_tail": output[-4000:]},
        )
    return _result(
        "ros_mock",
        "ROS Mock/Fake Sensor",
        "fail",
        "ROS mock launch failed or emitted a Python traceback.",
        details={**details, "returncode": completed.returncode, "output_tail": output[-8000:]},
        next_steps=["Open the ROS launch output and fix missing packages, imports, or config errors."],
    )


def validate_mujoco_readiness(
    paths: PanelPaths,
    manifest: PolicyManifest,
    *,
    model_path: Path,
    artifact_dir: Path,
    steps: int = DEFAULT_MUJOCO_STEPS,
) -> ValidationStageResult:
    if not model_path.is_file():
        return _result(
            "mujoco_readiness",
            "MuJoCo Readiness",
            "skipped",
            "No MuJoCo/MJCF/URDF model file was found for advisory rollout.",
            details={"model_path": str(model_path)},
            next_steps=["Add a calibrated RedRHex MuJoCo model and point the Deploy tab at it."],
        )
    mujoco_status = _module_status("mujoco")
    if not mujoco_status["installed"]:
        return _result(
            "mujoco_readiness",
            "MuJoCo Readiness",
            "skipped",
            f"mujoco Python package is not installed: {mujoco_status.get('error') or 'module not found'}",
            details={"model_path": str(model_path), "mujoco_status": mujoco_status},
            next_steps=["Install mujoco in the panel environment to run advisory sim-to-sim checks."],
        )
    ort_status = _module_status("onnxruntime")
    if not ort_status["installed"]:
        return _result(
            "mujoco_readiness",
            "MuJoCo Readiness",
            "skipped",
            f"onnxruntime Python package is not installed: {ort_status.get('error') or 'module not found'}",
            details={"model_path": str(model_path), "onnxruntime_status": ort_status},
            next_steps=["Install onnxruntime in the panel environment to run ONNX policy rollouts."],
        )
    try:
        config = load_calibration_config(repo_root=paths.repo_root, model_path=model_path, contract=_contract(paths))
        obs_dim = manifest.expected_obs_dim
        if Path(manifest.policy_onnx_path).is_file():
            try:
                session = _ort_session(Path(manifest.policy_onnx_path), ["CPUExecutionProvider"])
                input_meta = session.get_inputs()[0]
                obs_dim = _last_static_dim(list(input_meta.shape)) or manifest.expected_obs_dim
            except Exception:
                obs_dim = manifest.expected_obs_dim
        report = run_mujoco_rollouts(
            repo_root=paths.repo_root,
            model_path=model_path,
            policy_path=Path(manifest.policy_onnx_path),
            artifact_dir=artifact_dir,
            config=config,
            obs_dim=obs_dim,
            action_dim=manifest.expected_action_dim,
        )
        return _result(
            "mujoco_readiness",
            "MuJoCo Readiness",
            report.status,
            report.summary,
            details=report.to_dict(),
            artifacts=report.artifacts,
            next_steps=[] if report.calibrated else [
                "Calibrate MuJoCo masses, actuator mapping, contacts, and joint limits before treating this as a hard gate."
            ],
        )
    except Exception as exc:
        return _result(
            "mujoco_readiness",
            "MuJoCo Readiness",
            "warn",
            f"MuJoCo could not compile or step the model: {exc}",
            details={"model_path": str(model_path), "artifact_dir": str(artifact_dir), "advisory": True},
            next_steps=["Convert/repair the RedRHex model for MuJoCo and re-run the advisory stage."],
        )


def run_export_stage(paths: PanelPaths, run: dict[str, Any], *, device: str = "cuda:0") -> ValidationStageResult:
    checkpoint = str(run.get("latest_checkpoint") or "")
    if not checkpoint:
        return _result(
            "onnx_export",
            "ONNX Export",
            "fail",
            "Run has no checkpoint to export.",
            next_steps=["Train or select a run with a model_*.pt checkpoint."],
        )
    if not paths.isaaclab_launcher.is_file():
        return _result(
            "onnx_export",
            "ONNX Export",
            "fail",
            f"Isaac Lab launcher not found: {paths.isaaclab_launcher}",
            next_steps=["Set ISAACLAB_ROOT so isaaclab.sh can be found."],
        )
    spring_backend = resolve_spring_backend(run, checkpoint)
    task = str((run.get("params") or {}).get("task") or DEFAULT_TASK)
    script_argv = export_onnx_argv(checkpoint=checkpoint, task=task, device=device, spring_backend=spring_backend)
    shell = shell_for_isaaclab(paths, script_argv)
    start = time.monotonic()
    print(f"[deploy] launching ONNX export via Isaac: {paths.isaaclab_launcher} -p {' '.join(script_argv)}", flush=True)
    completed = subprocess.run(
        ["bash", "-lc", shell],
        cwd=str(paths.repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = round(time.monotonic() - start, 4)
    output = completed.stdout or ""
    sys.stdout.write(output)
    log_dir = Path(str(run.get("log_dir") or ""))
    onnx_path = latest_onnx(log_dir) if log_dir.is_dir() else None
    status = "pass" if completed.returncode == 0 and onnx_path else "fail"
    summary = "ONNX export completed and produced exported/policy.onnx."
    if status == "fail":
        summary = "ONNX export failed or did not produce exported/policy.onnx."
    return ValidationStageResult(
        name="onnx_export",
        title="ONNX Export",
        status=status,
        duration_s=duration,
        summary=summary,
        details={"returncode": completed.returncode, "command": shell, "output_tail": output[-8000:]},
        artifacts={"policy_onnx": onnx_path or ""},
        next_steps=[] if status == "pass" else ["Open the deploy process log and fix the Isaac/RSL-RL export error."],
    )


def evaluate_overall(stages: list[ValidationStageResult]) -> tuple[str, str]:
    required = {
        "onnx_export",
        "spring_calibration",
        "export_integrity",
        "static_onnx",
        "onnx_runtime",
        "torch_onnx_parity",
        "contract",
        "safety_faults",
    }
    blocking_failures = [stage for stage in stages if stage.name in required and stage.status == "fail"]
    if blocking_failures:
        return "fail", "blocked"
    if any(stage.status in {"warn", "skipped"} for stage in stages):
        return "warn", "review"
    return "pass", "ready"


def operator_checklist(report: DeploymentReport | None = None) -> list[str]:
    return [
        "Copy only policy.onnx, redrhex_policy.yaml, the manifest, and the readiness report to Jetson.",
        "Run ROS mock/fake-sensor validation on Jetson before connecting motor power.",
        "Keep enable_policy_on_start=false and enable_motor_output_on_start=false for bring-up.",
        "Verify E-stop, low-level heartbeat, single ABAD, and single main-drive tests before policy takeover.",
        "Do not enable motors from this panel; hardware enable remains a manual supervised operation.",
    ]


def assumptions() -> list[str]:
    return [
        "Jetson ROS2 with redrhex_rl_controller is the target deployment surface.",
        "The real robot currently has main-drive encoder feedback and commanded ABAD position estimation.",
        "MuJoCo readiness is advisory until the MuJoCo model is calibrated against the physical robot.",
        "Remote child-panel deployment controls are intentionally out of scope for v1.",
    ]


def write_report(report: DeploymentReport) -> tuple[Path, Path]:
    report_dir = Path(report.manifest.log_dir) / "deploy"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"readiness_{report.pipeline_id}.json"
    md_path = report_dir / f"readiness_{report.pipeline_id}.md"
    payload = report.to_dict()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_report_markdown(report, json_path), encoding="utf-8")
    return json_path, md_path


def render_report_markdown(report: DeploymentReport, json_path: Path) -> str:
    lines = [
        f"# RedRHex Deploy Readiness Report",
        "",
        f"- Pipeline: `{report.pipeline_id}`",
        f"- Run: `{report.manifest.run_id}`",
        f"- Status: `{report.overall_status}`",
        f"- Readiness: `{report.readiness_level}`",
        f"- JSON: `{json_path}`",
        f"- Runtime Python: `{report.runtime.get('python') or 'unknown'}`",
        "",
        "## Stages",
    ]
    for stage in report.stages:
        lines.append(f"- `{stage.status}` {stage.title}: {stage.summary}")
    lines.extend(["", "## Operator Checklist"])
    lines.extend(f"- {item}" for item in report.operator_checklist)
    lines.extend(["", "## Assumptions"])
    lines.extend(f"- {item}" for item in report.assumptions)
    lines.append("")
    return "\n".join(lines)


def run_deploy_validation(
    paths: PanelPaths,
    run: dict[str, Any],
    *,
    pipeline_id: str | None = None,
    export_first: bool = False,
    device: str = "cuda:0",
    include_ros_mock: bool = False,
    include_mujoco: bool = True,
    use_cuda: bool = False,
    use_tensorrt: bool = False,
    mujoco_model_path: str | None = None,
    mujoco_only: bool = False,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> DeploymentReport:
    pipeline_id = pipeline_id or timestamp_id()
    created_at = _iso_now()
    runtime = _deploy_runtime_info()
    stages: list[ValidationStageResult] = []
    if export_first:
        stages.append(run_export_stage(paths, run, device=device))
        if stages[-1].status == "fail":
            manifest = build_policy_manifest(paths, run)
            overall, readiness = evaluate_overall(stages)
            report = DeploymentReport(
                version=REPORT_VERSION,
                pipeline_id=pipeline_id,
                created_at=created_at,
                completed_at=_iso_now(),
                overall_status=overall,
                readiness_level=readiness,
                manifest=manifest,
                stages=stages,
                operator_checklist=operator_checklist(),
                assumptions=assumptions(),
                runtime=runtime,
            )
            json_path, md_path = write_report(report)
            report.artifacts.update({"json_report": str(json_path), "markdown_report": str(md_path)})
            json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            return report
    manifest = build_policy_manifest(paths, run)
    stage_plan = [
        ValidatorSpec(
            "spring_calibration",
            "Torsion-Spring Calibration",
            lambda: validate_spring_calibration(manifest),
            dependencies=[
                "params/torsion_spring.yaml",
                "params/physics_profile.json",
                "params/physics_profile_metadata.json",
                "managed torsion-spring calibration/holdout episodes",
            ],
        )
    ]
    if not mujoco_only:
        stage_plan.extend(
            [
                ValidatorSpec(
                    "export_integrity",
                    "Export Integrity",
                    lambda: validate_export_integrity(manifest),
                    dependencies=[
                        "policy.pt",
                        "policy.onnx",
                        "checkpoint",
                        "params/env.yaml",
                        "params/agent.yaml",
                    ],
                ),
                ValidatorSpec(
                    "static_onnx",
                    "Static ONNX",
                    lambda: validate_static_onnx(manifest),
                    dependencies=["policy.onnx", "onnx"],
                ),
                ValidatorSpec(
                    "onnx_runtime",
                    "ONNX Runtime",
                    lambda: validate_onnx_runtime(
                        manifest, use_cuda=use_cuda, use_tensorrt=use_tensorrt
                    ),
                    dependencies=["policy.onnx", "onnxruntime"],
                ),
                ValidatorSpec(
                    "torch_onnx_parity",
                    "Torch/ONNX Parity",
                    lambda: validate_torch_onnx_parity(
                        manifest, rtol=rtol, atol=atol
                    ),
                    dependencies=["policy.pt", "policy.onnx", "torch", "onnxruntime"],
                ),
                ValidatorSpec(
                    "contract",
                    "Observation/Action Contract",
                    lambda: validate_contract(paths, manifest),
                    dependencies=[
                        "redrhex_contract.py",
                        "params/env.yaml",
                        "params/agent.yaml",
                    ],
                ),
                ValidatorSpec(
                    "safety_faults",
                    "Safety Fault Injection",
                    lambda: validate_safety_faults(paths, manifest),
                    dependencies=["safety_filter.py", "redrhex_contract.py"],
                ),
                ValidatorSpec(
                    "ros_mock",
                    "ROS Mock/Fake Sensor",
                    lambda: validate_ros_mock(
                        paths, manifest, run_ros_mock=include_ros_mock
                    ),
                    dependencies=[
                        "ros2",
                        "fake_sensor_node.py",
                        "policy_onnx_runner.py",
                    ],
                ),
            ]
        )
    if include_mujoco:
        model_path = Path(mujoco_model_path).expanduser() if mujoco_model_path else default_mujoco_model_path(paths)
        artifact_dir = Path(manifest.log_dir) / "deploy" / f"mujoco_{pipeline_id}"
        stage_plan.append(
            ValidatorSpec(
                "mujoco_readiness",
                "MuJoCo Readiness",
                lambda: validate_mujoco_readiness(paths, manifest, model_path=model_path, artifact_dir=artifact_dir),
                dependencies=["policy.onnx", "onnxruntime", "mujoco", str(model_path)],
            )
        )
    else:
        model_path = default_mujoco_model_path(paths)
        artifact_dir = Path(manifest.log_dir) / "deploy" / f"mujoco_{pipeline_id}"
        stage_plan.append(
            ValidatorSpec(
                "mujoco_readiness",
                "MuJoCo Readiness",
                lambda: validate_mujoco_readiness(paths, manifest, model_path=model_path, artifact_dir=artifact_dir),
                dependencies=["policy.onnx", "onnxruntime", "mujoco"],
                skip_reason="MuJoCo advisory rollout was disabled for this run.",
                skip_next_steps=["Enable MuJoCo readiness after a calibrated model is available."],
            )
        )
    stages.extend(_stage(spec) for spec in stage_plan)
    overall, readiness = evaluate_overall(stages)
    report = DeploymentReport(
        version=REPORT_VERSION,
        pipeline_id=pipeline_id,
        created_at=created_at,
        completed_at=_iso_now(),
        overall_status=overall,
        readiness_level=readiness,
        manifest=manifest,
        stages=stages,
        operator_checklist=operator_checklist(),
        assumptions=assumptions(),
        runtime=runtime,
    )
    json_path, md_path = write_report(report)
    report.artifacts.update({"json_report": str(json_path), "markdown_report": str(md_path)})
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RedRHex deploy readiness validation for a panel run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pipeline-id", default="")
    parser.add_argument("--export-first", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-ros-mock", action="store_true")
    parser.add_argument("--no-mujoco", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--use-tensorrt", action="store_true")
    parser.add_argument("--mujoco-model-path", default="")
    parser.add_argument("--mujoco-only", action="store_true")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .history import HistoryStore

    args = _parser().parse_args(argv)
    paths = PanelPaths.from_env()
    history = HistoryStore(paths)
    run = history.get_run(args.run_id)
    if not run:
        print(json.dumps({"error": f"Run not found: {args.run_id}"}, indent=2), file=sys.stderr)
        return 2
    try:
        runtime = _deploy_runtime_info()
        print(f"[deploy] validation runtime python: {runtime['python']}", flush=True)
        print(
            "[deploy] validation runtime dependencies: "
            + json.dumps(runtime["dependencies"], sort_keys=True),
            flush=True,
        )
        report = run_deploy_validation(
            paths,
            run,
            pipeline_id=args.pipeline_id or None,
            export_first=args.export_first,
            device=args.device,
            include_ros_mock=args.include_ros_mock,
            include_mujoco=not args.no_mujoco,
            use_cuda=args.use_cuda,
            use_tensorrt=args.use_tensorrt,
            mujoco_model_path=args.mujoco_model_path or None,
            mujoco_only=args.mujoco_only,
            rtol=args.rtol,
            atol=args.atol,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.overall_status in {"pass", "warn"} else 1
