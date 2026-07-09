from __future__ import annotations

import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SCENARIO_STEPS = 250
DEFAULT_PLAYBACK_STEPS = 1250
DEFAULT_PLAYBACK_WIDTH = 1280
DEFAULT_PLAYBACK_HEIGHT = 720
DEFAULT_PLAYBACK_FPS = 30


@dataclass
class MujocoCalibrationConfig:
    calibrated: bool = False
    source_model_path: str = ""
    generated_model_path: str = ""
    package_roots: dict[str, str] = field(default_factory=dict)
    policy_hz: float = 60.0
    timestep: float = 0.002
    base_body_name: str = "base_link"
    main_drive_joint_names: list[str] = field(default_factory=list)
    abad_joint_names: list[str] = field(default_factory=list)
    actuator_ctrl_range: tuple[float, float] = (-1.0, 1.0)
    actuator_gear: float = 1.0
    min_base_height_m: float = -1.0
    max_abs_roll_pitch_rad: float = 1.2
    max_abs_qpos: float = 1000.0
    main_drive_vel_limit_rad_s: float = 30.0
    abad_pos_limit_rad: float = 0.7
    root_pos: tuple[float, float, float] = (0.0, 0.0, 0.14)
    cad_frame_quat: tuple[float, float, float, float] = (0.70710678, 0.70710678, 0.0, 0.0)
    floor_z: float = 0.0

    @property
    def action_joint_names(self) -> list[str]:
        return list(self.main_drive_joint_names) + list(self.abad_joint_names)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("actuator_ctrl_range", "root_pos", "cad_frame_quat"):
            data[key] = list(data[key])
        data["action_joint_names"] = self.action_joint_names
        return data


@dataclass(frozen=True)
class MujocoScenario:
    name: str
    command: tuple[float, float, float]
    steps: int = DEFAULT_SCENARIO_STEPS
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "command": list(self.command), "steps": self.steps, "seed": self.seed}


@dataclass
class MujocoRolloutMetrics:
    scenario: str
    status: str
    steps_requested: int
    steps_completed: int
    command: list[float]
    diverged: bool = False
    diverged_at: int | None = None
    fall_detected: bool = False
    nan_or_inf: bool = False
    base_height_min: float = 0.0
    base_height_max: float = 0.0
    max_abs_roll_rad: float = 0.0
    max_abs_pitch_rad: float = 0.0
    joint_limit_violations: int = 0
    actuator_saturation_steps: int = 0
    action_min: float = 0.0
    action_max: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    latency_ms_max: float = 0.0
    final_time_s: float = 0.0
    final_qpos_head: list[float] = field(default_factory=list)
    trace_path: str = ""
    policy_warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MujocoRolloutReport:
    status: str
    summary: str
    calibrated: bool
    config: MujocoCalibrationConfig
    scenarios: list[MujocoRolloutMetrics]
    artifacts: dict[str, str]
    model_info: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "calibrated": self.calibrated,
            "config": self.config.to_dict(),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "artifacts": dict(self.artifacts),
            "model_info": dict(self.model_info),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class MujocoPlaybackConfig:
    mode: str
    scenario: str = "stand_zero"
    steps: int = DEFAULT_PLAYBACK_STEPS
    width: int = DEFAULT_PLAYBACK_WIDTH
    height: int = DEFAULT_PLAYBACK_HEIGHT
    fps: int = DEFAULT_PLAYBACK_FPS
    show_ui: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MujocoPlaybackReport:
    status: str
    summary: str
    mode: str
    scenario: str
    steps_requested: int
    steps_completed: int
    video_path: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_scenarios(steps: int = DEFAULT_SCENARIO_STEPS) -> list[MujocoScenario]:
    return [
        MujocoScenario("stand_zero", (0.0, 0.0, 0.0), steps=steps, seed=1),
        MujocoScenario("forward_mid", (0.28, 0.0, 0.0), steps=steps, seed=2),
        MujocoScenario("yaw_mid", (0.0, 0.0, 0.35), steps=steps, seed=3),
        MujocoScenario("boundary_command", (0.56, 0.60, 0.70), steps=steps, seed=4),
    ]


def scenario_by_name(name: str, *, steps: int = DEFAULT_SCENARIO_STEPS) -> MujocoScenario:
    scenarios = {scenario.name: scenario for scenario in default_scenarios(steps=steps)}
    try:
        return scenarios[name]
    except KeyError as exc:
        raise ValueError(f"Unknown MuJoCo scenario: {name}") from exc


def policy_decimation(policy_hz: float, timestep: float) -> int:
    policy_period = 1.0 / max(float(policy_hz), 1.0e-6)
    return max(1, int(round(policy_period / max(float(timestep), 1.0e-9))))


def contract_joint_to_mujoco(joint_name: str) -> str:
    match = re.fullmatch(r"Revolute_(\d+)", str(joint_name))
    if match:
        return f"Revolute {match.group(1)}"
    return str(joint_name).replace("_", " ")


def resolve_package_uris(text: str, package_roots: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        package = match.group(1)
        suffix = match.group(2)
        root = package_roots.get(package)
        if not root:
            return match.group(0)
        return str(Path(root).resolve() / suffix)

    return re.sub(r"package://([^/]+)/([^\"'<>\s]+)", replace, text)


def load_calibration_config(
    *,
    repo_root: Path,
    model_path: Path,
    contract: Any,
    config_path: Path | None = None,
) -> MujocoCalibrationConfig:
    package_root = repo_root / "test_7_description" / "test_7_description"
    config = MujocoCalibrationConfig(
        source_model_path=str(model_path),
        package_roots={"test_7_description": str(package_root)},
        policy_hz=float(getattr(contract, "POLICY_HZ", 60.0)),
        main_drive_joint_names=list(getattr(contract, "MAIN_DRIVE_JOINT_NAMES", [])),
        abad_joint_names=list(getattr(contract, "ABAD_JOINT_NAMES", [])),
    )
    candidates = [path for path in (config_path, Path(f"{model_path}.calibration.json")) if path and path.is_file()]
    if candidates:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        tuple_fields = {"actuator_ctrl_range", "root_pos", "cad_frame_quat"}
        for key, value in payload.items():
            if hasattr(config, key):
                setattr(config, key, tuple(value) if key in tuple_fields else value)
    return config


def _ensure_mujoco_compiler(root: ET.Element) -> None:
    mujoco_elem = root.find("mujoco")
    if mujoco_elem is None:
        mujoco_elem = ET.Element("mujoco")
        root.insert(0, mujoco_elem)
    compiler = mujoco_elem.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_elem, "compiler")
    compiler.set("balanceinertia", "true")
    compiler.set("discardvisual", "false")


def _write_resolved_urdf(source_model_path: Path, artifact_dir: Path, config: MujocoCalibrationConfig) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    text = resolve_package_uris(source_model_path.read_text(encoding="utf-8"), config.package_roots)
    root = ET.fromstring(text)
    _ensure_mujoco_compiler(root)
    tree = ET.ElementTree(root)
    resolved_path = artifact_dir / "redrhex_resolved.urdf"
    tree.write(resolved_path, encoding="utf-8", xml_declaration=True)
    return resolved_path


def _ensure_mujoco_visual_style(root: ET.Element) -> None:
    visual = root.find("visual")
    if visual is None:
        compiler = root.find("compiler")
        insert_at = list(root).index(compiler) + 1 if compiler is not None else 0
        visual = ET.Element("visual")
        root.insert(insert_at, visual)
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.45 0.48 0.52")
    headlight.set("diffuse", "0.75 0.78 0.82")
    headlight.set("specular", "0.25 0.25 0.25")
    rgba = visual.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", "0.72 0.88 0.92 1")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("azimuth", "135")
    global_visual.set("elevation", "-18")

    asset = root.find("asset")
    if asset is None:
        worldbody = root.find("worldbody")
        insert_at = list(root).index(worldbody) if worldbody is not None else len(root)
        asset = ET.Element("asset")
        root.insert(insert_at, asset)
    if asset.find("texture[@name='deploy_mujoco_floor_grid']") is None:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "deploy_mujoco_floor_grid",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.02 0.12 0.14",
                "rgb2": "0.12 0.42 0.46",
                "mark": "edge",
                "markrgb": "0.95 0.90 0.55",
                "width": "512",
                "height": "512",
            },
        )
    if asset.find("texture[@name='deploy_mujoco_sky']") is None:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "deploy_mujoco_sky",
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.06 0.12 0.20",
                "rgb2": "0.62 0.82 0.88",
                "width": "512",
                "height": "512",
            },
        )
    if asset.find("material[@name='deploy_mujoco_floor_mat']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "deploy_mujoco_floor_mat",
                "texture": "deploy_mujoco_floor_grid",
                "texrepeat": "9 9",
                "reflectance": "0.18",
            },
        )

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    if worldbody.find("geom[@name='deploy_mujoco_visual_floor']") is None:
        worldbody.insert(
            0,
            ET.Element(
                "geom",
                {
                    "name": "deploy_mujoco_visual_floor",
                    "type": "plane",
                    "size": "5 5 0.02",
                    "pos": "0 0 0",
                    "material": "deploy_mujoco_floor_mat",
                    "contype": "1",
                    "conaffinity": "1",
                },
            ),
        )


def _format_float_tuple(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _set_floor_height(root: ET.Element, floor_z: float) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    floor = worldbody.find("geom[@name='deploy_mujoco_visual_floor']")
    if floor is not None:
        floor.set("pos", f"0 0 {float(floor_z):.8g}")


def _ensure_floating_root_body(root: ET.Element, config: MujocoCalibrationConfig) -> bool:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return False
    if worldbody.find(f"body[@name='{config.base_body_name}']") is not None:
        return False

    robot_children = [
        child
        for child in list(worldbody)
        if child.tag in {"body", "geom"} and child.attrib.get("name") != "deploy_mujoco_visual_floor"
    ]
    if not robot_children:
        return False

    root_body = ET.Element(
        "body",
        {
            "name": config.base_body_name,
            "pos": _format_float_tuple(config.root_pos),
        },
    )
    ET.SubElement(root_body, "freejoint", {"name": f"{config.base_body_name}_freejoint"})
    cad_frame = ET.SubElement(
        root_body,
        "body",
        {
            "name": f"{config.base_body_name}_cad_frame",
            "quat": _format_float_tuple(config.cad_frame_quat),
        },
    )
    for child in robot_children:
        worldbody.remove(child)
        cad_frame.append(child)
    worldbody.append(root_body)
    return True


def _inject_actuators(mjcf_path: Path, output_path: Path, config: MujocoCalibrationConfig) -> tuple[Path, list[str]]:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    _ensure_mujoco_visual_style(root)
    _set_floor_height(root, config.floor_z)
    _ensure_floating_root_body(root, config)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{float(config.timestep):.8g}")

    existing = root.find("actuator")
    if existing is not None:
        root.remove(existing)
    actuator = ET.SubElement(root, "actuator")
    injected = []
    ctrl_min, ctrl_max = config.actuator_ctrl_range
    available_joint_names = {joint.attrib.get("name") for joint in root.iter("joint") if joint.attrib.get("name")}
    for contract_name in config.action_joint_names:
        mujoco_name = contract_joint_to_mujoco(contract_name)
        if mujoco_name not in available_joint_names:
            raise ValueError(f"MuJoCo joint not found for contract joint {contract_name}: expected {mujoco_name}")
        motor_name = f"{contract_name}_motor"
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": motor_name,
                "joint": mujoco_name,
                "ctrllimited": "true",
                "ctrlrange": f"{ctrl_min:g} {ctrl_max:g}",
                "gear": f"{float(config.actuator_gear):g}",
            },
        )
        injected.append(motor_name)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    config.generated_model_path = str(output_path)
    return output_path, injected


def prepare_mujoco_model(*, model_path: Path, artifact_dir: Path, config: MujocoCalibrationConfig) -> tuple[Path, dict[str, Any]]:
    import mujoco

    resolved_urdf = _write_resolved_urdf(model_path, artifact_dir, config)
    compiled_model = mujoco.MjModel.from_xml_path(str(resolved_urdf))
    generated_mjcf = artifact_dir / "redrhex_generated.mjcf.xml"
    mujoco.mj_saveLastXML(str(generated_mjcf), compiled_model)
    rollout_model, actuators = _inject_actuators(generated_mjcf, artifact_dir / "redrhex_rollout_model.xml", config)
    model = mujoco.MjModel.from_xml_path(str(rollout_model))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    info = {
        "source_model_path": str(model_path),
        "resolved_urdf": str(resolved_urdf),
        "generated_mjcf": str(generated_mjcf),
        "rollout_model": str(rollout_model),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "timestep": float(model.opt.timestep),
        "actuators": actuators,
    }
    return rollout_model, info


class _ZeroPolicy:
    def __init__(self, action_dim: int, warning: str = "") -> None:
        self.action_dim = int(action_dim)
        self.warning = warning

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)


class _OnnxPolicy:
    def __init__(self, policy_path: Path, action_dim: int) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()[0]
        self.action_dim = int(action_dim)
        self.warning = ""

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        action = self.session.run([self.output_meta.name], {self.input_meta.name: obs})[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < self.action_dim:
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        return action[: self.action_dim]


def _policy_runner(policy_path: Path, action_dim: int):
    if not policy_path.is_file():
        return _ZeroPolicy(action_dim, f"No ONNX policy found at {policy_path}; rollout used zero actions.")
    try:
        return _OnnxPolicy(policy_path, action_dim)
    except Exception as exc:
        return _ZeroPolicy(action_dim, f"Could not run ONNX policy with onnxruntime ({exc}); rollout used zero actions.")


def _add_ros_controller_to_path(repo_root: Path) -> None:
    src = repo_root / "ros2_ws" / "src" / "redrhex_rl_controller"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _action_decoder(repo_root: Path):
    _add_ros_controller_to_path(repo_root)
    try:
        from redrhex_rl_controller.action_decoder import ActionDecoder

        decoder = ActionDecoder()
        return decoder, ""
    except Exception as exc:
        return None, f"ActionDecoder unavailable ({exc}); rollout used clipped policy actions directly."


def _joint_values(model: Any, data: Any, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    qpos = []
    qvel = []
    for name in joint_names:
        mujoco_name = contract_joint_to_mujoco(name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_name)
        if joint_id < 0:
            qpos.append(0.0)
            qvel.append(0.0)
            continue
        qpos_addr = model.jnt_qposadr[joint_id]
        qvel_addr = model.jnt_dofadr[joint_id]
        qpos.append(float(data.qpos[qpos_addr]))
        qvel.append(float(data.qvel[qvel_addr]))
    return np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)


def _rotation_roll_pitch(mat: np.ndarray) -> tuple[float, float]:
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    roll = math.atan2(mat[2, 1], mat[2, 2])
    pitch = math.asin(float(np.clip(-mat[2, 0], -1.0, 1.0)))
    return roll, pitch


def _observation(
    *,
    model: Any,
    data: Any,
    config: MujocoCalibrationConfig,
    command: np.ndarray,
    gait_phase: float,
    last_actions: np.ndarray,
    obs_dim: int,
) -> np.ndarray:
    import mujoco

    main_pos, main_vel = _joint_values(model, data, config.main_drive_joint_names)
    abad_pos, abad_vel = _joint_values(model, data, config.abad_joint_names)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)
    if base_id >= 0:
        xmat = data.xmat[base_id].reshape(3, 3)
        projected_gravity = xmat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    single = np.concatenate(
        [
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            projected_gravity,
            np.sin(main_pos),
            np.cos(main_pos),
            main_vel / 8.0,
            abad_pos / 0.61096,
            abad_vel,
            command,
            np.array([math.sin(gait_phase), math.cos(gait_phase)], dtype=np.float64),
            last_actions,
        ]
    ).astype(np.float32)
    if obs_dim <= single.size:
        return single[:obs_dim]
    history_len = max(1, int(math.ceil(obs_dim / single.size)))
    return np.tile(single, history_len)[:obs_dim].astype(np.float32)


def _control_from_action(
    *,
    action: np.ndarray,
    decoder: Any,
    decoder_warning: str,
    main_pos: np.ndarray,
    abad_pos: np.ndarray,
    command: np.ndarray,
    projected_gravity: np.ndarray,
    config: MujocoCalibrationConfig,
    dt: float,
    gait_phase: float,
) -> tuple[np.ndarray, str]:
    if decoder is None:
        return np.clip(action, -1.0, 1.0), decoder_warning
    try:
        decoded = decoder.decode(
            action,
            main_drive_pos=main_pos,
            abad_pos=abad_pos,
            command=command,
            projected_gravity=projected_gravity,
            dt=dt,
            gait_phase=gait_phase,
        )
        main = np.asarray(decoded.target_main_drive_velocity, dtype=np.float64) / max(config.main_drive_vel_limit_rad_s, 1.0e-6)
        abad = np.asarray(decoded.target_abad_position, dtype=np.float64) / max(config.abad_pos_limit_rad, 1.0e-6)
        return np.clip(np.concatenate([main, abad]), -1.0, 1.0), ""
    except Exception as exc:
        return np.clip(action, -1.0, 1.0), f"ActionDecoder failed ({exc}); rollout used clipped policy actions directly."


def _joint_limit_violations(model: Any, data: Any) -> int:
    count = 0
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        qpos_addr = model.jnt_qposadr[joint_id]
        value = float(data.qpos[qpos_addr])
        lo, hi = model.jnt_range[joint_id]
        if value < lo or value > hi:
            count += 1
    return count


def _base_state(model: Any, data: Any, base_body_name: str) -> tuple[int, float, float, float, np.ndarray]:
    import mujoco

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if base_id >= 0:
        height = float(data.xpos[base_id, 2])
        roll, pitch = _rotation_roll_pitch(data.xmat[base_id])
        projected_gravity = data.xmat[base_id].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return base_id, height, roll, pitch, projected_gravity
    return -1, 0.0, 0.0, 0.0, np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _scenario_metrics_from_samples(
    *,
    scenario: MujocoScenario,
    config: MujocoCalibrationConfig,
    steps_completed: int,
    diverged_at: int | None,
    nan_or_inf: bool,
    base_heights: list[float],
    rolls: list[float],
    pitches: list[float],
    joint_limit_violations: int,
    actuator_saturation_steps: int,
    action_values: list[float],
    latency_ms: list[float],
    final_time_s: float,
    final_qpos: np.ndarray,
    trace_path: Path,
    policy_warning: str,
) -> MujocoRolloutMetrics:
    max_roll = max((abs(value) for value in rolls), default=0.0)
    max_pitch = max((abs(value) for value in pitches), default=0.0)
    min_height = min(base_heights, default=0.0)
    fall = min_height < config.min_base_height_m or max(max_roll, max_pitch) > config.max_abs_roll_pitch_rad
    scenario_status = "fail" if (diverged_at is not None or nan_or_inf or fall or joint_limit_violations > 0) else "pass"
    latency = np.asarray(latency_ms, dtype=np.float64) if latency_ms else np.zeros(1)
    action_arr = np.asarray(action_values, dtype=np.float64) if action_values else np.zeros(1)
    return MujocoRolloutMetrics(
        scenario=scenario.name,
        status=scenario_status,
        steps_requested=int(scenario.steps),
        steps_completed=int(steps_completed),
        command=list(scenario.command),
        diverged=diverged_at is not None,
        diverged_at=diverged_at,
        fall_detected=fall,
        nan_or_inf=nan_or_inf,
        base_height_min=float(min_height),
        base_height_max=float(max(base_heights, default=0.0)),
        max_abs_roll_rad=float(max_roll),
        max_abs_pitch_rad=float(max_pitch),
        joint_limit_violations=int(joint_limit_violations),
        actuator_saturation_steps=int(actuator_saturation_steps),
        action_min=float(np.min(action_arr)),
        action_max=float(np.max(action_arr)),
        latency_ms_p50=float(np.percentile(latency, 50)),
        latency_ms_p95=float(np.percentile(latency, 95)),
        latency_ms_max=float(np.max(latency)),
        final_time_s=float(final_time_s),
        final_qpos_head=[float(value) for value in final_qpos[: min(12, final_qpos.size)]],
        trace_path=str(trace_path),
        policy_warning=policy_warning,
    )


def run_mujoco_rollouts(
    *,
    repo_root: Path,
    model_path: Path,
    policy_path: Path,
    artifact_dir: Path,
    config: MujocoCalibrationConfig,
    obs_dim: int,
    action_dim: int,
    scenarios: list[MujocoScenario] | None = None,
) -> MujocoRolloutReport:
    import mujoco

    artifact_dir.mkdir(parents=True, exist_ok=True)
    scenarios = scenarios or default_scenarios()
    rollout_model_path, model_info = prepare_mujoco_model(model_path=model_path, artifact_dir=artifact_dir, config=config)
    model = mujoco.MjModel.from_xml_path(str(rollout_model_path))
    policy = _policy_runner(policy_path, action_dim)
    decoder, decoder_warning = _action_decoder(repo_root)
    warnings = [warning for warning in (getattr(policy, "warning", ""), decoder_warning) if warning]
    metrics: list[MujocoRolloutMetrics] = []

    config_path = artifact_dir / "mujoco_calibration.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    for scenario in scenarios:
        decimation = policy_decimation(config.policy_hz, model.opt.timestep)
        rng = np.random.default_rng(scenario.seed)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        # Keep reset deterministic even if future scenarios add small perturbations.
        if data.qvel.size:
            data.qvel[:] += rng.normal(0.0, 0.0, size=data.qvel.shape)
        mujoco.mj_forward(model, data)
        last_action = np.zeros(action_dim, dtype=np.float32)
        last_ctrl = np.zeros(action_dim, dtype=np.float64)
        command = np.asarray(scenario.command, dtype=np.float64)
        latency_ms: list[float] = []
        action_values: list[float] = []
        base_heights: list[float] = []
        rolls: list[float] = []
        pitches: list[float] = []
        trace_rows: list[dict[str, Any]] = []
        diverged_at: int | None = None
        nan_or_inf = False
        actuator_saturation_steps = 0
        joint_limit_violations = 0
        warning = getattr(policy, "warning", "")

        for step in range(max(1, int(scenario.steps))):
            policy_tick = step // decimation
            gait_phase = 2.0 * math.pi * float(policy_tick) / max(float(config.policy_hz), 1.0)
            if step % decimation == 0:
                obs = _observation(
                    model=model,
                    data=data,
                    config=config,
                    command=command,
                    gait_phase=gait_phase,
                    last_actions=last_action,
                    obs_dim=obs_dim,
                )
                main_pos, _ = _joint_values(model, data, config.main_drive_joint_names)
                abad_pos, _ = _joint_values(model, data, config.abad_joint_names)
                _, _, _, _, projected_gravity = _base_state(model, data, config.base_body_name)

                started = time.perf_counter()
                action = np.asarray(policy(obs), dtype=np.float32).reshape(-1)[:action_dim]
                ctrl, control_warning = _control_from_action(
                    action=action,
                    decoder=decoder,
                    decoder_warning=decoder_warning,
                    main_pos=main_pos,
                    abad_pos=abad_pos,
                    command=command,
                    projected_gravity=projected_gravity,
                    config=config,
                    dt=float(model.opt.timestep) * decimation,
                    gait_phase=gait_phase,
                )
                latency_ms.append((time.perf_counter() - started) * 1000.0)
                if control_warning and control_warning not in warnings:
                    warnings.append(control_warning)
                    warning = control_warning
                last_action = action.copy()
                last_ctrl = np.asarray(ctrl, dtype=np.float64)
                action_values.extend(last_action.tolist())
            if model.nu:
                data.ctrl[: min(model.nu, last_ctrl.size)] = last_ctrl[: min(model.nu, last_ctrl.size)]
                ctrl_min, ctrl_max = config.actuator_ctrl_range
                if np.any(np.isclose(data.ctrl[: min(model.nu, last_ctrl.size)], ctrl_min, atol=1.0e-5)) or np.any(
                    np.isclose(data.ctrl[: min(model.nu, last_ctrl.size)], ctrl_max, atol=1.0e-5)
                ):
                    actuator_saturation_steps += 1
            mujoco.mj_step(model, data)
            joint_limit_violations += _joint_limit_violations(model, data)

            _, height, roll, pitch, _ = _base_state(model, data, config.base_body_name)
            base_heights.append(height)
            rolls.append(roll)
            pitches.append(pitch)
            if step == 0 or step == scenario.steps - 1 or step % 25 == 0:
                trace_rows.append(
                    {
                        "step": step,
                        "time_s": float(data.time),
                        "base_height": height,
                        "roll": roll,
                        "pitch": pitch,
                        "action_min": float(np.min(last_action)) if last_action.size else 0.0,
                        "action_max": float(np.max(last_action)) if last_action.size else 0.0,
                    }
                )
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                nan_or_inf = True
                diverged_at = step
                break
            if data.qpos.size and float(np.max(np.abs(data.qpos))) > config.max_abs_qpos:
                diverged_at = step
                break

        trace_path = artifact_dir / f"mujoco_trace_{scenario.name}.json"
        trace_path.write_text(json.dumps(trace_rows, indent=2), encoding="utf-8")
        steps_completed = int(scenario.steps if diverged_at is None else diverged_at + 1)
        metrics.append(
            _scenario_metrics_from_samples(
                scenario=scenario,
                config=config,
                steps_completed=steps_completed,
                diverged_at=diverged_at,
                nan_or_inf=nan_or_inf,
                base_heights=base_heights,
                rolls=rolls,
                pitches=pitches,
                joint_limit_violations=joint_limit_violations,
                actuator_saturation_steps=actuator_saturation_steps,
                action_values=action_values,
                latency_ms=latency_ms,
                final_time_s=float(data.time),
                final_qpos=data.qpos,
                trace_path=trace_path,
                policy_warning=warning,
            )
        )

    metrics_path = artifact_dir / "mujoco_rollout_metrics.json"
    status = "pass" if all(metric.status == "pass" for metric in metrics) else "fail"
    if not config.calibrated:
        status = "warn"
    summary = (
        "MuJoCo scenarios completed; results are advisory until calibration is marked true."
        if not config.calibrated
        else ("MuJoCo calibrated rollout checks passed." if status == "pass" else "MuJoCo calibrated rollout checks failed.")
    )
    report = MujocoRolloutReport(
        status=status,
        summary=summary,
        calibrated=bool(config.calibrated),
        config=config,
        scenarios=metrics,
        artifacts={
            "artifact_dir": str(artifact_dir),
            "resolved_urdf": model_info["resolved_urdf"],
            "generated_mjcf": model_info["generated_mjcf"],
            "rollout_model": model_info["rollout_model"],
            "calibration_config": str(config_path),
            "metrics": str(metrics_path),
        },
        model_info=model_info,
        warnings=warnings,
    )
    metrics_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def run_mujoco_playback(
    *,
    repo_root: Path,
    model_path: Path,
    policy_path: Path,
    artifact_dir: Path,
    config: MujocoCalibrationConfig,
    obs_dim: int,
    action_dim: int,
    playback: MujocoPlaybackConfig,
) -> MujocoPlaybackReport:
    import mujoco

    mode = str(playback.mode or "").lower()
    if mode not in {"viewer", "record"}:
        raise ValueError("MuJoCo playback mode must be viewer or record")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scenario = scenario_by_name(playback.scenario, steps=max(1, int(playback.steps)))
    rollout_model_path, model_info = prepare_mujoco_model(model_path=model_path, artifact_dir=artifact_dir, config=config)
    model = mujoco.MjModel.from_xml_path(str(rollout_model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    policy = _policy_runner(policy_path, action_dim)
    decoder, decoder_warning = _action_decoder(repo_root)
    warnings = [warning for warning in (getattr(policy, "warning", ""), decoder_warning) if warning]
    command = np.asarray(scenario.command, dtype=np.float64)
    decimation = policy_decimation(config.policy_hz, model.opt.timestep)
    last_action = np.zeros(action_dim, dtype=np.float32)
    last_ctrl = np.zeros(action_dim, dtype=np.float64)
    latency_ms: list[float] = []
    action_values: list[float] = []
    base_heights: list[float] = []
    rolls: list[float] = []
    pitches: list[float] = []
    trace_rows: list[dict[str, Any]] = []
    diverged_at: int | None = None
    nan_or_inf = False
    actuator_saturation_steps = 0
    joint_limit_violations = 0
    warning = getattr(policy, "warning", "")
    video_path = ""
    renderer = None
    writer = None
    viewer = None
    next_frame_time = 0.0
    frame_period = 1.0 / max(float(playback.fps), 1.0)

    try:
        if mode == "viewer":
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(
                model,
                data,
                show_left_ui=bool(playback.show_ui),
                show_right_ui=bool(playback.show_ui),
            )
        else:
            try:
                import imageio.v2 as imageio
            except Exception as exc:
                raise RuntimeError("imageio and imageio-ffmpeg are required for MuJoCo MP4 recording") from exc
            renderer = mujoco.Renderer(model, height=int(playback.height), width=int(playback.width))
            video = artifact_dir / f"mujoco_{scenario.name}.mp4"
            writer = imageio.get_writer(str(video), fps=int(playback.fps), codec="libx264", quality=8)
            video_path = str(video)

        for step in range(max(1, int(playback.steps))):
            if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                break
            loop_started = time.perf_counter()
            policy_tick = step // decimation
            gait_phase = 2.0 * math.pi * float(policy_tick) / max(float(config.policy_hz), 1.0)
            if step % decimation == 0:
                obs = _observation(
                    model=model,
                    data=data,
                    config=config,
                    command=command,
                    gait_phase=gait_phase,
                    last_actions=last_action,
                    obs_dim=obs_dim,
                )
                main_pos, _ = _joint_values(model, data, config.main_drive_joint_names)
                abad_pos, _ = _joint_values(model, data, config.abad_joint_names)
                _, _, _, _, projected_gravity = _base_state(model, data, config.base_body_name)
                started = time.perf_counter()
                action = np.asarray(policy(obs), dtype=np.float32).reshape(-1)[:action_dim]
                ctrl, control_warning = _control_from_action(
                    action=action,
                    decoder=decoder,
                    decoder_warning=decoder_warning,
                    main_pos=main_pos,
                    abad_pos=abad_pos,
                    command=command,
                    projected_gravity=projected_gravity,
                    config=config,
                    dt=float(model.opt.timestep) * decimation,
                    gait_phase=gait_phase,
                )
                latency_ms.append((time.perf_counter() - started) * 1000.0)
                if control_warning and control_warning not in warnings:
                    warnings.append(control_warning)
                    warning = control_warning
                last_action = action.copy()
                last_ctrl = np.asarray(ctrl, dtype=np.float64)
                action_values.extend(last_action.tolist())

            if model.nu:
                data.ctrl[: min(model.nu, last_ctrl.size)] = last_ctrl[: min(model.nu, last_ctrl.size)]
                ctrl_min, ctrl_max = config.actuator_ctrl_range
                if np.any(np.isclose(data.ctrl[: min(model.nu, last_ctrl.size)], ctrl_min, atol=1.0e-5)) or np.any(
                    np.isclose(data.ctrl[: min(model.nu, last_ctrl.size)], ctrl_max, atol=1.0e-5)
                ):
                    actuator_saturation_steps += 1
            mujoco.mj_step(model, data)
            joint_limit_violations += _joint_limit_violations(model, data)
            _, height, roll, pitch, _ = _base_state(model, data, config.base_body_name)
            base_heights.append(height)
            rolls.append(roll)
            pitches.append(pitch)

            if step == 0 or step == playback.steps - 1 or step % 25 == 0:
                trace_rows.append(
                    {
                        "step": step,
                        "time_s": float(data.time),
                        "base_height": height,
                        "roll": roll,
                        "pitch": pitch,
                        "action_min": float(np.min(last_action)) if last_action.size else 0.0,
                        "action_max": float(np.max(last_action)) if last_action.size else 0.0,
                    }
                )
            if renderer is not None and writer is not None and data.time >= next_frame_time:
                renderer.update_scene(data)
                writer.append_data(renderer.render())
                next_frame_time += frame_period
            if viewer is not None:
                viewer.sync()
                sleep_s = float(model.opt.timestep) - (time.perf_counter() - loop_started)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                nan_or_inf = True
                diverged_at = step
                break
            if data.qpos.size and float(np.max(np.abs(data.qpos))) > config.max_abs_qpos:
                diverged_at = step
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()

    trace_path = artifact_dir / f"mujoco_playback_trace_{scenario.name}.json"
    trace_path.write_text(json.dumps(trace_rows, indent=2), encoding="utf-8")
    steps_completed = len(base_heights)
    metrics = _scenario_metrics_from_samples(
        scenario=scenario,
        config=config,
        steps_completed=steps_completed,
        diverged_at=diverged_at,
        nan_or_inf=nan_or_inf,
        base_heights=base_heights,
        rolls=rolls,
        pitches=pitches,
        joint_limit_violations=joint_limit_violations,
        actuator_saturation_steps=actuator_saturation_steps,
        action_values=action_values,
        latency_ms=latency_ms,
        final_time_s=float(data.time),
        final_qpos=data.qpos,
        trace_path=trace_path,
        policy_warning=warning,
    )
    status = "completed" if metrics.status == "pass" else "failed"
    summary = f"MuJoCo {mode} playback completed for {scenario.name}."
    if status == "failed":
        summary = f"MuJoCo {mode} playback finished with rollout metric failures."
    report = MujocoPlaybackReport(
        status=status,
        summary=summary,
        mode=mode,
        scenario=scenario.name,
        steps_requested=int(playback.steps),
        steps_completed=steps_completed,
        video_path=video_path,
        artifacts={
            "artifact_dir": str(artifact_dir),
            "rollout_model": str(rollout_model_path),
            "trace": str(trace_path),
            "report": str(artifact_dir / "mujoco_playback_report.json"),
            **({"video": video_path} if video_path else {}),
        },
        metrics={**metrics.to_dict(), "policy_decimation": decimation, "model_info": model_info},
        warnings=warnings,
    )
    report_path = artifact_dir / "mujoco_playback_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report
