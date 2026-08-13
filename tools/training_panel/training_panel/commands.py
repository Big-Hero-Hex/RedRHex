from __future__ import annotations

import random
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import PanelPaths
from .reward_overrides import normalize_reward_overrides


DEFAULT_TASK = "Template-Redrhex-Direct-v0"
FORWARD_FAST_TASK = "Template-Redrhex-ForwardFast-Direct-v0"
DEFAULT_VIDEO_PRESET = "high"
DEFAULT_FOLLOW_CAMERA_EYE = (-3.0, -2.4, 1.6)
DEFAULT_FOLLOW_CAMERA_LOOKAT = (0.45, 0.0, 0.35)
SPRING_BACKENDS = ("explicit", "native")
_TOP_LEVEL_SPRING_BACKEND_RE = re.compile(r"^spring_backend\s*:\s*(.*)$")


def validate_spring_backend(spring_backend: str) -> None:
    if spring_backend not in SPRING_BACKENDS:
        raise ValueError(f"spring_backend must be one of: {', '.join(SPRING_BACKENDS)}")


def _spring_backend_from_yaml(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OSError(f"Could not read spring backend metadata: {path}") from exc
    values = []
    for line in lines:
        match = _TOP_LEVEL_SPRING_BACKEND_RE.match(line)
        if not match:
            continue
        value = match.group(1).split("#", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values.append(value)
    if len(values) > 1:
        raise ValueError(f"Duplicate top-level spring_backend entries in {path}")
    if values:
        validate_spring_backend(values[0])
        return values[0]
    return None


def resolve_spring_backend(run: dict[str, Any] | None, checkpoint: str | Path | None = None) -> str:
    params = (run or {}).get("params")
    if isinstance(params, dict) and "spring_backend" in params:
        spring_backend = params["spring_backend"]
        validate_spring_backend(spring_backend)
        return spring_backend

    log_dir = (run or {}).get("log_dir")
    if log_dir:
        spring_backend = _spring_backend_from_yaml(Path(str(log_dir)) / "params" / "torsion_spring.yaml")
        if spring_backend is not None:
            return spring_backend

    if checkpoint:
        spring_backend = _spring_backend_from_yaml(Path(checkpoint).parent / "params" / "torsion_spring.yaml")
        if spring_backend is not None:
            return spring_backend
    return "explicit"


@dataclass(frozen=True)
class VideoParams:
    preset: str
    width: int
    height: int
    length: int
    fps: int
    rendering_mode: str

    @classmethod
    def from_preset(cls, preset: str | None) -> "VideoParams":
        key = str(preset or DEFAULT_VIDEO_PRESET).lower()
        try:
            params = VIDEO_PRESETS[key]
        except KeyError as exc:
            raise ValueError(f"Unknown video preset: {preset}") from exc
        return params

    def validate(self) -> None:
        if self.width < 320 or self.width > 3840:
            raise ValueError("video width must be between 320 and 3840")
        if self.height < 240 or self.height > 2160:
            raise ValueError("video height must be between 240 and 2160")
        if self.length < 1 or self.length > 100000:
            raise ValueError("video length must be between 1 and 100000 steps")
        if self.fps < 1 or self.fps > 120:
            raise ValueError("video fps must be between 1 and 120")
        if self.rendering_mode not in ("performance", "balanced", "quality"):
            raise ValueError("rendering_mode must be performance, balanced, or quality")

    def to_dict(self) -> dict:
        return asdict(self)


VIDEO_PRESETS = {
    "high": VideoParams("high", width=1920, height=1080, length=1200, fps=30, rendering_mode="quality"),
}


def _normalize_override_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_override_value(item) for item in value]
    return value


@dataclass
class TrainingParams:
    task: str = DEFAULT_TASK
    num_envs: int = 4
    max_iterations: int = 1
    device: str = "cuda:0"
    spring_backend: str = "explicit"
    headless: bool = True
    seed: int | None = None
    resume: bool = False
    checkpoint: str | None = None
    reward_preset_id: str = "baseline"
    reward_overrides: dict = field(default_factory=dict)
    terrain_preset_id: str = "baseline"
    terrain_overrides: dict = field(default_factory=dict)
    tweak_source_run_id: str | None = None
    tweak_source_label: str | None = None
    requester_id: str | None = None
    requester_label: str | None = None
    display_name: str | None = None
    folder: str | None = None
    client_request_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingParams":
        raw_overrides = data.get("reward_overrides") or {}
        raw_terrain_overrides = data.get("terrain_overrides") or {}
        params = cls(
            task=str(data.get("task") or DEFAULT_TASK),
            num_envs=int(data.get("num_envs") or 4),
            max_iterations=int(data.get("max_iterations") or 1),
            device=str(data.get("device") or "cuda:0"),
            spring_backend=data["spring_backend"] if "spring_backend" in data else "explicit",
            headless=bool(data.get("headless", True)),
            # A blank seed used to mean "env default", which made runs
            # unreproducible. The panel now picks one and records it.
            seed=int(data["seed"]) if data.get("seed") not in (None, "") else random.randint(0, 2 ** 31 - 1),
            resume=bool(data.get("resume", False)),
            checkpoint=str(data["checkpoint"]) if data.get("checkpoint") else None,
            reward_preset_id=str(data.get("reward_preset_id") or "baseline"),
            reward_overrides=normalize_reward_overrides(raw_overrides),
            terrain_preset_id=str(data.get("terrain_preset_id") or "baseline"),
            terrain_overrides={str(k): _normalize_override_value(v) for k, v in raw_terrain_overrides.items()},
            tweak_source_run_id=str(data["tweak_source_run_id"]) if data.get("tweak_source_run_id") else None,
            tweak_source_label=str(data["tweak_source_label"]) if data.get("tweak_source_label") else None,
            requester_id=str(data["requester_id"]) if data.get("requester_id") else None,
            requester_label=str(data["requester_label"]) if data.get("requester_label") else None,
            display_name=str(data["display_name"]).strip() if data.get("display_name") else None,
            folder=str(data["folder"]).strip() if data.get("folder") else None,
            client_request_id=str(data["client_request_id"]).strip() if data.get("client_request_id") else None,
        )
        params.validate()
        return params

    def validate(self) -> None:
        if not self.task:
            raise ValueError("task is required")
        if self.num_envs < 1 or self.num_envs > 8192:
            raise ValueError("num_envs must be between 1 and 8192")
        if self.max_iterations < 1 or self.max_iterations > 100000:
            raise ValueError("max_iterations must be between 1 and 100000")
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError("device must be cpu or cuda[:index]")
        validate_spring_backend(self.spring_backend)
        if self.resume and not self.checkpoint:
            raise ValueError("checkpoint is required when resume is enabled")
        if self.display_name and len(self.display_name) > 120:
            raise ValueError("display_name must be 120 characters or fewer")
        if self.folder and len(self.folder) > 120:
            raise ValueError("folder must be 120 characters or fewer")
        if self.client_request_id and len(self.client_request_id) > 160:
            raise ValueError("client_request_id must be 160 characters or fewer")

    def to_dict(self) -> dict:
        return asdict(self)


def training_argv(params: TrainingParams) -> list[str]:
    argv = [
        "scripts/rsl_rl/train.py",
        "--task",
        params.task,
        "--num_envs",
        str(params.num_envs),
        "--max_iterations",
        str(params.max_iterations),
        "--device",
        params.device,
        "--spring-backend",
        params.spring_backend,
    ]
    # train.py only applies active_*_override.json when this flag is present.
    argv.append("--panel_overrides")
    if params.headless:
        argv.append("--headless")
    if params.seed is not None:
        argv.extend(["--seed", str(params.seed)])
    if params.resume:
        argv.append("--resume")
        argv.extend(["--checkpoint", params.checkpoint or ""])
    return argv


def play_argv(
    checkpoint: str,
    task: str = DEFAULT_TASK,
    num_envs: int = 1,
    device: str = "cuda:0",
    *,
    spring_backend: str = "explicit",
    headless: bool = False,
    video: bool = False,
    video_length: int | None = None,
    video_width: int | None = None,
    video_height: int | None = None,
    video_fps: int | None = None,
    rendering_mode: str | None = None,
    initial_command: str | None = None,
    terrain_override_file: str | None = None,
    camera_follow_robot: bool = False,
    camera_eye: tuple[float, float, float] | None = None,
    camera_lookat: tuple[float, float, float] | None = None,
    export_policy_only: bool = False,
) -> list[str]:
    validate_spring_backend(spring_backend)
    argv = [
        "scripts/rsl_rl/play.py",
        "--task",
        task,
        "--num_envs",
        str(num_envs),
        "--device",
        device,
        "--spring-backend",
        spring_backend,
    ]
    if headless:
        argv.append("--headless")
    if video:
        argv.append("--video")
        if video_length is not None:
            argv.extend(["--video_length", str(video_length)])
        if video_width is not None:
            argv.extend(["--video_width", str(video_width)])
        if video_height is not None:
            argv.extend(["--video_height", str(video_height)])
        if video_fps is not None:
            argv.extend(["--video_fps", str(video_fps)])
        if rendering_mode:
            argv.extend(["--rendering_mode", rendering_mode])
    if initial_command is not None:
        argv.extend(["--initial_command", initial_command])
    if terrain_override_file:
        argv.extend(["--terrain_override_file", terrain_override_file])
    if camera_follow_robot:
        eye = camera_eye or DEFAULT_FOLLOW_CAMERA_EYE
        lookat = camera_lookat or DEFAULT_FOLLOW_CAMERA_LOOKAT
        argv.append("--camera_follow_robot")
        argv.extend(["--camera_eye", *(str(value) for value in eye)])
        argv.extend(["--camera_lookat", *(str(value) for value in lookat)])
    if export_policy_only:
        argv.append("--export_policy_only")
    argv.extend(["--checkpoint", checkpoint])
    return argv


def export_onnx_argv(
    checkpoint: str,
    task: str = DEFAULT_TASK,
    num_envs: int = 1,
    device: str = "cuda:0",
    *,
    spring_backend: str = "explicit",
) -> list[str]:
    return play_argv(
        checkpoint=checkpoint,
        task=task,
        num_envs=num_envs,
        device=device,
        spring_backend=spring_backend,
        headless=True,
        export_policy_only=True,
    )


def tensorboard_argv(logdir: Path, host: str, port: int) -> list[str]:
    return ["tensorboard", "--logdir", str(logdir), "--host", host, "--port", str(port)]


def shell_env_prelude(paths: PanelPaths) -> list[str]:
    return [
        f"source {shlex.quote(str(paths.conda_sh))}",
        f"conda activate {shlex.quote(paths.conda_env)}",
        'export PATH="$CONDA_PREFIX/bin:$PATH"',
        'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"',
    ]


def shell_for_isaaclab(paths: PanelPaths, script_argv: list[str]) -> str:
    quoted_launcher = shlex.quote(str(paths.isaaclab_launcher))
    quoted_args = " ".join(shlex.quote(arg) for arg in script_argv)
    return "\n".join(
        [
            f"export REDRHEX_ROOT={shlex.quote(str(paths.repo_root))}",
            f"export ISAACLAB_ROOT={shlex.quote(str(paths.isaaclab_root))}",
            f"export ISAACSIM_ROOT={shlex.quote(str(paths.isaacsim_root))}",
            *shell_env_prelude(paths),
            "export TERM=xterm",
            f"cd {shlex.quote(str(paths.repo_root))}",
            f"exec {quoted_launcher} -p {quoted_args}",
        ]
    )


def shell_for_command(paths: PanelPaths, argv: list[str]) -> str:
    quoted = " ".join(shlex.quote(arg) for arg in argv)
    return "\n".join(
        [
            *shell_env_prelude(paths),
            f"cd {shlex.quote(str(paths.repo_root))}",
            f"exec {quoted}",
        ]
    )


def display_isaaclab_command(paths: PanelPaths, script_argv: list[str]) -> str:
    quoted_args = " ".join(shlex.quote(arg) for arg in script_argv)
    return f"{paths.isaaclab_launcher} -p {quoted_args}"
