from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


PRESETS: dict[str, dict] = {
    "loose":   {"window_iterations": 100, "min_improvement_pct": 5.0},
    "default": {"window_iterations": 200, "min_improvement_pct": 2.0},
    "strict":  {"window_iterations": 400, "min_improvement_pct": 1.0},
}
CONVERGENCE_MAX_EVENT_BYTES = 128 * 1024 * 1024
CONVERGENCE_MAX_SCALARS = 2000

_ALLOWED_FIELDS = {"enabled", "preset", "window_iterations", "min_improvement_pct",
                   "primary_tag", "min_iterations", "cooldown_minutes", "auto_record_video"}


@dataclass
class ConvergenceConfig:
    enabled: bool = True
    preset: str = "default"
    window_iterations: int = 200
    min_improvement_pct: float = 2.0
    primary_tag: str = "Train/mean_reward"
    min_iterations: int = 100
    cooldown_minutes: int = 60
    auto_record_video: bool = True


@dataclass
class ConvergenceResult:
    detected: bool
    iteration: int = 0
    window_max: float = 0.0
    window_min: float = 0.0
    improvement_pct: float = 0.0
    tag: str = "Train/mean_reward"
    reason: str = ""


class ConvergenceChecker:
    @staticmethod
    def _event_file_bytes(log_dir: Path) -> int:
        total = 0
        try:
            event_files = list(log_dir.glob("events.out.tfevents.*"))
        except OSError:
            return 0
        for path in event_files:
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def read_scalars(self, log_dir: Path, tag: str) -> list[tuple[int, float]]:
        """Read (step, value) pairs from TensorBoard event files in log_dir."""
        if self._event_file_bytes(log_dir) > CONVERGENCE_MAX_EVENT_BYTES:
            return []
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return []
        try:
            ea = EventAccumulator(str(log_dir), size_guidance={"scalars": CONVERGENCE_MAX_SCALARS})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            if tag not in tags:
                return []
            return [(e.step, e.value) for e in ea.Scalars(tag)]
        except Exception:
            return []

    def check(self, log_dir: Path, config: ConvergenceConfig) -> ConvergenceResult:
        """Apply plateau detection to TensorBoard scalars. Safe — never raises."""
        scalars = self.read_scalars(log_dir, config.primary_tag)
        if not scalars:
            return ConvergenceResult(detected=False, tag=config.primary_tag,
                                     reason="no data for tag")

        total = len(scalars)
        if total < config.min_iterations:
            return ConvergenceResult(detected=False, tag=config.primary_tag,
                                     reason=f"only {total} iterations, need {config.min_iterations}")

        window = scalars[-config.window_iterations:]
        values = [v for _, v in window]
        window_max = max(values)
        window_min = min(values)
        mean = sum(values) / len(values)
        latest_step = window[-1][0]

        if mean == 0.0:
            return ConvergenceResult(detected=False, iteration=latest_step,
                                     window_max=window_max, window_min=window_min,
                                     tag=config.primary_tag, reason="mean reward is zero")

        improvement_pct = abs(window_max - window_min) / abs(mean) * 100.0
        detected = improvement_pct < config.min_improvement_pct

        reason = (
            f"improvement {improvement_pct:.1f}% over last {len(window)} iters "
            f"({'< ' if detected else '>= '}{config.min_improvement_pct}% threshold)"
        )
        return ConvergenceResult(
            detected=detected,
            iteration=latest_step,
            window_max=round(window_max, 4),
            window_min=round(window_min, 4),
            improvement_pct=round(improvement_pct, 2),
            tag=config.primary_tag,
            reason=reason,
        )


def _apply_preset(config: ConvergenceConfig) -> ConvergenceConfig:
    """If preset is a named preset, overwrite window/threshold with preset values."""
    if config.preset in PRESETS:
        p = PRESETS[config.preset]
        config.window_iterations = p["window_iterations"]
        config.min_improvement_pct = p["min_improvement_pct"]
    return config


def load_convergence_config(config_file: Path) -> ConvergenceConfig:
    """Read convergence_config.json → ConvergenceConfig. Missing/corrupt file → defaults."""
    try:
        raw = json.loads(config_file.read_text())
        safe = {k: v for k, v in raw.items() if k in _ALLOWED_FIELDS}
        cfg = ConvergenceConfig(**safe)
    except Exception:
        cfg = ConvergenceConfig()
    return _apply_preset(cfg)


def save_convergence_config(config: ConvergenceConfig, config_file: Path) -> dict:
    """Write ConvergenceConfig → JSON. Returns the saved dict."""
    data = asdict(config)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(data, indent=2))
    return data


def apply_settings(updates: dict, config_file: Path) -> ConvergenceConfig:
    """
    Merge whitelisted fields from updates into existing config, save, and return.
    If preset changes to a named preset, window/threshold are overridden automatically.
    """
    cfg = load_convergence_config(config_file)
    for key, value in updates.items():
        if key not in _ALLOWED_FIELDS:
            continue
        if key == "enabled":
            cfg.enabled = bool(value)
        elif key == "auto_record_video":
            cfg.auto_record_video = bool(value)
        elif key == "preset":
            cfg.preset = str(value)
        elif key == "window_iterations":
            cfg.window_iterations = max(10, int(value))
        elif key == "min_improvement_pct":
            cfg.min_improvement_pct = max(0.01, float(value))
        elif key == "primary_tag":
            cfg.primary_tag = str(value)
        elif key == "min_iterations":
            cfg.min_iterations = max(1, int(value))
        elif key == "cooldown_minutes":
            cfg.cooldown_minutes = max(0, int(value))
    _apply_preset(cfg)
    save_convergence_config(cfg, config_file)
    return cfg


# Tags the panel is willing to plot. An allowlist, so a query parameter can
# never be used to sweep arbitrary strings through the event accumulator.
SCALAR_TAG_ALLOWLIST = frozenset({
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/learning_rate",
    "Policy/mean_noise_std",
})

DEFAULT_SCALAR_TAGS = ["Train/mean_reward", "Train/mean_episode_length"]
MAX_SCALAR_POINTS = 2000


def requested_tags(raw: str) -> list[str]:
    """Parse a comma-separated tag query into an allowlisted, de-duplicated list."""
    tags: list[str] = []
    for candidate in (raw or "").split(","):
        tag = candidate.strip()
        if tag in SCALAR_TAG_ALLOWLIST and tag not in tags:
            tags.append(tag)
    return tags or list(DEFAULT_SCALAR_TAGS)


def downsample(points: list[tuple[int, float]], limit: int) -> list[tuple[int, float]]:
    """Reduce a series to at most `limit` points, keeping the first and last."""
    if limit < 1:
        return []
    if len(points) <= limit:
        return list(points)
    if limit == 1:
        return [points[-1]]
    stride = (len(points) - 1) / (limit - 1)
    reduced = [points[int(round(i * stride))] for i in range(limit)]
    reduced[-1] = points[-1]
    return reduced
