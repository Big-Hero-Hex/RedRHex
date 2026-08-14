from __future__ import annotations

import ast
import re
from pathlib import Path


REWARD_SCALE_RE = re.compile(r"^\s*(rew_scale_[A-Za-z0-9_]+)\s*=\s*([^#\n]+)(?:#\s*(.*))?")
_YAML_SCALE_RE = re.compile(r"^\s*(rew_scale_[A-Za-z0-9_]+)\s*:\s*([^\s#]+)")
_YAML_V2_HEADER_RE = re.compile(r"^(\s*)v2_reward_scales\s*:\s*(?:#.*)?$")
_YAML_V2_SCALE_RE = re.compile(r"^(\s+)([A-Za-z0-9_]+)\s*:\s*([^\s#]+)")

_DIFF_TOLERANCE = 1e-9  # float comparison tolerance


def read_reward_scales_from_yaml(env_yaml_path: Path) -> dict[str, float]:
    """Parse flat and nested reward entries from a saved env.yaml without PyYAML."""
    if not env_yaml_path.exists():
        return {}
    scales: dict[str, float] = {}
    v2_indent: int | None = None
    for line in env_yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = _YAML_V2_HEADER_RE.match(line)
        if header:
            v2_indent = len(header.group(1))
            continue
        if v2_indent is not None:
            nested = _YAML_V2_SCALE_RE.match(line)
            if nested and len(nested.group(1)) > v2_indent:
                try:
                    scales[f"v2_reward_scales.{nested.group(2)}"] = float(nested.group(3))
                except ValueError:
                    pass
                continue
            if line.strip() and len(line) - len(line.lstrip()) <= v2_indent:
                v2_indent = None
        m = _YAML_SCALE_RE.match(line)
        if m:
            try:
                scales[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return scales


def reward_diff(
    yaml_scales: dict[str, float],
    defaults: dict[str, float],
) -> dict:
    """Compare yaml_scales against defaults. Returns changed/same/missing lists."""
    changed = []
    same = []
    for name, default_val in defaults.items():
        if name not in yaml_scales:
            continue
        yaml_val = yaml_scales[name]
        if abs(yaml_val - default_val) > _DIFF_TOLERANCE:
            delta_pct = round((yaml_val - default_val) / (abs(default_val) + 1e-12) * 100, 1)
            changed.append({
                "name": name,
                "yaml_value": yaml_val,
                "default_value": default_val,
                "delta_pct": delta_pct,
            })
        else:
            same.append(name)
    # Also flag values in YAML that have no matching default (new fields)
    for name, yaml_val in yaml_scales.items():
        if name not in defaults:
            changed.append({
                "name": name,
                "yaml_value": yaml_val,
                "default_value": None,
                "delta_pct": None,
            })
    return {"changed": changed, "same": same}



TWEAKABLE_FILES = [
    {
        "title": "Reward scales and environment constants",
        "path": "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py",
        "why": "Primary place to inspect reward weights, command ranges, episode limits, and joint mappings.",
    },
    {
        "title": "Reward calculation logic",
        "path": "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py",
        "why": "Shows how each reward component is computed and logged during training.",
    },
    {
        "title": "PPO training settings",
        "path": "source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py",
        "why": "Controls PPO network size, learning rate, entropy, batch settings, and save interval.",
    },
]


def _v2_reward_scales(cfg_path: Path, task: str | None) -> list[dict]:
    """Read the active class-level v2 reward mapping without importing Isaac Lab."""

    class_name = (
        "RedrhexForwardFastEnvCfg"
        if task is None or "ForwardFast" in str(task)
        else "RedrhexEnvCfg"
    )
    try:
        tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
    except (OSError, SyntaxError):
        return []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if not any(isinstance(target, ast.Name) and target.id == "v2_reward_scales" for target in targets):
                continue
            try:
                values = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                return []
            if not isinstance(values, dict):
                return []
            entries = []
            for key_node, (name, value) in zip(getattr(statement.value, "keys", []), values.items()):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                entries.append(
                    {
                        "name": f"v2_reward_scales.{name}",
                        "value": str(float(value)),
                        "comment": "",
                        "path": str(cfg_path),
                        "relative_path": TWEAKABLE_FILES[0]["path"],
                        "line": int(getattr(key_node, "lineno", statement.lineno)),
                    }
                )
            return entries
    return []


def scan_reward_scales(repo_root: Path, task: str | None = None) -> list[dict]:
    cfg_path = repo_root / TWEAKABLE_FILES[0]["path"]
    if not cfg_path.exists():
        return []
    scales = []
    for line_no, line in enumerate(cfg_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = REWARD_SCALE_RE.match(line)
        if not match:
            continue
        name, value, comment = match.groups()
        scales.append(
            {
                "name": name,
                "value": value.strip(),
                "comment": (comment or "").strip(),
                "path": str(cfg_path),
                "relative_path": TWEAKABLE_FILES[0]["path"],
                "line": line_no,
            }
        )
    scales.extend(_v2_reward_scales(cfg_path, task))
    return scales


def reward_defaults(repo_root: Path, task: str | None = None) -> dict[str, float]:
    """Return current flat and task-specific nested reward defaults."""
    scales = {}
    for item in scan_reward_scales(repo_root, task=task):
        try:
            scales[item["name"]] = float(item["value"])
        except ValueError:
            pass
    return scales


def reward_file_index(repo_root: Path) -> dict:
    files = []
    for item in TWEAKABLE_FILES:
        path = repo_root / item["path"]
        files.append({**item, "absolute_path": str(path), "exists": path.exists()})
    return {
        "files": files,
        "reward_scales": scan_reward_scales(repo_root),
        "reward_defaults": reward_defaults(repo_root),
        "mode": "read-only",
    }
