from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .contracts import ContractError
from .traces import sha256_file


_CONFIG_MODULE = "RedRhex.tasks.direct.redrhex.redrhex_env_cfg"
_CONFIG_RELATIVE = Path(
    "RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def redrhex_source_root(repo_root: str | Path | None = None) -> Path:
    root = repository_root() if repo_root is None else Path(repo_root).resolve()
    source = (root / "source" / "RedRhex").resolve()
    if not source.is_dir():
        raise ContractError(f"selected repository has no RedRhex source: {source}")
    return source


def _require_inside(path: Path, source: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(source)
    except ValueError as exc:
        raise ContractError(
            f"{label} resolves outside the selected repository RedRhex source: {resolved}"
        ) from exc
    return resolved


def _module_path(module: ModuleType, source: Path) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise ContractError(f"loaded module {module.__name__} has no source file")
    return _require_inside(Path(value), source, f"loaded module {module.__name__}")


def bind_redrhex_source(repo_root: str | Path | None = None) -> Path:
    """Prepend this repository's extension source and fail on a loaded decoy."""

    source = redrhex_source_root(repo_root)
    for name, module in tuple(sys.modules.items()):
        if name != "RedRhex" and not name.startswith("RedRhex."):
            continue
        if isinstance(module, ModuleType):
            _module_path(module, source)

    source_text = str(source)
    sys.path[:] = [
        source_text,
        *(
            entry
            for entry in sys.path
            if not entry or Path(entry).resolve() != source
        ),
    ]
    spec = importlib.util.find_spec("RedRhex")
    origin = getattr(spec, "origin", None)
    if spec is None or not isinstance(origin, str):
        raise ContractError(f"cannot resolve RedRhex from selected source: {source}")
    _require_inside(Path(origin), source, "RedRhex package")
    return source


def assert_redrhex_module_source(
    module: ModuleType,
    repo_root: str | Path | None = None,
) -> Path:
    source = redrhex_source_root(repo_root)
    return _module_path(module, source)


def redrhex_config_identity(
    repo_root: str | Path | None = None,
) -> dict[str, str]:
    source = redrhex_source_root(repo_root)
    loaded = sys.modules.get(_CONFIG_MODULE)
    if isinstance(loaded, ModuleType):
        path = _module_path(loaded, source)
    else:
        path = _require_inside(source / _CONFIG_RELATIVE, source, "RedRhex config")
    if not path.is_file():
        raise ContractError(f"RedRhex config module does not exist: {path}")
    return {
        "redrhex_module_path": str(path),
        "redrhex_module_sha256": sha256_file(path),
    }
