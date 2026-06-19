from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore
from tools.training_panel.training_panel.processes import ProcessRegistry


def build_panel_registry(repo_root: Path) -> ProcessRegistry:
    paths = replace(PanelPaths.from_env(), repo_root=Path(repo_root).resolve())
    history = HistoryStore(paths)
    return ProcessRegistry(paths, history)
