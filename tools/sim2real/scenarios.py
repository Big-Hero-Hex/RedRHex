from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ContractError, ScenarioSpecV1


SCENARIO_DIR = Path(__file__).with_name("scenario_specs")


def _scenario_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.suffix == ".json" or candidate.parent != Path("."):
        return candidate
    return SCENARIO_DIR / f"{candidate}.json"


def load_scenario(value: str | Path) -> ScenarioSpecV1:
    path = _scenario_path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read scenario {path}: {exc}") from exc
    scenario = ScenarioSpecV1.from_dict(payload)
    if path.parent == SCENARIO_DIR and path.stem != scenario.scenario_id:
        raise ContractError("scenario filename must match scenario_id")
    return scenario


def list_scenarios() -> list[dict[str, Any]]:
    result = []
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        scenario = load_scenario(path)
        result.append(
            {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "subsystem": scenario.subsystem,
                "experiment_kind": scenario.experiment_kind,
                "split": scenario.split,
                "scene_mode": scenario.scene_mode,
                "safety_class": scenario.safety_class,
            }
        )
    return result
