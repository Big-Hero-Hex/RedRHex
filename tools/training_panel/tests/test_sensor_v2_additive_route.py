from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / "source" / "RedRhex" / "RedRhex" / "tasks" / "direct" / "redrhex"


def _assignments(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    values: dict[str, object] = {}
    for item in node.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            try:
                values[item.targets[0].id] = ast.literal_eval(item.value)
            except (ValueError, TypeError):
                pass
    return values


def test_v1_dimensions_and_history_are_unchanged() -> None:
    values = _assignments(TASK_DIR / "redrhex_env_cfg.py", "RedrhexEnvCfg")
    assert values["action_space"] == 12
    assert values["observation_space"] == 56
    assert values["policy_history_length"] == 5
    assert values["critic_privileged_observation_space"] == 47


def test_sensor_v2_registration_is_additive() -> None:
    source = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
    assert 'id="Template-Redrhex-Direct-v0"' in source
    assert 'id="Template-Redrhex-ForwardFast-Direct-v0"' in source
    assert 'id="Template-Redrhex-ForwardSensorV2-Direct-v0"' in source
    assert "redrhex_sensor_v2_env:RedrhexForwardSensorV2Env" in source


def test_sensor_v2_contract_shape_and_reward_boundary() -> None:
    values = _assignments(
        TASK_DIR / "redrhex_sensor_v2_env_cfg.py", "RedrhexForwardSensorV2EnvCfg"
    )
    assert values["sensor_contract_id"] == "redrhex.student-observation.v2"
    assert values["action_contract_id"] == "redrhex.forward-residual-action.v2"
    assert values["sensor_frame_dim"] == 36
    assert values["sensor_history_length"] == 60
    assert values["sensor_sample_hz"] == 60
    assert values["command_dim"] == 3
    assert values["state_space"] == 65
    assert values["strict_forward_residual_actions"] is True
    assert values["sensor_history_order"] == "oldest_to_newest"
    scales = values["v2_reward_scales"]
    assert scales["forward_prior_coherence"] == 0.0
    assert scales["forward_prior_antiphase"] == 0.0
    assert scales["forward_prior_duty"] == 0.0


def test_sensor_v2_actor_groups_exclude_prohibited_features() -> None:
    tree = ast.parse(
        (TASK_DIR / "redrhex_sensor_v2_env.py").read_text(encoding="utf-8")
    )
    method = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "_get_observations"
    )
    source = ast.unparse(method)
    assert "sensor_history_v2" in source
    assert "command_v2" in source
    # Truth is allowed in named training targets, but the policy compatibility
    # tensor must consist solely of history and the separate current command.
    policy_assign = next(
        item for item in method.body if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "policy_compat" for target in item.targets
        )
    )
    policy_source = ast.unparse(policy_assign.value)
    assert "history" in policy_source
    assert "command" in policy_source
    assert "base_lin_vel" not in policy_source
    assert "gait_phase" not in policy_source
    assert "last_actions" not in policy_source
