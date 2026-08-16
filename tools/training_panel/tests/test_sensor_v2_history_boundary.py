from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / "source" / "RedRhex" / "RedRhex" / "tasks" / "direct" / "redrhex"


def test_sensor_v2_history_uses_only_real_samples_and_is_copied_on_handoff() -> None:
    config_source = (TASK_DIR / "redrhex_sensor_v2_env_cfg.py").read_text(encoding="utf-8")
    assert 'sensor_history_prefill = "real_samples_only"' in config_source

    source = (TASK_DIR / "redrhex_sensor_v2_env.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    append_method = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "_append_sensor_history_v2"
    )
    append_source = ast.unparse(append_method)
    assert "expand" not in append_source
    assert "repeat" not in append_source

    observation_method = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "_get_observations"
    )
    assert "self._sensor_history_v2.clone()" in ast.unparse(observation_method)


def test_sensor_v2_uses_shared_frame_composer_and_fails_on_invalid_gyro() -> None:
    source = (TASK_DIR / "redrhex_sensor_v2_env.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "_build_sensor_frame_v2"
    )
    method_source = ast.unparse(method)
    assert "build_sensor_frame_torch" in method_source
    assert "abad_neutral_position_rad=self._abad_rest_pos.squeeze(0)" in method_source
    assert "nan_to_num" not in method_source
    assert "contains NaN or Inf" in method_source


def test_student_residual_is_zero_until_full_history_is_ready() -> None:
    config_source = (TASK_DIR / "redrhex_sensor_v2_env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "sensor_history_action_gate = True" in config_source
    tree = ast.parse((TASK_DIR / "redrhex_sensor_v2_env.py").read_text(encoding="utf-8"))
    method = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "_pre_physics_step"
    )
    source = ast.unparse(method)
    assert "sensor_history_action_gate" in source
    assert "_sensor_history_ready_v2" in source
    assert "torch.zeros_like(strict_actions)" in source


def test_privileged_teacher_disables_student_history_action_gate() -> None:
    backend = (
        TASK_DIR / "agents" / "sensor_v2" / "backends.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(backend)
    teacher = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "VersionedTeacherBackendV2"
    )
    init = next(
        item
        for item in teacher.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    source = ast.unparse(init)
    assert "env.cfg.sensor_history_action_gate = False" in source
    assert source.index("sensor_history_action_gate = False") < source.index(
        "super().__init__"
    )


def test_encoder_stale_or_dropout_invalidates_history_and_velocity_baseline() -> None:
    tree = ast.parse((TASK_DIR / "redrhex_sensor_v2_env.py").read_text(encoding="utf-8"))
    method = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "_build_sensor_frame_v2"
    )
    source = ast.unparse(method)

    assert "invalid_sample = new_sample & (event.encoder_dropout | event.encoder_stale)" in source
    assert "self._sensor_history_valid_count_v2[invalid_ids] = 0" in source
    assert "self._sensor_history_ready_v2[invalid_ids] = False" in source
    assert "self._sensor_encoder_initialized_v2[invalid_ids] = False" in source
    assert "self._sensor_prev_main_pos_v2[invalid_ids] = 0.0" in source
    assert "self._sensor_prev_abad_pos_v2[invalid_ids] = 0.0" in source
    assert "self._sensor_prev_encoder_sample_step_v2[invalid_ids] = -1" in source


def test_physical_f4_f5_profiles_forbid_controller_target_fallback() -> None:
    tree = ast.parse((TASK_DIR / "redrhex_sensor_v2_env.py").read_text(encoding="utf-8"))
    method = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "_reset_idx"
    )
    source = ast.unparse(method)

    assert "sensor_dr_require_physical_material_writes" in source
    assert "_mass_physical_randomized" in source
    assert "_friction_physical_randomized" in source
    assert "controller-target fallback is forbidden" in source
