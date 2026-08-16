from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "rsl_rl" / "eval_command_sweep.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_v2_evaluation_preserves_training_decoder_semantics() -> None:
    source = _source()
    assert "if protocol.v2:" in source
    assert "unwrapped_env.cfg.play_forward_compat_enable = False" in source
    assert 'default="native"' in source


def test_acceptance_uses_contiguous_per_environment_windows() -> None:
    source = _source()
    tree = ast.parse(source)
    assert "cmd_current_contiguous_success" in source
    assert "cmd_max_contiguous_success" in source
    assert "contiguous_success_env_ratio" in source
    assert "accept_contiguous_env_ratio" in source
    # The historical aggregate duration is retained only as a diagnostic and
    # cannot be the command pass predicate.
    accepts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "accept_pass" for target in node.targets)
    ]
    assert accepts
    assert "contiguous_success_env_ratio" in ast.unparse(accepts[0].value)


def test_forward_contiguous_gate_averages_one_contract_scaled_gait_cycle() -> None:
    source = _source()
    assert "command_scaled_cycle_steps" in source
    assert "forward_rolling_speed_sum" in source
    assert "forward_rolling_lateral_sum" in source
    assert "forward_rolling_yaw_sum" in source
    assert "one_command_scaled_gait_cycle_velocity_means_with_" in source
    assert "pointwise_tilt_height_and_episode_boundary_safety" in source
    assert '"gait_cycle_window_steps"' in source
    assert '"gait_cycle_window_duration_s"' in source


def test_stability_is_reference_relative_and_failures_exit_nonzero() -> None:
    source = _source()
    assert "reference_relative_gravity" in source
    assert "reference_projected_gravity" in source
    assert "accept_forward_tilt_bound" in source
    assert "accept_forward_min_base_height" in source
    assert "if not overall_accept_pass:" in source
    assert "raise SystemExit(2)" in source
    assert "raise SystemExit(_exit_code)" in source


def test_v2_acceptance_exposes_split_action_saturation() -> None:
    source = _source()
    for metric in (
        "policy.main_action_saturation_ratio",
        "policy.abad_action_saturation_ratio",
        "policy.abad_action_magnitude_mean",
    ):
        assert metric in source
    assert "abad_action_magnitude_mean <= 1.0e-6" in source


def test_v2_summary_binds_canonical_training_config_and_seed() -> None:
    source = _source()
    assert '"checkpoint.canonical_config_sha256"' in source
    assert "checkpoint_manifest.canonical_config_hash" in source
    assert '"checkpoint.training_seed"' in source
    assert "checkpoint_manifest.training_seed" in source


def test_summary_records_every_threshold_used_by_command_acceptance() -> None:
    source = _source()
    for metric in (
        "acceptance.forward_lateral_leak",
        "acceptance.forward_yaw_leak",
        "acceptance.forward_tilt_bound_rad",
        "acceptance.forward_min_base_height_m",
        "acceptance.lateral_vy_abs",
        "acceptance.lateral_forward_leak",
        "acceptance.lateral_yaw_leak",
        "acceptance.diag_component_ratio",
        "acceptance.diag_yaw_leak",
        "acceptance.yaw_wz_abs",
        "acceptance.yaw_wz_ratio",
        "acceptance.yaw_tilt_bound_rad",
        "acceptance.yaw_linear_leak",
        "acceptance.yaw_min_base_height_m",
    ):
        assert f'"{metric}"' in source
