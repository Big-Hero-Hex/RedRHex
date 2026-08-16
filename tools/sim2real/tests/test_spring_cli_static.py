from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.sim2real.cli import build_parser


ROOT = Path(__file__).parents[3]
RSL_ENTRYPOINTS = (
    "scripts/rsl_rl/train.py",
    "scripts/rsl_rl/play.py",
    "scripts/rsl_rl/eval_command_sweep.py",
)


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _spring_backend_argument(relative: str) -> ast.Call:
    tree = ast.parse(_source(relative))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(
            isinstance(argument, ast.Constant)
            and argument.value == "--spring-backend"
            for argument in node.args
        )
    ]
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize(
    ("relative", "expected_default"),
    (
        ("scripts/rsl_rl/train.py", "native"),
        ("scripts/rsl_rl/play.py", "explicit"),
        # Evaluation must use the same default dynamics backend as training.
        # Legacy playback retains its explicit default for checkpoint replay.
        ("scripts/rsl_rl/eval_command_sweep.py", "native"),
    ),
)
def test_rsl_entrypoints_expose_provenance_aware_spring_backend_defaults(
    relative: str, expected_default: str
) -> None:
    call = _spring_backend_argument(relative)
    keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}

    assert keywords["choices"] == ("explicit", "native")
    assert keywords["default"] == expected_default


@pytest.mark.parametrize("relative", RSL_ENTRYPOINTS)
def test_rsl_entrypoints_apply_spring_backend_before_environment_creation(relative: str) -> None:
    source = _source(relative)

    assignment = "env_cfg.spring_backend = args_cli.spring_backend"
    assert source.index(assignment) < source.index("gym.make(")


@pytest.mark.parametrize("relative", ("scripts/rsl_rl/train.py", "scripts/rsl_rl/play.py"))
def test_profile_application_observes_cli_selected_spring_backend(relative: str) -> None:
    source = _source(relative)

    assignment = source.index("env_cfg.spring_backend = args_cli.spring_backend")
    profile_application = source.index("apply_profile_to_config(")
    calibration_status = source.index("spring_calibration_status =")

    assert assignment < profile_application < calibration_status


def test_training_writes_spring_backend_and_calibration_status_metadata() -> None:
    source = _source("scripts/rsl_rl/train.py")

    assert '"spring_backend": args_cli.spring_backend' in source
    assert '"calibration_status": spring_calibration_status' in source
    assert '"profile_id": spring_profile_id' in source
    assert '"profile_sha256": spring_profile_sha256' in source
    assert (
        '"resume_checkpoint_calibration_status": resume_checkpoint_calibration_status'
        in source
    )
    assert '"joint_aliases": [f"damper_{index}"' in source
    assert '"joint_names": list(env_cfg.damper_joint_names)' in source
    assert '"stiffness_nm_per_rad": list(env_cfg.spring_stiffness_nm_per_rad)' in source
    assert '"damping_nm_s_per_rad": list(env_cfg.spring_damping_nm_s_per_rad)' in source
    assert '"neutral_angle_rad": [' in source
    assert '"torsion_spring.yaml"' in source


def test_training_validates_calibrated_resume_binding_before_creation() -> None:
    source = _source("scripts/rsl_rl/train.py")

    resume_resolution = source.index("resume_path =")
    profile_hash = source.index("sha256_json(physics_profile.to_dict())")
    validation = source.index("validate_checkpoint_spring_evaluation(")
    environment_creation = source.index("gym.make(")

    assert resume_resolution < profile_hash < validation < environment_creation
    assert "selected_backend=args_cli.spring_backend" in source
    assert "selected_profile_id=spring_profile_id" in source
    assert "selected_profile_sha256=spring_profile_sha256" in source
    assert "calibrated checkpoint profile did not produce a calibrated spring configuration" in source


def test_playback_reports_spring_backend_and_calibration_status() -> None:
    source = _source("scripts/rsl_rl/play.py")

    assert "spring_calibration_status" in source
    assert "checkpoint_spring_calibration_status" in source
    assert "profile_id={spring_profile_id}" in source
    assert "profile_sha256={spring_profile_sha256}" in source
    assert "Torsion spring backend=" in source


def test_playback_validates_calibrated_checkpoint_binding_before_creation() -> None:
    source = _source("scripts/rsl_rl/play.py")

    profile_hash = source.index("sha256_json(physics_profile.to_dict())")
    validation = source.index("validate_checkpoint_spring_evaluation(")
    environment_creation = source.index("gym.make(")

    assert profile_hash < validation < environment_creation
    assert "selected_backend=args_cli.spring_backend" in source
    assert "selected_profile_id=spring_profile_id" in source
    assert "selected_profile_sha256=spring_profile_sha256" in source
    assert "calibrated checkpoint profile did not produce a calibrated spring configuration" in source


def test_command_sweep_records_spring_backend_and_calibration_status_in_summary() -> None:
    source = _source("scripts/rsl_rl/eval_command_sweep.py")

    assert '{"metric": "spring.backend", "value": args_cli.spring_backend}' in source
    assert '{"metric": "spring.calibration_status", "value": spring_calibration_status}' in source
    assert '"metric": "spring.checkpoint_calibration_status"' in source
    assert '"value": checkpoint_spring_calibration_status' in source
    assert '{"metric": "spring.profile_id", "value": spring_profile_id}' in source
    assert '{"metric": "spring.profile_sha256", "value": spring_profile_sha256}' in source


def test_command_sweep_spring_energy_uses_continuous_position_and_effective_rest() -> None:
    source = _source("scripts/rsl_rl/eval_command_sweep.py")

    assert 'getattr(unwrapped_env, "_spring_unwrapped_pos"' in source
    assert 'unwrapped_env, "_spring_rest_pos", unwrapped_env._damper_initial_pos' in source
    assert "damp_pos - spring_rest" in source


def test_command_sweep_applies_only_an_explicit_profile_before_and_after_creation() -> None:
    source = _source("scripts/rsl_rl/eval_command_sweep.py")
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(
            isinstance(argument, ast.Constant)
            and argument.value == "--physics-profile"
            for argument in node.args
        )
    )
    default = next(keyword.value for keyword in call.keywords if keyword.arg == "default")

    backend_assignment = source.index("env_cfg.spring_backend = args_cli.spring_backend")
    config_application = source.index("apply_profile_to_config(")
    environment_creation = source.index("gym.make(")
    runtime_application = source.index("apply_profile_to_runtime_env(")

    assert isinstance(default, ast.Constant) and default.value is None
    assert backend_assignment < config_application < environment_creation < runtime_application
    assert "sha256_json(physics_profile.to_dict())" in source


def test_command_sweep_validates_calibrated_checkpoint_binding_before_creation() -> None:
    source = _source("scripts/rsl_rl/eval_command_sweep.py")

    validation = source.index("validate_checkpoint_spring_evaluation(")
    environment_creation = source.index("gym.make(")

    assert validation < environment_creation
    assert "selected_backend=args_cli.spring_backend" in source
    assert "selected_profile_id=spring_profile_id" in source
    assert "selected_profile_sha256=spring_profile_sha256" in source
    assert "calibrated checkpoint profile did not produce a calibrated spring configuration" in source


def test_policy_training_quarantines_explicit_backend_but_characterization_retains_it() -> None:
    train_source = _source("scripts/rsl_rl/train.py")
    runner_source = _source("tools/sim2real/isaac_runner.py")

    assert 'if args_cli.spring_backend == "explicit":' in train_source
    assert "Explicit torsion-spring policy training is quarantined" in train_source
    pipeline_source = _source("scripts/rsl_rl/train_sensor_v2_pipeline.py")
    assert 'if args.spring_backend == "explicit":' in pipeline_source
    assert 'spring_backend not in {"explicit", "native"}' in runner_source


@pytest.mark.parametrize("command", ("run-sim", "sweep"))
def test_characterization_commands_expose_bounded_explicit_default_spring_backend(command: str) -> None:
    required = (
        ["--scenario", "audit", "--mode", "free-root", "--output", "/tmp/out"]
        if command == "run-sim"
        else [
            "profile.json",
            "--scenario",
            "main-step",
            "--mode",
            "one-factor",
            "--space-json",
            '{"simulation_physics.main_drive.damping":[0.3]}',
            "--output",
            "/tmp/out",
            "--generate-only",
        ]
    )

    defaults = build_parser().parse_args([command, *required])
    native = build_parser().parse_args([command, *required, "--spring-backend", "native"])

    assert defaults.spring_backend == "explicit"
    assert native.spring_backend == "native"
    with pytest.raises(SystemExit):
        build_parser().parse_args([command, *required, "--spring-backend", "invalid"])


def test_run_sim_exposes_supported_spring_characterization_frequencies() -> None:
    required = [
        "run-sim",
        "--scenario",
        "audit",
        "--mode",
        "free-root",
        "--output",
        "/tmp/out",
    ]

    defaults = build_parser().parse_args(required)
    high_rate = build_parser().parse_args([*required, "--physics-hz", "240"])

    assert defaults.physics_hz == 120
    assert high_rate.physics_hz == 240
    with pytest.raises(SystemExit):
        build_parser().parse_args([*required, "--physics-hz", "60"])
