from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).parents[3]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_run_sim_parser_is_available_without_importing_isaac() -> None:
    before = {name for name in sys.modules if name == "isaaclab" or name.startswith("isaaclab.")}
    from tools.sim2real.cli import build_parser

    args = build_parser().parse_args(
        [
            "run-sim",
            "--scenario",
            "audit",
            "--mode",
            "free-root",
            "--steps",
            "240",
            "--output",
            "/tmp/redrhex-audit",
            "--headless",
            "--device",
            "cuda:0",
        ]
    )

    after = {name for name in sys.modules if name == "isaaclab" or name.startswith("isaaclab.")}
    assert after == before
    assert args.command == "run-sim"
    assert args.steps == 240
    assert args.physics_profile is None
    assert args.replay_trace is None

    defaults = build_parser().parse_args(
        ["run-sim", "--scenario", "main-step", "--mode", "fixed-base", "--output", "/tmp/out"]
    )
    assert defaults.steps is None

    replay = build_parser().parse_args(
        [
            "run-sim",
            "--scenario",
            "main-step",
            "--mode",
            "fixed-base",
            "--output",
            "/tmp/out",
            "--replay-trace",
            "/tmp/real-episode",
        ]
    )
    assert replay.replay_trace == Path("/tmp/real-episode")


def test_isaac_bootstrap_launches_app_before_importing_runner() -> None:
    source = _source("tools/sim2real/isaac_main.py")
    assert source.index("AppLauncher(") < source.index("from .isaac_runner import")
    assert source.index("json.dumps(") < source.index("simulation_app.close()")


def test_characterization_runner_is_finite_one_env_and_direct_targeted() -> None:
    source = _source("tools/sim2real/isaac_runner.py")
    assert "InteractiveScene" in source
    assert "num_envs=1" in source
    assert "for step_index in range(request.steps)" in source
    assert "set_joint_velocity_target" in source
    assert "set_joint_position_target" in source
    assert "write_joint_stiffness_to_sim" in source
    assert "write_joint_damping_to_sim" in source
    assert "scene.write_data_to_sim()" in source
    assert "scene.update(physics_dt)" in source
    assert "contact_sensor.data.net_forces_w" in source
    assert "foot_contact_sensor" in source
    assert "body_contact_sensor" in source
    assert "left_feet_[1-3]" in source
    assert "right_feet_[1-3]" in source
    assert "base_link|top_bottom_connector_" in source
    assert "validate_foot_contact_probe" in source
    assert "controlled pull" in _source("tools/sim2real/characterization.py")
    assert "validate_scenario_mode" in source
    assert "root_physx_view.get_masses()" in source
    assert "root_physx_view.get_inertias()" in source
    assert "root_physx_view.get_coms()" in source
    assert "UsdGeom.Gprim" in source
    assert "Usd.TraverseInstanceProxies()" in source
    assert "joint_effort_estimate" in source
    assert '"Isaac Lab implicit-PD estimate' in source
    assert "sim.clear_all_callbacks()" in source
    assert "sim.clear_instance()" in source


def test_characterization_trace_binds_verified_runtime_provenance() -> None:
    source = _source("tools/sim2real/isaac_runner.py")

    assert "runtime_provenance = production_runtime_provenance()" in source
    for field in (
        "git_sha",
        "asset_sha256",
        "config_sha256",
        "characterization_runner_sha256",
        "runtime_bundle_sha256",
    ):
        assert f'"{field}": runtime_provenance["{field}"]' in source
    assert "def _git_sha(" not in source


def test_production_implicit_actuators_use_effective_sim_limit_fields() -> None:
    tree = ast.parse(_source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"))
    implicit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ImplicitActuatorCfg"
    ]
    assert len(implicit_calls) == 3
    for call in implicit_calls:
        keywords = {item.arg for item in call.keywords}
        assert "effort_limit_sim" in keywords
        assert "velocity_limit_sim" in keywords
        assert "effort_limit" not in keywords
        assert "velocity_limit" not in keywords


def test_train_and_play_expose_opt_in_profile_and_apply_before_creation() -> None:
    for relative in ("scripts/rsl_rl/train.py", "scripts/rsl_rl/play.py"):
        source = _source(relative)
        assert '"--physics-profile"' in source
        assert "default=None" in source
        assert source.index("apply_profile_to_config(") < source.index("gym.make(")
        assert source.index("gym.make(") < source.index("apply_profile_to_runtime_env(")


def test_train_and_play_import_profile_helpers_only_inside_explicit_guard() -> None:
    for relative in ("scripts/rsl_rl/train.py", "scripts/rsl_rl/play.py"):
        tree = ast.parse(_source(relative))
        guarded_modules: set[str] = set()
        unguarded_modules: set[str] = set()

        def visit(node: ast.AST, guarded: bool = False) -> None:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "args_cli.physics_profile is not None":
                for child in node.body:
                    visit(child, True)
                for child in node.orelse:
                    visit(child, guarded)
                return
            if isinstance(node, ast.ImportFrom) and node.module in {
                "tools.sim2real.physics_profile",
                "tools.sim2real.isaac_profile",
            }:
                (guarded_modules if guarded else unguarded_modules).add(node.module)
            for child in ast.iter_child_nodes(node):
                visit(child, guarded)

        visit(tree)
        assert unguarded_modules == set()
        assert guarded_modules == {
            "tools.sim2real.physics_profile",
            "tools.sim2real.isaac_profile",
        }
        assert "sys.path.insert(0, str(repo_root))" in _source(relative)


def test_training_and_playback_do_not_load_a_default_profile() -> None:
    for relative in ("scripts/rsl_rl/train.py", "scripts/rsl_rl/play.py"):
        tree = ast.parse(_source(relative))
        physics_profile_args = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "--physics-profile":
                    physics_profile_args.append(node)
        assert len(physics_profile_args) == 1
        defaults = {
            keyword.arg: keyword.value
            for keyword in physics_profile_args[0].keywords
            if keyword.arg is not None
        }
        assert isinstance(defaults.get("default"), ast.Constant)
        assert defaults["default"].value is None


def test_training_environment_delays_final_actuator_targets_only_when_configured() -> None:
    cfg_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py")
    env_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py")

    assert "sim2real_command_delay_steps = 0" in cfg_source
    assert "_apply_sim2real_command_delay" in env_source
    assert "_requested_target_drive_vel" in env_source
    assert "_requested_target_abad_pos" in env_source
    assert env_source.count("self._apply_sim2real_command_delay(") >= 2


def test_train_play_and_characterization_apply_measured_abad_target_mapping_at_boundary() -> None:
    cfg_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py")
    env_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py")
    runner_source = _source("tools/sim2real/isaac_runner.py")

    assert "sim2real_abad_target_scale" in cfg_source
    assert "sim2real_abad_target_offset_rad" in cfg_source
    delay_call = env_source.index("advance_target_delay(")
    mapping_call = env_source.index("mapped_abad = map_abad_targets(", delay_call)
    assert delay_call < mapping_call
    assert "lower=(self._abad_rest_pos - self._abad_pos_limit_rad)" in env_source
    assert "upper=(self._abad_rest_pos + self._abad_pos_limit_rad)" in env_source
    assert "apply_abad_target_mapping(" in runner_source
    assert "minimum_rad=abad_rest_rad - abad_limit_rad" in runner_source
    assert "maximum_rad=abad_rest_rad + abad_limit_rad" in runner_source
    assert "measurement_annotations(requested_schedule" in runner_source


def test_contact_holdout_reuses_native_runner_channels_and_adds_annotations() -> None:
    source = _source("tools/sim2real/isaac_runner.py")

    assert 'elif channel not in traces:' in source
    assert 'time_bases.get(channel) != scenario.time_bases[channel]' in source
    assert '"repeat_index": repeat_index' in source
    assert '"settled": settled' in source


def test_replay_initial_state_is_applied_before_stepping_and_audited() -> None:
    source = _source("tools/sim2real/isaac_runner.py")

    apply_state = source.index(
        "initial_joint_position[:, main_joint_indices] = replay_position"
    )
    apply_velocity = source.index(
        "initial_joint_velocity[:, main_joint_indices] = replay_velocity"
    )
    write_state = source.index("robot.write_joint_state_to_sim(")
    apply_root_orientation = source.index(
        "replay.initial_state.root_orientation_wxyz"
    )
    write_root_pose = source.index("robot.write_root_pose_to_sim(")
    physics_loop = source.index("for step_index in range(request.steps)")

    assert "main_joint_indices = _resolve_main_joint_indices(env_cfg, robot)" in source
    assert apply_state < apply_velocity < write_state < physics_loop
    assert apply_root_orientation < write_root_pose < physics_loop
    assert '"replay_initial_state": replay_initial_state' in source
    assert '"replay_initial_state_sha256": replay_initial_state_sha256' in source
