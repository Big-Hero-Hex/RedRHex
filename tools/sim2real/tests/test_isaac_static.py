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
    assert source.index("bind_redrhex_source(") < source.index("from isaaclab.app import")
    assert source.index("AppLauncher(") < source.index("from .isaac_runner import")
    assert source.index("json.dumps(") < source.index("simulation_app.close()")


def test_isaac_bootstrap_closes_app_on_success_and_failure() -> None:
    tree = ast.parse(_source("tools/sim2real/isaac_main.py"))
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    guarded = next(node for node in run.body if isinstance(node, ast.Try))

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
        for statement in guarded.finalbody
        for node in ast.walk(statement)
    )
    failure_handler = next(
        handler for handler in guarded.handlers if handler.type is not None
    )
    post_quit = next(
        node
        for statement in failure_handler.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "post_quit"
    )
    assert len(post_quit.args) == 1
    assert isinstance(post_quit.args[0], ast.Constant)
    assert post_quit.args[0].value != 0
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        and any(
            keyword.arg == "file"
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == "sys"
            and keyword.value.attr == "stderr"
            for keyword in node.keywords
        )
        for statement in failure_handler.body
        for node in ast.walk(statement)
    )


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
    assert "body_com_pos_w" in source
    assert ".to(robot.data.body_com_pos_w.device)" in source
    assert '"aggregate_com_body_m"' in source
    assert '"mass_profile_application"' in source
    assert '"joint_geometry"' in source
    assert "UsdPhysics.RevoluteJoint" in source
    assert '"runtime_audit_sha256"' in source
    assert "UsdGeom.Gprim" in source
    assert "Usd.TraverseInstanceProxies()" in source
    assert "joint_effort_estimate" in source
    assert '"Isaac Lab implicit-PD estimate' in source
    assert "sim.clear_all_callbacks()" in source
    assert "sim.clear_instance()" in source
    assert "spring_runtime = _configure_runner_torsion_springs(" in source
    assert "apply_profile_to_runtime_env(\n        spring_runtime, profile" in source


def test_characterization_trace_binds_verified_runtime_provenance() -> None:
    source = _source("tools/sim2real/isaac_runner.py")

    assert "runtime_provenance = production_runtime_provenance()" in source
    for field in (
        "git_sha",
        "asset_sha256",
        "config_sha256",
        "redrhex_module_path",
        "redrhex_module_sha256",
        "isaaclab_version",
        "isaacsim_version",
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
        assert source.index("bind_redrhex_source(") < source.index("import RedRhex.tasks")
        assert source.index("import RedRhex.tasks") < source.index(
            "assert_redrhex_module_source("
        )
        assert '"--physics-profile"' in source
        assert "default=None" in source
        assert source.index("apply_profile_to_config(") < source.index("gym.make(")
        assert source.index("gym.make(") < source.index("apply_profile_to_runtime_env(")


def test_training_snapshots_explicit_profile_after_runtime_application() -> None:
    source = _source("scripts/rsl_rl/train.py")

    assert source.index("apply_profile_to_runtime_env(") < source.index(
        "write_training_profile_snapshot("
    )
    assert source.index("write_training_profile_snapshot(") < source.index(
        "runner.learn("
    )


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


def test_energy_evaluation_accepts_per_joint_profiled_spring_parameters() -> None:
    source = _source("scripts/rsl_rl/eval_command_sweep.py")

    assert 'torch.as_tensor(getattr(unwrapped_env, "_spring_k"' in source
    assert 'torch.as_tensor(getattr(unwrapped_env, "_spring_d"' in source
    assert "spring_k * torch.square(damp_defl)" in source
    assert "spring_d * torch.square(damp_vel)" in source


def test_production_environment_exposes_two_passive_spring_backends() -> None:
    cfg_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py")
    env_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py")

    assert 'spring_backend = "native"' in cfg_source
    assert "spring_stiffness_nm_per_rad = (200.0,) * 6" in cfg_source
    assert "spring_damping_nm_s_per_rad = (0.0,) * 6" in cfg_source
    assert "velocity_limit_sim=1.0," not in cfg_source
    assert "def _configure_torsion_spring_backend(" in env_source
    assert "def _apply_explicit_torsion_spring(" in env_source
    assert "restoring_torque(" in env_source
    assert "self.robot.data.joint_pos[:, self._damper_indices]" in env_source
    assert "self.robot.set_joint_effort_target(" in env_source
    assert "self.robot.write_joint_stiffness_to_sim(" in env_source
    assert "self.robot.write_joint_damping_to_sim(" in env_source
    assert 'self.robot.actuators["damper"].stiffness = reorder_joint_parameters(' in env_source
    assert 'self.robot.actuators["damper"].damping = reorder_joint_parameters(' in env_source


def test_environment_spring_passivity_is_temporal_and_uses_fresh_velocity() -> None:
    env_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py")

    assert "def _advance_torsion_spring_accounting(" in env_source
    assert "trapezoidal_energy_increment(" in env_source
    assert "energy_work_residual(" in env_source
    diagnostics = env_source[env_source.index("def _compute_torsion_spring_diagnostics(") :]
    diagnostics = diagnostics[: diagnostics.index("def _compute_damper_diagnostics(")]
    assert "self.robot.data.joint_vel" in diagnostics
    assert "conservative_rate" not in diagnostics
    assert "self._spring_temporal_passivity_residual" in diagnostics


def test_damper_hold_is_replaced_by_substep_spring_effort_and_fixed_native_target() -> None:
    env_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py")

    assert "_apply_damper_hold" not in env_source
    apply_action = env_source[env_source.index("    def _apply_action(") :]
    assert apply_action.index("self._apply_explicit_torsion_spring()") < apply_action.index(
        'if getattr(self, "_action_targets_computed", False):'
    )
    assert "def _set_native_torsion_spring_target(" in env_source
    assert "rest_target = self._spring_rest_pos" in env_source


def test_environment_logs_passive_spring_state_without_changing_observation_contract() -> None:
    cfg_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py")
    env_source = _source("source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py")

    assert "action_space = 12" in cfg_source
    assert "observation_space = 56" in cfg_source
    for metric in (
        "diag_spring_deflection_mean",
        "diag_spring_torque_rms",
        "diag_spring_energy",
        "diag_spring_power",
        "diag_spring_passivity_residual",
    ):
        assert env_source.count(f'"{metric}"') >= 3


def test_characterization_runner_executes_and_stamps_selected_spring_backend() -> None:
    source = _source("tools/sim2real/isaac_runner.py")

    backend_assignment = "env_cfg.spring_backend = str(args.spring_backend)"
    assert source.index(backend_assignment) < source.index("apply_profile_to_config(")
    assert "def _configure_runner_torsion_springs(" in source
    assert "def _apply_runner_explicit_spring(" in source
    physics_loop = source.index("for step_index in range(request.steps)")
    explicit_apply = source.index("_apply_runner_explicit_spring(", physics_loop)
    write_to_sim = source.index("scene.write_data_to_sim()", physics_loop)
    assert physics_loop < explicit_apply < write_to_sim
    assert '"spring_backend": spring_backend' in source
    assert '"calibration_status": spring_calibration_status' in source
    assert '"profile_sha256": profile_sha256' in source


def test_spring_release_runner_uses_zero_gravity_and_records_passivity_channels() -> None:
    source = _source("tools/sim2real/isaac_runner.py")

    assert 'scenario.experiment_kind == "spring_release"' in source
    assert "sim_cfg.gravity = (0.0, 0.0, 0.0)" in source
    assert '"spring_deflection"' in source
    assert '"spring_model_torque"' in source
    assert '"spring_applied_torque_estimate"' in source
    assert '"spring_potential_energy"' in source
    assert '"spring_mechanical_power"' in source
    assert '"spring_passivity_residual"' in source
    assert '"spring_release_start"' in source
    assert '"spring_unwrap_ambiguous"' in source
    assert '"spring_pre_step_time_s"' in source
    assert "spring_system_energy - spring_energy_reference" in source
    assert "torch.clamp(\n            spring_system_energy - spring_energy_reference" not in source
    assert "def _author_spring_release_fixture(" in source
    assert "def _verify_spring_release_fixture(" in source
    assert "CreateLowerLimitAttr" in source
    assert "CreateUpperLimitAttr" in source
    assert "UsdPhysics.FixedJoint.Define(" in source
    assert "CreateExcludeFromArticulationAttr" in source
    assert '"locked_joint_names"' in source
    assert 'fixture_selected_range_kind = "continuous"' in source


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
