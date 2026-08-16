from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
SENSOR = ROOT / "source/RedRhex/RedRhex/tasks/direct/redrhex/agents/sensor_v2"


def _class_assignments(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    values: dict[str, object] = {}
    for item in node.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1:
            target = item.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(item.value)
                except (TypeError, ValueError):
                    pass
    return values


def test_f4_runner_is_additive_and_uses_a_distinct_checkpoint_stage() -> None:
    config = _class_assignments(SENSOR / "config.py", "ForwardSensorV2RobustPpoRunnerCfg")
    assert config["class_name"] == "SensorRobustnessRunnerV2"
    assert config["experiment_name"] == "redrhex_forward_v2_robust_ppo"

    backend = (SENSOR / "backends.py").read_text(encoding="utf-8")
    assert 'class SensorRobustnessBackendV2(SensorOnPolicyBackendV2):' in backend
    assert 'stage = "ppo_f4"' in backend
    assert "CheckpointIntentV2.ROBUSTNESS_BOOTSTRAP" in backend
    assert "manifest.stage not in {\"ppo_f3\", \"ppo_f4\"}" in backend
    assert "architecture_hash_v2(self.policy)" in backend
    assert 'self.cfg["environment_provenance_v2"] = environment_provenance' in backend
    assert '"sensor_dr_profile_sha256"' in backend

    registration = (
        ROOT / "source/RedRhex/RedRhex/tasks/direct/redrhex/__init__.py"
    ).read_text(encoding="utf-8")
    assert "rsl_rl_robust_ppo_v2_cfg_entry_point" in registration
    assert "ForwardSensorV2RobustPpoRunnerCfg" in registration
    script_factory = (ROOT / "scripts/rsl_rl/runner_factory.py").read_text(encoding="utf-8")
    assert '"SensorRobustnessRunnerV2": RunnerProtocol(' in script_factory
    assert script_factory.count('"student_ppo_v2"') >= 3


def test_train_cli_requires_bound_f4_profile_and_ppo_source() -> None:
    source = (ROOT / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")

    assert '"--ppo_checkpoint"' in source
    assert "runner.bootstrap_robustness_v2(resume_path)" in source
    assert 'robust_agent = "rsl_rl_robust_ppo_v2_cfg_entry_point"' in source
    assert 'parser.error("F4 --sensor-dr-profile requires --ppo_checkpoint")' in source
    assert 'parser.error("--ppo_checkpoint requires an evidence-bound --sensor-dr-profile")' in source


def test_bootstrap_flags_are_fail_closed_to_exact_runner_classes() -> None:
    path = ROOT / "scripts/rsl_rl/train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {
            "_V2_AGENT_CLASS_BY_ENTRY_POINT",
            "_V2_BOOTSTRAP_RUNNER_CLASS",
        }
    }
    assert assignments["_V2_BOOTSTRAP_RUNNER_CLASS"] == {
        "teacher_checkpoint": "SensorDistillationRunnerV2",
        "student_checkpoint": "SensorOnPolicyRunnerV2",
        "ppo_checkpoint": "SensorRobustnessRunnerV2",
    }
    assert assignments["_V2_AGENT_CLASS_BY_ENTRY_POINT"] == {
        "rsl_rl_distillation_v2_cfg_entry_point": "SensorDistillationRunnerV2",
        "rsl_rl_distillation_v2_no_aux_cfg_entry_point": "SensorDistillationRunnerV2",
        "rsl_rl_distillation_v2_velocity_cfg_entry_point": "SensorDistillationRunnerV2",
        "rsl_rl_distillation_v2_velocity_dynamics_cfg_entry_point": "SensorDistillationRunnerV2",
        "rsl_rl_ppo_v2_cfg_entry_point": "SensorOnPolicyRunnerV2",
        "rsl_rl_robust_ppo_v2_cfg_entry_point": "SensorRobustnessRunnerV2",
    }
    source = path.read_text(encoding="utf-8")
    assert "_validate_bootstrap_runner_class(agent_cfg.class_name)" in source


def test_required_v2_ablation_configs_are_registered_and_non_production() -> None:
    config_path = SENSOR / "config.py"
    registration = (
        ROOT / "source/RedRhex/RedRhex/tasks/direct/redrhex/__init__.py"
    ).read_text(encoding="utf-8")
    expected = {
        "ForwardSensorV2DistillationNoAuxRunnerCfg": (
            "v2_no_aux",
            "rsl_rl_distillation_v2_no_aux_cfg_entry_point",
        ),
        "ForwardSensorV2DistillationVelocityRunnerCfg": (
            "v2_velocity",
            "rsl_rl_distillation_v2_velocity_cfg_entry_point",
        ),
        "ForwardSensorV2DistillationVelocityDynamicsRunnerCfg": (
            "v2_velocity_dynamics",
            "rsl_rl_distillation_v2_velocity_dynamics_cfg_entry_point",
        ),
    }
    for class_name, (ablation_id, entry_point) in expected.items():
        values = _class_assignments(config_path, class_name)
        assert values["ablation_id"] == ablation_id
        assert values["run_name"] == ablation_id
        assert values["production_lineage_allowed"] is False
        assert entry_point in registration

    no_aux = _class_assignments(config_path, "SensorDistillationNoAuxAlgorithmCfgV2")
    velocity = _class_assignments(config_path, "SensorDistillationVelocityAlgorithmCfgV2")
    dynamics = _class_assignments(
        config_path, "SensorDistillationVelocityDynamicsAlgorithmCfgV2"
    )
    assert no_aux == {
        "velocity_loss_weight": 0.0,
        "dynamics_loss_weight": 0.0,
        "latent_regularization_weight": 0.0,
        "contact_loss_weight": 0.0,
    }
    assert velocity == {"velocity_loss_weight": 0.5}
    assert dynamics == {"dynamics_loss_weight": 0.1}


def test_v2_play_and_sweep_use_inference_intent_not_training_resume() -> None:
    backend = (SENSOR / "backends.py").read_text(encoding="utf-8")
    play = (ROOT / "scripts/rsl_rl/play.py").read_text(encoding="utf-8")
    sweep = (ROOT / "scripts/rsl_rl/eval_command_sweep.py").read_text(encoding="utf-8")

    assert backend.count("def load_inference_v2(") == 3
    assert "CheckpointIntentV2.INFERENCE" in backend
    assert "training-only config" in backend
    assert "runner.load_inference_v2(resume_path)" in play
    assert "runner.load_inference_v2(resume_path)" in sweep
