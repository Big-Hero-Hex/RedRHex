"""Parity test: ROS deploy contract vs. the IsaacLab training config.

redrhex_contract.py hand-mirrors constants from redrhex_env_cfg.py. This test
AST-parses the training config (importing it would require Isaac Sim) and
compares every mirrored constant, so drift like the old SIM_DT=1/250 (125 Hz
deploy vs 60 Hz training) fails CI instead of reaching hardware.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CFG_PATH = REPO_ROOT / "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
CONTRACT_PATH = (
    REPO_ROOT / "ros2_ws/src/redrhex_rl_controller/redrhex_rl_controller/redrhex_contract.py"
)

# contract constant -> stage-list attribute in RedrhexEnvCfg (stage 5 == index 4)
STAGE5_MAP = {
    "STAGE_DRIVE_VEL_SCALE": "stage_drive_vel_scale",
    "STAGE_MAIN_DRIVE_RESIDUAL_SCALE": "stage_main_drive_residual_scale",
    "STAGE_FORWARD_BIAS_SCALE": "stage_forward_bias_scale",
    "STAGE_YAW_DRIVE_BIAS_SCALE": "stage_yaw_drive_bias_scale",
    "STAGE_YAW_SAFE_MIN_SCALE": "stage_yaw_safe_min_scale",
    "STAGE_YAW_HARD_BRAKE_TILT": "stage_yaw_hard_brake_tilt",
    "STAGE_YAW_HARD_BRAKE_SCALE": "stage_yaw_hard_brake_scale",
    "STAGE_LATERAL_SOFT_LOCK_VELOCITY": "stage_lateral_soft_lock_velocity",
    "STAGE_LATERAL_POLICY_DRIVE_RESIDUAL_SCALE": "stage_lateral_policy_drive_residual_scale",
    "STAGE_LATERAL_ABAD_BASE_AMPLITUDE": "stage_lateral_abad_base_amplitude",
    "STAGE_LATERAL_ABAD_MAX_AMPLITUDE": "stage_lateral_abad_max_amplitude",
    "STAGE_LATERAL_ABAD_POLICY_BLEND": "stage_lateral_abad_policy_blend",
    "STAGE_DIAG_ABAD_BIAS_SCALE": "stage_diag_abad_bias_scale",
    "STAGE_DIAG_ABAD_POLICY_BLEND": "stage_diag_abad_policy_blend",
    "STAGE_YAW_ABAD_ACTION_SCALE": "stage_yaw_abad_action_scale",
    "STAGE_YAW_ABAD_STANCE_BIAS": "stage_yaw_abad_stance_bias",
    "STAGE_YAW_ABAD_POLICY_BLEND": "stage_yaw_abad_policy_blend",
    "STAGE_ABAD_POS_LIMIT": "stage_abad_pos_limit",
    "STAGE_ACTION_WARMUP_STEPS": "stage_action_warmup_steps",
    "STAGE_FORWARD_POLICY_DRIVE_RESIDUAL_SCALE": "stage_forward_policy_drive_residual_scale",
    "STAGE_DIAG_POLICY_DRIVE_RESIDUAL_SCALE": "stage_diag_policy_drive_residual_scale",
    "STAGE_YAW_POLICY_DRIVE_RESIDUAL_SCALE": "stage_yaw_policy_drive_residual_scale",
    "STAGE_FORWARD_RESIDUAL_CAP_RATIO": "stage_forward_residual_cap_ratio",
}

SCALAR_MAP = {
    "OBS_DIM_SINGLE": "observation_space",
    "ACTION_DIM": "action_space",
    "POLICY_HISTORY_LENGTH": "policy_history_length",
    "DECIMATION": "decimation",
    "BASE_GAIT_FREQUENCY_HZ": "base_gait_frequency",
    "STANCE_PHASE_START": "stance_phase_start",
    "STANCE_PHASE_END": "stance_phase_end",
    "STANCE_VELOCITY_RATIO": "stance_velocity_ratio",
    "SWING_VELOCITY_RATIO": "swing_velocity_ratio",
    "TRIPOD_PHASE_OFFSET": "tripod_phase_offset",
    "MAIN_DRIVE_VEL_SCALE": "main_drive_vel_scale",
    "ABAD_POS_SCALE": "abad_pos_scale",
}

LIST_MAP = {
    "MAIN_DRIVE_JOINT_NAMES": "main_drive_joint_names",
    "ABAD_JOINT_NAMES": "abad_joint_names",
    "DAMPER_JOINT_NAMES": "damper_joint_names",
    "TRIPOD_A_LEG_INDICES": "tripod_a_leg_indices",
    "TRIPOD_B_LEG_INDICES": "tripod_b_leg_indices",
    "LEG_DIRECTION_MULTIPLIER": "leg_direction_multiplier",
}


def _safe_eval(node: ast.AST):
    """Evaluate the constant-expression subset used by redrhex_env_cfg.py."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    if isinstance(node, ast.BinOp):
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        raise ValueError(f"unsupported operator: {ast.dump(node)}")
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math":
        return getattr(math, node.attr)
    if isinstance(node, ast.Call):
        func = _safe_eval(node.func)
        return func(*[_safe_eval(arg) for arg in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval(item) for item in node.elts]
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def _load_cfg_class_attr_nodes(tree: ast.Module, class_name: str) -> dict[str, ast.AST]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            attrs: dict[str, ast.AST] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    attrs[stmt.targets[0].id] = stmt.value
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    attrs[stmt.target.id] = stmt.value
            return attrs
    raise AssertionError(f"class {class_name} not found in {CFG_PATH}")


def _keyword_from_call(call_node: ast.AST, keyword: str) -> ast.AST | None:
    for node in ast.walk(call_node):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == keyword:
                    return kw.value
    return None


def _module_assign_value(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise AssertionError(f"module-level assignment {name} not found in {CFG_PATH}")


class TestContractParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("redrhex_contract_under_test", CONTRACT_PATH)
        cls.contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.contract)

        tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
        cls.cfg_attrs = _load_cfg_class_attr_nodes(tree, "RedrhexEnvCfg")
        cls.cfg_tree = tree

    def _cfg_value(self, attr: str):
        self.assertIn(attr, self.cfg_attrs, f"RedrhexEnvCfg.{attr} missing")
        return _safe_eval(self.cfg_attrs[attr])

    def test_control_rate_matches_training(self):
        sim_dt_node = _keyword_from_call(self.cfg_attrs["sim"], "dt")
        self.assertIsNotNone(sim_dt_node, "SimulationCfg(dt=...) not found")
        sim_dt = _safe_eval(sim_dt_node)
        decimation = self._cfg_value("decimation")
        self.assertAlmostEqual(self.contract.SIM_DT, sim_dt, places=9)
        self.assertEqual(self.contract.DECIMATION, decimation)
        self.assertAlmostEqual(self.contract.CONTROL_DT, sim_dt * decimation, places=9)
        self.assertAlmostEqual(self.contract.POLICY_HZ, 1.0 / (sim_dt * decimation), places=6)

    def test_scalar_constants_match(self):
        for contract_name, cfg_name in SCALAR_MAP.items():
            with self.subTest(constant=contract_name):
                self.assertAlmostEqual(
                    float(getattr(self.contract, contract_name)),
                    float(self._cfg_value(cfg_name)),
                    places=9,
                    msg=f"{contract_name} != RedrhexEnvCfg.{cfg_name}",
                )

    def test_list_constants_match(self):
        for contract_name, cfg_name in LIST_MAP.items():
            with self.subTest(constant=contract_name):
                self.assertEqual(
                    list(getattr(self.contract, contract_name)),
                    list(self._cfg_value(cfg_name)),
                    f"{contract_name} != RedrhexEnvCfg.{cfg_name}",
                )

    def test_stage5_constants_match(self):
        for contract_name, cfg_name in STAGE5_MAP.items():
            with self.subTest(constant=contract_name):
                stage_list = self._cfg_value(cfg_name)
                self.assertIsInstance(stage_list, list, f"RedrhexEnvCfg.{cfg_name} is not a list")
                expected = stage_list[min(4, len(stage_list) - 1)]
                self.assertAlmostEqual(
                    float(getattr(self.contract, contract_name)),
                    float(expected),
                    places=9,
                    msg=f"{contract_name} != RedrhexEnvCfg.{cfg_name}[stage5]",
                )

    def test_init_joint_pose_matches(self):
        redrhex_cfg_node = _module_assign_value(self.cfg_tree, "REDRHEX_CFG")
        joint_pos_node = _keyword_from_call(redrhex_cfg_node, "joint_pos")
        self.assertIsNotNone(joint_pos_node, "ArticulationCfg init_state joint_pos dict not found")
        self.assertIsInstance(joint_pos_node, ast.Dict)
        joint_pos = {
            _safe_eval(key): _safe_eval(value)
            for key, value in zip(joint_pos_node.keys, joint_pos_node.values)
        }
        for names_attr, contract_attr in (
            ("MAIN_DRIVE_JOINT_NAMES", "INIT_MAIN_DRIVE_POS"),
            ("ABAD_JOINT_NAMES", "INIT_ABAD_POS"),
            ("DAMPER_JOINT_NAMES", "INIT_DAMPER_POS"),
        ):
            names = list(getattr(self.contract, names_attr))
            expected = [joint_pos[name] for name in names]
            actual = list(getattr(self.contract, contract_attr))
            for name, exp, act in zip(names, expected, actual):
                self.assertAlmostEqual(act, exp, places=9, msg=f"{contract_attr}[{name}]")

    def test_derived_gait_velocities(self):
        base_vel = 2.0 * math.pi * float(self._cfg_value("base_gait_frequency"))
        self.assertAlmostEqual(
            self.contract.STANCE_VELOCITY, base_vel * float(self._cfg_value("stance_velocity_ratio")), places=9
        )
        self.assertAlmostEqual(
            self.contract.SWING_VELOCITY, base_vel * float(self._cfg_value("swing_velocity_ratio")), places=9
        )


if __name__ == "__main__":
    unittest.main()
