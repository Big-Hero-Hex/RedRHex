"""RedRhex IsaacLab deployment contract.

All constants here are copied from the local IsaacLab task in:
source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py
source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EXPECTED_OBS_DIM = 56
EXPECTED_ACTION_DIM = 12
SIM_DT = 1.0 / 250.0
DECIMATION = 2
CONTROL_DT = SIM_DT * DECIMATION
POLICY_HZ = 1.0 / CONTROL_DT

MAIN_DRIVE_JOINT_NAMES = [
    "Revolute_15",  # RF
    "Revolute_7",   # RM
    "Revolute_12",  # RR
    "Revolute_18",  # LF
    "Revolute_23",  # LM
    "Revolute_24",  # LR
]
ABAD_JOINT_NAMES = [
    "Revolute_14",  # RF
    "Revolute_6",   # RM
    "Revolute_11",  # RR
    "Revolute_17",  # LF
    "Revolute_22",  # LM
    "Revolute_21",  # LR
]
DAMPER_JOINT_NAMES = [
    "Revolute_5",
    "Revolute_8",
    "Revolute_13",
    "Revolute_25",
    "Revolute_26",
    "Revolute_27",
]
MOTOR_JOINT_NAMES = MAIN_DRIVE_JOINT_NAMES + ABAD_JOINT_NAMES
LEG_ORDER = ["RF", "RM", "RR", "LF", "LM", "LR"]

LEG_DIRECTION_MULTIPLIER = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
TRIPOD_A_LEG_INDICES = [0, 3, 5]
TRIPOD_B_LEG_INDICES = [1, 2, 4]

MAIN_DRIVE_VEL_SCALE = 8.0
MAIN_DRIVE_RESIDUAL_SCALE = 0.40
ABAD_POS_SCALE = 0.61096
BASE_GAIT_FREQUENCY_HZ = 1.0
BASE_GAIT_ANGULAR_VEL = 2.0 * math.pi * BASE_GAIT_FREQUENCY_HZ
STANCE_VELOCITY = BASE_GAIT_ANGULAR_VEL * 0.15
SWING_VELOCITY = BASE_GAIT_ANGULAR_VEL * 1.5
TRIPOD_PHASE_OFFSET = math.pi
INIT_MAIN_DRIVE_POS_RAD = [
    math.pi / 4.0,
    math.pi / 4.0,
    math.pi / 4.0,
    -math.pi / 4.0,
    -math.pi / 4.0,
    -math.pi / 4.0,
]
INIT_ABAD_POS_RAD = [0.0] * 6

COMMAND_LIMITS = {
    "vx_min": 0.0,
    "vx_max": 0.45,
    "vy_min": -0.40,
    "vy_max": 0.40,
    "wz_min": -1.00,
    "wz_max": 1.00,
}


@dataclass(frozen=True)
class ObsSlice:
    name: str
    start: int
    stop: int


OBS_SLICES = [
    ObsSlice("base_lin_vel", 0, 3),
    ObsSlice("base_ang_vel", 3, 6),
    ObsSlice("projected_gravity", 6, 9),
    ObsSlice("main_drive_pos_sin", 9, 15),
    ObsSlice("main_drive_pos_cos", 15, 21),
    ObsSlice("main_drive_vel_div_base_gait_angular_vel", 21, 27),
    ObsSlice("abad_pos_div_abad_pos_scale", 27, 33),
    ObsSlice("abad_vel", 33, 39),
    ObsSlice("velocity_command", 39, 42),
    ObsSlice("gait_phase_sin_cos", 42, 44),
    ObsSlice("last_actions", 44, 56),
]


def command_mode(vx: float, vy: float, wz: float) -> str:
    lin_zero = 0.08
    yaw_zero = 0.10
    if vx > 0.10 and abs(vy) < lin_zero and abs(wz) < yaw_zero:
        return "FWD"
    if abs(vx) < lin_zero and abs(vy) > 0.12 and abs(wz) < yaw_zero:
        return "LAT"
    if vx > 0.10 and abs(vy) > 0.10 and abs(wz) < yaw_zero:
        return "DIAG"
    if abs(vx) < lin_zero and abs(vy) < lin_zero and abs(wz) > 0.15:
        return "YAW"
    return "OTHER"
