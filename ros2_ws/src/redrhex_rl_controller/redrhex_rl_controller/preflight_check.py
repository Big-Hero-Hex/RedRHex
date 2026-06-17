from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .policy_onnx_runner import PolicyONNXRunner
from .redrhex_contract import EXPECTED_ACTION_DIM, EXPECTED_OBS_DIM


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RedRhex ONNX and bench-safety preflight check.")
    parser.add_argument("--onnx", default="/home/jetson/RedRHex/policy.onnx")
    parser.add_argument("--hardware-profile", default="bench", choices=["bench", "lab", "custom"])
    parser.add_argument("--main-drive-vel-limit-rad-s", type=float, default=1.0)
    parser.add_argument("--abad-pos-limit-rad", type=float, default=0.15)
    parser.add_argument("--enable-policy-on-start", action="store_true")
    parser.add_argument("--enable-motor-output-on-start", action="store_true")
    args = parser.parse_args(argv)

    failures: list[str] = []
    warnings: list[str] = []
    if not os.path.exists(args.onnx):
        failures.append(f"ONNX does not exist: {args.onnx}")
    if args.hardware_profile == "bench":
        if args.main_drive_vel_limit_rad_s > 1.0:
            failures.append("bench profile main drive velocity limit must be <= 1.0 rad/s")
        if args.abad_pos_limit_rad > 0.15:
            failures.append("bench profile ABAD position limit must be <= 0.15 rad")
        if args.enable_policy_on_start:
            failures.append("enable_policy_on_start=true is not allowed for first bench runs")
        if args.enable_motor_output_on_start:
            failures.append("enable_motor_output_on_start=true is not allowed for first bench runs")
    if failures:
        for item in failures:
            print(f"FAIL: {item}", file=sys.stderr)
        return 2

    try:
        runner = PolicyONNXRunner(args.onnx, EXPECTED_OBS_DIM, EXPECTED_ACTION_DIM)
        print(runner.describe())
        action = runner.zero_check()
        print("zero obs action:", np.array2string(action, precision=5, suppress_small=True))
        print("action_dim:", action.shape[0])
        if action.shape[0] != EXPECTED_ACTION_DIM:
            failures.append(f"output dim is {action.shape[0]}, expected {EXPECTED_ACTION_DIM}")
        if not np.isfinite(action).all():
            failures.append("zero obs output has NaN/Inf")
    except Exception as exc:
        failures.append(str(exc))

    for item in warnings:
        print(f"WARN: {item}")
    if failures:
        for item in failures:
            print(f"FAIL: {item}", file=sys.stderr)
        return 2
    print("PASS: ONNX and bench-safe preflight checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
