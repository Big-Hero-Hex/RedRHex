from __future__ import annotations

import argparse
import sys

import numpy as np

from .policy_onnx_runner import PolicyONNXRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect RedRhex policy.onnx input/output.")
    parser.add_argument("onnx", nargs="?", default="/home/jetson/RedRHex/policy.onnx")
    parser.add_argument("--obs-dim", type=int, default=56)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--tensorrt", action="store_true")
    args = parser.parse_args(argv)

    runner = PolicyONNXRunner(args.onnx, args.obs_dim, args.action_dim, args.cuda, args.tensorrt)
    print(runner.describe())
    action = runner.zero_check()
    print("zero observation output:")
    print(np.array2string(action, precision=6, suppress_small=True))
    print(f"shape={action.shape} finite={np.isfinite(action).all()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
