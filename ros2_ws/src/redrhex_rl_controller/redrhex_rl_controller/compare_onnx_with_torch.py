from __future__ import annotations

import argparse
import sys

import numpy as np

from .policy_onnx_runner import PolicyONNXRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ONNX output with exported IsaacLab obs/action pairs stored in an NPZ file."
    )
    parser.add_argument("--onnx", default="/home/jetson/RedRHex/policy.onnx")
    parser.add_argument("--npz", required=True, help="NPZ containing obs [N,56] and action [N,12]")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args(argv)

    data = np.load(args.npz)
    obs = np.asarray(data["obs"], dtype=np.float32)
    expected = np.asarray(data["action"], dtype=np.float32)
    runner = PolicyONNXRunner(args.onnx)
    outputs = np.stack([runner.run(x) for x in obs], axis=0)
    err = outputs - expected
    print(f"max_abs_error={float(np.max(np.abs(err))):.8f}")
    print(f"mean_abs_error={float(np.mean(np.abs(err))):.8f}")
    if not np.allclose(outputs, expected, rtol=args.rtol, atol=args.atol):
        print("FAIL: ONNX output differs. Check normalizer, export path, and eval mode.", file=sys.stderr)
        return 2
    print("PASS: ONNX matches exported action pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
