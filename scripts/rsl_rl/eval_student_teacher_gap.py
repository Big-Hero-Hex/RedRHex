#!/usr/bin/env python3
"""Aggregate hash-bound command-sweep evidence across V2 policy lineages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sim2real.student_teacher_gap import (  # noqa: E402
    evaluate_student_teacher_gap,
    write_gap_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Teacher A, legacy, distilled V2, PPO V2, and required ablations."
    )
    parser.add_argument("manifest", type=Path, help="Evaluation run manifest JSON.")
    parser.add_argument("--json", type=Path, required=True, help="Output report JSON.")
    parser.add_argument("--csv", type=Path, required=True, help="Output aggregate CSV.")
    args = parser.parse_args()

    result = evaluate_student_teacher_gap(args.manifest)
    write_gap_report(result, args.json, args.csv)
    print(json.dumps(result["promotion"], indent=2, sort_keys=True))
    return 0 if result["promotion"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
