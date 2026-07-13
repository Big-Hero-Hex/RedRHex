from __future__ import annotations

import argparse
import json
from typing import Any

from isaaclab.app import AppLauncher


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Launch Isaac Sim before importing the characterization implementation."""

    app_launcher = AppLauncher(
        {
            "headless": bool(args.headless),
            "device": str(args.device),
        }
    )
    simulation_app = app_launcher.app
    try:
        from .isaac_runner import run_characterization

        result = run_characterization(args)
    except BaseException:
        # Isaac Sim's fast shutdown can terminate the interpreter before the
        # original exception is reported. Leave exceptional teardown to the
        # process so CLI failures retain their traceback and non-zero status.
        raise
    else:
        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            flush=True,
        )
        simulation_app.close()
        return result
