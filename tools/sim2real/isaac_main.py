from __future__ import annotations

import argparse
import json
from typing import Any

from .repo_binding import bind_redrhex_source


bind_redrhex_source()

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
        return result
    finally:
        simulation_app.close()
