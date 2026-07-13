from __future__ import annotations

import argparse
import json
import sys
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
    except BaseException as exc:
        # Kit owns process shutdown. Preserve a failing process status even when
        # framework teardown ends the interpreter before Python can re-raise.
        print(f"error: {exc}", file=sys.stderr, flush=True)
        simulation_app.app.post_quit(2)
        raise
    finally:
        simulation_app.close()
