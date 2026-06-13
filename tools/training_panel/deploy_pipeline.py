"""Command entry point for RedRHex deploy readiness validation."""

from .training_panel.deploy import main


if __name__ == "__main__":
    raise SystemExit(main())
