from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_redrhex_package_declares_and_discovers_sensor_v2_dependencies() -> None:
    source = (ROOT / "source" / "RedRhex" / "setup.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert '"redrhex-policy-io==2.0.0"' in source
    assert "find_packages" in source
    assert '("RedRhex", "RedRhex.*")' in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "find_packages"
        for node in ast.walk(tree)
    )


def test_repository_installer_orders_local_policy_io_before_redrhex() -> None:
    path = ROOT / "scripts" / "install_redrhex.py"
    spec = importlib.util.spec_from_file_location("redrhex_install_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    commands = module.install_commands(
        repo_root=ROOT,
        python_executable="/test/python",
    )

    assert commands == (
        (
            "/test/python",
            "-m",
            "pip",
            "install",
            "-e",
            str(ROOT / "source" / "redrhex_policy_io"),
        ),
        (
            "/test/python",
            "-m",
            "pip",
            "install",
            "-e",
            str(ROOT / "source" / "RedRhex"),
        ),
    )
