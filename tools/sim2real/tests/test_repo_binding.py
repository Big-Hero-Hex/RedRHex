from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[3]


def _package(root: Path, marker: str) -> Path:
    package = root / "source" / "RedRhex" / "RedRhex"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
    return package


def test_repo_binding_precedes_a_decoy_checkout_in_subprocess(tmp_path: Path) -> None:
    expected_root = tmp_path / "expected"
    decoy_root = tmp_path / "decoy"
    expected_package = _package(expected_root, "expected")
    _package(decoy_root, "decoy")
    script = """
from pathlib import Path
import sys
from tools.sim2real.repo_binding import bind_redrhex_source
bind_redrhex_source(Path(sys.argv[1]))
import RedRhex
print(RedRhex.MARKER)
print(Path(RedRhex.__file__).resolve())
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(decoy_root / "source" / "RedRhex"), str(ROOT)]
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(expected_root)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "expected",
        str((expected_package / "__init__.py").resolve()),
    ]


def test_repo_binding_rejects_an_already_loaded_decoy(tmp_path: Path) -> None:
    expected_root = tmp_path / "expected"
    decoy_root = tmp_path / "decoy"
    _package(expected_root, "expected")
    _package(decoy_root, "decoy")
    script = """
from pathlib import Path
import sys
import RedRhex
from tools.sim2real.repo_binding import bind_redrhex_source
bind_redrhex_source(Path(sys.argv[1]))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(decoy_root / "source" / "RedRhex"), str(ROOT)]
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(expected_root)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )

    assert completed.returncode != 0
    assert "outside the selected repository" in completed.stderr
