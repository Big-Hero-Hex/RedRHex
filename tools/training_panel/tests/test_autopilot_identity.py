from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from importlib import metadata as importlib_metadata
from pathlib import Path
from unittest import mock

from tools.training_panel.training_panel.autopilot_identity import (
    AUTOPILOT_CODE_IDENTITY_PATHS,
    DEPENDENCY_DISTRIBUTIONS,
    SIMULATOR_COMPONENT_PATTERNS,
    _git_worktree_identity,
    build_dependency_manifest,
    build_simulator_manifest,
    build_source_manifest,
    canonical_json_bytes,
    dependency_manifest_sha256,
    dependency_manifest_for_python,
    runtime_source_identities,
)


class AutopilotIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative in AUTOPILOT_CODE_IDENTITY_PATHS:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture:{relative}\n", encoding="utf-8")
        self.simulator_root = self.root / "isaacsim"
        for name, pattern in SIMULATOR_COMPONENT_PATTERNS:
            path = self.simulator_root / pattern.replace("*", "fixture")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture:{name}\n", encoding="utf-8")
        self.distribution_source_state = "fixture-source-v1"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _versions(name: str) -> str:
        versions = {
            "gymnasium": "1.2.3",
            "hydra-core": "1.3.2",
            "isaaclab": "0.54.2",
            "isaaclab-assets": "0.2.4",
            "isaaclab-rl": "0.4.7",
            "isaaclab-tasks": "0.11.12",
            "redrhex": "0.1.0",
            "rsl-rl-lib": "3.1.2",
        }
        return versions[name]

    def _distribution_identity(self, name: str, version: str) -> dict[str, str]:
        return {
            "install_kind": "editable_tree",
            "content_sha256": hashlib.sha256(
                f"{name}:{version}:{self.distribution_source_state}".encode("utf-8")
            ).hexdigest(),
        }

    def test_dependency_manifest_is_resolved_canonical_and_fail_closed(self) -> None:
        manifest = build_dependency_manifest(
            self.root,
            simulator_root=self.simulator_root,
            version_resolver=self._versions,
            distribution_identity_resolver=self._distribution_identity,
            python_version=(3, 11, 15),
            implementation="CPython",
        )

        self.assertEqual(manifest["python"], {
            "implementation": "CPython",
            "version": "3.11.15",
        })
        distributions = {
            item["name"]: item for item in manifest["distributions"]
        }
        self.assertEqual(set(distributions), set(DEPENDENCY_DISTRIBUTIONS))
        self.assertEqual(distributions["isaaclab-tasks"]["version"], "0.11.12")
        simulator = build_simulator_manifest(self.simulator_root)
        self.assertEqual(manifest["simulator"], simulator)
        self.assertEqual(
            {item["name"] for item in simulator["components"]},
            {name for name, _pattern in SIMULATOR_COMPONENT_PATTERNS},
        )
        self.assertEqual(
            json.loads(canonical_json_bytes(manifest)),
            manifest,
        )

        first = dependency_manifest_sha256(manifest)
        changed = build_dependency_manifest(
            self.root,
            simulator_root=self.simulator_root,
            version_resolver=lambda name: (
                "0.55.0" if name == "isaaclab" else self._versions(name)
            ),
            distribution_identity_resolver=self._distribution_identity,
            python_version=(3, 11, 15),
            implementation="CPython",
        )
        self.assertNotEqual(first, dependency_manifest_sha256(changed))

        self.distribution_source_state = "fixture-source-v2"
        source_changed = build_dependency_manifest(
            self.root,
            simulator_root=self.simulator_root,
            version_resolver=self._versions,
            distribution_identity_resolver=self._distribution_identity,
            python_version=(3, 11, 15),
            implementation="CPython",
        )
        self.assertNotEqual(first, dependency_manifest_sha256(source_changed))

        def missing_hydra(name: str) -> str:
            if name == "hydra-core":
                raise importlib_metadata.PackageNotFoundError(name)
            return self._versions(name)

        with self.assertRaisesRegex(RuntimeError, "required dependency.*hydra-core"):
            build_dependency_manifest(
                self.root,
                simulator_root=self.simulator_root,
                version_resolver=missing_hydra,
                distribution_identity_resolver=self._distribution_identity,
                python_version=(3, 11, 15),
                implementation="CPython",
            )

    def test_editable_git_identity_changes_without_a_version_bump(self) -> None:
        origin = self.root / "editable"
        origin.mkdir()
        source = origin / "runtime.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        for args in (
            ("git", "init", "-q"),
            ("git", "config", "user.email", "autopilot@example.invalid"),
            ("git", "config", "user.name", "Autopilot Test"),
            ("git", "add", "runtime.py"),
            ("git", "commit", "-qm", "fixture"),
        ):
            subprocess.run(args, cwd=origin, check=True)

        baseline = _git_worktree_identity(origin)
        source.write_text("VALUE = 2\n", encoding="utf-8")
        changed = _git_worktree_identity(origin)

        self.assertIsNotNone(baseline)
        self.assertIsNotNone(changed)
        self.assertNotEqual(baseline, changed)

    def test_service_and_evaluator_identity_entrypoint_share_the_manifest(self) -> None:
        identities, dependency_manifest = runtime_source_identities(
            self.root,
            simulator_root=self.simulator_root,
            version_resolver=self._versions,
            distribution_identity_resolver=self._distribution_identity,
            python_version=(3, 11, 15),
            implementation="CPython",
        )
        source_manifest = build_source_manifest(self.root)

        self.assertEqual(
            [item["path"] for item in source_manifest["files"]],
            list(AUTOPILOT_CODE_IDENTITY_PATHS),
        )
        self.assertEqual(
            identities["dependency"],
            dependency_manifest_sha256(dependency_manifest),
        )
        self.assertEqual(len(identities["code"]), 64)
        self.assertEqual(len(identities["config"]), 64)

    def test_simulator_component_drift_and_ambiguity_fail_closed(self) -> None:
        baseline = build_dependency_manifest(
            self.root,
            simulator_root=self.simulator_root,
            version_resolver=self._versions,
            distribution_identity_resolver=self._distribution_identity,
        )
        torch_entry = next(
            item
            for item in baseline["simulator"]["components"]
            if item["name"] == "torch"
        )
        torch_metadata = self.simulator_root / torch_entry["path"]
        torch_metadata.write_text("Version: changed\n", encoding="utf-8")
        changed = build_dependency_manifest(
            self.root,
            simulator_root=self.simulator_root,
            version_resolver=self._versions,
            distribution_identity_resolver=self._distribution_identity,
        )
        self.assertNotEqual(
            dependency_manifest_sha256(baseline),
            dependency_manifest_sha256(changed),
        )

        duplicate = (
            self.simulator_root
            / "exts/omni.isaac.ml_archive/pip_prebundle/torch-second.dist-info/METADATA"
        )
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        duplicate.write_text("Version: duplicate\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "torch must resolve exactly once"):
            build_simulator_manifest(self.simulator_root)

    def test_dependency_probe_resolves_the_selected_interpreter(self) -> None:
        expected = build_dependency_manifest(
            self.root,
            simulator_root=self.simulator_root,
            version_resolver=self._versions,
            distribution_identity_resolver=self._distribution_identity,
        )
        with mock.patch(
            "tools.training_panel.training_panel.autopilot_identity.build_dependency_manifest",
            return_value=expected,
        ):
            manifest = dependency_manifest_for_python(
                self.root,
                sys.executable,
                simulator_root=self.simulator_root,
            )

        self.assertEqual(
            manifest["python"]["version"],
            ".".join(str(value) for value in sys.version_info[:3]),
        )
        with self.assertRaisesRegex(FileNotFoundError, "interpreter is missing"):
            dependency_manifest_for_python(
                self.root,
                self.root / "missing-python",
                simulator_root=self.simulator_root,
            )

    def test_controller_or_runner_drift_changes_code_identity(self) -> None:
        baseline, _manifest = runtime_source_identities(
            self.root,
            simulator_root=self.simulator_root,
            version_resolver=self._versions,
            distribution_identity_resolver=self._distribution_identity,
            python_version=(3, 11, 15),
            implementation="CPython",
        )
        for relative in (
            "tools/training_panel/training_panel/autopilot.py",
            "scripts/rsl_rl/runner_factory.py",
        ):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b"# drift\n")
                try:
                    changed, _manifest = runtime_source_identities(
                        self.root,
                        simulator_root=self.simulator_root,
                        version_resolver=self._versions,
                        distribution_identity_resolver=self._distribution_identity,
                        python_version=(3, 11, 15),
                        implementation="CPython",
                    )
                    self.assertNotEqual(changed["code"], baseline["code"])
                finally:
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
