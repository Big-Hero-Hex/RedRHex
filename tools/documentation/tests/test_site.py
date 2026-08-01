from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class SiteManifestTests(unittest.TestCase):
    def write_manifest(self, repo: Path, value: object) -> Path:
        path = repo / "docs/site-manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_checked_in_default_and_valid_colocated_sources_load(self) -> None:
        try:
            from tools.documentation.site import SiteSource, load_site_sources
        except ImportError as error:
            self.fail(f"site manifest loader is missing: {error}")

        checked_in = PROJECT_ROOT / "docs/site-manifest.json"
        self.assertEqual(
            json.loads(checked_in.read_text(encoding="utf-8")),
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": "docs"}],
            },
        )
        self.assertEqual(
            load_site_sources(PROJECT_ROOT),
            (SiteSource(PROJECT_ROOT / "docs", Path(".")),),
        )

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "docs").mkdir()
            (repo / "component/manual").mkdir(parents=True)
            self.write_manifest(
                repo,
                {
                    "schema_version": 1,
                    "sources": [
                        {"destination": ".", "source": "docs"},
                        {
                            "destination": "components/leg",
                            "source": "component/manual",
                        },
                    ],
                },
            )
            self.assertEqual(
                load_site_sources(repo),
                (
                    SiteSource(repo / "docs", Path(".")),
                    SiteSource(repo / "component/manual", Path("components/leg")),
                ),
            )

    def test_manifest_rejects_malformed_structure_paths_and_sources(self) -> None:
        from tools.documentation.errors import DocumentationOperationError
        from tools.documentation.site import load_site_sources

        valid_entry = {"destination": ".", "source": "docs"}
        invalid_values = (
            [],
            {},
            {"schema_version": 1},
            {"schema_version": 1, "sources": [], "extra": True},
            {"schema_version": True, "sources": [valid_entry]},
            {"schema_version": "1", "sources": [valid_entry]},
            {"schema_version": 2, "sources": [valid_entry]},
            {"schema_version": 1, "sources": {}},
            {"schema_version": 1, "sources": []},
            {"schema_version": 1, "sources": ["docs"]},
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": "docs", "extra": 1}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": 1}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": [], "source": "docs"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": "/docs"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": "/site", "source": "docs"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": "C:/docs"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": "C:/site", "source": "docs"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": "docs\\nested"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": "site\\nested", "source": "docs"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": "../outside"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": "../site", "source": "docs"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": "docs/../outside"}],
            },
            {
                "schema_version": 1,
                "sources": [valid_entry, valid_entry],
            },
            {
                "schema_version": 1,
                "sources": [
                    valid_entry,
                    {"destination": ".", "source": "component"},
                ],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": "missing"}],
            },
            {
                "schema_version": 1,
                "sources": [{"destination": ".", "source": "plain-file"}],
            },
        )

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            repo = Path(directory)
            (repo / "docs").mkdir()
            (repo / "component").mkdir()
            (repo / "plain-file").write_text("not a directory\n", encoding="utf-8")
            manifest_path = repo / "docs/site-manifest.json"

            with self.assertRaisesRegex(
                DocumentationOperationError,
                "^invalid documentation site manifest$",
            ):
                load_site_sources(repo)

            manifest_path.write_text("{invalid", encoding="utf-8")
            with self.assertRaisesRegex(
                DocumentationOperationError,
                "^invalid documentation site manifest$",
            ):
                load_site_sources(repo)

            for value in invalid_values:
                with self.subTest(value=value):
                    self.write_manifest(repo, value)
                    with self.assertRaisesRegex(
                        DocumentationOperationError,
                        "^invalid documentation site manifest$",
                    ):
                        load_site_sources(repo)

            (repo / "escape").symlink_to(Path(outside), target_is_directory=True)
            self.write_manifest(
                repo,
                {
                    "schema_version": 1,
                    "sources": [{"destination": ".", "source": "escape"}],
                },
            )
            with self.assertRaisesRegex(
                DocumentationOperationError,
                "^invalid documentation site manifest$",
            ):
                load_site_sources(repo)

    def test_manifest_rejects_nul_source_and_destination_strings(self) -> None:
        from tools.documentation.errors import DocumentationOperationError
        from tools.documentation.site import load_site_sources

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "docs").mkdir()
            for field in ("source", "destination"):
                with self.subTest(field=field):
                    entry = {"destination": ".", "source": "docs"}
                    entry[field] = "\0"
                    self.write_manifest(
                        repo,
                        {"schema_version": 1, "sources": [entry]},
                    )
                    with self.assertRaisesRegex(
                        DocumentationOperationError,
                        "^invalid documentation site manifest$",
                    ):
                        load_site_sources(repo)


class SiteStagingTests(unittest.TestCase):
    def write_manifest(self, repo: Path, sources: list[dict[str, str]]) -> None:
        path = repo / "docs/site-manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": 1, "sources": sources}),
            encoding="utf-8",
        )

    def write(self, repo: Path, relative: str, content: bytes) -> Path:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_canonical_sources_stage_with_deterministic_layout_and_exclusions(self) -> None:
        try:
            from tools.documentation.site import stage_site
        except ImportError as error:
            self.fail(f"site staging is missing: {error}")

        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory)
            repo = top / "repo"
            (repo / "docs").mkdir(parents=True)
            (repo / "component/manual").mkdir(parents=True)
            self.write_manifest(
                repo,
                [
                    {"destination": ".", "source": "docs"},
                    {
                        "destination": "components/leg",
                        "source": "component/manual",
                    },
                ],
            )
            expected_sources = {
                "docs/guide.en.md": b"english\n",
                "docs/guide.zh-TW.md": "中文\n".encode(),
                "docs/資料 空間/nested.en.md": b"nested english\n",
                "docs/資料 空間/nested.zh-TW.md": "巢狀中文\n".encode(),
                "component/manual/local.en.md": b"local english\n",
                "component/manual/local.zh-TW.md": "本地中文\n".encode(),
            }
            fixed_time = 1_700_000_000_000_000_000
            for relative, content in expected_sources.items():
                source = self.write(repo, relative, content)
                os.utime(source, ns=(fixed_time, fixed_time))

            for relative in (
                "docs/README.md",
                "docs/legacy.md",
                "docs/data.csv",
                "docs/governance/templates/reference.en.md",
                "docs/__pycache__/cached.en.md",
                "docs/.pytest_cache/generated.zh-TW.md",
                "docs/site/generated.en.md",
                "docs/Bad.en.md",
                "component/manual/notes.md",
            ):
                self.write(repo, relative, b"excluded\n")

            output = top / "staged"
            count = stage_site(repo, output)

            self.assertEqual(count, 6)
            actual_paths = sorted(
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            )
            self.assertEqual(
                actual_paths,
                [
                    "components/leg/local.en.md",
                    "components/leg/local.zh-TW.md",
                    "guide.en.md",
                    "guide.zh-TW.md",
                    "資料 空間/nested.en.md",
                    "資料 空間/nested.zh-TW.md",
                ],
            )
            for relative, content in expected_sources.items():
                if relative.startswith("docs/"):
                    destination = output / relative.removeprefix("docs/")
                else:
                    destination = output / "components/leg" / Path(relative).name
                self.assertEqual(destination.read_bytes(), content)
                self.assertEqual(destination.stat().st_mtime_ns, fixed_time)

    def test_copy_plan_collisions_fail_before_output_is_created(self) -> None:
        from tools.documentation.errors import DocumentationOperationError
        from tools.documentation.site import stage_site

        for collision_kind in ("same-file", "file-directory"):
            with self.subTest(collision_kind=collision_kind), tempfile.TemporaryDirectory() as directory:
                top = Path(directory)
                repo = top / "repo"
                (repo / "docs").mkdir(parents=True)
                if collision_kind == "same-file":
                    self.write(repo, "one/nested/guide.en.md", b"one\n")
                    self.write(repo, "two/guide.en.md", b"two\n")
                    sources = [
                        {"destination": ".", "source": "one"},
                        {"destination": "nested", "source": "two"},
                    ]
                else:
                    self.write(repo, "one/node.en.md", b"file\n")
                    self.write(repo, "two/child.en.md", b"child\n")
                    sources = [
                        {"destination": ".", "source": "one"},
                        {"destination": "node.en.md", "source": "two"},
                    ]
                self.write_manifest(repo, sources)
                output = top / "staged"

                with self.assertRaisesRegex(
                    DocumentationOperationError,
                    "^site staging plan has destination collisions$",
                ):
                    stage_site(repo, output)
                self.assertFalse(output.exists())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX file types")
    def test_symlinked_sources_candidates_and_nonregular_files_are_rejected(self) -> None:
        from tools.documentation.errors import DocumentationOperationError
        from tools.documentation.site import stage_site

        for unsafe_kind in (
            "configured-symlink",
            "source-ancestor-symlink",
            "candidate-symlink",
            "candidate-nonregular",
        ):
            with self.subTest(unsafe_kind=unsafe_kind), tempfile.TemporaryDirectory() as directory:
                top = Path(directory)
                repo = top / "repo"
                (repo / "docs").mkdir(parents=True)
                if unsafe_kind == "configured-symlink":
                    self.write(repo, "real-source/guide.en.md", b"guide\n")
                    (repo / "linked-source").symlink_to(
                        repo / "real-source", target_is_directory=True
                    )
                    source = "linked-source"
                elif unsafe_kind == "source-ancestor-symlink":
                    self.write(repo, "real-parent/nested/guide.en.md", b"guide\n")
                    (repo / "alias").symlink_to(
                        repo / "real-parent", target_is_directory=True
                    )
                    source = "alias/nested"
                elif unsafe_kind == "candidate-symlink":
                    self.write(repo, "source/target.txt", b"target\n")
                    (repo / "source/guide.en.md").symlink_to(repo / "source/target.txt")
                    source = "source"
                else:
                    (repo / "source").mkdir()
                    os.mkfifo(repo / "source/guide.en.md")
                    source = "source"
                self.write_manifest(
                    repo,
                    [{"destination": ".", "source": source}],
                )
                output = top / "staged"

                with self.assertRaisesRegex(
                    DocumentationOperationError,
                    "^unsafe documentation site source$",
                ):
                    stage_site(repo, output)
                self.assertFalse(output.exists())

    def test_output_must_be_external_real_and_empty_without_overwrite(self) -> None:
        from tools.documentation.errors import DocumentationOperationError
        from tools.documentation.site import stage_site

        for output_kind in (
            "inside-repository",
            "inside-source",
            "symlink",
            "file",
            "nonempty-directory",
        ):
            with self.subTest(output_kind=output_kind), tempfile.TemporaryDirectory() as directory:
                top = Path(directory)
                repo = top / "repo"
                self.write(repo, "docs/guide.en.md", b"guide\n")
                self.write_manifest(
                    repo,
                    [{"destination": ".", "source": "docs"}],
                )
                if output_kind == "inside-repository":
                    output = repo / "staged"
                elif output_kind == "inside-source":
                    output = repo / "docs/staged"
                elif output_kind == "symlink":
                    target = top / "target"
                    target.mkdir()
                    output = top / "staged"
                    output.symlink_to(target, target_is_directory=True)
                elif output_kind == "file":
                    output = top / "staged"
                    output.write_text("keep\n", encoding="utf-8")
                else:
                    output = top / "staged"
                    output.mkdir()
                    (output / "keep.txt").write_text("keep\n", encoding="utf-8")

                with self.assertRaisesRegex(
                    DocumentationOperationError,
                    "^unsafe or nonempty site staging output$",
                ):
                    stage_site(repo, output)
                if output_kind == "nonempty-directory":
                    self.assertEqual(
                        (output / "keep.txt").read_text(encoding="utf-8"),
                        "keep\n",
                    )
                    self.assertEqual(
                        sorted(path.name for path in output.iterdir()),
                        ["keep.txt"],
                    )
                if output_kind == "file":
                    self.assertEqual(output.read_text(encoding="utf-8"), "keep\n")

        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory)
            repo = top / "repo"
            self.write(repo, "docs/guide.en.md", b"guide\n")
            self.write_manifest(
                repo,
                [{"destination": ".", "source": "docs"}],
            )
            output = top / "staged"
            output.mkdir()
            self.assertEqual(stage_site(repo, output), 1)
            self.assertEqual((output / "guide.en.md").read_bytes(), b"guide\n")

    def test_missing_output_below_file_ancestor_is_rejected_before_staging(self) -> None:
        from tools.documentation.errors import DocumentationOperationError
        from tools.documentation.site import stage_site

        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory)
            repo = top / "repo"
            self.write(repo, "docs/guide.en.md", b"guide\n")
            self.write_manifest(
                repo,
                [{"destination": ".", "source": "docs"}],
            )
            blocking_ancestor = top / "regular-parent"
            blocking_ancestor.write_text("keep\n", encoding="utf-8")
            output = blocking_ancestor / "child"

            with self.assertRaisesRegex(
                DocumentationOperationError,
                "^unsafe or nonempty site staging output$",
            ):
                stage_site(repo, output)

            self.assertEqual(
                blocking_ancestor.read_text(encoding="utf-8"),
                "keep\n",
            )
            self.assertFalse(output.exists())

    def test_staging_filesystem_failures_use_operational_errors(self) -> None:
        from tools.documentation.errors import DocumentationOperationError
        from tools.documentation.site import stage_site

        failure_cases = (
            (
                "output-mkdir",
                "pathlib.Path.mkdir",
                OSError("injected mkdir failure"),
                "unsafe or nonempty site staging output",
            ),
            (
                "copy",
                "tools.documentation.site.shutil.copy2",
                ValueError("injected copy failure"),
                "unable to stage documentation site",
            ),
        )
        for label, target, error, message in failure_cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                top = Path(directory)
                repo = top / "repo"
                self.write(repo, "docs/guide.en.md", b"guide\n")
                self.write_manifest(
                    repo,
                    [{"destination": ".", "source": "docs"}],
                )
                with mock.patch(target, side_effect=error):
                    with self.assertRaisesRegex(
                        DocumentationOperationError,
                        f"^{message}$",
                    ):
                        stage_site(repo, top / "staged")

if __name__ == "__main__":
    unittest.main()
