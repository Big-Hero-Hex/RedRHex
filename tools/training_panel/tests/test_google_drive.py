import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.google_drive import (
    EXPORT_FIELD,
    GoogleDriveBusyError,
    GoogleDriveExporter,
    GoogleDrivePathError,
    GoogleDriveUnavailableError,
    normalize_destination_folder,
    parse_drive_destination,
)
from tools.training_panel.training_panel.history import HistoryStore
from tools.training_panel.training_panel.server import PanelHandler


class GoogleDriveExporterTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        return PanelPaths(
            repo_root=root,
            isaaclab_root=root / "IsaacLab",
            isaacsim_root=root / "isaacsim",
            conda_sh=root / "conda.sh",
            conda_env="env",
        )

    def make_run(self, root: Path, run_id: str = "run fixture") -> tuple[PanelPaths, HistoryStore, Path]:
        paths = self.make_paths(root)
        log_dir = paths.rsl_rl_log_root / run_id
        video = log_dir / "videos" / "play" / "model 10 result.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"fixture video")
        history = HistoryStore(paths)
        history.add_run(
            {
                "id": run_id,
                "source": "training_panel",
                "status": "completed",
                "created_at": "2026-08-14T10:00:00",
                "log_dir": str(log_dir),
            }
        )
        return paths, history, video

    @staticmethod
    def result(args, returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)

    def test_destination_folder_validation_accepts_nested_paths_and_rejects_unsafe_values(self):
        self.assertEqual(
            normalize_destination_folder(" Robotics / Panel Exports "),
            "Robotics/Panel Exports",
        )
        for value in ("", "/root", "root/", "root//child", "root/../child", "drive:path", "a\\b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_destination_folder(value)

    def test_destination_parser_accepts_a_drive_folder_link_and_rejects_other_urls(self):
        destination = parse_drive_destination(
            "https://drive.google.com/drive/u/0/folders/1AbCdEfGhIjKlMnOp?resourcekey=resource_Key-1&usp=sharing"
        )

        self.assertEqual(destination["destination_mode"], "folder_link")
        self.assertEqual(destination["root_folder_id"], "1AbCdEfGhIjKlMnOp")
        self.assertEqual(destination["resource_key"], "resource_Key-1")
        self.assertEqual(
            destination["folder_url"],
            "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp",
        )
        for value in (
            "https://example.com/drive/folders/1AbCdEfGhIjKlMnOp",
            "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view",
            "http://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_drive_destination(value)

    def test_status_reports_missing_rclone_without_exposing_configuration(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "tools.training_panel.training_panel.google_drive.shutil.which", return_value=None
        ):
            paths, history, _video = self.make_run(Path(tmp))
            exporter = GoogleDriveExporter(paths, history)

            status = exporter.status()

            self.assertFalse(status["available"])
            self.assertFalse(status["configured"])
            self.assertEqual(status["remote"], "redrhex-drive:")
            self.assertEqual(status["folder"], "RedRHex Videos")
            self.assertNotIn("token", str(status).lower())

    def test_success_uses_argument_arrays_persists_private_link_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, video = self.make_run(Path(tmp))
            commands = []
            command_kwargs = []

            def run_command(args, **kwargs):
                commands.append(args)
                command_kwargs.append(kwargs)
                if args[1] == "listremotes":
                    return self.result(args, stdout="redrhex-drive:\n")
                if args[1] == "copyto":
                    return self.result(args)
                if args[1] == "lsjson":
                    return self.result(args, stdout=f'{{"ID":"drive-file-10","Size":{video.stat().st_size}}}')
                raise AssertionError(args)

            exporter = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=run_command,
                start_background=lambda target: target(),
            )

            _started_record, started, deduplicated = exporter.start_export(
                "run fixture", video, checkpoint_iteration=10
            )
            completed = history.get_run("run fixture")[EXPORT_FIELD]["videos/play/model 10 result.mp4"]
            repeated, repeated_started, repeated_deduplicated = exporter.start_export(
                "run fixture", video, checkpoint_iteration=10
            )

            self.assertTrue(started)
            self.assertFalse(deduplicated)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["file_id"], "drive-file-10")
            self.assertEqual(
                completed["web_view_url"],
                "https://drive.google.com/file/d/drive-file-10/view",
            )
            self.assertIn(
                "redrhex-drive:RedRHex Videos/run_fixture/model_10_result.mp4",
                commands[1],
            )
            self.assertEqual(sum(command[1] == "copyto" for command in commands), 1)
            self.assertFalse(any("link" in command for command in commands))
            self.assertTrue(all(kwargs["shell"] is False for kwargs in command_kwargs))
            self.assertFalse(repeated_started)
            self.assertTrue(repeated_deduplicated)
            self.assertEqual(repeated["file_id"], "drive-file-10")

            video.write_bytes(b"changed fixture video")
            _changed, changed_started, changed_deduplicated = exporter.start_export(
                "run fixture", video, checkpoint_iteration=10
            )
            self.assertTrue(changed_started)
            self.assertFalse(changed_deduplicated)
            self.assertEqual(sum(command[1] == "copyto" for command in commands), 2)

    def test_destination_change_is_persisted_and_invalidates_old_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, video = self.make_run(Path(tmp))
            commands = []

            def run_command(args, **_kwargs):
                commands.append(args)
                if args[1] == "listremotes":
                    return self.result(args, stdout="redrhex-drive:\n")
                if args[1] == "lsjson":
                    return self.result(args, stdout=f'{{"ID":"drive-id","Size":{video.stat().st_size}}}')
                return self.result(args)

            exporter = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=run_command,
                start_background=lambda target: target(),
            )
            exporter.start_export("run fixture", video)
            status = exporter.save_destination("Robotics/Panel Exports")
            _record, started, reused = exporter.start_export("run fixture", video)

            self.assertEqual(status["folder"], "Robotics/Panel Exports")
            self.assertEqual(status["destination_revision"], 2)
            self.assertTrue(started)
            self.assertFalse(reused)
            self.assertEqual(sum(command[1] == "copyto" for command in commands), 2)
            self.assertIn(
                "redrhex-drive:Robotics/Panel Exports/run_fixture/model_10_result.mp4",
                commands[-2],
            )
            saved = json.loads(exporter.settings_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["destination_folder"], "Robotics/Panel Exports")
            self.assertNotIn("token", str(saved).lower())

    def test_pasted_folder_link_is_validated_and_used_as_the_private_export_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, video = self.make_run(Path(tmp))
            commands = []
            folder_link = (
                "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp"
                "?resourcekey=resource_Key-1&usp=sharing"
            )

            def run_command(args, **_kwargs):
                commands.append(args)
                if args[1] == "listremotes":
                    return self.result(args, stdout="redrhex-drive:\n")
                if args[1] == "lsjson" and args[3] == "redrhex-drive:":
                    return self.result(args, stdout='{"IsDir":true}')
                if args[1] == "lsjson":
                    return self.result(args, stdout=f'{{"ID":"linked-id","Size":{video.stat().st_size}}}')
                return self.result(args)

            exporter = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=run_command,
                start_background=lambda target: target(),
            )

            status = exporter.save_destination(folder_link)
            exporter.start_export("run fixture", video)

            self.assertEqual(status["destination_mode"], "folder_link")
            self.assertEqual(status["destination_display"], "Linked Google Drive folder")
            self.assertEqual(
                status["folder_url"],
                "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp",
            )
            self.assertNotIn("resource", str(status).lower())
            validation = next(command for command in commands if command[1:4] == ["lsjson", "--stat", "redrhex-drive:"])
            copy_command = next(command for command in commands if command[1] == "copyto")
            self.assertIn("--drive-root-folder-id", validation)
            self.assertIn("1AbCdEfGhIjKlMnOp", copy_command)
            self.assertIn("--drive-resource-key", copy_command)
            self.assertIn("redrhex-drive:run_fixture/model_10_result.mp4", copy_command)
            record = history.get_run("run fixture")[EXPORT_FIELD]["videos/play/model 10 result.mp4"]
            self.assertNotIn("resource", str(record).lower())
            saved = json.loads(exporter.settings_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["resource_key"], "resource_Key-1")
            self.assertEqual(exporter.settings_file.stat().st_mode & 0o777, 0o600)

    def test_reconnect_uses_argument_array_and_versions_the_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, _video = self.make_run(Path(tmp))
            popen_calls = []

            class FakeProcess:
                returncode = 0

                def communicate(self, timeout=None):
                    self.timeout = timeout
                    return "", ""

            def popen_factory(args, **kwargs):
                popen_calls.append((args, kwargs))
                return FakeProcess()

            exporter = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=lambda args, **kwargs: self.result(args, stdout="redrhex-drive:\n"),
                popen_factory=popen_factory,
                start_background=lambda target: target(),
            )

            reconnect, started = exporter.start_reconnect()

            self.assertTrue(started)
            self.assertEqual(reconnect["status"], "completed")
            self.assertEqual(exporter.status()["destination_revision"], 2)
            self.assertEqual(
                popen_calls[0][0],
                ["/usr/bin/rclone", "config", "reconnect", "redrhex-drive:", "--auto-confirm"],
            )
            self.assertIs(popen_calls[0][1]["shell"], False)
            self.assertNotIn("token", str(exporter.status()).lower())

    def test_destination_and_reconnect_refuse_to_change_during_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, video = self.make_run(Path(tmp))
            callbacks = []
            exporter = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=lambda args, **kwargs: self.result(args, stdout="redrhex-drive:\n"),
                start_background=callbacks.append,
            )
            exporter.start_export("run fixture", video)

            with self.assertRaises(GoogleDriveBusyError):
                exporter.save_destination("Other Folder")
            with self.assertRaises(GoogleDriveBusyError):
                exporter.start_reconnect()

    def test_accepts_a_run_directory_under_another_rsl_rl_task_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            log_dir = paths.rsl_rl_log_root.parent / "redrhex_forward_fast" / "run_fast"
            video = log_dir / "videos" / "play" / "result.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            history = HistoryStore(paths)
            history.add_run({"id": "run_fast", "created_at": "2026-08-14T10:00:00", "log_dir": str(log_dir)})

            def run_command(args, **_kwargs):
                if args[1] == "listremotes":
                    return self.result(args, stdout="redrhex-drive:\n")
                if args[1] == "lsjson":
                    return self.result(args, stdout=f'{{"ID":"fast-id","Size":{video.stat().st_size}}}')
                return self.result(args)

            exporter = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=run_command,
                start_background=lambda target: target(),
            )

            exporter.start_export("run_fast", video)

            record = history.get_run("run_fast")[EXPORT_FIELD]["videos/play/result.mp4"]
            self.assertEqual(record["status"], "completed")

    def test_fake_rclone_executable_completes_copy_and_stat_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, history, video = self.make_run(root)
            executable = root / "fake-rclone"
            command_log = root / "rclone-commands.jsonl"
            executable.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["FAKE_RCLONE_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[0] == "listremotes":
    print("redrhex-drive:")
elif args[0] == "lsjson":
    print(json.dumps({"ID": "fake-executable-id", "Size": int(os.environ["FAKE_VIDEO_SIZE"])}))
""",
                encoding="utf-8",
            )
            executable.chmod(0o755)

            with mock.patch.dict(
                os.environ,
                {
                    "FAKE_RCLONE_LOG": str(command_log),
                    "FAKE_VIDEO_SIZE": str(video.stat().st_size),
                },
            ):
                exporter = GoogleDriveExporter(
                    paths,
                    history,
                    rclone_path=str(executable),
                    start_background=lambda target: target(),
                )
                exporter.start_export("run fixture", video)

            commands = [json.loads(line) for line in command_log.read_text(encoding="utf-8").splitlines()]
            completed = history.get_run("run fixture")[EXPORT_FIELD]["videos/play/model 10 result.mp4"]
            self.assertEqual([command[0] for command in commands], ["listremotes", "copyto", "lsjson"])
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["file_id"], "fake-executable-id")

    def test_second_click_coalesces_while_same_file_is_uploading(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, video = self.make_run(Path(tmp))
            callbacks = []

            def run_command(args, **_kwargs):
                if args[1] == "listremotes":
                    return self.result(args, stdout="redrhex-drive:\n")
                if args[1] == "copyto":
                    return self.result(args)
                return self.result(args, stdout=f'{{"ID":"drive-id","Size":{video.stat().st_size}}}')

            exporter = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=run_command,
                start_background=callbacks.append,
            )

            first, first_started, _ = exporter.start_export("run fixture", video)
            second, second_started, second_deduplicated = exporter.start_export("run fixture", video)

            self.assertTrue(first_started)
            self.assertFalse(second_started)
            self.assertFalse(second_deduplicated)
            self.assertEqual(first["started_at"], second["started_at"])
            self.assertEqual(len(callbacks), 1)
            callbacks[0]()
            self.assertEqual(
                history.get_run("run fixture")[EXPORT_FIELD]["videos/play/model 10 result.mp4"]["status"],
                "completed",
            )

    def test_failed_upload_is_retryable_and_redacts_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, video = self.make_run(Path(tmp))

            def run_command(args, **_kwargs):
                if args[1] == "listremotes":
                    return self.result(args, stdout="redrhex-drive:\n")
                return self.result(
                    args,
                    returncode=1,
                    stderr=(
                        '{"detail":"'
                        + ("x" * 3000)
                        + '","access_token":"do-not-store"} --drive-resource-key do-not-store-resource'
                    ),
                )

            exporter = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=run_command,
                start_background=lambda target: target(),
            )

            exporter.start_export("run fixture", video)
            failed = history.get_run("run fixture")[EXPORT_FIELD]["videos/play/model 10 result.mp4"]

            self.assertEqual(failed["status"], "failed")
            self.assertIn("<redacted>", failed["error"])
            self.assertNotIn("do-not-store", failed["error"])
            self.assertNotIn("do-not-store-resource", failed["error"])
            self.assertLessEqual(len(failed["error"]), 2000)

    def test_reconcile_marks_unfinished_export_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, video = self.make_run(Path(tmp))
            history.patch_run_metadata(
                "run fixture",
                **{
                    EXPORT_FIELD: {
                        "videos/play/model 10 result.mp4": {
                            "status": "uploading",
                            "source_size": video.stat().st_size,
                        }
                    }
                },
            )

            GoogleDriveExporter(paths, history, rclone_path="/usr/bin/rclone")
            export = history.get_run("run fixture")[EXPORT_FIELD]["videos/play/model 10 result.mp4"]

            self.assertEqual(export["status"], "interrupted")
            self.assertIn("Retry", export["error"])

    def test_rejects_source_outside_run_and_unconfigured_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, history, video = self.make_run(root)
            outside = paths.rsl_rl_log_root / "other" / "outside.mp4"
            outside.parent.mkdir(parents=True)
            outside.write_bytes(b"outside")

            def configured(args, **_kwargs):
                return self.result(args, stdout="redrhex-drive:\n")

            exporter = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=configured,
                start_background=lambda target: target(),
            )
            with self.assertRaisesRegex(GoogleDrivePathError, "outside the selected run"):
                exporter.start_export("run fixture", outside)

            not_video = video.with_suffix(".txt")
            not_video.write_text("not an MP4", encoding="utf-8")
            with self.assertRaisesRegex(GoogleDrivePathError, "MP4"):
                exporter.start_export("run fixture", not_video)

            unavailable = GoogleDriveExporter(
                paths,
                history,
                rclone_path="/usr/bin/rclone",
                run_command=lambda args, **kwargs: self.result(args, stdout="other-drive:\n"),
            )
            with self.assertRaises(GoogleDriveUnavailableError):
                unavailable.start_export("run fixture", video)


class GoogleDriveHandlerTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        return PanelPaths(
            repo_root=root,
            isaaclab_root=root / "IsaacLab",
            isaacsim_root=root / "isaacsim",
            conda_sh=root / "conda.sh",
            conda_env="env",
        )

    def make_history(self, root: Path) -> tuple[PanelPaths, HistoryStore, Path, Path]:
        paths = self.make_paths(root)
        log_dir = paths.rsl_rl_log_root / "drive_run"
        log_dir.mkdir(parents=True)
        (log_dir / "model_0.pt").write_bytes(b"checkpoint 0")
        (log_dir / "model_10.pt").write_bytes(b"checkpoint 10")
        video_dir = log_dir / "videos" / "play"
        video_dir.mkdir(parents=True)
        old_video = video_dir / "model_0_old.mp4"
        latest_video = video_dir / "model_10_latest.mp4"
        old_video.write_bytes(b"old")
        latest_video.write_bytes(b"latest")
        os.utime(old_video, (100, 100))
        os.utime(latest_video, (200, 200))
        history = HistoryStore(paths)
        history.add_run(
            {
                "id": "drive_run",
                "source": "training_panel",
                "status": "completed",
                "created_at": "2026-08-14T10:00:00",
                "log_dir": str(log_dir),
            }
        )
        return paths, history, old_video, latest_video

    @staticmethod
    def handler(paths, history, exporter, payload):
        handler = object.__new__(PanelHandler)
        handler.path = "/api/runs/drive_run/export-video-to-drive"
        handler.state = type(
            "FakeState",
            (),
            {"paths": paths, "history": history, "google_drive": exporter},
        )()
        responses = []
        handler._payload = lambda: payload
        handler._json = lambda body, status=200: responses.append((body, status))
        handler._record_activity = lambda *args, **kwargs: None
        return handler, responses

    def test_system_payload_exposes_secret_free_drive_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))

            class FakeProcesses:
                def cuda_health(self):
                    return {"ok": True}

            class FakeExporter:
                def status(self):
                    return {
                        "available": True,
                        "configured": True,
                        "remote": "redrhex-drive:",
                        "folder": "RedRHex Videos",
                        "remediation": "",
                    }

            handler = object.__new__(PanelHandler)
            handler.path = "/api/system"
            handler.state = type(
                "FakeState",
                (),
                {"paths": paths, "processes": FakeProcesses(), "google_drive": FakeExporter()},
            )()
            responses = []
            handler._json = lambda body, status=200: responses.append((body, status))

            handler._do_GET()

            readiness = responses[0][0]["google_drive_export"]
            self.assertTrue(readiness["configured"])
            self.assertNotIn("token", str(readiness).lower())

    def test_settings_routes_save_destination_and_start_secret_free_reconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))

            class FakeExporter:
                def __init__(self):
                    self.saved = []

                def status(self):
                    return {
                        "available": True,
                        "configured": True,
                        "remote": "redrhex-drive:",
                        "folder": self.saved[-1] if self.saved else "RedRHex Videos",
                        "destination_mode": "my_drive_path",
                        "destination_display": self.saved[-1] if self.saved else "RedRHex Videos",
                        "folder_url": "",
                        "destination_revision": 2 if self.saved else 1,
                        "reconnect": {"status": "authorizing"},
                        "reconnect_command": "rclone config reconnect redrhex-drive:",
                        "remediation": "",
                    }

                def save_destination(self, folder):
                    self.saved.append(folder)
                    return self.status()

                def start_reconnect(self):
                    return {"status": "authorizing", "error": ""}, True

            exporter = FakeExporter()

            def invoke(path, payload):
                handler = object.__new__(PanelHandler)
                handler.path = path
                handler.state = type("FakeState", (), {"paths": paths, "google_drive": exporter})()
                responses = []
                activities = []
                handler._payload = lambda: payload
                handler._json = lambda body, status=200: responses.append((body, status))
                handler._record_activity = lambda *args, **kwargs: activities.append((args, kwargs))
                handler._do_POST()
                return responses, activities

            saved, saved_activity = invoke(
                "/api/google-drive/settings",
                {"destination": "Robotics/Panel Exports"},
            )
            reconnect, reconnect_activity = invoke("/api/google-drive/reconnect", {})

            self.assertEqual(saved[0][1], 200)
            self.assertEqual(saved[0][0]["google_drive_export"]["folder"], "Robotics/Panel Exports")
            self.assertEqual(reconnect[0][1], 202)
            self.assertEqual(reconnect[0][0]["reconnect"]["status"], "authorizing")
            self.assertTrue(saved_activity)
            self.assertTrue(reconnect_activity)
            self.assertNotIn("token", str(saved + reconnect).lower())

    def test_settings_routes_map_busy_and_unavailable_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))

            class FakeExporter:
                def save_destination(self, _folder):
                    raise GoogleDriveUnavailableError("configure rclone")

                def start_reconnect(self):
                    raise GoogleDriveBusyError("upload active")

            def invoke(path, payload):
                handler = object.__new__(PanelHandler)
                handler.path = path
                handler.state = type("FakeState", (), {"paths": paths, "google_drive": FakeExporter()})()
                responses = []
                handler._payload = lambda: payload
                handler._json = lambda body, status=200: responses.append((body, status))
                handler._record_activity = lambda *args, **kwargs: None
                handler._do_POST()
                return responses

            unavailable = invoke("/api/google-drive/settings", {"destination": "Exports"})
            busy = invoke("/api/google-drive/reconnect", {})

            self.assertEqual(unavailable, [({"error": "configure rclone"}, 503)])
            self.assertEqual(busy, [({"error": "upload active"}, 409)])

    def test_route_exports_latest_or_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, old_video, latest_video = self.make_history(Path(tmp))

            class FakeExporter:
                calls = []

                def start_export(self, run_id, source, checkpoint_iteration=None):
                    self.calls.append((run_id, Path(source), checkpoint_iteration))
                    return ({"status": "uploading"}, True, False)

            exporter = FakeExporter()
            latest_handler, latest_responses = self.handler(paths, history, exporter, {})
            latest_handler._do_POST()
            checkpoint_handler, checkpoint_responses = self.handler(
                paths, history, exporter, {"checkpoint_iteration": 0}
            )
            checkpoint_handler._do_POST()

            self.assertEqual(exporter.calls[0], ("drive_run", latest_video.resolve(), None))
            self.assertEqual(exporter.calls[1], ("drive_run", old_video.resolve(), 0))
            self.assertEqual(latest_responses[0][1], 202)
            self.assertEqual(checkpoint_responses[0][1], 202)

    def test_route_returns_200_for_unchanged_completed_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, history, _old_video, _latest_video = self.make_history(Path(tmp))

            class FakeExporter:
                def start_export(self, *args, **kwargs):
                    return (
                        {
                            "status": "completed",
                            "web_view_url": "https://drive.google.com/file/d/id/view",
                        },
                        False,
                        True,
                    )

            handler, responses = self.handler(paths, history, FakeExporter(), {})
            handler._do_POST()

            self.assertEqual(responses[0][1], 200)
            self.assertTrue(responses[0][0]["deduplicated"])

    def test_route_maps_invalid_missing_unsafe_and_unconfigured_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, history, _old_video, _latest_video = self.make_history(root)

            class FakeExporter:
                def start_export(self, *args, **kwargs):
                    raise GoogleDriveUnavailableError("configure rclone")

            invalid, invalid_responses = self.handler(
                paths, history, FakeExporter(), {"checkpoint_iteration": "bad"}
            )
            invalid._do_POST()
            missing, missing_responses = self.handler(
                paths, history, FakeExporter(), {"checkpoint_iteration": 999}
            )
            missing._do_POST()
            unavailable, unavailable_responses = self.handler(paths, history, FakeExporter(), {})
            unavailable._do_POST()

            outside_log = root / "outside"
            outside_video = outside_log / "outside.mp4"
            outside_log.mkdir()
            outside_video.write_bytes(b"outside")

            class UnsafeHistory:
                def get_run(self, _run_id):
                    return {
                        "id": "drive_run",
                        "log_dir": str(outside_log),
                        "latest_video": str(outside_video),
                    }

            unsafe, unsafe_responses = self.handler(paths, UnsafeHistory(), FakeExporter(), {})
            unsafe._do_POST()

            self.assertEqual(invalid_responses, [({"error": "checkpoint_iteration must be an integer"}, 400)])
            self.assertEqual(missing_responses[0][1], 404)
            self.assertEqual(unavailable_responses, [({"error": "configure rclone"}, 503)])
            self.assertEqual(unsafe_responses[0][1], 403)


if __name__ == "__main__":
    unittest.main()
