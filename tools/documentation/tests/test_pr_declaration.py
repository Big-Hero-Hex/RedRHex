from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PullRequestDeclarationTests(unittest.TestCase):
    def test_accepts_every_allowed_impact_with_normalized_values_and_extra_text(
        self,
    ) -> None:
        try:
            from tools.documentation.pr_declaration import validate_declaration
        except ImportError as error:
            self.fail(f"pull-request declaration validator is missing: {error}")

        for impact in (
            "none",
            "operator",
            "developer",
            "shared",
            "release",
            "experiment",
        ):
            with self.subTest(impact=impact):
                body = (
                    "Summary of the pull request.\n\n"
                    f"Docs impact: \t{impact}  \n"
                    f"Docs reason:  Concrete explanation for {impact}. \t\n\n"
                    "Additional review context is allowed.\n"
                )
                self.assertEqual(validate_declaration(body), [])

    def test_rejects_an_unknown_impact_with_a_deterministic_error(self) -> None:
        from tools.documentation.pr_declaration import validate_declaration

        self.assertEqual(
            validate_declaration(
                "Docs impact: documentation\n"
                "Docs reason: This category is not part of the contract.\n"
            ),
            [
                "Docs impact must be one of: none, operator, developer, shared, "
                "release, experiment"
            ],
        )

    def test_requires_both_exact_case_sensitive_field_names(self) -> None:
        from tools.documentation.pr_declaration import validate_declaration

        cases = (
            (
                "Docs reason: The impact field is absent.\n",
                ["missing Docs impact field"],
            ),
            (
                "Docs impact: none\n",
                ["missing Docs reason field"],
            ),
            (
                "docs impact: none\nDocs Reason: Wrong field case.\n",
                ["missing Docs impact field", "missing Docs reason field"],
            ),
        )
        for body, expected in cases:
            with self.subTest(body=body):
                self.assertEqual(validate_declaration(body), expected)

    def test_rejects_duplicate_fields_once_per_field(self) -> None:
        from tools.documentation.pr_declaration import validate_declaration

        cases = (
            (
                "Docs impact: none\n"
                "Docs impact: shared\n"
                "Docs reason: One reason is present.\n",
                ["duplicate Docs impact field"],
            ),
            (
                "Docs impact: developer\n"
                "Docs reason: First reason.\n"
                "Docs reason: Second reason.\n",
                ["duplicate Docs reason field"],
            ),
        )
        for body, expected in cases:
            with self.subTest(body=body):
                self.assertEqual(validate_declaration(body), expected)

    def test_rejects_empty_or_whitespace_only_reasons(self) -> None:
        from tools.documentation.pr_declaration import validate_declaration

        for reason in ("", " \t "):
            with self.subTest(reason=reason):
                self.assertEqual(
                    validate_declaration(
                        f"Docs impact: none\nDocs reason:{reason}\n"
                    ),
                    ["Docs reason must contain non-whitespace prose"],
                )

    def test_html_comments_do_not_count_as_reason_prose(self) -> None:
        from tools.documentation.pr_declaration import validate_declaration

        for reason in (
            "<!-- Replace with a concrete explanation. -->",
            "<!-- first --><!-- second -->",
        ):
            with self.subTest(reason=reason):
                self.assertEqual(
                    validate_declaration(
                        f"Docs impact: shared\nDocs reason: {reason}\n"
                    ),
                    ["Docs reason must contain non-whitespace prose"],
                )

        self.assertEqual(
            validate_declaration(
                "Docs impact: shared\n"
                "Docs reason: <!-- context --> This updates both reader paths.\n"
            ),
            [],
        )

    def test_multiline_and_unclosed_html_comments_cannot_hide_declarations(
        self,
    ) -> None:
        from tools.documentation.pr_declaration import validate_declaration

        cases = (
            (
                "Docs impact: none\nDocs reason: <!--\nonly comment\n-->\n",
                ["Docs reason must contain non-whitespace prose"],
            ),
            (
                "Docs impact: none\nDocs reason: <!-- placeholder\n",
                ["Docs reason must contain non-whitespace prose"],
            ),
            (
                "<!--\nDocs impact: none\n"
                "Docs reason: Hidden declaration.\n-->\n",
                ["missing Docs impact field", "missing Docs reason field"],
            ),
        )
        for body, expected in cases:
            with self.subTest(body=body):
                self.assertEqual(validate_declaration(body), expected)

        self.assertEqual(
            validate_declaration(
                "<!-- introductory context\ninside a complete comment -->\n"
                "Docs impact: shared\n"
                "Docs reason: <!-- hidden\ncontext --> Visible explanation.\n"
            ),
            [],
        )

    def test_rejects_the_required_explanation_placeholder(self) -> None:
        from tools.documentation.pr_declaration import validate_declaration

        self.assertEqual(
            validate_declaration(
                "Docs impact: operator\nDocs reason: <required explanation>\n"
            ),
            ["Docs reason must contain non-whitespace prose"],
        )


class PullRequestDeclarationCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(PROJECT_ROOT), existing_pythonpath)
            if part
        )
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.documentation.pr_declaration",
                *arguments,
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_invalid_argument_shapes_use_argparse_exit_two(self) -> None:
        malformed = (
            (),
            ("--event-json",),
            ("--event", "event.json"),
            ("--event-json", "event.json", "extra"),
            (
                "--event-json",
                "first.json",
                "--event-json",
                "second.json",
            ),
        )
        for arguments in malformed:
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("usage:", result.stderr)

    def test_valid_utf8_event_exits_zero_with_exact_success_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "事件.json"
            event_path.write_text(
                json.dumps(
                    {
                        "pull_request": {
                            "body": (
                                "Summary.\nDocs impact: developer\n"
                                "Docs reason: 更新開發者驗證流程。\n"
                            )
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = self.run_cli("--event-json", str(event_path))

        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            result.stdout,
            "documentation impact declaration passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_invalid_declaration_exits_one_with_ordered_errors_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text(
                json.dumps(
                    {
                        "pull_request": {
                            "body": "Docs impact: unknown\nDocs reason: \n"
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_cli("--event-json", str(event_path))

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "Docs impact must be one of: none, operator, developer, shared, "
            "release, experiment\n"
            "Docs reason must contain non-whitespace prose\n"
            "documentation impact declaration failed (2 errors)\n",
        )
        self.assertNotIn("Traceback", result.stderr)

    def test_html_comment_bypasses_fail_cleanly_without_tracebacks(self) -> None:
        cases = (
            (
                "Docs impact: none\nDocs reason: <!--\nonly comment\n-->\n",
                "Docs reason must contain non-whitespace prose\n"
                "documentation impact declaration failed (1 errors)\n",
            ),
            (
                "Docs impact: none\nDocs reason: <!-- placeholder\n",
                "Docs reason must contain non-whitespace prose\n"
                "documentation impact declaration failed (1 errors)\n",
            ),
            (
                "<!--\nDocs impact: none\n"
                "Docs reason: Hidden declaration.\n-->\n",
                "missing Docs impact field\n"
                "missing Docs reason field\n"
                "documentation impact declaration failed (2 errors)\n",
            ),
        )
        for body, expected_stderr in cases:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                event_path = Path(directory) / "event.json"
                event_path.write_text(
                    json.dumps({"pull_request": {"body": body}}),
                    encoding="utf-8",
                )

                result = self.run_cli("--event-json", str(event_path))

                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, expected_stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_missing_event_file_fails_cleanly_without_a_traceback(self) -> None:
        result = self.run_cli(
            "--event-json",
            str(PROJECT_ROOT / "missing-task-3c-event.json"),
        )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "unable to read GitHub event JSON\n"
            "documentation impact declaration failed (1 errors)\n",
        )
        self.assertNotIn("Traceback", result.stderr)

    def test_invalid_utf8_event_fails_cleanly_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_bytes(b"\xff\xfe")

            result = self.run_cli("--event-json", str(event_path))

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "GitHub event JSON is not valid UTF-8\n"
            "documentation impact declaration failed (1 errors)\n",
        )
        self.assertNotIn("Traceback", result.stderr)

    def test_malformed_json_event_fails_cleanly_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text("{invalid", encoding="utf-8")

            result = self.run_cli("--event-json", str(event_path))

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "GitHub event JSON is malformed\n"
            "documentation impact declaration failed (1 errors)\n",
        )
        self.assertNotIn("Traceback", result.stderr)

    def test_oversized_numeric_body_uses_json_operational_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text(
                '{"pull_request":{"body":' + "9" * 5000 + "}}",
                encoding="utf-8",
            )

            result = self.run_cli("--event-json", str(event_path))

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "GitHub event JSON is malformed\n"
            "documentation impact declaration failed (1 errors)\n",
        )
        self.assertNotIn("Traceback", result.stderr)

    def test_event_requires_a_pull_request_object(self) -> None:
        for payload in ({}, {"pull_request": None}, {"pull_request": []}):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                event_path = Path(directory) / "event.json"
                event_path.write_text(json.dumps(payload), encoding="utf-8")

                result = self.run_cli("--event-json", str(event_path))

                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertEqual(
                    result.stderr,
                    "GitHub event is missing a pull_request object\n"
                    "documentation impact declaration failed (1 errors)\n",
                )
                self.assertNotIn("Traceback", result.stderr)

    def test_pull_request_body_must_be_a_string(self) -> None:
        for pull_request in ({}, {"body": None}, {"body": 42}, {"body": []}):
            with self.subTest(pull_request=pull_request), tempfile.TemporaryDirectory() as directory:
                event_path = Path(directory) / "event.json"
                event_path.write_text(
                    json.dumps({"pull_request": pull_request}),
                    encoding="utf-8",
                )

                result = self.run_cli("--event-json", str(event_path))

                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertEqual(
                    result.stderr,
                    "pull_request.body must be a string\n"
                    "documentation impact declaration failed (1 errors)\n",
                )
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
