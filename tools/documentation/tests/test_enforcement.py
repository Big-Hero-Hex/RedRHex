from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PANEL_WORKFLOW = PROJECT_ROOT / ".github/workflows/training-panel-pages.yml"
PULL_REQUEST_TEMPLATE = PROJECT_ROOT / ".github/pull_request_template.md"
PRE_COMMIT_CONFIG = PROJECT_ROOT / ".pre-commit-config.yaml"
DOCUMENTATION_WORKFLOW = PROJECT_ROOT / ".github/workflows/documentation.yml"
DOCUMENTATION_IMPACT = (
    PROJECT_ROOT / "docs/governance/documentation-impact.en.md",
    PROJECT_ROOT / "docs/governance/documentation-impact.zh-TW.md",
)
DOCUMENTATION_PLAN = (
    PROJECT_ROOT
    / "docs/plans/active/2026-08-01-documentation-system-reorganization.en.md",
    PROJECT_ROOT
    / "docs/plans/active/2026-08-01-documentation-system-reorganization.zh-TW.md",
)
DOCUMENTATION_DESIGN = tuple(
    PROJECT_ROOT
    / f"docs/designs/active/2026-08-01-documentation-system-design.{locale}.md"
    for locale in ("en", "zh-TW")
)
DOCUMENTATION_RELEASE = tuple(
    PROJECT_ROOT
    / f"docs/releases/2026-08-07-documentation-system-v1.{locale}.md"
    for locale in ("en", "zh-TW")
)
DOCUMENTATION_AUDIT = tuple(
    PROJECT_ROOT
    / f"docs/research/2026-08-07-documentation-system-audit.{locale}.md"
    for locale in ("en", "zh-TW")
)


class PanelWorkflowPreservationTests(unittest.TestCase):
    def test_panel_remains_at_pages_root_and_docs_build_below_docs(self) -> None:
        text = PANEL_WORKFLOW.read_text(encoding="utf-8")
        for required in (
            "          REDRHEX_DOCS_DIR: ${{ runner.temp }}/redrhex-doc-source\n",
            "          REDRHEX_DOCS_SITE_DIR: ${{ runner.temp }}/redrhex-pages/docs\n",
            "          REDRHEX_PAGES_DIR: ${{ runner.temp }}/redrhex-pages\n",
            '          cp -R tools/training_panel/remote_web/. "$REDRHEX_PAGES_DIR/"\n',
            '          python -m tools.documentation stage-site --output "$REDRHEX_DOCS_DIR"\n',
            "          mkdocs build --strict -f mkdocs.yml\n",
            "          path: ${{ runner.temp }}/redrhex-pages\n",
        ):
            self.assertEqual(text.count(required), 1, required)
        self.assertNotIn("path: tools/training_panel/remote_web", text)


class PullRequestTemplateContractTests(unittest.TestCase):
    def test_template_prompts_for_an_unselected_rejected_declaration(self) -> None:
        from tools.documentation.pr_declaration import (
            ALLOWED_IMPACTS,
            validate_declaration,
        )

        self.assertTrue(PULL_REQUEST_TEMPLATE.is_file(), "PR template is missing")
        text = PULL_REQUEST_TEMPLATE.read_text(encoding="utf-8")
        impact_lines = [
            line for line in text.splitlines() if line.startswith("Docs impact:")
        ]
        reason_lines = [
            line for line in text.splitlines() if line.startswith("Docs reason:")
        ]
        self.assertEqual(len(impact_lines), 1)
        self.assertEqual(len(reason_lines), 1)
        for impact in ALLOWED_IMPACTS:
            self.assertIn(impact, impact_lines[0])
        selected = impact_lines[0].removeprefix("Docs impact:").strip()
        self.assertNotIn(selected, ALLOWED_IMPACTS)
        self.assertEqual(
            validate_declaration("\n".join((*impact_lines, *reason_lines))),
            [
                "Docs impact must be one of: none, operator, developer, shared, "
                "release, experiment",
                "Docs reason must contain non-whitespace prose",
            ],
        )


class PreCommitContractTests(unittest.TestCase):
    def test_exact_local_bilingual_documentation_hook_is_always_run(self) -> None:
        text = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
        self.assertEqual(text.count("  - repo: local\n"), 1)
        self.assertEqual(text.count("      - id: redrhex-documentation\n"), 1)
        local = text.split("  - repo: local\n", 1)[1]
        for required_line in (
            "    hooks:\n",
            "      - id: redrhex-documentation\n",
            "        name: RedRHex bilingual documentation validation\n",
            "        entry: python -m tools.documentation validate --staged\n",
            "        language: system\n",
            "        pass_filenames: false\n",
            "        always_run: true\n",
        ):
            self.assertEqual(local.count(required_line), 1)


class DocumentationWorkflowContractTests(unittest.TestCase):
    def test_workflow_enforces_all_contracts_without_body_injection(self) -> None:
        self.assertTrue(DOCUMENTATION_WORKFLOW.is_file(), "workflow is missing")
        text = DOCUMENTATION_WORKFLOW.read_text(encoding="utf-8")
        for required in (
            "  pull_request:\n",
            "  push:\n",
            "    branches: [main]\n",
            "permissions:\n  contents: read\n",
            "      - uses: actions/checkout@v4\n",
            "          fetch-depth: 0\n",
            "      - uses: actions/setup-python@v5\n",
            '          python-version: "3.13"\n',
            "        run: python -m pip install -r docs/requirements-site.txt\n",
            "        run: python -m unittest discover tools/documentation/tests -v\n",
            "        run: python -m tools.documentation validate --all\n",
            '          python -m tools.documentation stage-site --output "$REDRHEX_DOCS_DIR"\n',
            "          mkdocs build --strict -f mkdocs.yml\n",
            "        if: github.event_name == 'pull_request'\n",
            '        run: python -m tools.documentation.pr_declaration --event-json "$GITHUB_EVENT_PATH"\n',
        ):
            self.assertEqual(text.count(required), 1, required)
        self.assertNotIn("paths:", text)
        self.assertNotIn("paths-ignore:", text)
        self.assertNotIn("contents: write", text)
        self.assertNotIn("github.event.pull_request.body", text)
        commands = (
            "python -m unittest discover tools/documentation/tests -v",
            "python -m tools.documentation validate --all",
            'python -m tools.documentation stage-site --output "$REDRHEX_DOCS_DIR"',
            "mkdocs build --strict -f mkdocs.yml",
            'python -m tools.documentation.pr_declaration --event-json "$GITHUB_EVENT_PATH"',
        )
        self.assertEqual([text.count(command) for command in commands], [1, 1, 1, 1, 1])
        self.assertEqual(
            [text.index(command) for command in commands],
            sorted(text.index(command) for command in commands),
        )


class GovernanceEnforcementStateTests(unittest.TestCase):
    def test_stable_interface_is_implemented_with_locale_parity(self) -> None:
        english, chinese = (
            path.read_text(encoding="utf-8") for path in DOCUMENTATION_IMPACT
        )
        self.assertIn(
            "Phase 3 implements the following stable interface.",
            english,
        )
        self.assertNotIn("will implement", english)
        self.assertNotIn("not yet available", english)
        self.assertIn("階段 3 已實作下列穩定介面。", chinese)
        self.assertNotIn("將實作", chinese)
        self.assertNotIn("尚不可用", chinese)
        expected_commands = (
            "python -m tools.documentation validate --all",
            "python -m tools.documentation validate --staged",
            "python -m tools.documentation validate --changed-from REF",
            "python -m tools.documentation inventory --format json",
            "python -m tools.documentation stage-site --output DIR",
        )
        for text in (english, chinese):
            self.assertEqual(
                tuple(
                    line
                    for line in text.splitlines()
                    if line.startswith("python -m tools.documentation ")
                ),
                expected_commands,
            )
        anchor_pattern = re.compile(r'<a id="([a-z0-9-]+)"></a>')
        self.assertEqual(
            anchor_pattern.findall(english),
            anchor_pattern.findall(chinese),
        )


class DocumentationLifecycleCompletionTests(unittest.TestCase):
    def test_completed_design_and_plan_resolve_to_release_and_audit(self) -> None:
        for temporary in (*DOCUMENTATION_DESIGN, *DOCUMENTATION_PLAN):
            self.assertFalse(temporary.exists(), temporary)
        for durable in (*DOCUMENTATION_RELEASE, *DOCUMENTATION_AUDIT):
            self.assertTrue(durable.is_file(), durable)

        release_texts = tuple(
            path.read_text(encoding="utf-8") for path in DOCUMENTATION_RELEASE
        )
        audit_texts = tuple(
            path.read_text(encoding="utf-8") for path in DOCUMENTATION_AUDIT
        )
        for text in release_texts:
            self.assertIn("id: documentation-system-v1-release", text)
            self.assertIn("status: published", text)
            self.assertIn("docs-reorg-v1", text)
        for text in audit_texts:
            self.assertIn("id: documentation-system-v1-audit", text)
            self.assertIn("`pending`", text)
            self.assertIn("stale", text)
            self.assertTrue("zero" in text or "零" in text)

        anchor_pattern = re.compile(r'<a id="([a-z0-9-]+)"></a>')
        self.assertEqual(
            anchor_pattern.findall(release_texts[0]),
            anchor_pattern.findall(release_texts[1]),
        )
        self.assertEqual(
            anchor_pattern.findall(audit_texts[0]),
            anchor_pattern.findall(audit_texts[1]),
        )


if __name__ == "__main__":
    unittest.main()
