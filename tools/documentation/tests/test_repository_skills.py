from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKILLS_ROOT = PROJECT_ROOT / ".agents/skills"
CLAUDE_ROOT = PROJECT_ROOT / ".claude/skills"


AUTHORING_SCENARIOS = {
    "operator": ("operator", "how-to", "safety sequencing"),
    "developer": ("developer", "architecture", "tests"),
    "release": ("release", "shipped behavior", "component version"),
    "adr": ("adr", "supersede", "decision"),
    "design": ("design", "proposed", "architecture"),
    "plan": ("plan", "temporary", "completed or cancelled"),
    "experiment": ("experiment", "evidence", "dated addendum"),
    "no-doc-impact": ("docs impact: none", "concrete reason", "internal-only"),
}

REVIEW_SCENARIOS = {
    "operator": ("operator", "hardware instructions", "physical e-stop"),
    "developer": ("developer", "interfaces", "tests"),
    "release": ("releases", "shipped behavior", "evidence-backed"),
    "adr": ("adrs", "superseded", "rewritten"),
    "design": ("designs", "durable records", "lifecycle"),
    "plan": ("plans", "durable records", "temporary"),
    "experiment": ("experiment", "immutable", "addenda"),
    "no-doc-impact": ("docs impact: none", "concrete reason", "actual change"),
}


def missing_signals(text: str, signals: tuple[str, ...]) -> set[str]:
    lowered = text.casefold()
    return {signal for signal in signals if signal.casefold() not in lowered}


class RepositoryAuthoringSkillScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skill = (SKILLS_ROOT / "writing-redrhex-docs/SKILL.md").read_text(
            encoding="utf-8"
        )

    def test_guided_contract_covers_every_document_scenario(self) -> None:
        for scenario, signals in AUTHORING_SCENARIOS.items():
            with self.subTest(scenario=scenario):
                self.assertEqual(missing_signals(self.skill, signals), set())

    def test_five_no_guidance_controls_each_miss_the_authoring_contract(self) -> None:
        controls = (
            "Write an English setup page in docs/setup.md.",
            "Copy the old report and call every statement active.",
            "Add an ADR and rewrite it later if the team changes its mind.",
            "Update a command without checking code or running validation.",
            "Docs impact: none",
        )
        required = (
            "name.en.md",
            "name.zh-tw.md",
            "verify commands",
            "stable-anchor",
            "concrete reason",
        )
        self.assertEqual(len(controls), 5)
        for repetition, control in enumerate(controls, start=1):
            with self.subTest(repetition=repetition):
                self.assertTrue(missing_signals(control, required))
        self.assertEqual(missing_signals(self.skill, required), set())

    def test_loophole_attempt_that_only_names_two_files_is_rejected(self) -> None:
        loophole = "Create name.en.md and name.zh-TW.md, then finish."
        required = ("metadata", "anchor", "meaning", "validate --staged")
        self.assertTrue(missing_signals(loophole, required))
        self.assertEqual(missing_signals(self.skill, required), set())


class RepositoryReviewSkillScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skill = (SKILLS_ROOT / "reviewing-redrhex-docs/SKILL.md").read_text(
            encoding="utf-8"
        )

    def test_guided_contract_covers_every_review_scenario(self) -> None:
        for scenario, signals in REVIEW_SCENARIOS.items():
            with self.subTest(scenario=scenario):
                self.assertEqual(missing_signals(self.skill, signals), set())

    def test_five_no_guidance_controls_each_miss_the_review_contract(self) -> None:
        controls = (
            "Looks good; spelling is clean.",
            "English exists, so translation can follow later.",
            "The command looks plausible and does not need code inspection.",
            "The old file was deleted; Git history is enough.",
            "Docs impact none is acceptable because the author selected it.",
        )
        required = (
            "current sources",
            "pair meaning",
            "migration-manifest.csv",
            "site-manifest.json",
            "ordered by severity",
        )
        self.assertEqual(len(controls), 5)
        for repetition, control in enumerate(controls, start=1):
            with self.subTest(repetition=repetition):
                self.assertTrue(missing_signals(control, required))
        self.assertEqual(missing_signals(self.skill, required), set())

    def test_loophole_attempt_that_only_runs_the_validator_is_rejected(self) -> None:
        loophole = "Run validate --all. If it passes, approve the documentation."
        required = ("accuracy", "evidence", "pair meaning", "lifecycle")
        self.assertTrue(missing_signals(loophole, required))
        self.assertEqual(missing_signals(self.skill, required), set())


class RepositorySkillAdapterTests(unittest.TestCase):
    def test_claude_adapters_point_to_canonical_skills_without_policy_copy(self) -> None:
        for name in ("writing-redrhex-docs", "reviewing-redrhex-docs"):
            with self.subTest(name=name):
                adapter = (CLAUDE_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
                self.assertIn(f".agents/skills/{name}/SKILL.md", adapter)
                self.assertIn("intentionally contains no duplicate policy", adapter)
                self.assertLess(len(adapter.splitlines()), 12)


if __name__ == "__main__":
    unittest.main()
