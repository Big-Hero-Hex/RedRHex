# Repository instructions

Before creating, moving, renaming, deleting, or meaningfully editing human-facing documentation, read the English [documentation governance index](docs/governance/index.en.md) and the relevant governance files linked there. English is used for agent instructions only; the English and Traditional Chinese canonical files remain equal sources for humans.

Meaning-changing documentation changes must update both locale files together and preserve matching explicit anchors, correct metadata, type, and location, migration traceability for removals, and the [documentation-impact declaration](docs/governance/documentation-impact.en.md). Follow the linked governance details rather than restating them here.

Once Phase 3 provides it, run `python -m tools.documentation validate --staged`. Until then, run `git diff --check` before committing documentation changes.

When present, use `.agents/skills/writing-redrhex-docs/SKILL.md` for authoring and `.agents/skills/reviewing-redrhex-docs/SKILL.md` for review. These are the canonical skill implementations; Claude discovery adapters will live under `.claude/skills` and point to them.

CI validation is authoritative if agent guidance is missed.
