---
name: writing-redrhex-docs
description: Create or update maintained RedRHex operator, developer, shared-reference, governance, decision, design, plan, roadmap, release, or research documentation. Use whenever a change adds documentation, changes documented behavior, or requires a Docs impact declaration; also use for bilingual pairs, component README routers, and no-doc-impact decisions.
---

# Write RedRHex documentation

Produce accurate English and Traditional Chinese documentation that fits the repository information architecture and passes its validator. Canonical governance is authoritative; this skill supplies the workflow, not a second policy copy.

## Load the contract

Read the relevant canonical files before editing:

- [Documentation policy](../../../docs/governance/documentation-policy.en.md)
- [Metadata schema](../../../docs/governance/metadata-schema.en.md)
- [Naming conventions](../../../docs/governance/naming-conventions.en.md)
- [Document lifecycle](../../../docs/governance/document-lifecycle.en.md)
- [Translation guide](../../../docs/governance/translation-guide.en.md)
- [Documentation impact](../../../docs/governance/documentation-impact.en.md)
- [README router convention](../../../docs/governance/readme-router-convention.en.md)
- [Templates and authoring contract](../../../docs/governance/document-templates.en.md)

Read the paired `zh-TW` governance page when wording in that locale matters. Follow root `AGENTS.md` and any nearer instructions.

## Choose the document contract

1. State the user journey or durable question the document answers.
2. Choose the narrowest allowed type from the metadata schema: use a tutorial for learning, a how-to for a goal-oriented operator task, reference for exact facts, and explanation for understanding. Do not turn temporary execution details into architecture or an ADR.
3. Choose the audience and owner, then select the central portal or colocated component `docs/` directory.
4. Choose lifecycle status from the type-specific lifecycle. Record uncertain or unimplemented behavior as proposed or roadmap work, never as current behavior.
5. Use a dated filename only when chronology is intrinsic. Use the ADR sequence for decisions and component version for component releases. A release records evidence-backed shipped behavior, not aspiration.

If no maintained document should change, prepare `Docs impact: none` with a concrete reason. Internal-only refactoring is not self-explanatory.

## Verify before writing

Inspect current code, configuration, tests, and existing canonical documents. Verify commands, paths, environment names, ports, versions, topic names, dimensions, compatibility claims, and safety gates at their source. Distinguish:

- implemented behavior;
- evidence-backed result;
- limitation or unknown;
- proposal or future priority.

Never copy an unsupported historical claim into a maintained guide. For hardware procedures, preserve the repository's actual safety sequencing and do not invent a separate approval gate.

## Create or update the pair

Use the matching templates linked by the governance index.

- Create `name.en.md` and `name.zh-TW.md` together.
- Keep `id`, `audience`, `type`, `status`, `owner`, and `last_reviewed` identical. Localize only `title` and `lang`.
- Put `<a id="stable-anchor"></a>` immediately before every heading and preserve the same anchor sequence in both files.
- Translate meaning, caveats, and safety force; do not mechanically mirror sentence structure.
- Keep links within the same locale when an equivalent page exists.
- Make a root or component `README.md` a short bilingual router without frontmatter; put detail in canonical paired pages.
- Update the relevant portal and `docs/site-manifest.json` when a new colocated documentation source is introduced.

For meaning-changing edits, update both locales in the same change. A spelling-only localized correction may remain local only when it does not change meaning.

## Apply lifecycle and traceability

- Supersede an ADR; never rewrite its historical decision.
- Move completed roadmap work to a release or milestone.
- Resolve a design into architecture, an ADR, release, or roadmap before removing it.
- Summarize a completed or cancelled plan into durable documentation before removing it.
- Preserve published experiment summaries; issue a dated addendum for corrections.
- During migration, update every source heading in `docs/governance/migration-manifest.csv` before removing the source, then record the full removal commit in a follow-up change.

## Validate the result

Run the smallest relevant check while iterating, then finish with:

```bash
python -m tools.documentation validate --staged
python -m unittest discover -s tools/documentation/tests
```

For a repository-wide documentation change, also run:

```bash
python -m tools.documentation validate --all
python -m tools.documentation inventory --format json
```

Fix the document rather than weakening validation. Report the selected type, location, verification source, pair status, lifecycle effect, and command results.
