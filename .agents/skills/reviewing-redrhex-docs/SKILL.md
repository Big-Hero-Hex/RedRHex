---
name: reviewing-redrhex-docs
description: Review RedRHex documentation changes for technical accuracy, bilingual meaning parity, metadata and naming compliance, lifecycle correctness, migration traceability, link health, site inclusion, and Docs impact declarations. Use for documentation PRs, source changes with documentation impact, releases, ADRs, designs, plans, experiment summaries, or claims that no documentation is needed.
---

# Review RedRHex documentation

Review evidence before prose style. Report actionable findings with exact file locations and do not edit unless the user also asks for changes.

## Load the contract

Read the canonical rules relevant to the change:

- [Documentation policy](../../../docs/governance/documentation-policy.en.md)
- [Metadata schema](../../../docs/governance/metadata-schema.en.md)
- [Naming conventions](../../../docs/governance/naming-conventions.en.md)
- [Document lifecycle](../../../docs/governance/document-lifecycle.en.md)
- [Translation guide](../../../docs/governance/translation-guide.en.md)
- [Migration manifest contract](../../../docs/governance/migration-manifest.en.md)
- [Documentation impact](../../../docs/governance/documentation-impact.en.md)
- [README router convention](../../../docs/governance/readme-router-convention.en.md)

Follow root `AGENTS.md` and any nearer instructions. Treat English and Traditional Chinese as equal canonical sources.

## Establish review scope

Inspect the diff, affected source code/configuration/tests, and the relevant portal pages. Identify:

- operator, developer, shared, release, experiment, or no-documentation impact;
- new, modified, superseded, moved, or removed knowledge;
- any component documentation that must appear in the central portal and staged site;
- any historical sources that require heading-level traceability.

Do not accept `Docs impact: none` without a concrete reason tied to the actual change.

## Review in risk order

### 1. Accuracy and safety

Verify commands, paths, versions, ports, environment names, interfaces, dimensions, topic names, compatibility, and shipped status against current sources. Flag proposals presented as implementation, results without evidence, missing limitations, or hardware instructions that bypass a gate, physical E-stop, current limit, preview, heartbeat, or staged enable sequence.

### 2. Pair meaning

Confirm both locale files express equivalent behavior, prerequisites, warnings, limitations, and outcomes. Heading anchor sequences and nonlocalized frontmatter must match. Do not demand word-for-word structure.

### 3. Structure and lifecycle

Check type, owner, audience, status, location, filename, date use, and portal links. Confirm README files are concise bilingual routers. Ensure ADRs are superseded rather than rewritten; temporary designs/plans resolve into durable records; releases describe evidence-backed shipped behavior; published experiments are immutable or use addenda.

### 4. Traceability and publication

For removed legacy material, inspect `docs/governance/migration-manifest.csv` at heading granularity. A removed source must have no pending rows, migrated rows must name valid canonical IDs, and a later traceability commit must record the full removal hash. Confirm colocated sources appear in `docs/site-manifest.json` and do not collide when staged.

### 5. Mechanical validation

Run:

```bash
python -m tools.documentation validate --all
python -m tools.documentation validate --staged
python -m unittest discover -s tools/documentation/tests
```

For publication changes, also stage and strictly build the site using the repository workflow. Inspect inventory and stale-document results rather than treating structural success as semantic freshness.

## Report the review

Lead with findings ordered by severity. Each finding states location, incorrect or risky behavior, evidence, and required fix. Then list unresolved assumptions and validation results. If no findings remain, say so explicitly and name residual risks such as unexecuted hardware steps or evidence that could not be reproduced.
