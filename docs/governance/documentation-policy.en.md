---
id: documentation-policy
title: Documentation Policy
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="audiences"></a>
## Audience paths

RedRHex maintains two distinct bilingual entry paths:

- **Operator Documentation** is for people who set up, run, calibrate, deploy, or troubleshoot the robot.
- **Developer Documentation** is for people who understand, test, extend, or maintain the software.

Shared reference material may serve both paths, but each portal must preserve a clear reader journey.

<a id="bilingual-scope"></a>
## Bilingual scope and exceptions

English and Traditional Chinese are equal canonical sources. The bilingual requirement covers current operator and developer knowledge, shared references, governance, designs, plans, ADRs, roadmaps, releases, and published evidence.

Automation, generated files, skills, raw run notes, and deleted historical material do not require translation. Root and component `README.md` files are concise, single-file bilingual routers and are the only human-facing bilingual exception.

<a id="placement"></a>
## Hybrid placement

Central documentation owns portals, governance, roadmaps, decisions, and cross-cutting architecture or references. Detailed component-specific documentation is colocated beside the component it explains. Central navigation and the checked-in site manifest include both central and colocated documents so that readers have one connected system.

<a id="document-types"></a>
## Document types and use

- `index` provides a portal or section landing page.
- `tutorial` teaches through a guided learning sequence.
- `how-to` gives steps for a specific task.
- `reference` records exact facts, interfaces, commands, schemas, or contracts.
- `explanation` develops concepts, rationale, or architecture.
- `safety` records safety-critical constraints and procedures.
- `troubleshooting` diagnoses symptoms and recovery actions.
- `decision` is an ADR for a durable cross-cutting decision.
- `design` specifies an approved feature or material change before implementation.
- `plan` sequences a multi-step implementation.
- `roadmap` records current priorities and unresolved future work.
- `release` records shipped component behavior or a dated project milestone.
- `experiment-summary` publishes evidence that changes a baseline, recommendation, decision, or result.
- `audit` preserves a durable review or compliance finding.

<a id="retention-and-review"></a>
## Retention and review

Git is the archive; the live documentation tree is not an attic. Durable content is migrated into maintained paired documents, and redundant historical source documents are deleted only after replacement traceability and validation succeed. Safety documents use the normal review process and do not require a separate human safety gate.

<a id="related-governance"></a>
## Related governance

Use the [metadata schema](metadata-schema.en.md), [naming conventions](naming-conventions.en.md), [document lifecycle](document-lifecycle.en.md), [translation guide](translation-guide.en.md), [documentation-impact rules](documentation-impact.en.md), and [README router convention](readme-router-convention.en.md) for their detailed contracts.
