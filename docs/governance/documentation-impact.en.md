---
id: documentation-impact
title: Documentation Impact Rules
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-02
---

<a id="change-triggers"></a>
## Change triggers

| Change trigger | Required target family |
| --- | --- |
| CLI, workflow, environment, path, setup, hardware, or safety change | Operator documentation or shared reference |
| API, schema, architecture, dependency, testing, or extension change | Developer documentation or shared reference |
| Shipped behavior or compatibility change | Affected component release |
| Cross-cutting decision | ADR (`decision`) |
| Approved feature | `design` |
| Multi-step implementation | `plan` |
| Internal refactor with no reader-visible effect | Concrete no-impact declaration |
| Evidence that changes a baseline, recommendation, decision, or result | `experiment-summary` |

Update every affected audience or family when a change crosses rows; selecting one target does not waive another.

<a id="pull-request-declaration"></a>
## Pull-request declaration

Every pull request declares exactly these fields:

```text
Docs impact: none | operator | developer | shared | release | experiment
Docs reason: <required explanation>
```

Internal refactors still require a concrete `Docs impact: none` reason that explains why no maintained reader journey, reference, release, decision, design, plan, or evidence summary changes.

<a id="review-responsibility"></a>
## Review responsibility

Automation validates declaration presence and shape. Semantic correctness remains a reviewer responsibility; automation does not infer documentation impact from changed source paths.

<a id="stable-tool-interface"></a>
## Stable tool interface

Phase 3 implements the following stable interface. These commands are the stable documentation-tool contract:

```text
python -m tools.documentation validate --all
python -m tools.documentation validate --staged
python -m tools.documentation validate --changed-from REF
python -m tools.documentation inventory --format json
python -m tools.documentation stage-site --output DIR
```

Commands exit `0` on success; validation exits `1` on failure. Pre-commit uses `validate --staged`, and CI uses `validate --all`.
