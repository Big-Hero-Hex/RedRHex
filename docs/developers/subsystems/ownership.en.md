---
id: subsystem-ownership
title: Subsystem Ownership and Documentation
lang: en
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-07
---

<a id="ownership"></a>
## Ownership map

| Subsystem | Owner | Canonical detail |
| --- | --- | --- |
| Isaac task and core behavior | `core` | `docs/developers/architecture/` and source-adjacent code |
| Training scripts and evaluation | `training` | operator training/evaluation and developer architecture |
| Training Panel and remote UI | `panel` | `tools/training_panel/docs/` |
| Reward Agent | `reward-agent` | `tools/reward_agent/docs/` |
| Sim-to-real calibration | `sim2real` | operator calibration, developer architecture, `tools/sim2real/` |
| ROS2 policy deployment | `deployment` | `ros2_ws/src/redrhex_rl_controller/docs/` |
| Documentation system | `project` | `docs/governance/` |

<a id="change-routing"></a>
## Change routing

Component-specific behavior is documented beside its code and linked from central portals. Cross-project contracts, decisions, roadmaps, releases, and governance remain central. Component READMEs are bilingual routers and must not duplicate maintained procedures.

<a id="release-routing"></a>
## Release routing

The Training Panel is independently versioned and owns component release notes. The repository as a whole uses dated milestones, not invented global SemVer. ROS, reward-agent, or sim-to-real shipped changes receive a component release record when an independent versioning stream is introduced; until then they are included in dated project milestones.

<a id="review-routing"></a>
## Review routing

The metadata `owner` identifies technical review responsibility. It does not waive bilingual parity, documentation-impact rules, or review by another affected subsystem when a change crosses a contract boundary.
