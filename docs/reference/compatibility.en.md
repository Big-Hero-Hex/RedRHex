---
id: compatibility-reference
title: Versions and Compatibility
lang: en
audience: shared
type: reference
status: active
owner: project
last_reviewed: 2026-08-07
---

<a id="project"></a>
## Project packages

The `RedRhex` Python extension reports version `0.1.0` and requires Python 3.10 or newer. Its classifiers list Isaac Sim 4.5, 5.0, and 5.1; actual operation also depends on a compatible Isaac Lab checkout. Training scripts require `rsl-rl-lib` 3.0.1 or newer.

<a id="panel"></a>
## Training Panel

The independently versioned Training Panel is `3.4.10` (`Sync Health Repair`). Its local API/UI, remote worker, remote web assets, and schema tag must remain aligned. Do not infer nonexistent releases 3.4.4 through 3.4.9; the consolidated 3.4.10 release record describes the evidenced change range.

<a id="deployment"></a>
## Deployment

The current ROS2 workflow targets ROS 2 Humble and a Jetson-class host. The ONNX contract is 56/280 observations, 12 actions, and 60 Hz. Hardware transport and sbRIO/RINBO assumptions remain site-specific and must be checked during bring-up.

<a id="documentation"></a>
## Documentation tooling

Documentation validation uses repository Python and no generated HTML. The published site pins MkDocs, Material, and `mkdocs-static-i18n` separately from runtime/training dependencies.

<a id="truth-boundary"></a>
## Truth boundary

Version declarations show supported or tested intent, not proof for every combination. Record the exact Isaac Lab/Sim, Python, CUDA, RSL-RL, ONNX Runtime, ROS, hardware, source commit, and dirty-state policy in experiments and deployment evidence.
