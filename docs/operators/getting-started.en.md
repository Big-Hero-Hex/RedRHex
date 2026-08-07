---
id: operator-getting-started
title: Get Started with RedRHex
lang: en
audience: operator
type: tutorial
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="prerequisites"></a>
## Prerequisites

Use Ubuntu with a working NVIDIA driver, an Isaac Lab installation, Git LFS, and Python 3.10 or newer. The extension declares compatibility with Isaac Sim 4.5, 5.0, and 5.1; the active Isaac Lab checkout and its Python environment must agree.

Set the launcher path for the current shell:

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
```

<a id="install"></a>
## Install the project extension

From the repository root:

```bash
git lfs install
git lfs pull
"$ISAACLAB_ROOT/isaaclab.sh" -p -m pip install -e source/RedRhex
```

The USD assets are stored through Git LFS. A tiny pointer file in place of `RedRhex.usd` means `git lfs pull` has not completed.

<a id="verify"></a>
## Verify the environment

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/list_envs.py
```

Confirm that `Template-Redrhex-Direct-v0` and `Template-Redrhex-ForwardFast-Direct-v0` appear. Then run the disposable smoke pipeline:

```bash
python -m tools.training_panel.smoke_pipeline
```

It launches one iteration and checks the checkpoint, TensorBoard event, saved parameters, and panel history discovery. Use `--dry-run` first when you only want to inspect the generated Isaac command.

<a id="next"></a>
## Next steps

- [Launch training](training/launch-training.en.md)
- [Use staged training](training/staged-training.en.md)
- [Operate the Training Panel](panel/training-panel.en.md)
- [Troubleshoot setup or training](troubleshooting/training-troubleshooting.en.md)
