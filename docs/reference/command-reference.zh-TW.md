---
id: command-reference
title: 指令參考
lang: zh-TW
audience: shared
type: reference
status: active
owner: project
last_reviewed: 2026-08-07
---

<a id="environment"></a>
## 環境與安裝

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
git lfs install
git lfs pull
"$ISAACLAB_ROOT/isaaclab.sh" -p -m pip install -e source/RedRhex
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/list_envs.py
```

<a id="training"></a>
## 訓練

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 1 --headless
bash scripts/rsl_rl/train_stage_pipeline.sh --run_tag NAME --num_envs 4096
python -m tools.training_panel.smoke_pipeline --dry-run
```

<a id="operation"></a>
## 面板與監看

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
ssh -L 8080:127.0.0.1:8080 -L 6006:127.0.0.1:6006 user@host
```

<a id="reward-agent"></a>
## Reward Agent

```bash
python -m tools.reward_agent status
python -m tools.reward_agent create-session --objective "OBJECTIVE"
python -m tools.reward_agent propose-candidates --session-id ID --base-overrides-json '{}' --scale velocity_tracking:3.5:4.5
python -m tools.reward_agent queue-trials --session-id ID --base-params-json '{}' --dry-run
```

檢查 dry-run 參數後才可使用 `--launch`。

<a id="documentation"></a>
## 文件

```bash
python -m tools.documentation validate --all
python -m tools.documentation validate --staged
python -m tools.documentation validate --changed-from REF
python -m tools.documentation inventory --format json
python -m tools.documentation stage-site --output EMPTY_DIRECTORY
```
