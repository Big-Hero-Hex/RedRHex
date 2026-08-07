---
id: developer-validation
title: 測試與驗證 RedRHex
lang: zh-TW
audience: developer
type: how-to
status: active
owner: core
last_reviewed: 2026-08-07
---

<a id="tiers"></a>
## 驗證層級

先使用成本最低但足夠的層級：pure Python unit test、component integration test、documentation validation、受限 Isaac smoke、短 PPO/teacher/distillation smoke、command-sweep evaluation、deployment readiness、ROS mock/preflight，最後才是受限制的 hardware evidence。低層級通過不能取代必要的高層級結果。

<a id="cpu"></a>
## CPU 與元件測試

```bash
python -m unittest discover -s tools/documentation/tests -p 'test_*.py'
python -m unittest discover -s tools/reward_agent/tests -p 'test_*.py'
pytest -q tools/sim2real/tests tools/training_panel/tests
```

部分 Training Panel UI test 另有瀏覽器或執行環境需求；修改 UI 行為時必須執行。

<a id="isaac"></a>
## Isaac 驗證

使用 `scripts/rsl_rl/validate_reform_stack.py` 檢查 observation group、terrain、fault、PPO、teacher 與 distillation wiring。先使用少量 environment 與 step。Random rollout 通過不代表學習結果；若訓練程式有變更，還需執行 runner smoke。

<a id="deployment"></a>
## 部署驗證

從選定的 training checkpoint 匯出，執行面板 readiness，驗證 56/280 observation、12 action、60 Hz、joint order、limit、safety fault 與 Torch/ONNX parity，再執行 ROS preflight 與 mock mode。Hardware test 必須依 operator 安全順序並使用已審查證據。

<a id="docs"></a>
## 文件驗證

```bash
python -m tools.documentation validate --all
python -m tools.documentation inventory --format json
```

Commit 前使用 `validate --staged`；PR 必須包含正確的 `Docs impact` 與 `Docs reason` 欄位。語意新鮮度仍由 review 負責。
