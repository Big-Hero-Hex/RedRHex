---
id: developer-validation
title: 測試與驗證 RedRHex
lang: zh-TW
audience: developer
type: how-to
status: active
owner: core
last_reviewed: 2026-08-15
---

<a id="tiers"></a>
## 驗證層級

先使用成本最低但足夠的層級：pure Python unit test、component integration test、documentation validation、受限 Isaac smoke、短 PPO/teacher/distillation smoke、command-sweep evaluation、deployment readiness、ROS mock/preflight，最後才是受限制的 hardware evidence。低層級通過不能取代必要的高層級結果。

<a id="cpu"></a>
## CPU 與元件測試

從 repository root 執行下列 dependency-light contract，作為 merge 的最低門檻。目前 mainline code CI 涵蓋 Training Panel service、CPU sim-to-real subset、browser-independent JavaScript 與 desktop-launcher source test；在 CI coverage 擴充前，也要於本機執行 Reward Agent、Autopilot MCP 與 ROS contract。Autopilot MCP HTTP test 需要綁定 loopback socket 的權限，process test 可能使用 `tmux`。

```bash
python -m unittest discover -s tools/documentation/tests -p 'test_*.py'
python -m unittest discover -s tools/reward_agent/tests -p 'test_*.py'
python -m unittest discover -s plugins/redrhex-autopilot/tests -p 'test_*.py'
python -m pytest -q tools/training_panel/tests
python -m pytest -q tools/sim2real/tests \
  --ignore=tools/sim2real/tests/test_abad_target_mapping.py \
  --ignore=tools/sim2real/tests/test_physics_profile.py \
  --ignore=tools/sim2real/tests/test_target_delay.py \
  --ignore=tools/sim2real/tests/test_torsion_spring_model.py
PYTHONPATH="$PWD:$PWD/source/redrhex_policy_io:$PWD/ros2_ws/src/redrhex_rl_controller:$PWD/ros2_ws/src/redrhex_lowlevel_bridge" \
  python -m pytest -q ros2_ws/src/redrhex_lowlevel_bridge/test ros2_ws/src/redrhex_rl_controller/test
node --check tools/training_panel/static/app.js
node --check tools/training_panel/remote_web/remote_app.js
node --test tools/training_panel/remote_web/*.test.mjs
```

四個被排除的 sim-to-real module 需要較完整的 project runtime；其 contract 或 consumer 有變更時，這些 targeted test 仍是必要項。Lightweight suite 通過不能取代這些測試。

<a id="ui-and-launchers"></a>
## Browser 與 desktop launcher

Mother 或 Child 的 markup、style、navigation、role 或 action 有變更時，必須執行完整 browser suite。請先安裝 repository 的 Playwright browser/runtime prerequisite。

```bash
python -m pytest -q tools/training_panel/ui_tests
bash tools/macos/tests/test_redrhex_remote.sh
pwsh -NoProfile -File tools/windows/tests/test_redrhex_remote.ps1
```

PowerShell source test 需要 `pwsh`；此外也必須在受支援的 Windows 與 macOS host 上執行 active launcher plan 的 interactive smoke checklist。Source 與 mocked browser test 不能證明 first-launch security、SSH authentication、tunnel lifetime 或 target-workstation behavior。

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

Publication 變更需先安裝 `docs/requirements-site.txt`，將 canonical source stage 到空的 temporary directory，再透過 `mkdocs.yml` 執行 strict bilingual MkDocs build。Commit 前使用 `validate --staged`；PR 必須包含正確的 `Docs impact` 與 `Docs reason` 欄位。語意新鮮度仍由 review 負責。
