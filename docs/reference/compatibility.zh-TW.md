---
id: compatibility-reference
title: 版本與相容性
lang: zh-TW
audience: shared
type: reference
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="project"></a>
## 專案套件

`RedRhex` Python 擴充套件回報版本 `0.1.0`，需要 Python 3.10 以上。Classifier 列出 Isaac Sim 4.5、5.0 與 5.1；實際操作還需要相容的 Isaac Lab checkout。訓練 script 需要 `rsl-rl-lib` 3.0.1 以上。

<a id="panel"></a>
## Training Panel

獨立版本化的 Mother package 目前為 `3.8.0-autopilot-preview`。其 Autopilot campaign feature 為 local、simulation-only 且預設關閉。Remote worker、Child web asset/cache key、heartbeat compatibility signal、capability row、sync summary 與 schema label 仍對齊獨立 protocol `3.7.0-remote-parity`；3.8 不新增 remote campaign control。3.7 migration 為 additive，既有 3.4.10 row 仍可讀取；但它不會讓舊 worker 具備 mutation compatibility。Worker 或 schema 任一較舊時，Child 會刻意保留 sign-in 與 inspection，同時停用 mutation 並顯示 migration/restart 指引。

不可推測不存在的 3.4.4 至 3.4.9 release；合併的 3.4.10 release record 只描述該段有證據的變更範圍。3.6.4 Drive exporter 維持 prerequisite baseline，credential 與 sharing policy 都留在 Mother。

<a id="deployment"></a>
## 部署

目前 ROS2 workflow 目標為 ROS 2 Humble 與 Jetson 類型主機。ONNX contract 是 56/280 observation、12 action 與 60 Hz。Hardware transport 與 sbRIO/RINBO 假設依現場而異，bring-up 時必須檢查。

<a id="documentation"></a>
## 文件工具

文件驗證使用 repository Python，且不提交 generated HTML。發布網站的 MkDocs、Material 與 `mkdocs-static-i18n` 另外 pin 版本，不與 runtime/training dependency 混用。

<a id="truth-boundary"></a>
## 真實邊界

版本宣告表示支援或測試意圖，不代表所有組合都已證明。實驗與部署證據必須記錄確切 Isaac Lab/Sim、Python、CUDA、RSL-RL、ONNX Runtime、ROS、hardware、source commit 與 dirty-state policy。
