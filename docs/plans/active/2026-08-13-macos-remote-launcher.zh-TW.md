---
id: macos-remote-launcher-plan
title: macOS 遠端啟動器實作計畫
lang: zh-TW
audience: developer
type: plan
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## 目標

依[已核准 macOS launcher design](../../designs/active/2026-08-13-macos-remote-launcher.zh-TW.md)，以一個可自我安裝的 POSIX shell `.command` 檔案、dependency-free test 與精簡雙語元件 README 完成實作。

<a id="context"></a>
## 背景

Launcher 保留 Windows launcher 固定的 SSH target、port forward、remote panel command 與 no-secret boundary，並以 macOS 原生 command 和 desktop `.command` file 取代 PowerShell 與 Windows shortcut。Implementation environment 可以證明 portable shell behavior，但無法證明 Finder、Terminal 或 end-to-end macOS behavior。

<a id="phased-checklist"></a>
## 分階段檢查清單

<a id="implementation"></a>
### 實作

- [x] 新增 `tools/macos/redrhex_remote.command`，包含 per-user installation、endpoint probe、foreground SSH、background readiness polling、browser launch 與固定 forward。
- [x] 新增 `tools/macos/tests/test_redrhex_remote.sh`，檢查 deterministic remote command、SSH argument、install path、executable mode 與 source preservation。
- [x] 新增雙語 router 與成對的 operator／developer 文件，再連至中央 portal 與 site manifest。
- [x] 不儲存 password、private key、Tailscale credential 或其他 secret。

<a id="macos-smoke"></a>
### macOS smoke 驗證

- [ ] 不使用 administrator 權限安裝，並驗證 desktop `.command` file。
- [ ] 沒有既有 tunnel 時雙擊，確認 Terminal 顯示可見的 host-key 或 SSH authentication prompt。
- [ ] Authentication 後驗證 Training Panel、TensorBoard forward 與 browser behavior。
- [ ] 關閉 Terminal，確認兩個 forward 都停止。
- [ ] 已有 tunnel 時重新啟動，確認不建立重複 SSH process。
- [ ] 驗證 first-launch security handling、Tailscale 未連線、timeout 與 port 被占用時產生清楚結果。

<a id="verification"></a>
## 驗證

執行 `sh tools/macos/tests/test_redrhex_remote.sh`、`python -m tools.documentation validate --all` 與 `python -m unittest discover -s tools/documentation/tests`。Release 前另行記錄 macOS smoke observation。

<a id="completion-summary"></a>
## 完成摘要

Implementation、portable test 與 canonical documentation 已備妥。此 plan 維持 active，等待 macOS 與 workstation smoke evidence；取得 evidence 後，將 design 更新為 `implemented`、發布 release record，並依 documentation lifecycle 解決此 plan。
