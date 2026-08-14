---
id: windows-remote-launcher-plan
title: Windows 遠端啟動器實作計畫
lang: zh-TW
audience: developer
type: plan
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## 目標

依[已核准 Windows launcher design](../../designs/active/2026-08-01-windows-remote-launcher.zh-TW.md)，以一個可自我安裝的 PowerShell 檔案、dependency-free test 與精簡雙語元件 README 完成實作。

<a id="implementation"></a>
## 實作工作

- [x] 新增 `tools/windows/redrhex_remote.ps1`，包含 pure argument/path/command helper、endpoint probe、visible tunnel startup、readiness timeout、browser launch 與 `-Install` mode。
- [ ] 新增 `tools/windows/tests/test_redrhex_remote.ps1`，證明測試在 implementation 前失敗，之後於 Windows PowerShell 5.1+ 通過。Test source 已提交；仍待 Windows execution evidence。
- [x] 在 `tools/windows/README.md` 新增雙語 router，連到 canonical operator 與 developer 文件。
- [x] 保留 design 的固定 SSH/port contract，不儲存任何 secret。
- [x] 標記 Windows launcher session、透過固定 forward `6006` 路由單一 all-runs TensorBoard、強制 headless training，並停用 host-only file-manager 與 live-viewer control。
- [x] 新增 browser regression proof，涵蓋 Windows/macOS marker、fixed-forward TensorBoard、停用的 host-only control 與維持可用的 browser-safe action。

<a id="windows-smoke"></a>
## Windows smoke 驗證

- [ ] 不使用 administrator 權限安裝，並驗證 per-user script 與 desktop shortcut。
- [ ] 沒有既有 tunnel 時雙擊，確認可見 SSH authentication terminal。
- [ ] Authentication 後驗證帶 marker 的 panel URL 與 on-demand TensorBoard；確認 host-only control 顯示為灰色，且 recorded media 仍可檢視。
- [ ] 關閉 SSH terminal，確認兩個 forward 都停止。
- [ ] 已有 tunnel 時重新啟動，確認不建立重複 SSH process。
- [ ] 驗證缺少 OpenSSH、Tailscale 未連線、timeout 與 port 被占用時都有清楚錯誤。

<a id="completion"></a>
## 完成條件

Implementation 與 Windows evidence 完成後，將 design 更新為 `implemented`、發布 component release 或日期式 project milestone、遷移 durable operating instruction、移除此 completed plan，並在 migration manifest 記錄 removal。
