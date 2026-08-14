---
id: macos-remote-launcher-design
title: macOS 遠端啟動器設計
lang: zh-TW
audience: developer
type: design
status: approved
owner: panel
last_reviewed: 2026-08-14
---

<a id="problem"></a>
## 問題

僅限 Windows 的 launcher 無法讓 macOS operator 從 Finder 啟動可見的 SSH workflow，以使用遠端 Training Panel 與 TensorBoard。macOS 路徑必須保留 interactive authentication、固定 local forward 與 per-user ownership，且不得儲存 credential。

<a id="goals-and-non-goals"></a>
## 目標與非目標

- 目標：提供單一 macOS `.command` launcher，安裝到目前使用者桌面、啟動或重用 workstation panel、開啟具 on-demand TensorBoard 的 remote-aware panel，並保持 tunnel 可見。
- 非目標：修改既有 Windows launcher、Tailscale、SSH server、Isaac Lab 或 training artifact。共用 Training Panel change 僅限 marker-driven remote capability presentation。

<a id="proposal-and-interfaces"></a>
## 提案與介面

使用 POSIX shell script 與 macOS 內建的 `ssh`、`curl`、`base64`、`open` command。透過 `lab_user1@100.90.246.97`，將本機 `8080` 與 `6006` 轉送到 workstation loopback。SSH 在 foreground 保留 host-key 與 password prompt，background monitor 最多等待 45 秒，再開啟 `http://localhost:8080/?remote_client=macos`。此 marker 會透過 `6006` 路由單一 all-runs TensorBoard、強制 headless training，並停用 host-only file-manager 與 live-viewer control。`--install` 不需 administrator 權限，會將 executable launcher 複製至 `~/Desktop/RedRHex Remote.command`。

<a id="failure-modes"></a>
## 失敗模式

缺少本機 command 時會在連線前失敗。Tailscale、authentication、host-key、remote startup 與 port occupied failure 會保持顯示於 Terminal。Readiness monitor 會回報 timeout；remote command 啟動失敗時則印出 tmux pane 或 fallback log。關閉 Terminal 或按 Control-C 會結束 forward。不會內嵌或保存 password、private key 或 Tailscale credential。

<a id="acceptance"></a>
## 驗收

- [x] 提供 launcher、dependency-free source test，以及成對的 operator 與 developer 文件。
- [x] 提供 marker-driven capability state 與 fixed-forward TensorBoard 的 browser regression proof。
- [ ] 在支援的 macOS host 驗證 installation 與 first-launch behavior。
- [ ] 對 workstation 驗證 interactive authentication、兩個 forward、browser launch、UI capability state、timeout、tunnel shutdown 與 existing-tunnel reuse。

<a id="resolution"></a>
## 結果

Implementation 與 portable check 已備妥。在 macOS smoke evidence 滿足剩餘驗收條件前，此 design 維持 `approved`；之後再發布 shipped outcome 並解決 active plan。
