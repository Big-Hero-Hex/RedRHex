---
id: windows-remote-launcher-architecture
title: Windows 遠端啟動器架構與驗證
lang: zh-TW
audience: developer
type: explanation
status: draft
owner: panel
last_reviewed: 2026-08-14
---

<a id="boundary"></a>
## 元件邊界

`redrhex_remote.ps1` 是單一 PowerShell 5.1-compatible、自我安裝 launcher。它負責 deterministic configuration、installation path、shortcut creation、endpoint probe、SSH argument、可見 tunnel window、readiness polling、browser launch 與 panel 的 Windows remote-mode marker。它不改變 Tailscale、SSH server、Isaac Lab 或 training artifact。

<a id="connection"></a>
## 連線 contract

Launcher 透過 `lab_user1@100.90.246.97` 將本機 `8080` 與 `6006` 轉送到 workstation loopback。Remote command 在傳輸前以 UTF-8 編碼為 Base64，由 workstation shell 解碼，並在 `/home/lab_user1/Py/RedRHex` 使用 Conda environment `env_isaaclab_bin` 啟動面板。

Remote process 會重用已有回應的 panel；否則在 detached tmux 啟動 `redrhex_panel`，或 fallback 至 `nohup` 與 `logs/training_panel/remote_panel.log`。它透過 SSH 保持 attached，因此關閉可見 Windows terminal 就會關閉 forward。不會內嵌或保存 credential。

Launcher 會開啟 `/?remote_client=windows`。這是 capability marker，不是 authorization boundary。Panel 使用它強制 headless training、把 **TensorBoard** 導向固定 `6006` forward 上的單一 all-runs process，並停用只能在 workstation 開啟 file-manager 或 live viewer window 的 action。Recorded media、console output、copy control 與其他 HTTP/API action 維持可用。

<a id="installation"></a>
## 安裝 contract

`-Install` 將目前 script 複製至 `%LOCALAPPDATA%\RedRHex Remote\redrhex_remote.ps1`，並透過 `WScript.Shell` 建立目前使用者的 desktop shortcut。Shortcut 使用 `-NoProfile` 與已安裝檔案呼叫 `powershell.exe`。不會修改 administrator-owned location。

<a id="verification"></a>
## 驗證 contract

`tests/test_redrhex_remote.ps1` 會檢查 remote-aware panel URL、SSH option/forward ordering、單一 quote-safe remote command argument、deterministic install path 與可見 tunnel command。Training Panel browser test 會驗證兩種 desktop marker、固定 port TensorBoard routing 與停用的 host-only control。在 Windows PowerShell 5.1 或更新版本執行 launcher test：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\windows\tests\test_redrhex_remote.ps1
```

Git 已驗證 static source preservation，但 release 前仍需完成[進行中的 Windows smoke checklist](../../../docs/plans/active/2026-08-01-windows-remote-launcher.zh-TW.md#windows-smoke)。Evidence 完成前，design 維持 `approved`，這些 component document 維持 `draft`。
