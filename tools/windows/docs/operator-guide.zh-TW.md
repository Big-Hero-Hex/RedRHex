---
id: windows-remote-launcher-operator-guide
title: 在 Windows 安裝與使用 RedRHex Remote
lang: zh-TW
audience: operator
type: how-to
status: draft
owner: panel
last_reviewed: 2026-08-13
---

<a id="prerequisites"></a>
## 前置條件與狀態

使用 Windows PowerShell 5.1 或更新版本，並安裝 Windows OpenSSH Client、連上 Tailscale。Active Windows smoke checklist 完成前，此 launcher 仍是 release candidate。它不會儲存 password、private key 或 Tailscale credential。

<a id="install"></a>
## 為目前使用者安裝

開啟 Windows PowerShell 並執行：

```powershell
$download = Join-Path $env:TEMP "redrhex_remote.ps1"
scp lab_user1@100.90.246.97:/home/lab_user1/Py/RedRHex/tools/windows/redrhex_remote.ps1 $download
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $download -Install
```

只在 `scp` 或 SSH prompt 輸入 Ubuntu password。安裝不需要 administrator 權限；script 會複製到 `%LOCALAPPDATA%\RedRHex Remote\`，並在目前使用者桌面建立 `RedRHex Remote.lnk`。

<a id="use"></a>
## 連線

1. 確認 Tailscale 已連線。
2. 雙擊 `RedRHex Remote`。
3. 若可見的 `RedRHex SSH Tunnel` 視窗出現 prompt，確認 host 後輸入 Ubuntu password。
4. 使用轉送服務期間保持該視窗開啟。

Launcher 會重用已有回應的 tunnel；否則連到 `lab_user1@100.90.246.97`、轉送本機 port `8080` 與 `6006`，並在 workstation 的 `env_isaaclab_bin` environment 啟動面板。它會開啟 `http://localhost:8080` 的 Training Panel。只有 TensorBoard 已有回應時，才會開啟 `http://localhost:6006`。

<a id="stop"></a>
## 停止與移除

關閉 `RedRHex SSH Tunnel` 視窗即可停止兩個 forward。移除 per-user installation：

```powershell
Remove-Item (Join-Path ([Environment]::GetFolderPath("Desktop")) "RedRHex Remote.lnk") -Force
Remove-Item (Join-Path $env:LOCALAPPDATA "RedRHex Remote") -Recurse -Force
```

這些 command 只會移除 shortcut 與已安裝的 launcher copy。

<a id="troubleshoot"></a>
## 疑難排解

- 缺少 `ssh.exe`：從 Windows Optional Features 安裝 OpenSSH Client。
- Authentication 或 reachability failure：在可見 tunnel 視窗確認 Tailscale、workstation address、host-key prompt 與 Ubuntu password。
- Local bind failure：停止已使用 port `8080` 或 `6006` 的 process，再重試。
- Panel readiness timeout：檢查 SSH 視窗與 workstation 的 `redrhex_panel` tmux session；沒有 tmux 時檢查 `logs/training_panel/remote_panel.log`。
- TensorBoard 未開啟：先從 Training Panel 啟動，再開啟 `http://localhost:6006`。
