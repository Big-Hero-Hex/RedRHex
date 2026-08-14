---
id: macos-remote-launcher-operator-guide
title: 在 macOS 安裝與使用 RedRHex Remote
lang: zh-TW
audience: operator
type: how-to
status: draft
owner: panel
last_reviewed: 2026-08-14
---

<a id="prerequisites"></a>
## 前置條件與狀態

使用 macOS Terminal，並連上 Tailscale。Launcher 使用 macOS 內建的 `ssh`、`curl`、`base64` 與 `open` command。Active macOS smoke checklist 完成前，此 launcher 仍是 release candidate；它不會儲存 password、private key 或 Tailscale credential。

<a id="install"></a>
## 為目前使用者安裝

開啟 Terminal 並執行：

```sh
download="$(mktemp -t redrhex_remote)"
scp lab_user1@100.90.246.97:/home/lab_user1/Py/RedRHex/tools/macos/redrhex_remote.command "$download"
sh "$download" --install
rm -f "$download"
```

只在 `scp` 或 SSH prompt 輸入 Ubuntu password。安裝不需要 administrator 權限；script 會在目前使用者桌面建立可執行的 `RedRHex Remote.command`。

<a id="use"></a>
## 連線

1. 確認 Tailscale 已連線。
2. 雙擊桌面的 `RedRHex Remote.command`。
3. 若 Terminal 顯示 host-key 或 password prompt，確認 host 後輸入 Ubuntu password。
4. 使用轉送服務期間保持該 Terminal 視窗開啟。

Launcher 會重用已有回應的 tunnel；否則連到 `lab_user1@100.90.246.97`、轉送本機 port `8080` 與 `6006`，並在 workstation 的 `env_isaaclab_bin` environment 啟動面板。它會開啟 `http://localhost:8080/?remote_client=macos` 的 Training Panel。按下 **TensorBoard** 會在轉送的 `http://localhost:6006` 啟動或重用單一 all-runs TensorBoard。

此 marker 會保留 browser-safe action，並明確顯示 remote boundary。Training 固定使用 Headless。**Play**、**Open MuJoCo Viewer** 與開啟 folder 的 button 會顯示為停用，因為這些視窗只會開在 Ubuntu workstation，不會出現在 Mac。請改用 Record Video、Record MuJoCo MP4、Process Console 與可用的 copy-path control。

<a id="stop"></a>
## 停止與移除

關閉 launcher 的 Terminal 視窗，或按 Control-C，即可停止兩個 forward。移除 per-user installation：

```sh
rm "$HOME/Desktop/RedRHex Remote.command"
```

此 command 只會移除已安裝的 launcher copy。

<a id="troubleshoot"></a>
## 疑難排解

- macOS 拒絕開啟檔案：按住 Control 點擊 `RedRHex Remote.command`，選擇**打開**，並確認第一次啟動。
- Authentication 或 reachability failure：在 Terminal 確認 Tailscale、workstation address、host-key prompt 與 Ubuntu password。
- Local bind failure：停止已使用 port `8080` 或 `6006` 的 process，再重試。
- Panel readiness timeout：檢查 Terminal 與 workstation 的 `redrhex_panel` tmux session；沒有 tmux 時檢查 `logs/training_panel/remote_panel.log`。
- TensorBoard 未開啟：確認本機 port `6006` 未被占用，在 History 按下 **TensorBoard**；若啟動失敗，檢查 Process Console。
