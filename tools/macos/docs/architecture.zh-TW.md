---
id: macos-remote-launcher-architecture
title: macOS 遠端啟動器架構與驗證
lang: zh-TW
audience: developer
type: explanation
status: draft
owner: panel
last_reviewed: 2026-08-14
---

<a id="boundary"></a>
## 元件邊界

`redrhex_remote.command` 是單一 POSIX shell、自我安裝且以 macOS 為目標的 launcher。它負責 deterministic configuration、desktop installation、endpoint probe、SSH argument、readiness polling、browser launch 與 panel 的 macOS remote-mode marker。它不改變 Tailscale、SSH server、Isaac Lab 或 training artifact。

<a id="connection"></a>
## 連線 contract

Launcher 透過 `lab_user1@100.90.246.97` 將本機 `8080` 與 `6006` 轉送到 workstation loopback。Remote command 在傳輸前以 UTF-8 編碼為 Base64，由 workstation shell 解碼，並在 `/home/lab_user1/Py/RedRHex` 使用 Conda environment `env_isaaclab_bin` 啟動面板。

Remote process 會重用已有回應的 panel；否則在 detached tmux 啟動 `redrhex_panel`，或 fallback 至 `nohup` 與 `logs/training_panel/remote_panel.log`。本機 background monitor 會等待轉送後的 panel，foreground SSH 則保留 host-key 與 password prompt 的存取能力。關閉 launcher Terminal 視窗就會關閉 forward。不會內嵌或保存 credential。

Launcher 會開啟 `/?remote_client=macos`。這是 capability marker，不是 authorization boundary。Panel 使用它強制 headless training、把 **TensorBoard** 導向固定 `6006` forward 上的單一 all-runs process，並停用只能在 workstation 開啟 file-manager 或 live viewer window 的 action。Recorded media、console output、copy control 與其他 HTTP/API action 維持可用。

<a id="installation"></a>
## 安裝 contract

`--install` 將目前 script 複製至 `~/Desktop/RedRHex Remote.command`，並設定 mode `0700`。`.command` 副檔名讓 Finder 能透過 Terminal 啟動 script。不會修改 administrator-owned location。

<a id="verification"></a>
## 驗證 contract

`tests/test_redrhex_remote.sh` 會檢查 remote-aware panel URL、remote panel command、SSH option/forward ordering、Base64 transport shape、deterministic install path、executable installation 與 byte-for-byte source preservation。Training Panel browser test 會驗證兩種 desktop marker、固定 port TensorBoard routing 與停用的 host-only control。執行 dependency-free launcher check：

```sh
sh tools/macos/tests/test_redrhex_remote.sh
```

Portable source check 可在 macOS 以外執行，但 release 前仍需完成[進行中的 macOS smoke checklist](../../../docs/plans/active/2026-08-13-macos-remote-launcher.zh-TW.md#macos-smoke)。Evidence 完成前，design 維持 `approved`，這些 component document 維持 `draft`。
