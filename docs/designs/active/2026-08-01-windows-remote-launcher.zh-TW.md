---
id: windows-remote-launcher-design
title: Windows 遠端啟動器設計
lang: zh-TW
audience: developer
type: design
status: approved
owner: panel
last_reviewed: 2026-08-07
---

<a id="goal"></a>
## 目標

提供 Windows 桌面捷徑，讓筆電透過 Tailscale 連到 RedRHex workstation，以 SSH 轉送本機 Training Panel 與 TensorBoard port，等待面板就緒，再用預設瀏覽器開啟兩項服務。

<a id="scope"></a>
## 範圍

由一個可自我安裝的 PowerShell 5.1+ script 負責設定、per-user 安裝、shortcut 建立、SSH command construction、readiness polling 與 browser launch。它不修改面板、Isaac Lab、Tailscale、SSH server 或訓練程式。Workstation 仍是執行主機與 artifact source。

<a id="connection"></a>
## 連線 contract

- SSH target：`lab_user1@100.90.246.97`
- Panel：本機 `8080` 轉 workstation `127.0.0.1:8080`
- TensorBoard：本機 `6006` 轉 workstation `127.0.0.1:6006`
- URL：`http://localhost:8080` 與 `http://localhost:6006`
- 安裝目錄：`%LOCALAPPDATA%\RedRHex Remote\`
- Shortcut：目前使用者桌面的 `RedRHex Remote.lnk`

SSH terminal 保持可見，用於顯示 host-key、password、bind 與 connectivity error。關閉 terminal 就關閉 tunnel。Launcher 不儲存 password、private key、Tailscale credential 或其他 secret。

<a id="flow"></a>
## 啟動流程

1. 要求 Windows OpenSSH `ssh.exe`。
2. 若 panel tunnel 已可回應，直接重用，不啟動重複程序。
3. 否則以兩個 forward、`ExitOnForwardFailure` 與 keepalive 啟動可見 SSH process。
4. 最多等待 panel 45 秒。
5. 成功後開啟兩個 URL；失敗時說明 Tailscale、authentication、SSH 與 port conflict 原因。

<a id="verification"></a>
## 驗證

Dependency-free PowerShell test 涵蓋 deterministic argument、install path 與 tunnel command construction。Windows smoke verification 涵蓋 install/uninstall ownership、shortcut launch、authentication、兩個 forward、timeout、關閉 tunnel 與重用現有 tunnel。

<a id="status"></a>
## 狀態邊界

此 design 已核准，但 implementation 不在 documentation source branch。Implementation 與 Windows smoke evidence 尚未提交、release record 尚未發布前，不可描述 launcher 已交付。
