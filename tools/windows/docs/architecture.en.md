---
id: windows-remote-launcher-architecture
title: Windows Remote Launcher Architecture and Verification
lang: en
audience: developer
type: explanation
status: draft
owner: panel
last_reviewed: 2026-08-13
---

<a id="boundary"></a>
## Component boundary

`redrhex_remote.ps1` is one PowerShell 5.1-compatible, self-installing launcher. It owns deterministic configuration, installation paths, shortcut creation, endpoint probes, SSH arguments, the visible tunnel window, readiness polling, and browser launch. It does not change Tailscale, the SSH server, Training Panel internals, Isaac Lab, or training artifacts.

<a id="connection"></a>
## Connection contract

The launcher forwards local `8080` and `6006` to workstation loopback through `lab_user1@100.90.246.97`. The remote command is UTF-8 encoded as Base64 before transport, decoded by the workstation shell, and starts the panel in `/home/lab_user1/Py/RedRHex` with Conda environment `env_isaaclab_bin`.

The remote process reuses a responding panel, otherwise starts `redrhex_panel` in detached tmux or falls back to `nohup` and `logs/training_panel/remote_panel.log`. It remains attached through SSH so closing the visible Windows terminal closes the forwards. No credential is embedded or persisted.

<a id="installation"></a>
## Installation contract

`-Install` copies the current script to `%LOCALAPPDATA%\RedRHex Remote\redrhex_remote.ps1` and creates the current user's desktop shortcut through `WScript.Shell`. The shortcut invokes `powershell.exe` with `-NoProfile` and the installed file. No administrator-owned location is modified.

<a id="verification"></a>
## Verification contract

`tests/test_redrhex_remote.ps1` checks SSH option/forward ordering, one quote-safe remote command argument, deterministic install paths, and the visible tunnel command. Run it on Windows PowerShell 5.1 or newer:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\windows\tests\test_redrhex_remote.ps1
```

Static source preservation is verified in Git, but release still requires the [active Windows smoke checklist](../../../docs/plans/active/2026-08-01-windows-remote-launcher.en.md#windows-smoke). The design stays `approved` and these component documents stay `draft` until that evidence exists.
