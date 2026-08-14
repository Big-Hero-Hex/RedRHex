---
id: macos-remote-launcher-architecture
title: macOS Remote Launcher Architecture and Verification
lang: en
audience: developer
type: explanation
status: draft
owner: panel
last_reviewed: 2026-08-14
---

<a id="boundary"></a>
## Component boundary

`redrhex_remote.command` is one POSIX-shell, self-installing launcher intended for macOS. It owns deterministic configuration, desktop installation, endpoint probes, SSH arguments, readiness polling, browser launch, and the panel's macOS remote-mode marker. It does not change Tailscale, the SSH server, Isaac Lab, or training artifacts.

<a id="connection"></a>
## Connection contract

The launcher forwards local `8080` and `6006` to workstation loopback through `lab_user1@100.90.246.97`. The remote command is UTF-8 encoded as Base64 before transport, decoded by the workstation shell, and starts the panel in `/home/lab_user1/Py/RedRHex` with Conda environment `env_isaaclab_bin`.

The remote process reuses a responding panel, otherwise starts `redrhex_panel` in detached tmux or falls back to `nohup` and `logs/training_panel/remote_panel.log`. A background local monitor waits for the forwarded panel while foreground SSH retains access to host-key and password prompts. Closing the launcher Terminal window closes the forwards. No credential is embedded or persisted.

The launcher opens `/?remote_client=macos`. This is a capability marker, not an authorization boundary. The panel uses it to force headless training, route **TensorBoard** to one all-runs process on the fixed `6006` forward, and disable actions whose file-manager or live viewer windows can only open on the workstation. Recorded media, console output, copy controls, and other HTTP/API actions remain available.

<a id="installation"></a>
## Installation contract

`--install` copies the current script to `~/Desktop/RedRHex Remote.command` and sets mode `0700`. The `.command` extension makes the script launchable from Finder through Terminal. No administrator-owned location is modified.

<a id="verification"></a>
## Verification contract

`tests/test_redrhex_remote.sh` checks the remote-aware panel URL, remote panel command, SSH option/forward ordering, Base64 transport shape, deterministic install path, executable installation, and byte-for-byte source preservation. Training Panel browser tests verify both desktop markers, fixed-port TensorBoard routing, and disabled host-only controls. Run the dependency-free launcher checks with:

```sh
sh tools/macos/tests/test_redrhex_remote.sh
```

Portable source checks can run outside macOS, but release still requires the [active macOS smoke checklist](../../../docs/plans/active/2026-08-13-macos-remote-launcher.en.md#macos-smoke). The design stays `approved` and these component documents stay `draft` until that evidence exists.
