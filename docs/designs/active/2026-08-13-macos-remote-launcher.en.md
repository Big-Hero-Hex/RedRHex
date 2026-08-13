---
id: macos-remote-launcher-design
title: macOS Remote Launcher Design
lang: en
audience: developer
type: design
status: approved
owner: panel
last_reviewed: 2026-08-14
---

<a id="problem"></a>
## Problem

The Windows-only launcher does not give macOS operators a Finder-launchable, visible SSH workflow for the remote Training Panel and TensorBoard. The macOS path must preserve interactive authentication, fixed local forwards, and per-user ownership without storing credentials.

<a id="goals-and-non-goals"></a>
## Goals and non-goals

- Goal: provide one macOS `.command` launcher that installs on the current user's desktop, starts or reuses the workstation panel, opens responding services, and keeps the tunnel visible.
- Non-goal: modify the existing Windows launcher, Tailscale, the SSH server, Training Panel internals, Isaac Lab, or training artifacts.

<a id="proposal-and-interfaces"></a>
## Proposal and interfaces

Use a POSIX-shell script with the macOS-provided `ssh`, `curl`, `base64`, and `open` commands. Forward local `8080` and `6006` to workstation loopback through `lab_user1@100.90.246.97`. Keep SSH in the foreground for host-key and password prompts while a background monitor waits up to 45 seconds for the panel, then opens the responding browser pages. `--install` copies the executable launcher to `~/Desktop/RedRHex Remote.command` without administrator access.

<a id="failure-modes"></a>
## Failure modes

Missing local commands fail before connection. Tailscale, authentication, host-key, remote startup, and occupied-port failures remain visible in Terminal. The readiness monitor reports a timeout, while the remote command prints the tmux pane or fallback log when startup fails. Closing Terminal or pressing Control-C ends the forwards. No password, private key, or Tailscale credential is embedded or persisted.

<a id="acceptance"></a>
## Acceptance

- [x] Provide the launcher, dependency-free source tests, and paired operator and developer documentation.
- [ ] Verify installation and first-launch behavior on a supported macOS host.
- [ ] Verify interactive authentication, both forwards, browser launch, timeout behavior, tunnel shutdown, and existing-tunnel reuse against the workstation.

<a id="resolution"></a>
## Resolution

Implementation and portable checks are present. Keep this design `approved` until macOS smoke evidence satisfies the remaining acceptance criteria; then publish the shipped outcome and resolve the active plan.
