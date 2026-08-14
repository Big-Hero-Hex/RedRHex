---
id: macos-remote-launcher-plan
title: macOS Remote Launcher Implementation Plan
lang: en
audience: developer
type: plan
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## Objective

Implement the [approved macOS launcher design](../../designs/active/2026-08-13-macos-remote-launcher.en.md) as one self-installing POSIX-shell `.command` file with dependency-free tests and a concise bilingual component README.

<a id="context"></a>
## Context

The launcher preserves the Windows launcher's fixed SSH target, port forwards, remote panel command, and no-secret boundary while replacing PowerShell and Windows shortcuts with macOS-native commands and a desktop `.command` file. The implementation environment can prove portable shell behavior, but not Finder, Terminal, or end-to-end macOS behavior.

<a id="phased-checklist"></a>
## Phased checklist

<a id="implementation"></a>
### Implementation

- [x] Add `tools/macos/redrhex_remote.command` with per-user installation, endpoint probes, foreground SSH, background readiness polling, browser launch, and fixed forwards.
- [x] Add `tools/macos/tests/test_redrhex_remote.sh` for the deterministic remote command, SSH arguments, install path, executable mode, and source preservation.
- [x] Add a bilingual router and paired operator and developer documentation, then connect them to the central portals and site manifest.
- [x] Store no password, private key, Tailscale credential, or other secret.
- [x] Mark macOS launcher sessions, route one all-runs TensorBoard through fixed forward `6006`, force headless training, and disable host-only file-manager and live-viewer controls.
- [x] Add browser regression proof for both desktop markers and preserved browser-safe actions.

<a id="macos-smoke"></a>
### macOS smoke verification

- [ ] Install without administrator privileges and verify the desktop `.command` file.
- [ ] Double-click with no existing tunnel and observe visible host-key or SSH authentication prompts in Terminal.
- [ ] Verify the marked Training Panel, on-demand TensorBoard, grey host-only controls, and browser-viewable recorded media after authentication.
- [ ] Close Terminal and verify both forwards stop.
- [ ] Relaunch with an existing tunnel and verify no duplicate SSH process.
- [ ] Verify first-launch security handling, disconnected Tailscale, timeout, and occupied ports produce understandable results.

<a id="verification"></a>
## Verification

Run `sh tools/macos/tests/test_redrhex_remote.sh`, `python -m tools.documentation validate --all`, and `python -m unittest discover -s tools/documentation/tests`. Record the separate macOS smoke observations before release.

<a id="completion-summary"></a>
## Completion summary

The implementation, portable tests, and canonical documentation are present. The plan remains active pending macOS and workstation smoke evidence; after that evidence, update the design to `implemented`, publish a release record, and resolve this plan according to the documentation lifecycle.
