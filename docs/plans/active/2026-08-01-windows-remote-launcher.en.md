---
id: windows-remote-launcher-plan
title: Windows Remote Launcher Implementation Plan
lang: en
audience: developer
type: plan
status: active
owner: panel
last_reviewed: 2026-08-13
---

<a id="objective"></a>
## Objective

Implement the [approved Windows launcher design](../../designs/active/2026-08-01-windows-remote-launcher.en.md) as one self-installing PowerShell file with dependency-free tests and a concise bilingual component README.

<a id="implementation"></a>
## Implementation tasks

- [x] Add `tools/windows/redrhex_remote.ps1` with pure argument/path/command helpers, endpoint probing, visible tunnel startup, readiness timeout, browser launch, and `-Install` mode.
- [ ] Add `tools/windows/tests/test_redrhex_remote.ps1` and prove the test fails before implementation and passes afterward on Windows PowerShell 5.1+. The test source is committed; Windows execution evidence remains pending.
- [x] Add a bilingual router at `tools/windows/README.md` that links to canonical operator and developer documentation.
- [x] Preserve the fixed SSH/port contract from the design and store no secrets.

<a id="windows-smoke"></a>
## Windows smoke verification

- [ ] Install without administrator privileges and verify the per-user script and desktop shortcut.
- [ ] Double-click with no existing tunnel and observe a visible SSH authentication terminal.
- [ ] Verify panel and TensorBoard URLs after authentication.
- [ ] Close the SSH terminal and verify both forwards stop.
- [ ] Relaunch with an existing tunnel and verify no duplicate SSH process.
- [ ] Verify missing OpenSSH, disconnected Tailscale, timeout, and occupied ports produce understandable errors.

<a id="completion"></a>
## Completion

After implementation and Windows evidence, update the design to `implemented`, publish a component release or dated project milestone, migrate any durable operating instructions, remove this completed plan, and record the removal in the migration manifest.
