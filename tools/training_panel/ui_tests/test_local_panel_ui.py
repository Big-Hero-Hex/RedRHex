import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright


REPO_ROOT = Path(__file__).resolve().parents[3]
CHROME = shutil.which("google-chrome")
pytestmark = pytest.mark.skipif(not CHROME, reason="google-chrome is required for local panel UI tests")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_panel(url: str, proc: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"panel exited early with {proc.returncode}\n{output}")
        try:
            with urllib.request.urlopen(f"{url}/api/system", timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - diagnostic only
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"panel did not start: {last_error}")


def wait_for_missing(path: Path, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"path still exists: {path}")


def write_run(log_root: Path, run_id: str, checkpoints: tuple[int, ...]) -> Path:
    run_dir = log_root / run_id
    run_dir.mkdir(parents=True)
    for iteration in checkpoints:
        (run_dir / f"model_{iteration}.pt").write_text(f"model {iteration}", encoding="utf-8")
    (run_dir / "events.out.tfevents.test").write_text("event", encoding="utf-8")
    return run_dir


def make_fixture_root(root: Path) -> dict:
    log_root = root / "logs" / "rsl_rl" / "redrhex_wheg"
    alpha = write_run(log_root, "run_alpha", (0,))
    beta = write_run(log_root, "run_beta", (0, 10))
    (beta / "params").mkdir()
    (beta / "params" / "env.yaml").write_text("env: test\n", encoding="utf-8")
    (beta / "videos" / "play").mkdir(parents=True)
    (beta / "videos" / "play" / "rl-video-step-0.mp4").write_text("video", encoding="utf-8")
    (beta / "exported").mkdir()
    (beta / "exported" / "policy.onnx").write_text("onnx", encoding="utf-8")
    (beta / "deploy").mkdir()
    (beta / "deploy" / "readiness_deploy_fixture.json").write_text(
        json.dumps(
            {
                "version": 1,
                "pipeline_id": "deploy_fixture",
                "created_at": "2026-05-20T10:06:00",
                "completed_at": "2026-05-20T10:07:00",
                "overall_status": "warn",
                "readiness_level": "software-ready-with-warnings",
                "manifest": {
                    "run_id": "panel_run_beta",
                    "display_name": "Beta Foldered",
                    "target": "Jetson ROS2",
                    "expected_obs_dim": 56,
                    "history_obs_dim": 280,
                    "expected_action_dim": 12,
                },
                "stages": [
                    {
                        "name": "export_integrity",
                        "title": "Export Integrity",
                        "status": "pass",
                        "duration_s": 0.01,
                        "summary": "policy.onnx is present",
                        "details": {},
                        "artifacts": {},
                        "next_steps": [],
                    },
                    {
                        "name": "mujoco_readiness",
                        "title": "MuJoCo Readiness",
                        "status": "warn",
                        "duration_s": 0.01,
                        "summary": "model is advisory until calibrated",
                        "details": {},
                        "artifacts": {},
                        "next_steps": ["Calibrate the MJCF model before hardware signoff."],
                    },
                ],
                "artifacts": {},
                "operator_checklist": ["Keep motor output disabled for mock validation."],
                "assumptions": ["Fixture report for UI tests."],
            }
        ),
        encoding="utf-8",
    )

    panel_root = root / "logs" / "training_panel"
    process_logs = panel_root / "process_logs"
    process_logs.mkdir(parents=True)
    beta_log = process_logs / "panel_run_beta.log"
    beta_log.write_text("finished beta\n", encoding="utf-8")
    history = {
        "runs": [
            {
                "id": "panel_run_beta",
                "source": "training_panel",
                "status": "completed",
                "created_at": "2026-05-20T10:00:00",
                "updated_at": "2026-05-20T10:05:00",
                "log_dir": str(beta),
                "process_log": str(beta_log),
                "display_name": "Beta Foldered",
                "folder": "Good Runs",
                "params": {"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 10},
            },
            {
                "id": "queued_run",
                "source": "training_panel",
                "status": "queued",
                "created_at": "2026-05-20T11:00:00",
                "updated_at": "2026-05-20T11:00:00",
                "log_dir": None,
                "params": {"task": "Template-Redrhex-Direct-v0", "num_envs": 2, "max_iterations": 4},
            },
        ],
        "folders": ["Good Runs"],
        "deleted_runs": [],
        "deleted_folders": [],
    }
    (panel_root / "runs.json").write_text(json.dumps(history), encoding="utf-8")
    return {"alpha": alpha, "beta": beta}


@pytest.fixture()
def panel(tmp_path):
    paths = make_fixture_root(tmp_path)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["REDRHEX_ROOT"] = str(tmp_path)
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "tools.training_panel", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wait_for_panel(url, proc)
    try:
        yield {"url": url, "root": tmp_path, **paths}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture()
def page():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 950})
        page = context.new_page()
        yield page
        context.close()
        browser.close()


def open_history(page, url: str) -> None:
    page.goto(url)
    page.locator('.nav-button[data-view="history"]').click()
    page.wait_for_selector(".run-card")


def open_deploy(page, url: str) -> None:
    page.goto(url)
    page.locator('.nav-button[data-view="deploy"]').click()
    page.wait_for_selector("#deploy-run-select")
    page.select_option("#deploy-run-select", "panel_run_beta")
    expect(page.locator("#deploy-artifact-status")).to_contain_text("ONNX")
    expect(page.locator("#deploy-artifact-status")).to_contain_text("ready")


def test_history_defaults_to_all_runs_and_filters(panel, page):
    open_history(page, panel["url"])

    all_class = page.locator('.folder-item[data-folder="__all__"]').get_attribute("class") or ""
    assert "active" in all_class
    expect(page.locator("#runs")).to_contain_text("run_alpha")
    expect(page.locator("#runs")).to_contain_text("Beta Foldered")
    expect(page.locator("#runs")).to_contain_text("queued_run")

    assert page.locator('#status-filter option[value="queued"]').count() == 1
    assert page.locator('#status-filter option[value="stopping"]').count() == 1
    assert page.locator('#status-filter option[value="cancelled"]').count() == 1
    assert page.locator('#status-filter option[value="unknown"]').count() == 1

    page.select_option("#status-filter", "completed")
    expect(page.locator("#runs")).to_contain_text("Beta Foldered")
    expect(page.locator("#runs")).not_to_contain_text("queued_run")

    page.select_option("#status-filter", "")
    page.locator('.folder-item[data-folder="Good Runs"] .folder-name').click()
    expect(page.locator("#runs")).to_contain_text("Beta Foldered")
    expect(page.locator("#runs")).not_to_contain_text("run_alpha")

    page.locator('.folder-item[data-folder="__all__"]').click()
    expect(page.locator("#runs")).to_contain_text("run_alpha")


def test_single_delete_requires_exact_run_id(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="run_alpha").click()
    page.locator("#delete-run").click()
    expect(page.locator("#confirm-dialog")).to_be_visible()
    page.fill("#confirm-dialog-input", "wrong")
    expect(page.locator("#confirm-dialog-confirm")).to_be_disabled()
    page.locator("#confirm-dialog-cancel").click()
    expect(page.locator("#runs")).to_contain_text("run_alpha")
    assert panel["alpha"].exists()

    page.locator("#delete-run").click()
    page.fill("#confirm-dialog-input", "run_alpha")
    page.locator("#confirm-dialog-confirm").click()
    expect(page.locator("#runs")).not_to_contain_text("run_alpha")
    assert not panel["alpha"].exists()


def test_compact_preserves_artifacts_and_requires_exact_run_id(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()

    page.locator("#compact-run").click()
    expect(page.locator("#confirm-dialog")).to_be_visible()
    page.fill("#confirm-dialog-input", "wrong")
    expect(page.locator("#confirm-dialog-confirm")).to_be_disabled()
    page.locator("#confirm-dialog-cancel").click()
    expect(page.locator("#compact-run")).to_be_enabled()
    assert (panel["beta"] / "model_0.pt").exists()

    page.locator("#compact-run").click()
    page.fill("#confirm-dialog-input", "panel_run_beta")
    page.locator("#confirm-dialog-confirm").click()
    expect(page.locator("#panel-status")).to_contain_text("Compacted")
    expect(page.locator("#runs")).to_contain_text("Beta Foldered")

    wait_for_missing(panel["beta"] / "model_0.pt")
    assert (panel["beta"] / "model_10.pt").exists()
    assert (panel["beta"] / "events.out.tfevents.test").exists()
    assert (panel["beta"] / "params" / "env.yaml").exists()
    assert (panel["beta"] / "videos" / "play" / "rl-video-step-0.mp4").exists()
    assert (panel["beta"] / "exported" / "policy.onnx").exists()


def test_rename_notes_refresh_and_snapshot_debug_does_not_poll(panel, page):
    debug_calls = {"count": 0}

    def count_debug(route):
        debug_calls["count"] += 1
        route.continue_()

    page.route("**/api/runs/panel_run_beta/debug", count_debug)
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()
    expect(page.locator("#debug-live")).to_have_text("Snapshot")
    calls_after_snapshot = debug_calls["count"]
    page.wait_for_timeout(2200)
    assert debug_calls["count"] == calls_after_snapshot

    page.fill("#run-name", "Better Beta")
    page.locator("#save-name").click()
    expect(page.locator("#details-title")).to_have_text("Better Beta")

    page.fill("#notes-editor", "stable note")
    page.locator("#save-notes").click()
    expect(page.locator("#panel-status")).to_contain_text("Notes saved")

    page.locator("#refresh-button").click()
    expect(page.locator("#details-title")).to_have_text("Better Beta")
    expect(page.locator("#notes-editor")).to_have_value("stable note")


def test_history_mobile_layout_has_no_horizontal_overflow(panel, page):
    page.set_viewport_size({"width": 390, "height": 900})
    open_history(page, panel["url"])

    expect(page.locator("#folder-sidebar")).to_be_visible()
    expect(page.locator("#runs")).to_contain_text("Beta Foldered")
    expect(page.locator("#details-title")).to_be_visible()
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1


def test_deploy_tab_renders_report_and_controls(panel, page):
    open_deploy(page, panel["url"])

    expect(page.locator("#deploy-target")).to_have_value("Jetson ROS2")
    expect(page.locator("#deploy-readiness-badge")).to_contain_text("warn")
    expect(page.locator("#deploy-stage-list")).to_contain_text("Export Integrity")
    expect(page.locator("#deploy-stage-list")).to_contain_text("MuJoCo Readiness")
    expect(page.locator("#deploy-report-json")).to_contain_text('"pipeline_id": "deploy_fixture"')
    expect(page.locator("#deploy-validate-existing")).to_be_enabled()
    expect(page.locator("#deploy-export-validate")).to_be_enabled()
    expect(page.locator("#deploy-stop")).to_be_hidden()
    expect(page.locator("#deploy-console-live")).to_contain_text("Idle")


def test_deploy_mobile_layout_has_no_horizontal_overflow(panel, page):
    page.set_viewport_size({"width": 390, "height": 900})
    open_deploy(page, panel["url"])

    expect(page.locator("#deploy-stage-list")).to_contain_text("Export Integrity")
    expect(page.locator("#deploy-report-json")).to_be_visible()
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1


def test_status_toast_appears_for_actions_outside_history(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.evaluate("setStatus('Fixture status message.')")
    expect(page.locator("#panel-status .toast")).to_contain_text("Fixture status message.")


def test_error_toast_persists_and_closes_on_click(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.evaluate("setStatusTone('Fixture failure.', 'error')")
    toast = page.locator("#panel-status .toast.toast-error")
    expect(toast).to_contain_text("Fixture failure.")
    page.wait_for_timeout(7000)
    expect(toast).to_be_visible()  # errors must not auto-dismiss
    toast.locator(".toast-close").click()
    expect(page.locator("#panel-status .toast")).to_have_count(0)


def test_info_toast_auto_dismisses(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.evaluate("setStatus('Transient message.')")
    expect(page.locator("#panel-status .toast")).to_have_count(1)
    page.wait_for_timeout(7000)
    expect(page.locator("#panel-status .toast")).to_have_count(0)


def test_repeated_message_collapses_with_counter(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.evaluate("setStatus('Same message.'); setStatus('Same message.'); setStatus('Same message.')")
    expect(page.locator("#panel-status .toast")).to_have_count(1)
    expect(page.locator("#panel-status .toast-count")).to_contain_text("3")


def test_view_switch_updates_hash(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.click('.nav-button[data-view="terrain"]')
    expect(page).to_have_url(re.compile(r"#/terrain$"))


def test_run_selection_updates_hash_and_deep_link_restores(panel, page):
    open_history(page, panel["url"])
    page.click(".run-card")
    page.wait_for_function("location.hash.startsWith('#/history/')")
    deep_link = page.evaluate("location.href")

    page.goto(deep_link)
    page.wait_for_selector("#history.view.active")
    expect(page.locator("#details-subtitle")).not_to_contain_text("Select a run")


def test_unknown_hash_falls_back_to_train(panel, page):
    page.goto(f"{panel['url']}#/not-a-view")
    page.wait_for_selector("#train.view.active")
    expect(page.locator("#view-title")).to_contain_text("Train")


def test_dead_run_id_in_hash_opens_history_without_error(panel, page):
    page.goto(f"{panel['url']}#/history/run_that_does_not_exist")
    page.wait_for_selector("#history.view.active")
    expect(page.locator("#panel-status .toast")).to_have_count(0)


def test_malformed_hash_degrades_quietly(panel, page):
    page.goto(f"{panel['url']}#/history/%")
    # Applying the malformed route happens inside the boot promise chain,
    # after refreshAll()'s network calls settle; wait for that to finish
    # so the assertion below isn't racing the async error handling and
    # passing "by accident" while the toast is still on its way.
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#train.view.active")
    expect(page.locator("#panel-status .toast")).to_have_count(0)


def test_tooltip_is_reachable_by_keyboard(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.click('.nav-button[data-view="convergence"]')
    target = page.locator('#convergence-presets [data-preset="default"]')
    page.wait_for_selector("#convergence-presets")
    assert target.get_attribute("data-tooltip") is not None

    # Reach the target purely by keyboard so Chromium's :focus-visible
    # heuristic engages (locator.focus() is programmatic focus and never
    # matches :focus-visible). Land programmatic focus on the checkbox
    # immediately preceding the segmented control in DOM order, then walk
    # forward with real Tab keypresses until the target itself is focused.
    page.locator("#convergence-enabled").focus()
    for _ in range(10):
        page.keyboard.press("Tab")
        is_target = page.evaluate(
            "document.activeElement === document.querySelector('#convergence-presets [data-preset=\"default\"]')"
        )
        if is_target:
            break
    else:
        raise AssertionError("keyboard Tab traversal never reached the target element")

    # opacity is animated by a 160ms CSS transition; give it a moment to
    # settle before reading the computed value.
    page.wait_for_timeout(250)
    opacity = target.evaluate("(el) => getComputedStyle(el, '::after').opacity")
    assert float(opacity) > 0, f"tooltip not visible on keyboard focus (opacity={opacity})"

    # Sanity check: a mouse click on a sibling element (one that has never
    # received keyboard focus) does focus it, but Chromium does not treat
    # mouse-driven focus as focus-visible, so its tooltip must not appear.
    # This distinguishes the :focus-visible rule from a plain :focus rule
    # that would fire for keyboard and mouse alike. Using a fresh element
    # (rather than re-clicking `target`, which is already focus-visible from
    # the Tab traversal above and would keep that state on a same-element
    # click regardless of input modality) is what actually exercises the
    # mouse-focus code path.
    strict_button = page.locator('#convergence-presets [data-preset="strict"]')
    strict_button.click()
    is_focused = page.evaluate(
        "document.activeElement === document.querySelector('#convergence-presets [data-preset=\"strict\"]')"
    )
    assert is_focused, "mouse click did not focus the sibling element"
    click_opacity = strict_button.evaluate("(el) => getComputedStyle(el, '::after').opacity")
    assert float(click_opacity) == 0, (
        f"tooltip should not appear on mouse click focus (opacity={click_opacity})"
    )


def test_stop_controls_are_never_inside_the_menu(panel, page):
    open_history(page, panel["url"])
    menu_text = page.evaluate(
        "Array.from(document.querySelectorAll('.run-menu')).map((m) => m.textContent).join(' ')"
    )
    assert "Stop Training" not in menu_text
    assert "Stop Recording" not in menu_text
    assert "Cancel Queue" not in menu_text


def test_overflow_menu_opens_and_closes_on_escape(panel, page):
    open_history(page, panel["url"])
    trigger = page.locator(".run-card .run-menu-trigger").first
    trigger.click()
    expect(page.locator(".run-menu[data-open='true']")).to_have_count(1)
    expect(trigger).to_have_attribute("aria-expanded", "true")
    page.keyboard.press("Escape")
    expect(page.locator(".run-menu[data-open='true']")).to_have_count(0)
    expect(trigger).to_be_focused()


def test_overflow_menu_holds_secondary_actions(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card .run-menu-trigger").first.click()
    menu = page.locator(".run-menu[data-open='true']")
    expect(menu).to_contain_text("Resume to Train")
    expect(menu).to_contain_text("Tweak")


def test_open_menu_survives_a_runs_refresh(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card .run-menu-trigger").first.click()
    expect(page.locator(".run-menu[data-open='true']")).to_have_count(1)
    page.evaluate("loadRuns()")
    page.wait_for_timeout(500)
    expect(page.locator(".run-menu[data-open='true']")).to_have_count(1)
