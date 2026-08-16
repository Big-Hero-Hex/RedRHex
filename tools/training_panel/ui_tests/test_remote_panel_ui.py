import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright


REMOTE_WEB = Path(__file__).resolve().parents[1] / "remote_web"
CHROME = shutil.which("google-chrome")
pytestmark = pytest.mark.skipif(not CHROME, reason="google-chrome is required for Child UI tests")
ACTOR_ID = "11111111-1111-4111-8111-111111111111"
PROTOCOL = "3.7.0-remote-parity"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def child_url():
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(REMOTE_WEB)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.05)
    else:
        proc.terminate()
        raise RuntimeError("Child static server did not start")
    try:
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def mock_rows(role: str) -> dict[str, list[dict]]:
    now = datetime.now(timezone.utc).isoformat()
    run = {
        "id": "run-one",
        "machine_id": "biorolapc2-ubuntu",
        "status": "completed",
        "display_name": "Parity run",
        "folder": "Acceptance",
        "params": {"training_route": "standard", "spring_backend": "native", "max_iterations": 8},
        "latest_checkpoint": "/mother/logs/run-one/model_8.pt",
        "latest_video": "/mother/logs/run-one/videos/model_8.mp4",
        "onnx_path": "/mother/logs/run-one/exported/policy.onnx",
        "progress": {"iteration": 8, "max_iterations": 8, "percent": 100},
        "deploy_state": {
            "status": "completed",
            "overall_status": "pass",
            "readiness_level": "software-ready",
            "completed_at": now,
            "report": {"stages": [{"name": "contract", "label": "Contract", "status": "pass", "summary": "Dimensions match"}]},
        },
        "created_at": now,
        "updated_at": now,
    }
    return {
        "machines": [{
            "machine_id": "biorolapc2-ubuntu",
            "online": True,
            "accept_jobs": True,
            "gpu_locked": False,
            "panel_version": PROTOCOL,
            "remote_protocol_version": PROTOCOL,
            "heartbeat_at": now,
        }],
        "profiles": [{"id": ACTOR_ID, "email": f"{role}@example.com", "display_name": role.title(), "role": role}],
        "jobs": [],
        "runs": [run],
        "run_deletions": [],
        "artifacts": [
            {"id": "cp", "run_id": "run-one", "machine_id": "biorolapc2-ubuntu", "kind": "checkpoint", "local_path": run["latest_checkpoint"], "checkpoint_iteration": 8},
            {"id": "video", "run_id": "run-one", "machine_id": "biorolapc2-ubuntu", "kind": "video", "local_path": run["latest_video"], "storage_path": "runs/run-one/videos/model_8.mp4", "checkpoint_iteration": 8},
        ],
        "reward_presets": [{"id": "baseline", "name": "Baseline", "description": "Defaults", "values": {}, "built_in": True, "updated_at": now}],
        "terrain_presets": [{"id": "baseline", "name": "Baseline", "description": "Defaults", "values": {}, "built_in": True, "updated_at": now}],
        "physics_presets": [{"id": "baseline", "name": "Baseline", "description": "Defaults", "values": {}, "built_in": True, "updated_at": now}],
        "team_folders": [{"id": "folder", "machine_id": "biorolapc2-ubuntu", "name": "Acceptance", "folder_key": "acceptance"}],
        "notification_settings": [],
        "machine_capabilities": [{
            "machine_id": "biorolapc2-ubuntu",
            "protocol_version": PROTOCOL,
            "feature_flags": {"remote_deploy": True},
            "training_routes": ["standard", "sensor_v2_full", "sensor_v2_teacher", "sensor_v2_distillation", "sensor_v2_ppo"],
            "physics": {"field_schema": [{"key": "simulation_physics.main_drive.stiffness", "label": "Stiffness", "category": "Main drive actuator", "description": "Gain", "unit": "N m/rad", "step": 0.1, "min": 0, "max": 1000}]},
            "deploy": {"scenarios": ["stand_zero", "forward_mid"], "target": "Jetson ROS2", "mujoco_installed": True},
            "detection": {"enabled": True, "preset": "default", "divergence_enabled": True, "divergence_action": "notify"},
        }],
        "team_activity_events": [{"id": "event", "machine_id": "biorolapc2-ubuntu", "actor_name": "Operator", "actor_role": "operator", "event_type": "remote_job_completed", "category": "training", "outcome": "completed", "run_id": "run-one", "points": 4, "created_at": now}],
        "team_activity_member_7d": [{"actor_name": "Operator", "actor_role": "operator", "events": 1, "points": 4}],
        "team_activity_experiment_7d": [{"category": "training", "event_type": "remote_job_completed", "outcome": "completed", "events": 1, "points": 4}],
    }


def open_child(page, url: str, *, role: str = "operator", path: str = "") -> None:
    rows = mock_rows(role)
    page.add_init_script(
        """
        localStorage.setItem('redrhex_child_access_token', 'test-token');
        localStorage.setItem('redrhex_child_refresh_token', 'refresh-token');
        localStorage.setItem('redrhex_machine_id', 'biorolapc2-ubuntu');
        """
    )

    def route_api(route):
        request_url = route.request.url
        if "/auth/v1/user" in request_url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"id": ACTOR_ID, "email": f"{role}@example.com"}))
            return
        if "/rest/v1/rpc/" in request_url:
            route.fulfill(status=200, content_type="application/json", body="[]")
            return
        marker = "/rest/v1/"
        if marker in request_url:
            table = request_url.split(marker, 1)[1].split("?", 1)[0]
            if route.request.method in {"POST", "PATCH", "DELETE"}:
                payload = json.loads(route.request.post_data or "{}") if route.request.post_data else {}
                response = [{**payload, "id": payload.get("id", "job-mock"), "status": payload.get("status", "queued")}]
            else:
                response = rows.get(table, [])
            route.fulfill(status=200, content_type="application/json", body=json.dumps(response))
            return
        if "jsdelivr.net" in request_url:
            route.abort()
            return
        route.continue_()

    page.route("**/*", route_api)
    page.goto(f"{url}/{path}")
    expect(page.locator("#child-release-badge")).to_contain_text("3.7.0")
    expect(page.locator("#protocol-mode-badge")).to_have_text("read-write")


@pytest.mark.parametrize("width", [390, 768, 1440])
def test_child_has_no_horizontal_overflow_at_acceptance_widths(child_url, width):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": width, "height": 900})
        open_child(page, child_url)
        if width <= 820:
            expect(page.locator(".mobile-nav")).to_be_visible()
            expect(page.locator(".desktop-sidebar")).to_be_hidden()
        else:
            expect(page.locator(".desktop-sidebar")).to_be_visible()
            expect(page.locator(".mobile-nav")).to_be_hidden()
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 1
        browser.close()


def test_child_deep_links_train_routes_more_views_and_themes(child_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 900})
        open_child(page, child_url, path="?view=deploy&run=run-one")
        expect(page.locator("h2", has_text="Remote-safe Deploy")).to_be_visible()
        page.locator('.mobile-nav [data-action="view"][data-view="train"]').click()
        page.select_option("#training-route", "sensor_v2_full")
        expect(page.locator("#teacher-iterations")).to_be_visible()
        expect(page.locator("#f0-evidence")).to_be_visible()
        expect(page.locator("#task")).to_be_hidden()
        page.select_option("#training-route", "sensor_v2_distillation")
        expect(page.locator("#resume-run")).to_be_visible()
        page.locator('[data-action="toggle-mobile-more"]').click()
        expect(page.locator(".mobile-more-sheet")).to_be_visible()
        page.locator('.mobile-more-sheet [data-view="physics"]').click()
        expect(page.locator("h2", has_text="Physics Calibration")).to_be_visible()
        page.locator('[data-action="toggle-theme"]').click()
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")
        browser.close()


def test_dark_theme_selected_cards_keep_dark_contrast_surface(child_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        open_child(page, child_url, path="?view=history&run=run-one")
        page.evaluate("document.documentElement.dataset.theme = 'dark'")
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        page.wait_for_timeout(200)

        expect(page.locator(".run-card.active")).to_be_visible()
        run_colors = page.locator(".run-card.active").evaluate(
            """element => ({
                theme: document.documentElement.dataset.theme,
                token: getComputedStyle(document.documentElement).getPropertyValue('--accent-soft').trim(),
                background: getComputedStyle(element).backgroundColor,
            })"""
        )
        assert run_colors == {
            "theme": "dark",
            "token": "#122a2e",
            "background": "rgb(18, 42, 46)",
        }

        page.locator('.sidebar-link[data-view="physics"]').click()
        expect(page.locator(".preset-button.active")).to_be_visible()
        assert page.locator(".preset-button.active").evaluate(
            "element => getComputedStyle(element).backgroundColor"
        ) == "rgb(18, 42, 46)"
        browser.close()


@pytest.mark.parametrize("role,queue_disabled,delete_disabled", [("viewer", True, True), ("operator", False, True), ("admin", False, False)])
def test_child_role_gates_mutations(child_url, role, queue_disabled, delete_disabled):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        open_child(page, child_url, role=role, path="?view=train")
        assert page.locator('[data-action="queue-training"]').is_disabled() is queue_disabled
        page.locator('.sidebar-link[data-view="history"]').click()
        expect(page.locator(".run-details")).to_be_visible()
        assert page.locator('[data-action="delete-run"]').is_disabled() is delete_disabled
        browser.close()
