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
    # The Physics robot preview parses the real URDF, so the fixture root needs the
    # robot description alongside the fake run logs. Symlinked rather than copied: it is
    # read-only to the panel, and copying the meshes would cost megabytes per test.
    description = REPO_ROOT / "test_7_description"
    if description.exists():
        (root / "test_7_description").symlink_to(description, target_is_directory=True)

    log_root = root / "logs" / "rsl_rl" / "redrhex_wheg"
    alpha = write_run(log_root, "run_alpha", (0,))
    beta = write_run(log_root, "run_beta", (0, 10))
    (beta / "params").mkdir()
    (beta / "params" / "env.yaml").write_text("env: test\n", encoding="utf-8")
    (beta / "params" / "torsion_spring.yaml").write_text("spring_backend: native\n", encoding="utf-8")
    (beta / "videos" / "play").mkdir(parents=True)
    old_video = beta / "videos" / "play" / "model_0_video_fixture.mp4"
    latest_video = beta / "videos" / "play" / "rl-video-step-0.mp4"
    old_video.write_text("old video", encoding="utf-8")
    latest_video.write_text("latest video", encoding="utf-8")
    os.utime(old_video, (100, 100))
    os.utime(latest_video, (200, 200))
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
        yield {"url": url, "root": tmp_path, "proc": proc, **paths}
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


def open_physics(page, url: str) -> None:
    page.goto(url)
    page.locator('.nav-button[data-view="physics"]').click()
    page.wait_for_selector('[data-physics-key="simulation_physics.mass.scale"]')


def autopilot_snapshot(state: str = "draft", revision: int = 1) -> dict:
    return {
        "schema_version": "redrhex.autopilot.campaign.v1",
        "id": "campaign_demo",
        "revision": revision,
        "state": state,
        "goal": {
            "schema_version": "redrhex.autopilot.goal.v1",
            "description": "Run diagonally without falling",
            "task": "Template-Redrhex-Direct-v0",
            "stage": 3,
            "evaluation_profile": "stage3",
            "gait": "run",
            "directions": ["forward_left", "forward_right"],
            "command_envelope": {
                "vx": [[0.47, 0.60]],
                "vy": [[-0.48, -0.38], [0.38, 0.48]],
                "wz": [[0.0, 0.0]],
            },
            "skill_gates": {
                "min_command_pass_ratio": 0.70,
                "min_skill_pass_ratio": 0.60,
                "max_fall_rate": 0.20,
                "min_tracking_quality": 0.0,
                "min_stability_quality": 0.0,
                "min_direction_sign_ratio": 0.70,
                "max_linear_leak": 0.18,
                "max_yaw_leak": 0.35,
                "max_energy_per_distance": 500.0,
            },
            "baseline_run_id": None,
            "baseline_checkpoint_iteration": None,
            "physics_profile_sha256": "a" * 64,
            "spring_profile_sha256": "b" * 64,
            "code_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "command_profile_sha256": "e" * 64,
            "budget": {"max_training_trials": 24, "max_gpu_hours": 72},
        },
        "leader": {"candidate_id": "candidate_1", "evaluation_id": "eval_1"},
        "budget": {
            "max_training_trials": 24,
            "max_gpu_hours": 72,
            "training_trials_used": 2,
            "gpu_hours_used": 1.5,
            "remaining_training_trials": 22,
            "remaining_gpu_hours": 70.5,
        },
        "active_process": None,
        "candidate_lineage": [
            {
                "candidate_id": "candidate_1",
                "reward_key": "v2_reward_scales.velocity_tracking",
                "proposed_value": 1.1,
                "delta": 0.1,
                "state": "completed",
                "seed": 42,
                "rank": 1,
            }
        ],
        "decisions": [{"hypothesis": "More XY tracking should improve the diagonal gate."}],
        "evaluations": [
            {
                "id": "eval_1",
                "trial_id": "candidate_1",
                "seed": 42,
                "hard_gates": {"passed": True},
                "soft_ranking": {"tracking_quality": 0.82, "energy": 4.2},
                "artifact_ids": ["command_csv_1"],
            }
        ],
        "connector": {
            "last_heartbeat_at": "2026-08-14T10:00:00Z",
            "polls_used": 4,
            "max_polls": 300,
        },
        "terminal_reason": None,
        "next_permitted_actions": ["arm"] if state == "draft" else ["pause", "stop"],
        "created_at": "2026-08-14T09:00:00Z",
        "updated_at": "2026-08-14T10:00:00Z",
    }


def install_autopilot_routes(page, campaigns: list[dict], writes: list[dict]) -> None:
    capabilities = {
        "schema_version": "redrhex.autopilot.capabilities.v1",
        "enabled": True,
        "max_armed_campaigns": 1,
        "default_reward_keys": {
            "Template-Redrhex-Direct-v0": {
                "stage3": ["v2_reward_scales.velocity_tracking"],
            }
        },
        "reward_catalog": {
            "Template-Redrhex-Direct-v0": {
                "stage3": [
                    {
                        "key": "v2_reward_scales.velocity_tracking",
                        "description": "Command velocity tracking",
                        "start_value": 6.0,
                        "minimum": 4.8,
                        "maximum": 7.2,
                        "enabled": True,
                    }
                ],
            }
        },
        "command_profiles": {
            "Template-Redrhex-Direct-v0": {
                "stage3": {
                    "walk": {
                        "vx": [[0.34, 0.47]],
                        "vy": [[-0.38, -0.28], [0.28, 0.38]],
                        "wz": [[0.0, 0.0]],
                    },
                    "run": {
                        "vx": [[0.47, 0.60]],
                        "vy": [[-0.48, -0.38], [0.38, 0.48]],
                        "wz": [[0.0, 0.0]],
                    },
                },
            }
        },
        "default_skill_gates": {
            "min_command_pass_ratio": 0.70,
            "min_skill_pass_ratio": 0.60,
            "max_fall_rate": 0.20,
            "min_tracking_quality": 0.0,
            "min_stability_quality": 0.0,
            "min_direction_sign_ratio": 0.70,
            "max_linear_leak": 0.18,
            "max_yaw_leak": 0.35,
            "max_energy_per_distance": 500.0,
        },
    }

    def handle(route):
        request = route.request
        path = request.url.split("?", 1)[0].split("/api/autopilot", 1)[-1]
        if path == "/capabilities":
            route.fulfill(status=200, content_type="application/json", body=json.dumps(capabilities))
            return
        if path == "/campaigns" and request.method == "GET":
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"campaigns": campaigns}))
            return
        if path == "/campaigns" and request.method == "POST":
            writes.append(
                {
                    "path": path,
                    "headers": request.headers,
                    "payload": request.post_data_json,
                }
            )
            created = autopilot_snapshot()
            campaigns[:] = [created]
            route.fulfill(status=201, content_type="application/json", body=json.dumps({"campaign": created}))
            return
        if path == "/campaigns/campaign_demo":
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"campaign": campaigns[0]}))
            return
        if path == "/campaigns/campaign_demo/decision-context":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "campaign_id": "campaign_demo",
                        "revision": campaigns[0]["revision"],
                        "state": campaigns[0]["state"],
                        "next_permitted_actions": campaigns[0]["next_permitted_actions"],
                        "eligible_move_count": 6,
                        "recent_evidence": ["eval_1"],
                        "recent_decisions": campaigns[0]["decisions"],
                    }
                ),
            )
            return
        if path == "/campaigns/campaign_demo/events":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "events": [
                            {
                                "sequence": 1,
                                "type": "campaign_created",
                                "created_at": "2026-08-14T09:00:00Z",
                            }
                        ]
                    }
                ),
            )
            return
        if path == "/campaigns/campaign_demo/artifacts":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"artifacts": [{"id": "command_csv_1", "kind": "command CSV", "sha256": "abc123"}]}),
            )
            return
        action = path.rsplit("/", 1)[-1]
        if action in {"arm", "pause", "resume", "stop"} and request.method == "POST":
            writes.append(
                {
                    "path": path,
                    "headers": request.headers,
                    "payload": request.post_data_json,
                }
            )
            next_state = {"arm": "armed", "pause": "paused", "resume": "armed", "stop": "stopped"}[action]
            next_actions = {
                "armed": ["pause", "stop"],
                "paused": ["resume", "stop"],
                "stopped": [],
            }[next_state]
            campaigns[0] = {
                **campaigns[0],
                "revision": campaigns[0]["revision"] + 1,
                "state": next_state,
                "next_permitted_actions": next_actions,
            }
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"campaign": campaigns[0]}))
            return
        route.fulfill(status=404, content_type="application/json", body=json.dumps({"error": f"unhandled {path}"}))

    page.route("**/api/autopilot/**", handle)


def test_autopilot_is_hidden_when_capability_is_disabled(panel, page):
    page.route(
        "**/api/autopilot/capabilities",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"enabled": False}),
        ),
    )
    page.goto(panel["url"])
    expect(page.locator("#autopilot-nav")).to_be_hidden()


def test_autopilot_goal_draft_uses_exact_envelope_and_idempotency(panel, page):
    campaigns = []
    writes = []
    install_autopilot_routes(page, campaigns, writes)
    page.goto(panel["url"])

    nav = page.locator("#autopilot-nav")
    expect(nav).to_be_visible()
    nav.click()
    expect(page.locator("#autopilot-workspace")).to_be_visible()
    page.select_option("#autopilot-task", "Template-Redrhex-Direct-v0")
    page.select_option("#autopilot-stage", "3")
    page.select_option("#autopilot-gait", "run")
    assert page.locator('#autopilot-gait option[value="explicit"]').count() == 0
    assert page.evaluate("autopilotDirections(2)") == ["left", "right"]
    assert page.evaluate("autopilotDirections(4)") == ["yaw_ccw", "yaw_cw"]
    expect(page.locator("#autopilot-command-preview")).to_contain_text("vx [0.47, 0.6]")
    expect(page.locator("#autopilot-command-intervals")).to_contain_text("[-0.48, -0.38] ∪ [0.38, 0.48]")

    page.fill("#autopilot-goal-description", "Run diagonally without falling")
    page.fill("#autopilot-iterations", "1500")
    page.fill('input[name="autopilot_reward_min"]', "5.4")
    page.fill('input[name="autopilot_reward_max"]', "6.6")
    page.locator("#autopilot-create").click()
    expect(page.locator("#autopilot-campaign-title")).to_contain_text("Run diagonally")

    create = next(item for item in writes if item["path"] == "/campaigns")
    assert create["headers"]["idempotency-key"].startswith("panel-create-")
    assert create["headers"]["if-match"] == '"0"'
    assert create["payload"]["expected_revision"] == 0
    assert create["payload"]["schema_version"] == "redrhex.autopilot.goal.v1"
    assert create["payload"]["evaluation_profile"] == "stage3"
    assert create["payload"]["gait"] == "run"
    assert create["payload"]["directions"] == ["forward_left", "forward_right"]
    assert create["payload"]["command_envelope"] == {
        "vx": [[0.47, 0.60]],
        "vy": [[-0.48, -0.38], [0.38, 0.48]],
        "wz": [[0.0, 0.0]],
    }
    assert create["payload"]["budget"] == {
        "max_training_trials": 24,
        "max_gpu_hours": 72,
    }
    assert create["payload"]["skill_gates"]["max_fall_rate"] == 0.20
    assert "target_gates" not in create["payload"]
    assert "checkpoint_path" not in create["payload"]
    assert create["payload"]["baseline_checkpoint_iteration"] is None
    assert "physics_profile_sha256" not in create["payload"]
    assert "command_profile_sha256" not in create["payload"]
    assert create["payload"]["tunable_reward_keys"] == ["v2_reward_scales.velocity_tracking"]
    assert create["payload"]["reward_bounds"] == {
        "v2_reward_scales.velocity_tracking": [5.4, 6.6]
    }


def test_autopilot_campaign_renders_evidence_and_sends_revision(panel, page):
    campaigns = [autopilot_snapshot()]
    writes = []
    install_autopilot_routes(page, campaigns, writes)
    page.goto(f"{panel['url']}#/autopilot")
    page.locator('[data-autopilot-campaign-id="campaign_demo"]').click()

    expect(page.locator("#autopilot-candidate-rows")).to_contain_text("v2_reward_scales.velocity_tracking")
    expect(page.locator("#autopilot-evaluation-rows")).to_contain_text("command_csv_1")
    expect(page.locator("#autopilot-event-list")).to_contain_text("campaign created")
    expect(page.locator("#autopilot-artifact-list")).to_contain_text("command CSV")
    expect(page.locator("#autopilot-connector-state")).to_contain_text("4/300 polls")
    expect(page.locator('[data-autopilot-action="stop-after-current"]')).to_be_visible()

    page.locator('[data-autopilot-action="arm"]').click()
    expect(page.locator("#confirm-dialog")).to_be_visible()
    expect(page.locator("#confirm-dialog-body")).to_contain_text("vx [0.47, 0.6]")
    expect(page.locator("#confirm-dialog-body")).to_contain_text("vy [-0.48, -0.38] ∪ [0.38, 0.48]")
    expect(page.locator("#confirm-dialog-body")).to_contain_text("Training trial budget: 24")
    expect(page.locator("#confirm-dialog-body")).to_contain_text("Active GPU-hour budget: 72")
    page.locator("#confirm-dialog-confirm").click()
    expect(page.locator("#autopilot-campaign-badge")).to_contain_text("armed")
    arm = next(item for item in writes if item["path"].endswith("/arm"))
    assert arm["payload"] == {"expected_revision": 1}
    assert arm["headers"]["idempotency-key"].startswith("panel-arm-")
    assert arm["headers"]["if-match"] == '"1"'

    page.locator('[data-autopilot-action="stop"]').click()
    expect(page.locator("#confirm-dialog-title")).to_have_text("Emergency Stop Campaign Work")
    expect(page.locator("#confirm-dialog-body")).to_contain_text("Unrelated Training Panel processes are never stopped")
    page.locator("#confirm-dialog-confirm").click()
    expect(page.locator("#autopilot-campaign-badge")).to_contain_text("stopped")
    stop = next(item for item in writes if item["path"].endswith("/stop"))
    assert stop["payload"] == {"expected_revision": 2, "mode": "emergency"}
    assert stop["headers"]["if-match"] == '"2"'


def test_training_route_only_shows_relevant_controls(panel, page):
    page.goto(panel["url"])

    route = page.locator("#training-route")
    single_iterations = page.locator("#single-stage-iterations-field")
    pipeline_iterations = page.locator(".sensor-v2-pipeline-field")
    checkpoint = page.locator("#training-checkpoint-field")
    task = page.locator("#training-task-field")
    reward_preset = page.locator("#train-reward-preset")
    terrain_preset = page.locator("#train-terrain-preset")
    physics_preset = page.locator("#train-physics-preset")

    expect(route).to_have_value("standard")
    expect(page.locator("#training-route-summary-title")).to_have_text("Standard PPO")
    expect(task).to_be_visible()
    expect(single_iterations).to_be_visible()
    expect(single_iterations).to_contain_text("Iterations")
    for field in pipeline_iterations.all():
        expect(field).to_be_hidden()
    expect(checkpoint).to_be_visible()
    expect(reward_preset).to_be_visible()
    expect(terrain_preset).to_be_visible()
    expect(physics_preset).to_be_visible()

    route.select_option("sensor_v2_full")
    expect(page.locator("#training-route-summary-title")).to_have_text("Full Sensor V2 Pipeline")
    expect(task).to_be_hidden()
    expect(single_iterations).to_be_hidden()
    for field in pipeline_iterations.all():
        expect(field).to_be_visible()
    expect(checkpoint).to_be_hidden()
    expect(reward_preset).to_be_hidden()
    expect(terrain_preset).to_be_hidden()
    expect(physics_preset).to_be_visible()
    page.locator("#smoke-button").click()
    expect(page.locator('input[name="teacher_iterations"]')).to_have_value("1")
    expect(page.locator('input[name="distillation_iterations"]')).to_have_value("1")
    expect(page.locator('input[name="ppo_iterations"]')).to_have_value("1")
    full_payload = page.evaluate("formData(document.querySelector('#train-form'))")
    assert "max_iterations" not in full_payload
    assert full_payload["teacher_iterations"] == 1
    assert full_payload["distillation_iterations"] == 1
    assert full_payload["ppo_iterations"] == 1
    assert "task" not in full_payload
    assert "reward_preset_id" not in full_payload
    assert "reward_overrides" not in full_payload
    assert "terrain_preset_id" not in full_payload
    assert "terrain_overrides" not in full_payload

    route.select_option("sensor_v2_teacher")
    expect(single_iterations).to_be_visible()
    expect(page.locator("#single-stage-iterations-label")).to_have_text("F1 Teacher Iterations")
    expect(checkpoint).to_be_visible()
    expect(page.locator('input[name="checkpoint"]')).not_to_have_attribute("required", "")

    route.select_option("sensor_v2_distillation")
    expect(page.locator("#single-stage-iterations-label")).to_have_text("F2 Distillation Iterations")
    expect(page.locator("#training-checkpoint-label")).to_have_text("Teacher Checkpoint")
    expect(page.locator('input[name="checkpoint"]')).to_have_attribute("required", "")
    stage_payload = page.evaluate("formData(document.querySelector('#train-form'))")
    assert "teacher_iterations" not in stage_payload
    assert "distillation_iterations" not in stage_payload
    assert "ppo_iterations" not in stage_payload

    route.select_option("sensor_v2_ppo")
    expect(page.locator("#single-stage-iterations-label")).to_have_text("F3 Student PPO Iterations")
    expect(page.locator("#training-checkpoint-label")).to_have_text("Distilled Student Checkpoint")
    expect(page.locator('input[name="checkpoint"]')).to_have_attribute("required", "")


def test_history_defaults_to_all_runs_and_filters(panel, page):
    open_history(page, panel["url"])

    all_class = (
        page.locator('.folder-item:has(.folder-select[data-folder="__all__"])').get_attribute("class") or ""
    )
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
    page.locator('.folder-select[data-folder="Good Runs"]').click()
    expect(page.locator("#runs")).to_contain_text("Beta Foldered")
    expect(page.locator("#runs")).not_to_contain_text("run_alpha")

    page.locator('.folder-select[data-folder="__all__"]').click()
    expect(page.locator("#runs")).to_contain_text("run_alpha")


def test_history_checkpoint_evolution_is_manual_and_records_selected_save_point(panel, page):
    recorded_requests = []

    def record_selected(route, request):
        recorded_requests.append(json.loads(request.post_data or "{}"))
        route.fulfill(
            status=201,
            content_type="application/json",
            body=json.dumps({"id": "video_fixture", "checkpoint_iteration": 0}),
        )

    page.route("**/api/runs/panel_run_beta/record-video", record_selected)
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()

    evolution = page.locator("#checkpoint-evolution")
    expect(evolution).to_be_visible()
    expect(evolution).not_to_have_attribute("open", "")
    expect(page.locator("#checkpoint-timeline")).not_to_be_visible()
    expect(page.locator("#video-state")).to_have_text("Latest Video")
    expect(page.locator("#result-video")).not_to_have_attribute("src", re.compile("checkpoint_iteration"))
    expect(page.locator("#record-video")).to_have_text("Record Latest")

    evolution.locator("summary").click()
    expect(page.locator("#checkpoint-timeline")).to_be_visible()
    expect(page.locator("#checkpoint-timeline")).to_contain_text("Iteration 10")
    expect(page.locator("#checkpoint-timeline")).to_contain_text("Iteration 0")
    page.locator('[data-checkpoint-iteration="0"]').click()

    expect(page.locator("#video-state")).to_have_text("Iter 0 Video")
    expect(page.locator("#result-video")).to_have_attribute("src", re.compile("checkpoint_iteration=0"))
    expect(page.locator("#record-video")).to_have_text("Record Iter 0")
    page.locator("#record-video").click()
    expect(page.locator("#panel-status")).to_contain_text("Recording iteration 0")
    assert recorded_requests == [{"device": "cuda:0", "checkpoint_iteration": 0}]

    page.locator("#show-latest-video").click()
    expect(page.locator("#video-state")).to_have_text("Latest Video")
    expect(page.locator("#result-video")).not_to_have_attribute("src", re.compile("checkpoint_iteration"))
    expect(page.locator("#record-video")).to_have_text("Record Latest")


def test_history_drive_export_explains_missing_training_pc_setup(panel, page):
    page.route(
        "**/api/system",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "google_drive_export": {
                        "available": False,
                        "configured": False,
                        "remote": "redrhex-drive:",
                        "folder": "RedRHex Videos",
                        "remediation": "Install rclone and create the redrhex-drive remote.",
                    }
                }
            ),
        ),
    )
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()

    expect(page.locator("#export-video-drive")).to_be_visible()
    expect(page.locator("#export-video-drive")).to_be_disabled()
    expect(page.locator("#drive-export-hint")).to_contain_text("Install rclone")


def test_history_drive_export_tracks_checkpoint_background_success_and_retry(panel, page):
    requests = []

    page.route(
        "**/api/system",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "google_drive_export": {
                        "available": True,
                        "configured": True,
                        "remote": "redrhex-drive:",
                        "folder": "RedRHex Videos",
                        "remediation": "",
                    }
                }
            ),
        ),
    )

    def export_video(route, request):
        payload = json.loads(request.post_data or "{}")
        requests.append(payload)
        route.fulfill(
            status=202,
            content_type="application/json",
            body=json.dumps(
                {
                    "run_id": "panel_run_beta",
                    "checkpoint_iteration": payload.get("checkpoint_iteration"),
                    "started": True,
                    "deduplicated": False,
                    "export": {
                        "status": "uploading",
                        "remote_path": "redrhex-drive:RedRHex Videos/panel_run_beta/video.mp4",
                    },
                }
            ),
        )

    page.route("**/api/runs/panel_run_beta/export-video-to-drive", export_video)
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()
    page.locator("#checkpoint-evolution summary").click()
    page.locator('[data-checkpoint-iteration="0"]').click()

    expect(page.locator("#export-video-drive")).to_be_enabled()
    page.locator("#export-video-drive").click()
    expect(page.locator("#export-video-drive")).to_have_text("Exporting…")
    expect(page.locator("#export-video-drive")).to_be_disabled()
    expect(page.locator("#drive-export-hint")).to_contain_text("Uploading in the background")
    assert requests == [{"checkpoint_iteration": 0}]

    page.evaluate(
        """() => {
          const run = state.selectedRun;
          const key = displayedVideoRelativePath(run);
          run.google_drive_video_exports[key] = {
            status: 'completed',
            web_view_url: 'https://drive.google.com/file/d/private-file-id/view',
          };
          renderVideoPanel(run);
        }"""
    )
    expect(page.locator("#export-video-drive")).to_have_text("Export to Drive")
    expect(page.locator("#export-video-drive")).to_be_enabled()
    expect(page.locator("#open-video-drive")).to_be_visible()
    expect(page.locator("#open-video-drive")).to_have_attribute(
        "href", "https://drive.google.com/file/d/private-file-id/view"
    )
    expect(page.locator("#open-video-drive")).to_have_attribute("rel", "noopener noreferrer")

    page.locator("#show-latest-video").click()
    page.locator("#export-video-drive").click()
    assert requests == [{"checkpoint_iteration": 0}, {}]
    expect(page.locator("#export-video-drive")).to_have_text("Exporting…")
    page.evaluate(
        """() => {
          const run = state.selectedRun;
          const key = displayedVideoRelativePath(run);
          run.google_drive_video_exports[key] = { status: 'failed', error: 'Drive quota exceeded.' };
          renderVideoPanel(run);
        }"""
    )
    expect(page.locator("#export-video-drive")).to_have_text("Retry Drive Export")
    expect(page.locator("#export-video-drive")).to_be_enabled()
    expect(page.locator("#drive-export-hint")).to_contain_text("Drive quota exceeded")


def test_settings_organizes_control_center_and_saves_drive_location(panel, page):
    saved_requests = []
    drive_status = {
        "available": True,
        "configured": True,
        "remote": "redrhex-drive:",
        "folder": "RedRHex Videos",
        "destination_mode": "my_drive_path",
        "destination_display": "RedRHex Videos",
        "folder_url": "",
        "destination_revision": 1,
        "reconnect": {"status": "idle", "started_at": "", "finished_at": "", "error": ""},
        "reconnect_command": "rclone config reconnect redrhex-drive:",
        "remediation": "",
    }

    page.route(
        "**/api/system",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"google_drive_export": drive_status}),
        ),
    )

    def save_location(route, request):
        payload = json.loads(request.post_data or "{}")
        saved_requests.append(payload)
        updated = {
            **drive_status,
            "folder": "",
            "destination_mode": "folder_link",
            "destination_display": "Linked Google Drive folder",
            "folder_url": "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp",
            "destination_revision": 2,
        }
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"saved": True, "google_drive_export": updated}),
        )

    page.route("**/api/google-drive/settings", save_location)
    page.goto(panel["url"])
    page.locator('.nav-button[data-view="access"]').click()

    expect(page.locator("#view-title")).to_have_text("Settings")
    expect(page.locator(".settings-section-nav")).to_be_visible()
    expect(page.locator("#settings-panel-remote")).to_be_visible()
    expect(page.locator("#settings-panel-drive")).not_to_be_visible()

    page.locator("#settings-tab-drive").click()
    expect(page.locator("#settings-panel-drive")).to_be_visible()
    expect(page.locator("#settings-panel-remote")).not_to_be_visible()
    expect(page.locator("#drive-settings-badge")).to_have_text("Connected")
    expect(page.locator("#drive-destination-folder")).to_have_value("RedRHex Videos")

    folder_link = "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp?usp=sharing"
    page.locator("#drive-destination-folder").fill(folder_link)
    expect(page.locator("#drive-destination-preview")).to_have_text("Pasted Google Drive folder")
    expect(page.locator("#drive-destination-preview")).not_to_contain_text("redrhex-drive")
    expect(page.locator("#drive-save-location")).to_be_enabled()
    page.locator("#drive-save-location").click()

    expect(page.locator("#panel-status")).to_contain_text("Video folder updated")
    expect(page.locator("#drive-open-folder")).to_be_visible()
    expect(page.locator("#drive-open-folder")).to_have_attribute(
        "href", "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp"
    )
    assert saved_requests == [{"destination": folder_link}]


def test_settings_starts_drive_account_reconnect_without_exposing_credentials(panel, page):
    reconnect_requests = []
    drive_status = {
        "available": True,
        "configured": True,
        "remote": "redrhex-drive:",
        "folder": "RedRHex Videos",
        "destination_mode": "my_drive_path",
        "destination_display": "RedRHex Videos",
        "folder_url": "",
        "destination_revision": 1,
        "reconnect": {"status": "idle", "started_at": "", "finished_at": "", "error": ""},
        "reconnect_command": "rclone config reconnect redrhex-drive:",
        "remediation": "",
    }
    page.route(
        "**/api/system",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"google_drive_export": drive_status}),
        ),
    )

    def reconnect(route, request):
        reconnect_requests.append(json.loads(request.post_data or "{}"))
        authorizing = {
            **drive_status,
            "reconnect": {
                "status": "authorizing",
                "started_at": "2026-08-14T10:00:00+00:00",
                "finished_at": "",
                "error": "",
            },
        }
        route.fulfill(
            status=202,
            content_type="application/json",
            body=json.dumps(
                {
                    "started": True,
                    "reconnect": authorizing["reconnect"],
                    "google_drive_export": authorizing,
                }
            ),
        )

    page.route("**/api/google-drive/reconnect", reconnect)
    page.goto(panel["url"])
    page.locator('.nav-button[data-view="access"]').click()
    page.locator("#settings-tab-drive").click()
    expect(page.locator("#drive-reconnect-account")).to_have_text("Change Google account")
    page.locator("#drive-reconnect-account").click()

    expect(page.locator("#confirm-dialog")).to_be_visible()
    expect(page.locator("#confirm-dialog-body")).to_contain_text("Existing Drive files")
    page.locator("#confirm-dialog-confirm").click()

    expect(page.locator("#drive-reconnect-state")).to_contain_text("Google sign-in window is open")
    expect(page.locator("#drive-account-title")).to_have_text("Waiting for Google sign-in")
    expect(page.locator("#drive-settings-badge")).to_have_text("Authorizing")
    expect(page.locator("#drive-reconnect-command")).to_have_text("rclone config reconnect redrhex-drive:")
    assert reconnect_requests == [{}]
    assert "token" not in page.locator("#settings-panel-drive").inner_text().lower()


def test_settings_mobile_layout_has_no_horizontal_overflow(panel, page):
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(panel["url"])
    page.locator('.nav-button[data-view="access"]').click()
    page.locator("#settings-tab-drive").click()

    expect(page.locator("#settings-panel-drive")).to_be_visible()
    dimensions = page.evaluate(
        """() => ({
          viewport: document.documentElement.clientWidth,
          page: document.documentElement.scrollWidth,
          navClient: document.querySelector('.settings-section-nav').clientWidth,
          navScroll: document.querySelector('.settings-section-nav').scrollWidth,
        })"""
    )
    assert dimensions["page"] <= dimensions["viewport"]
    assert dimensions["navScroll"] >= dimensions["navClient"]


def test_history_shows_evolution_entry_for_a_single_checkpoint(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="run_alpha").click()

    expect(page.locator("#video-panel")).to_be_visible()
    expect(page.locator("#video-panel h3")).to_have_text("Recorded Video & Evolution")
    expect(page.locator("#checkpoint-evolution")).to_be_visible()
    expect(page.locator("#checkpoint-evolution-count")).to_have_text("1 save point")
    expect(page.locator("#checkpoint-evolution-help")).to_contain_text("One checkpoint is saved")


def test_history_evolution_explains_gpu_lock_and_keeps_long_list_position(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()
    page.locator("#checkpoint-evolution summary").click()

    scroll = page.evaluate(
        """
        () => {
        const run = state.selectedRun;
        run.checkpoint_history = Array.from({ length: 51 }, (_, index) => ({
          iteration: index * 50,
          created_at: '2026-05-20T10:00:00',
          is_latest: index === 50,
          video: null,
        }));
        renderVideoPanel(run);
        const timeline = document.querySelector('#checkpoint-timeline');
        const target = timeline.querySelector('[data-checkpoint-iteration="2000"]');
        timeline.scrollTop = target.offsetTop - timeline.offsetTop - 40;
        const before = timeline.scrollTop;
        selectCheckpointForVideo(2000);
        return {
          before,
          after: timeline.scrollTop,
          overflowY: getComputedStyle(timeline).overflowY,
        };
        }
        """
    )
    assert scroll["before"] > 0
    assert abs(scroll["after"] - scroll["before"]) <= 1
    assert scroll["overflowY"] == "auto"
    expect(page.locator('[data-checkpoint-iteration="2000"]')).to_have_class(re.compile("active"))

    page.evaluate(
        """
        state.activeProcesses = [{ kind: 'training', run_id: 'active_training' }];
        renderVideoPanel(state.selectedRun);
        """
    )
    expect(page.locator("#record-video")).to_be_disabled()
    expect(page.locator("#record-video-hint")).to_be_visible()
    expect(page.locator("#record-video-hint")).to_contain_text("GPU busy with training")
    assert page.locator("#record-video").get_attribute("data-tooltip") is None


def test_physics_preset_is_searchable_sparse_and_persistent(panel, page):
    open_physics(page, panel["url"])
    expect(page.locator("#physics-status")).to_contain_text("113 validated")
    expect(page.locator('[data-physics-key="simulation_physics.mass.scale"]')).to_be_disabled()

    page.once("dialog", lambda dialog: dialog.accept("Measured Lab Robot"))
    page.locator("#physics-preset-duplicate-btn").click()
    expect(page.locator("#physics-profile-name")).to_have_value("Measured Lab Robot")

    page.fill("#physics-search", "mass scale")
    mass_scale = page.locator('[data-physics-key="simulation_physics.mass.scale"]')
    expect(mass_scale).to_be_visible()
    mass_scale.fill("1.05")
    expect(page.locator("#physics-change-summary")).to_have_text("1 override")
    page.locator("#physics-preset-save-btn").click()
    expect(page.locator("#physics-status")).to_have_text("Physics preset saved and selected for the next training run.")

    preset_file = panel["root"] / "logs" / "training_panel" / "physics_presets.json"
    payload = json.loads(preset_file.read_text(encoding="utf-8"))
    saved = next(item for item in payload["presets"] if item["name"] == "Measured Lab Robot")
    assert saved["values"] == {"simulation_physics.mass.scale": 1.05}

    page.locator('.nav-button[data-view="train"]').click()
    expect(page.locator("#train-active-physics-preset-name")).to_have_text("Measured Lab Robot")


def open_physics_schematic(page, url: str) -> None:
    """Open Physics with the 3D path forced off.

    Headless Chrome may or may not expose a working WebGL context depending on the host,
    so interaction tests drive the SVG fallback: it is deterministic, and it is also the
    path a machine without GPU support actually gets.
    """

    page.goto(f"{url}/?robot3d=off")
    page.locator('.nav-button[data-view="physics"]').click()
    page.wait_for_selector(".robot-fallback-leg")


def test_physics_viewport_is_built_from_the_urdf_and_reports_its_mode(panel, page):
    open_physics(page, panel["url"])
    expect(page.locator("#physics-viewport")).to_be_visible()

    layout = page.evaluate("fetch('/api/physics/robot-geometry').then((r) => r.json())")
    assert layout["source"] == "urdf"
    assert len(layout["legs"]) == 6
    assert [leg["joints"]["main"]["canonical_id"] for leg in layout["legs"]] == [
        f"main_{index}" for index in range(6)
    ]

    expect(page.locator("#physics-viewport-mode")).to_have_text(re.compile(r"3D|Schematic"))
    # Damping and command delay have no honest spatial depiction, so they are reported
    # as text beside the model instead.
    expect(page.locator(".physics-readout-chip")).to_have_count(6)
    expect(page.locator("#physics-viewport-readout")).to_contain_text("Command delay")


def test_physics_viewport_flags_legs_whose_name_disagrees_with_the_model(panel, page):
    """_LEG_LABELS calls index 0 'Right front' but the URDF mounts it at the right middle.

    The joint indices and tripod grouping are correct, so this is a labelling defect
    rather than a training one. The viewer must say so rather than quietly drawing one
    interpretation. If _LEG_LABELS is corrected, this test should be inverted.
    """

    open_physics(page, panel["url"])
    warning = page.locator("#physics-viewport-warning")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("do not match")
    expect(page.locator(".physics-label-swap")).to_have_count(3)
    expect(warning).to_contain_text("Right front")
    expect(warning).to_contain_text("right middle")


def test_physics_schematic_leg_click_filters_the_editor(panel, page):
    open_physics_schematic(page, panel["url"])
    expect(page.locator("#physics-viewport-mode")).to_have_text("Schematic")
    expect(page.locator(".robot-fallback-leg")).to_have_count(6)

    total_rows = page.locator(".physics-row").count()
    page.locator('.robot-fallback-leg[data-leg-index="0"]').click()

    # Selecting a leg reuses the existing search filter rather than adding a second one.
    expect(page.locator("#physics-search")).to_have_value("Right front")
    expect(page.locator("#physics-status")).to_contain_text("leg 0")
    assert page.locator(".physics-row").count() < total_rows
    expect(page.locator(".physics-row-meta").first).to_contain_text("Right front")


def test_physics_viewport_mirrors_edited_values_and_pauses_when_hidden(panel, page):
    open_physics_schematic(page, panel["url"])
    page.once("dialog", lambda dialog: dialog.accept("Viewer Preset"))
    page.locator("#physics-preset-duplicate-btn").click()
    expect(page.locator("#physics-profile-name")).to_have_value("Viewer Preset")

    page.fill("#physics-search", "mass scale")
    page.locator('[data-physics-key="simulation_physics.mass.scale"]').fill("1.8")
    chip = page.locator(".physics-readout-chip.is-overridden")
    expect(chip).to_have_count(1)
    expect(chip).to_contain_text("1.8")

    # The render loop must not keep running once the operator leaves the Physics view;
    # the panel is left open for hours beside a training run.
    frames_before = page.evaluate("window.RedRHexRobotView.frameCount()")
    page.locator('.nav-button[data-view="train"]').click()
    page.wait_for_timeout(400)
    frames_after = page.evaluate("window.RedRHexRobotView.frameCount()")
    assert frames_after == frames_before


def test_physics_preview_follows_the_operator_down_the_field_list(panel, page):
    """The field list is long; the preview is useless if it scrolls out of sight."""

    def scroll_to(y: int) -> None:
        # Synchronise on the settled scroll position rather than a fixed sleep: the
        # panel's startup fetches can land mid-scroll and make a timed wait flaky.
        page.evaluate(f"window.scrollTo(0, {y})")
        page.wait_for_function(f"Math.abs(window.scrollY - {y}) <= 1")

    open_physics_schematic(page, panel["url"])
    viewport = page.locator("#physics-viewport")
    expect(viewport).not_to_have_class(re.compile(r"is-stuck"))
    resting_height = viewport.bounding_box()["height"]

    scroll_to(2200)
    expect(viewport).to_have_class(re.compile(r"is-stuck"))

    box = viewport.bounding_box()
    assert box["y"] <= 1, "preview should be pinned to the top of the window"
    # Pinned it competes with the fields, so it must give height back.
    assert box["height"] < resting_height * 0.6
    # The caveat stays readable in the collapsed bar, just shorter.
    expect(page.locator(".physics-label-short")).to_be_visible()
    expect(page.locator(".physics-label-swap").first).to_be_hidden()

    scroll_to(0)
    expect(viewport).not_to_have_class(re.compile(r"is-stuck"))


def test_physics_mobile_layout_has_no_horizontal_overflow(panel, page):
    page.set_viewport_size({"width": 390, "height": 900})
    open_physics(page, panel["url"])
    expect(page.locator("#physics-categories")).to_contain_text("Mass scale")
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    offenders = page.evaluate(
        """[...document.querySelectorAll('*')]
          .filter((element) => element.getBoundingClientRect().right > window.innerWidth + 1)
          .map((element) => ({tag: element.tagName, id: element.id, cls: element.className,
            right: Math.round(element.getBoundingClientRect().right), width: element.scrollWidth}))
          .slice(0, 12)"""
    )
    assert overflow <= 1, offenders


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
    page.locator(".run-card", has_text="Beta Foldered").click()
    expect(page.locator("#details-title")).to_be_visible()
    expect(page.locator("#export-video-drive")).to_be_visible()
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1


def test_native_spring_backend_is_safe_default_submitted_restored_and_displayed(panel, page):
    submitted_payloads = []

    def capture_training_start(route):
        submitted_payloads.append(route.request.post_data_json)
        route.fulfill(
            status=201,
            content_type="application/json",
            body=json.dumps({"id": "native_submit", "status": "queued", "display_name": ""}),
        )

    page.route("**/api/training/start", capture_training_start)
    page.goto(panel["url"])

    backend = page.locator('select[name="spring_backend"]')
    assert backend.locator('option[value="explicit"]').count() == 1
    assert backend.locator('option[value="native"]').count() == 1
    expect(backend).to_have_value("native")
    expect(backend.locator('option[value="explicit"]')).to_have_attribute("disabled", "")
    expect(backend.locator("xpath=following-sibling::small")).to_contain_text(
        "numerically unstable"
    )
    page.locator("#train-form button[type=submit]").click()
    expect(page.locator("#train-status")).to_contain_text("Queued native_submit")
    assert submitted_payloads == [{
        "training_route": "standard",
        "task": "Template-Redrhex-ForwardFast-Direct-v0",
        "display_name": "",
        "num_envs": 4,
        "max_iterations": 1,
        "device": "cuda:0",
        "spring_backend": "native",
        "seed": "",
        "checkpoint": "",
        "headless": True,
        "resume": False,
        "reward_preset_id": "speed-focus",
        "reward_overrides": {
            "v2_reward_scales.forward_progress": 3,
            "v2_reward_scales.velocity_tracking": 6,
            "v2_reward_scales.axis_suppression": 2,
            "v2_reward_scales.height_maintain": 1,
            "v2_reward_scales.height_low_penalty": 1.5,
            "v2_reward_scales.leg_moving": 0.25,
            "v2_reward_scales.stall_penalty": -3,
            "v2_reward_scales.energy_per_distance": 0.0005,
        },
        "terrain_preset_id": "baseline",
        "terrain_overrides": {},
        "physics_preset_id": "baseline",
        "physics_overrides": {},
    }]

    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()
    expect(page.locator("#runs")).to_contain_text("spring backend: native")
    expect(page.locator("#run-info-grid")).to_contain_text("Spring Backend")
    expect(page.locator("#run-info-grid")).to_contain_text("native")

    page.locator("#resume-run").click()
    expect(page.locator('select[name="spring_backend"]')).to_have_value("native")

    page.locator('.nav-button[data-view="history"]').click()
    page.locator(".run-card", has_text="Beta Foldered").click()
    page.locator("#tweak-run").click()
    expect(page.locator('select[name="spring_backend"]')).to_have_value("native")

    page.locator('.nav-button[data-view="history"]').click()
    page.locator(".run-card", has_text="run_alpha").click()
    expect(page.locator("#details-title")).to_have_text("run_alpha")
    expect(page.locator("#runs")).to_contain_text("spring backend: explicit")
    expect(page.locator("#resume-run")).to_be_disabled()
    expect(page.locator("#resume-run")).to_have_attribute("title", re.compile("Explicit.*cannot be resumed"))
    beta_card = page.locator(".run-card", has_text="Beta Foldered")
    beta_card.locator(".run-menu-trigger").click()
    beta_card.locator('button[data-action="compare"]').click()
    expect(page.locator(".comparison-grid")).to_contain_text("Spring Backend")
    expect(page.locator(".comparison-grid")).to_contain_text("native")


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


def test_deploy_stage_counts_and_report_collapse(panel, page):
    open_deploy(page, panel["url"])

    # Stage outcome counts belong beside the heading, not buried under the list.
    expect(page.locator("#deploy-stage-summary")).to_contain_text("1 pass")
    expect(page.locator("#deploy-stage-summary")).to_contain_text("1 warn")

    # The raw JSON is available but no longer occupies the panel by default.
    expect(page.locator("#deploy-report-json")).to_be_hidden()
    page.locator("#deploy-report-details > summary").click()
    expect(page.locator("#deploy-report-json")).to_be_visible()

    # Runtime knobs stay collapsed until an operator asks for them.
    expect(page.locator("#deploy-use-tensorrt")).to_be_hidden()
    page.locator("#deploy-advanced > summary").click()
    expect(page.locator("#deploy-use-tensorrt")).to_be_visible()


def test_deploy_next_step_names_the_next_action(panel, page):
    open_deploy(page, panel["url"])

    # A run with an ONNX and a warning report should be told to read the stages.
    expect(page.locator("#deploy-next-step")).to_contain_text("warnings")
    expect(page.locator("#deploy-next-step")).to_have_class(re.compile("deploy-next-warn"))
    expect(page.locator("#deploy-artifact-status .status-completed")).to_have_count(2)

    # A run with a checkpoint but no export must be pointed at the export path,
    # and the action it cannot take has to say why it is disabled.
    page.select_option("#deploy-run-select", "run_alpha")
    expect(page.locator("#deploy-next-step")).to_contain_text("Export ONNX + Validate")
    expect(page.locator("#deploy-validate-existing")).to_be_disabled()
    expect(page.locator("#deploy-validate-existing")).to_have_attribute(
        "data-tooltip", re.compile("no exported ONNX")
    )
    expect(page.locator("#deploy-export-validate")).to_be_enabled()


def test_deploy_mobile_layout_has_no_horizontal_overflow(panel, page):
    page.set_viewport_size({"width": 390, "height": 900})
    open_deploy(page, panel["url"])

    expect(page.locator("#deploy-stage-list")).to_contain_text("Export Integrity")
    page.locator("#deploy-advanced > summary").click()
    page.locator("#deploy-report-details > summary").click()
    page.locator("#deploy-console-details > summary").click()
    expect(page.locator("#deploy-report-json")).to_be_visible()
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1


def test_status_toast_appears_for_actions_outside_history(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.evaluate("setStatus('Fixture status message.')")
    expect(page.locator("#panel-status .toast")).to_contain_text("Fixture status message.")


@pytest.mark.parametrize("remote_client", ["windows", "macos"])
def test_desktop_remote_mode_disables_host_only_controls(panel, page, remote_client):
    page.goto(f"{panel['url']}/?remote_client={remote_client}#/history/panel_run_beta")
    expect(page.locator("#details-title")).to_contain_text("Beta Foldered")

    for selector in (
        "#play-run",
        "#open-run-folder",
        "#open-onnx-folder",
        "#open-video-folder",
        "#open-process-log-folder",
        "#deploy-mujoco-viewer",
    ):
        expect(page.locator(selector)).to_be_disabled()

    expect(page.locator("#play-run")).to_have_attribute("data-tooltip", re.compile("opens on the training PC"))
    expect(page.locator("#open-run-folder")).to_have_attribute("data-tooltip", re.compile("folders open on the training PC"))
    expect(page.locator("#tensorboard-run")).to_be_enabled()
    expect(page.locator("#record-video")).to_be_enabled()
    expect(page.locator("#copy-video-path")).to_be_enabled()
    expect(page.locator("#copy-onnx-path")).to_be_enabled()

    headless = page.locator('#train-form input[name="headless"]')
    expect(headless).to_be_checked()
    expect(headless).to_be_disabled()


def test_desktop_remote_tensorboard_uses_fixed_forward_for_all_runs(panel, page):
    requests = []

    def start_tensorboard(route, request):
        requests.append(json.loads(request.post_data or "{}"))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "id": "tensorboard_6006",
                    "already_running": False,
                    "url": panel["url"],
                    "host": "127.0.0.1",
                    "port": 6006,
                }
            ),
        )

    page.route("**/api/tensorboard/start", start_tensorboard)
    page.goto(f"{panel['url']}/?remote_client=windows#/history/panel_run_beta")
    expect(page.locator("#tensorboard-run")).to_be_enabled()

    with page.expect_popup() as popup_info:
        page.locator("#tensorboard-run").click()
    popup = popup_info.value
    expect(page.locator("#panel-status")).to_contain_text("Started TensorBoard for all runs on port 6006.")
    assert requests == [{"host": "127.0.0.1", "port": 6006}]
    popup.close()


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


def test_error_toast_survives_a_flood_of_info_toasts(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.evaluate("setStatusTone('Deploy blew up.', 'error')")
    page.evaluate("setStatus('one'); setStatus('two'); setStatus('three'); setStatus('four')")
    # Eviction must drop info toasts first — an unread failure may not be pushed out.
    expect(page.locator("#panel-status .toast")).to_have_count(3)
    expect(page.locator("#panel-status .toast.toast-error")).to_contain_text("Deploy blew up.")


def test_selecting_a_run_does_not_emit_a_toast(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card").first.click()
    page.wait_for_timeout(800)
    # Checkpoint metadata belongs in the details pane, not in a corner toast that
    # would evict unread errors on every ordinary click.
    expect(page.locator("#panel-status .toast")).to_have_count(0)


def test_refresh_failure_leaves_a_persistent_error_toast(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.route(
        "**/api/system",
        lambda route: route.fulfill(
            status=500,
            content_type="application/json",
            body=json.dumps({"error": "System probe exploded."}),
        ),
    )
    page.locator("#refresh-button").click()
    toast = page.locator("#panel-status .toast.toast-error")
    expect(toast).to_contain_text("System probe exploded.")
    page.wait_for_timeout(7000)
    expect(toast).to_be_visible()  # must not erase itself after 6s


def test_action_failure_outside_history_reaches_the_screen(panel, page):
    """The bug Task 1 fixed: a failure raised while a non-History view is active."""
    page.goto(panel["url"])
    page.locator('.nav-button[data-view="rewards"]').click()
    page.wait_for_selector(".preset-card")

    def fail_writes(route):
        if route.request.method == "POST":
            route.fulfill(
                status=500,
                content_type="application/json",
                body=json.dumps({"error": "Preset store is read-only."}),
            )
        else:
            route.continue_()

    page.route("**/api/presets", fail_writes)
    page.on("dialog", lambda dialog: dialog.accept("Copy of baseline"))
    page.locator(".preset-card").first.click()
    page.locator("#preset-duplicate-btn").click()
    toast = page.locator("#panel-status .toast.toast-error")
    expect(toast).to_contain_text("Preset store is read-only.")


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


def test_convergence_shows_effective_settings_and_unsaved_state(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.click('.nav-button[data-view="convergence"]')
    page.wait_for_selector("#convergence-presets")

    # Cooldown and the scalar tag are named in Definitions, so the page has to
    # show what mother is actually using for them.
    info = page.locator("#convergence-info-grid")
    expect(info).to_contain_text("200 iterations")
    expect(info).to_contain_text("60 minutes")
    expect(info).to_contain_text("Train/mean_reward")

    # A form that matches the saved config offers nothing to save.
    expect(page.locator("#convergence-dirty-hint")).to_be_hidden()
    expect(page.locator("#convergence-save")).to_be_disabled()

    page.locator('#convergence-presets [data-preset="strict"]').click()
    expect(page.locator("#convergence-dirty-hint")).to_be_visible()
    expect(page.locator("#convergence-save")).to_be_enabled()

    page.locator("#convergence-save").click()
    expect(page.locator("#convergence-info-grid")).to_contain_text("400 iterations")
    expect(page.locator("#convergence-dirty-hint")).to_be_hidden()


def test_convergence_dependent_controls_follow_their_master_switch(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.click('.nav-button[data-view="convergence"]')
    page.wait_for_selector("#convergence-presets")

    expect(page.locator("#divergence-patience")).to_be_enabled()
    expect(page.locator("#divergence-action-hint")).to_contain_text("reported only")

    page.locator("#divergence-auto-stop").check()
    expect(page.locator("#divergence-action-hint")).to_contain_text("stopped automatically")

    page.locator("#divergence-enabled").uncheck()
    expect(page.locator("#divergence-patience")).to_be_disabled()
    expect(page.locator("#divergence-auto-stop")).to_be_disabled()
    expect(page.locator("#divergence-action-hint")).to_contain_text("keeps training")

    page.locator("#convergence-enabled").uncheck()
    expect(page.locator('#convergence-presets [data-preset="strict"]')).to_be_disabled()
    expect(page.locator("#convergence-auto-record")).to_be_disabled()


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
    # The click leaves the pointer resting on the button, so :hover is still
    # driving the 160ms fade. Move the pointer away and let the transition
    # settle, otherwise this reads a mid-fade value rather than the
    # focus-visible state under test.
    page.mouse.move(0, 0)
    page.wait_for_timeout(250)
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


def test_menu_closes_after_choosing_an_action(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card .run-menu-trigger").first.click()
    expect(page.locator(".run-menu[data-open='true']")).to_have_count(1)
    page.locator(".run-menu[data-open='true'] button[data-action='tweak']").click()
    expect(page.locator(".run-menu[data-open='true']")).to_have_count(0)


def test_menu_does_not_reopen_on_a_later_render(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card .run-menu-trigger").first.click()
    page.locator(".run-menu[data-open='true'] button[data-action='tweak']").click()
    page.evaluate("loadRuns()")
    page.wait_for_timeout(500)
    expect(page.locator(".run-menu[data-open='true']")).to_have_count(0)


# Holds every GET /api/runs open until the test releases it, so a skeleton can be
# observed during a genuinely in-flight fetch rather than by hand-setting state.
RUNS_FETCH_GATE = """
(() => {
  const realFetch = window.fetch.bind(window);
  window.__pendingRuns = [];
  window.fetch = (input, init) => {
    const url = String(typeof input === "string" ? input : input.url);
    if (url.split("?")[0].endsWith("/api/runs")) {
      return new Promise((resolve, reject) => {
        window.__pendingRuns.push(() => realFetch(input, init).then(resolve, reject));
      });
    }
    return realFetch(input, init);
  };
  window.__releaseRuns = () => {
    const queued = window.__pendingRuns;
    window.__pendingRuns = [];
    queued.forEach((run) => run());
    return queued.length;
  };
})();
"""


def test_skeleton_shows_during_an_in_flight_runs_fetch(panel, page):
    page.add_init_script(RUNS_FETCH_GATE)
    page.goto(panel["url"])
    page.locator('.nav-button[data-view="history"]').click()

    # The real boot path — no hand-set state — must paint a skeleton.
    expect(page.locator("#runs .skeleton-row").first).to_be_visible()
    expect(page.locator(".run-card")).to_have_count(0)

    page.evaluate("window.__releaseRuns()")
    expect(page.locator(".run-card").first).to_be_visible()
    expect(page.locator("#runs .skeleton-row")).to_have_count(0)

    # A background refresh of already-loaded data must not reflash the skeleton.
    # `void` matters: the gated promise never settles and evaluate() awaits it.
    page.evaluate("void loadRuns()")
    page.wait_for_function("window.__pendingRuns.length > 0")
    expect(page.locator("#runs .skeleton-row")).to_have_count(0)
    expect(page.locator(".run-card").first).to_be_visible()
    page.evaluate("window.__releaseRuns()")
    expect(page.locator("#runs .skeleton-row")).to_have_count(0)


def test_skeleton_shows_before_runs_arrive(panel, page):
    page.goto(panel["url"])
    page.locator('.nav-button[data-view="history"]').click()
    page.wait_for_selector("#runs")
    page.evaluate("state.runs = []; beginLoading('runs'); renderRuns()")
    expect(page.locator("#runs .skeleton-row").first).to_be_visible()
    expect(page.locator("#runs")).not_to_contain_text("No training history found yet")


def test_empty_state_shows_when_not_loading(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.evaluate("state.runs = []; endLoading('runs'); renderRuns()")
    expect(page.locator("#runs")).to_contain_text("No training history found yet")
    expect(page.locator("#runs .skeleton-row")).to_have_count(0)


def test_background_refresh_does_not_reflash_skeleton(panel, page):
    open_history(page, panel["url"])
    page.wait_for_selector(".run-card")
    page.evaluate("beginLoading('runs'); renderRuns()")
    expect(page.locator(".run-card").first).to_be_visible()
    expect(page.locator("#runs .skeleton-row")).to_have_count(0)


def test_freshness_reports_live_after_a_successful_poll(panel, page):
    page.goto(panel["url"])
    page.wait_for_selector("#train-form")
    page.wait_for_function("document.querySelector('#freshness')?.dataset.state === 'live'")
    expect(page.locator("#freshness-label")).to_contain_text("updated")


def test_freshness_reports_failed_when_the_backend_stops(panel, page):
    # The 45s margin below assumes the fixture seeds a queued run, which keeps the
    # panel on its 10s active poll interval rather than the 30s idle one.
    page.goto(panel["url"])
    page.wait_for_function("document.querySelector('#freshness')?.dataset.state === 'live'")
    panel["proc"].terminate()
    panel["proc"].wait(timeout=10)
    page.wait_for_function(
        "document.querySelector('#freshness')?.dataset.state === 'failed'",
        timeout=45000,
    )


def test_closing_a_comparison_leaves_the_details_panel_usable(panel, page):
    # Comparison used to overwrite .details-panel's innerHTML, which destroyed the
    # element ids renderRunDetails() writes to. Closing the comparison then threw
    # on every later render, including the one inside the runs poll, so History
    # stopped refreshing with no visible error.
    page_errors: list[str] = []
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()
    expect(page.locator("#details-title")).to_have_text("Beta Foldered")

    alpha_card = page.locator(".run-card", has_text="run_alpha")
    alpha_card.locator(".run-menu-trigger").click()
    alpha_card.locator("[data-action='compare']").click()
    expect(page.locator("#comparison-panel")).to_be_visible()
    expect(page.locator("#comparison-grid")).to_contain_text("run_alpha")

    page.locator("#exit-comparison-btn").click()
    expect(page.locator("#comparison-panel")).to_be_hidden()
    expect(page.locator("#details-title")).to_have_text("Beta Foldered")
    expect(page.locator("#notes-editor")).to_be_visible()

    # The details pane must still respond to a fresh selection.
    page.locator(".run-card", has_text="run_alpha").click()
    expect(page.locator("#details-title")).to_have_text("run_alpha")

    # The runs poll keeps calling renderRunDetails(); it must not be throwing.
    page.wait_for_timeout(1200)
    assert page_errors == []


def test_escape_closes_a_comparison(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()
    alpha_card = page.locator(".run-card", has_text="run_alpha")
    alpha_card.locator(".run-menu-trigger").click()
    alpha_card.locator("[data-action='compare']").click()
    expect(page.locator("#comparison-panel")).to_be_visible()

    page.keyboard.press("Escape")
    expect(page.locator("#comparison-panel")).to_be_hidden()


def test_running_run_card_renders_a_single_progress_bar(panel, page):
    open_history(page, panel["url"])
    page.evaluate(
        """
        const run = state.runs.find((item) => item.id === 'panel_run_beta');
        run.status = 'running';
        run.progress = {
          iteration: 5,
          total_iterations: 10,
          percent: 50,
          updated_at: new Date().toISOString(),
        };
        renderRuns();
        """
    )
    beta_card = page.locator(".run-card", has_text="Beta Foldered")
    expect(beta_card.locator(".run-progress")).to_have_count(1)


def test_status_sort_ranks_by_urgency_not_alphabet(panel, page):
    # Alphabetically "completed" sorts above "failed", which buried the runs an
    # operator actually needs to look at. The order is now operational.
    open_history(page, panel["url"])
    page.select_option("#sort-runs", "status")
    expect(page.locator(".run-card").first.locator(".status-pill")).to_have_text("failed")
    expect(page.locator(".run-card").last.locator(".status-pill")).to_have_text("completed")


def test_history_filters_survive_a_reload(panel, page):
    open_history(page, panel["url"])
    page.fill("#run-search", "beta")
    expect(page.locator("#runs")).not_to_contain_text("run_alpha")
    expect(page.locator("#run-count-badge")).to_contain_text("of")

    page.reload()
    page.wait_for_selector(".run-card")
    expect(page.locator("#run-search")).to_have_value("beta")
    expect(page.locator("#runs")).not_to_contain_text("run_alpha")

    page.locator("#clear-run-filters").click()
    expect(page.locator("#runs")).to_contain_text("run_alpha")
    expect(page.locator("#run-search")).to_have_value("")


def test_search_matches_folder_and_notes(panel, page):
    open_history(page, panel["url"])
    page.fill("#run-search", "Good Runs")
    expect(page.locator("#runs")).to_contain_text("Beta Foldered")
    expect(page.locator("#runs")).not_to_contain_text("run_alpha")


def test_unsaved_notes_survive_switching_runs(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").click()
    expect(page.locator("#notes-editor")).to_be_enabled()
    page.fill("#notes-editor", "half-written observation")

    page.locator(".run-card", has_text="run_alpha").click()
    expect(page.locator("#details-title")).to_have_text("run_alpha")

    page.locator(".run-card", has_text="Beta Foldered").click()
    expect(page.locator("#notes-editor")).to_have_value("half-written observation")
    expect(page.locator("#notes-dirty-flag")).to_be_visible()


def test_slash_focuses_search_and_j_k_move_the_selection(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card").first.click()
    first_title = page.locator("#details-title").inner_text()

    page.keyboard.press("j")
    expect(page.locator("#details-title")).not_to_have_text(first_title)
    page.keyboard.press("k")
    expect(page.locator("#details-title")).to_have_text(first_title)

    page.keyboard.press("/")
    assert page.evaluate("document.activeElement.id") == "run-search"


def test_bulk_toolbar_keeps_a_stable_height_across_selection(panel, page):
    # Mounting the bulk controls only once a run is ticked reflowed the run list
    # under the pointer, so they stay mounted and merely disable.
    open_history(page, panel["url"])
    toolbar = page.locator(".bulk-toolbar")
    expect(page.locator("#delete-selected-runs")).to_be_visible()
    expect(page.locator("#delete-selected-runs")).to_be_disabled()
    before = toolbar.bounding_box()["height"]

    page.locator(".run-card", has_text="Beta Foldered").locator(".run-select-checkbox").check()
    expect(page.locator("#bulk-selected-count")).to_have_text("1 selected")
    expect(page.locator("#delete-selected-runs")).to_be_enabled()
    assert toolbar.bounding_box()["height"] == before


def test_dragging_a_run_onto_a_folder_moves_it(panel, page):
    open_history(page, panel["url"])
    page.locator('.folder-select[data-folder="Good Runs"]').click()
    expect(page.locator("#runs")).not_to_contain_text("run_alpha")

    page.locator('.folder-select[data-folder="__all__"]').click()
    page.locator(".run-card", has_text="run_alpha").drag_to(
        page.locator('.folder-item[data-drop-folder="Good Runs"]')
    )
    expect(page.locator("#panel-status")).to_contain_text("Moved 1 run to Good Runs")

    page.locator('.folder-select[data-folder="Good Runs"]').click()
    expect(page.locator("#runs")).to_contain_text("run_alpha")


def test_all_runs_is_not_a_drop_target(panel, page):
    open_history(page, panel["url"])
    assert page.locator('.folder-item[data-drop-folder="__all__"]').count() == 0
    assert page.locator('.folder-item[data-drop-folder="__uncategorized__"]').count() == 1


def test_bulk_delete_requires_a_typed_acknowledgement(panel, page):
    open_history(page, panel["url"])
    page.locator(".run-card", has_text="Beta Foldered").locator(".run-select-checkbox").check()
    page.locator("#delete-selected-runs").click()

    expect(page.locator("#confirm-dialog")).to_be_visible()
    expect(page.locator("#confirm-dialog-input-wrap")).to_be_visible()
    expect(page.locator("#confirm-dialog-confirm")).to_be_disabled()
    # Nothing is marked as deleting while the operator is still deciding.
    expect(page.locator(".run-card.deleting")).to_have_count(0)

    page.fill("#confirm-dialog-input", "DELETE")
    expect(page.locator("#confirm-dialog-confirm")).to_be_enabled()
    page.locator("#confirm-dialog-cancel").click()
    expect(page.locator("#runs")).to_contain_text("Beta Foldered")


def test_folder_creation_uses_the_in_app_dialog(panel, page):
    open_history(page, panel["url"])
    page.locator("#create-folder-btn").click()
    expect(page.locator("#confirm-dialog")).to_be_visible()
    expect(page.locator("#confirm-dialog-title")).to_have_text("New Folder")

    page.fill("#confirm-dialog-input", "Sweep A")
    page.locator("#confirm-dialog-confirm").click()
    expect(page.locator("#folder-sidebar")).to_contain_text("Sweep A")
    assert page.locator('.folder-select[data-folder="Sweep A"]').count() == 1


def test_run_list_popups_stay_inside_the_scrolling_list(panel, page):
    # The run list is its own scroll container, so a popup drawn past a card's
    # edge is clipped at the container boundary rather than overlaying the page.
    open_history(page, panel["url"])
    list_box = page.locator("#runs").bounding_box()

    checkbox = page.locator(".run-card").first.locator(".run-select-checkbox")
    checkbox.hover()
    tip_top = page.evaluate(
        """() => {
          const box = document.querySelector('.run-card .run-select-checkbox');
          const style = getComputedStyle(box, '::after');
          return box.getBoundingClientRect().bottom + parseFloat(style.marginTop || 0);
        }"""
    )
    assert tip_top >= list_box["y"]

    page.locator(".run-card").first.locator(".run-menu-trigger").click()
    menu = page.locator(".run-menu[data-open='true']").bounding_box()
    assert menu["y"] >= list_box["y"]
    assert menu["y"] + menu["height"] <= list_box["y"] + list_box["height"] + 1


def drag_run_over_folder(page, run_text: str, folder_selector: str):
    """Dispatch real HTML5 drag events; synthetic mouse moves do not emit them."""
    transfer = page.evaluate_handle("() => new DataTransfer()")
    page.locator(".run-card", has_text=run_text).dispatch_event(
        "dragstart", {"dataTransfer": transfer}
    )
    page.locator(folder_selector).dispatch_event("dragover", {"dataTransfer": transfer})
    return transfer


def test_drag_marks_valid_targets_and_states_the_outcome(panel, page):
    open_history(page, panel["url"])
    alpha = page.locator(".run-card", has_text="run_alpha")
    good_runs = page.locator('.folder-item[data-drop-folder="Good Runs"]')
    uncategorized = page.locator('.folder-item[data-drop-folder="__uncategorized__"]')

    transfer = drag_run_over_folder(page, "run_alpha", '.folder-item[data-drop-folder="Good Runs"]')

    expect(page.locator("body")).to_have_class(re.compile("dragging-runs"))
    expect(alpha).to_have_class(re.compile("dragging"))
    expect(good_runs).to_have_attribute("data-drop-label", "Move 1 run here")
    expect(good_runs).to_have_class(re.compile("drop-target"))
    # The outcome line is drawn from the attribute the row carries.
    label = page.evaluate(
        """() => getComputedStyle(
             document.querySelector('.folder-item.drop-target'), '::after'
           ).content"""
    )
    assert "Move 1 run here" in label
    # run_alpha is already uncategorized, so that row cannot receive it.
    expect(uncategorized).to_have_class(re.compile("drop-unavailable"))

    good_runs.dispatch_event("drop", {"dataTransfer": transfer})
    expect(page.locator("#panel-status")).to_contain_text("Moved 1 run to Good Runs")
    expect(page.locator("body")).not_to_have_class(re.compile("dragging-runs"))


def test_a_selection_drag_reports_its_whole_count(panel, page):
    open_history(page, panel["url"])
    page.locator("#select-visible-runs").click()
    expect(page.locator("#bulk-selected-count")).to_have_text("3 selected")

    drag_run_over_folder(page, "run_alpha", '.folder-item[data-drop-folder="Good Runs"]')
    expect(page.locator('.folder-item[data-drop-folder="Good Runs"]')).to_have_attribute(
        "data-drop-label", "Move 3 runs here"
    )


def test_drag_state_is_cleared_when_a_drag_is_abandoned(panel, page):
    open_history(page, panel["url"])
    alpha = page.locator(".run-card", has_text="run_alpha")
    transfer = page.evaluate_handle("() => new DataTransfer()")
    alpha.dispatch_event("dragstart", {"dataTransfer": transfer})
    expect(page.locator("body")).to_have_class(re.compile("dragging-runs"))

    alpha.dispatch_event("dragend", {"dataTransfer": transfer})
    expect(page.locator("body")).not_to_have_class(re.compile("dragging-runs"))
    expect(page.locator(".run-card.dragging")).to_have_count(0)
    assert page.locator("[data-drop-folder][data-drop-label]").count() == 0
