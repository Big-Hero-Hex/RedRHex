import { escapeHtml, formatRelativeTime, statusTone } from "./core.js?v=3.7.0-remote-parity";

export function deployViewHtml({ runs = [], selectedRunId = "", capability = null, canOperate = false, compatible = false } = {}) {
  const run = runs.find((item) => String(item.id) === String(selectedRunId)) || runs[0] || null;
  const scenarios = capability?.deploy?.scenarios || [];
  const state = run?.deploy_state || {};
  const stages = state.report?.stages || [];
  const disabled = !canOperate || !compatible || !run;
  return `<section class="deploy-grid">
    <article class="panel"><div class="section-head"><div><h2>Remote-safe Deploy</h2><p class="muted">Validation and recording only. No terminal, viewer, host paths, or hardware actuation.</p></div><span class="badge ${compatible ? "good" : "warning"}">${compatible ? "Compatible" : "Read-only"}</span></div>
      <label>Run <select id="deploy-run">${runs.map((item) => `<option value="${escapeHtml(item.id)}" ${item.id === run?.id ? "selected" : ""}>${escapeHtml(item.display_name || item.id)}</option>`).join("")}</select></label>
      <label class="check-row"><input id="deploy-ros-mock" type="checkbox"> Include optional ROS mock</label>
      <label>MuJoCo recording scenario <select id="deploy-scenario">${scenarios.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name.replaceAll("_", " "))}</option>`).join("")}</select></label>
      <div class="button-row wrap"><button data-action="deploy-validate" ${disabled ? "disabled" : ""}>Validate Existing ONNX</button><button class="primary" data-action="deploy-export-validate" ${disabled ? "disabled" : ""}>Export + Validate</button><button data-action="deploy-mujoco-smoke" ${disabled ? "disabled" : ""}>MuJoCo Smoke</button><button data-action="deploy-mujoco-video" ${disabled || !scenarios.length ? "disabled" : ""}>Record MP4</button></div>
    </article>
    <article class="panel"><h2>Latest Readiness</h2>${run ? `<div class="health-row"><span>Status</span><strong class="badge ${statusTone(state.overall_status || state.status)}">${escapeHtml(state.overall_status || state.status || "Not run")}</strong></div><div class="health-row"><span>Readiness</span><strong>${escapeHtml(state.readiness_level || "Unknown")}</strong></div>${state.completed_at ? `<p class="muted">Updated ${escapeHtml(formatRelativeTime(state.completed_at))}</p>` : ""}${stages.length ? `<div class="stage-list">${stages.map((stage) => `<div><span class="badge ${statusTone(stage.status)}">${escapeHtml(stage.status || "unknown")}</span><strong>${escapeHtml(stage.title || stage.label || stage.name || "Stage")}</strong><small>${escapeHtml(stage.summary || stage.error || "")}</small></div>`).join("")}</div>` : `<p class="empty-state">No readiness report yet.</p>`}${state.error ? `<p class="notice danger">${escapeHtml(state.error)}</p>` : ""}` : `<p class="empty-state">No runs are available.</p>`}</article>
  </section>`;
}

export function detectionViewHtml({ detection = {}, runs = [] } = {}) {
  const outcomes = runs.filter((run) => run.convergence_detected || run.divergence_detected).slice(0, 20);
  return `<section class="split-grid"><article class="panel"><div class="section-head"><div><h2>Detection</h2><p class="muted">Active Mother settings are read-only in To Go.</p></div><span class="badge info">Read-only</span></div><div class="health-list">${Object.entries(detection || {}).map(([key, value]) => `<div class="health-row"><span>${escapeHtml(key.replaceAll("_", " "))}</span><strong>${escapeHtml(value)}</strong></div>`).join("") || `<p class="empty-state">Restart the 3.7 worker to publish settings.</p>`}</div></article><article class="panel"><h2>Recent Outcomes</h2>${outcomes.map((run) => `<button class="run-card compact" data-action="open-run-from-detection" data-run-id="${escapeHtml(run.id)}"><strong>${escapeHtml(run.display_name || run.id)}</strong><small>${run.divergence_detected ? `Diverged: ${escapeHtml(run.divergence_reason || run.divergence_kind || "detected")}` : `Converged at ${escapeHtml(run.convergence_iteration ?? "unknown")}`}</small></button>`).join("") || `<p class="empty-state">No convergence or divergence outcomes yet.</p>`}</article></section>`;
}
