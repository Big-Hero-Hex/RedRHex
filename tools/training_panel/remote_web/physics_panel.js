import { escapeHtml, normalizePhysicsValues } from "./core.js?v=3.7.0-remote-parity";

export function normalizePhysicsPreset(raw = {}) {
  return {
    id: String(raw.id || "baseline"),
    name: String(raw.name || "Baseline"),
    description: String(raw.description || ""),
    values: raw.values && typeof raw.values === "object" ? { ...raw.values } : {},
    built_in: Boolean(raw.built_in),
    created_by: raw.created_by || null,
    updated_at: raw.updated_at || raw.created_at || "",
  };
}

export function physicsValuesFromForm(root, fieldSchema = []) {
  const values = {};
  root?.querySelectorAll?.(".physics-input").forEach((input) => {
    if (input.value !== "") values[input.dataset.key] = input.value;
  });
  return normalizePhysicsValues(values, fieldSchema);
}

export function physicsViewHtml({ presets = [], selectedId = "baseline", fieldSchema = [], editable = false, search = "" } = {}) {
  const normalized = presets.map(normalizePhysicsPreset);
  const preset = normalized.find((item) => item.id === selectedId) || normalized[0] || normalizePhysicsPreset();
  const query = String(search || "").trim().toLowerCase();
  const categories = new Map();
  for (const field of fieldSchema || []) {
    if (query && !`${field.label} ${field.key} ${field.category} ${field.description}`.toLowerCase().includes(query)) continue;
    if (!categories.has(field.category)) categories.set(field.category, []);
    categories.get(field.category).push(field);
  }
  return `<section class="rewards-page physics-page">
    <aside class="panel preset-list rewards-rail">
      <div class="section-head compact"><div><h2>Physics</h2><p class="muted">Sparse team calibration profiles</p></div>
        <button class="icon-action" title="New Physics preset" data-action="new-physics-preset" ${editable ? "" : "disabled"}>+</button></div>
      <div class="preset-scroll">${normalized.map((item) => `<button class="preset-button ${item.id === preset.id ? "active" : ""}" data-action="select-physics-preset" data-id="${escapeHtml(item.id)}"><strong>${escapeHtml(item.name)}</strong><small>${item.built_in ? "Built-in" : "Team preset"}</small></button>`).join("")}</div>
    </aside>
    <article class="panel reward-workspace">
      <div class="section-head reward-head"><div><h2>Physics Calibration</h2><p class="muted">Only changed values are sent. Native torsion springs remain the safe default.</p></div>
        <div class="button-row"><button data-action="duplicate-physics-preset" ${editable ? "" : "disabled"}>Duplicate</button><button class="danger" data-action="delete-physics-preset" ${editable && !preset.built_in ? "" : "disabled"}>Delete</button><button class="primary" data-action="save-physics-preset" ${editable && !preset.built_in ? "" : "disabled"}>Save</button></div></div>
      <div class="preset-meta-grid"><label>Name <input id="physics-preset-name" maxlength="120" value="${escapeHtml(preset.name)}" ${editable && !preset.built_in ? "" : "disabled"}></label><label>Description <textarea id="physics-preset-description" ${editable && !preset.built_in ? "" : "disabled"}>${escapeHtml(preset.description)}</textarea></label></div>
      <label>Search fields <input id="physics-search" type="search" value="${escapeHtml(search)}" placeholder="stiffness, damping, friction…"></label>
      <div class="reward-editor physics-editor">${[...categories.entries()].map(([category, fields]) => `<section class="reward-group editor-group" data-kind="physics"><button class="group-toggle" data-action="toggle-editor-group"><span>${escapeHtml(category)}</span><span class="chevron">▾</span></button><div class="group-body">${fields.map((field) => `<label class="reward-row"><span><strong>${escapeHtml(field.label)}</strong><small>${escapeHtml(field.description || "")}</small><code>${escapeHtml(field.key)}</code></span><span><input class="physics-input" data-key="${escapeHtml(field.key)}" type="number" step="${escapeHtml(field.step ?? 0.01)}" ${field.min === null || field.min === undefined ? "" : `min="${escapeHtml(field.min)}"`} ${field.max === null || field.max === undefined ? "" : `max="${escapeHtml(field.max)}"`} value="${escapeHtml(preset.values[field.key] ?? "")}" ${editable && !preset.built_in ? "" : "disabled"}><small>${escapeHtml(field.unit || "")}</small></span></label>`).join("")}</div></section>`).join("") || `<p class="empty-state">No Physics fields match this search.</p>`}</div>
    </article>
  </section>`;
}
