const SETTINGS_SECTION_KEY = "redrhex-settings-section-v1";
const DRIVE_POLL_MS = 1800;

const sectionButtons = Array.from(document.querySelectorAll("[data-settings-section]"));
const sectionPanels = Array.from(document.querySelectorAll("[data-settings-panel]"));
const driveFolderInput = document.querySelector("#drive-destination-folder");
const driveSaveButton = document.querySelector("#drive-save-location");
const driveReconnectButton = document.querySelector("#drive-reconnect-account");
const driveCheckButton = document.querySelector("#drive-check-connection");
const driveOpenFolder = document.querySelector("#drive-open-folder");
const settingsRefreshButton = document.querySelector("#settings-refresh");

let driveStatus = null;
let driveFolderDirty = false;
let drivePollTimer = null;

function notify(message, tone = "info") {
  if (typeof window.setStatusTone === "function") {
    window.setStatusTone(message, tone);
    return;
  }
  const status = document.querySelector("#panel-status");
  if (status) status.textContent = message;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (_) {
      payload = { error: text };
    }
  }
  if (!response.ok) throw new Error(payload.error || response.statusText || "Settings request failed");
  return payload;
}

function activateSettingsSection(name, { focus = false } = {}) {
  const selected = sectionButtons.some((button) => button.dataset.settingsSection === name)
    ? name
    : "remote";
  sectionButtons.forEach((button) => {
    const active = button.dataset.settingsSection === selected;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    if (active && focus) button.focus();
  });
  sectionPanels.forEach((panel) => {
    panel.hidden = panel.dataset.settingsPanel !== selected;
  });
  sessionStorage.setItem(SETTINGS_SECTION_KEY, selected);
}

function isDriveFolderLink(value = "") {
  try {
    const url = new URL(String(value).trim());
    return url.protocol === "https:"
      && url.hostname === "drive.google.com"
      && /\/folders\/[A-Za-z0-9_-]{10,}(?:\/|$)/.test(url.pathname);
  } catch (_) {
    return false;
  }
}

function configuredDestinationValue(status = driveStatus) {
  if (status?.destination_mode === "folder_link") return String(status.folder_url || "");
  return String(status?.folder || "RedRHex Videos");
}

function drivePathPreview(destination = "") {
  const normalized = String(destination || "RedRHex Videos").trim() || "RedRHex Videos";
  if (isDriveFolderLink(normalized)) return "Pasted Google Drive folder";
  if (normalized.includes("://")) return "Paste a valid Google Drive folder link";
  return `My Drive › ${normalized.split("/").map((part) => part.trim()).filter(Boolean).join(" › ")}`;
}

function updateDrivePreview() {
  const preview = document.querySelector("#drive-destination-preview");
  if (preview) preview.textContent = drivePathPreview(driveFolderInput?.value);
}

function scheduleDrivePoll() {
  if (drivePollTimer) clearTimeout(drivePollTimer);
  if (driveStatus?.reconnect?.status !== "authorizing") return;
  drivePollTimer = setTimeout(() => {
    loadDriveSettings().catch((error) => notify(error.message, "error"));
  }, DRIVE_POLL_MS);
}

function renderDriveSettings(status) {
  driveStatus = status || {};
  const configured = Boolean(driveStatus.configured);
  const authorizing = driveStatus.reconnect?.status === "authorizing";
  const folder = driveStatus.folder || "RedRHex Videos";
  const badge = document.querySelector("#drive-settings-badge");
  const summary = document.querySelector("#drive-settings-summary");
  const remote = document.querySelector("#drive-settings-remote");
  const accountTitle = document.querySelector("#drive-account-title");
  const reconnectState = document.querySelector("#drive-reconnect-state");
  const reconnectCommand = document.querySelector("#drive-reconnect-command");

  if (badge) {
    badge.textContent = authorizing ? "Authorizing" : configured ? "Connected" : "Needs setup";
    badge.className = authorizing
      ? "status-badge status-running"
      : configured
        ? "status-badge status-completed"
        : "status-badge status-interrupted";
  }
  if (remote) remote.textContent = driveStatus.remote || "redrhex-drive:";
  if (accountTitle) {
    accountTitle.textContent = authorizing
      ? "Waiting for Google sign-in"
      : configured
        ? "Google Drive connected"
        : "Google Drive not connected";
  }
  if (reconnectCommand) {
    reconnectCommand.textContent = driveStatus.reconnect_command || "rclone config reconnect redrhex-drive:";
  }

  if (summary) {
    if (!driveStatus.available || !configured) {
      summary.textContent = driveStatus.remediation || "Configure rclone on the training PC to enable exports.";
    } else if (authorizing) {
      summary.textContent = "Finish choosing the Google account on the training PC.";
    } else if (driveStatus.reconnect?.status === "failed") {
      summary.textContent = driveStatus.reconnect.error || "Google authorization did not complete.";
    } else {
      summary.textContent = driveStatus.destination_mode === "folder_link"
        ? "New History exports will use the folder from your pasted link."
        : "New History exports will use this folder in My Drive.";
    }
  }

  if (reconnectState) {
    const reconnect = driveStatus.reconnect || {};
    if (reconnect.status === "authorizing") {
      reconnectState.textContent = "A Google sign-in window is open on the training PC. Choose the account that should own new uploads.";
      reconnectState.dataset.state = "running";
    } else if (reconnect.status === "completed") {
      reconnectState.textContent = "Account changed. New History exports will use the newly approved Google account.";
      reconnectState.dataset.state = "success";
    } else if (reconnect.status === "failed") {
      reconnectState.textContent = reconnect.error || "Authorization failed. Retry or use the terminal command.";
      reconnectState.dataset.state = "error";
    } else {
      reconnectState.textContent = configured
        ? "This shared Mother connection is used for every one-touch History export."
        : "Connect a Google account on the training PC before exporting videos.";
      reconnectState.dataset.state = "idle";
    }
  }

  if (driveFolderInput && (!driveFolderDirty || document.activeElement !== driveFolderInput)) {
    driveFolderInput.value = configuredDestinationValue(driveStatus) || folder;
    driveFolderDirty = false;
  }
  if (driveFolderInput) driveFolderInput.disabled = !configured || authorizing;
  if (driveSaveButton) {
    driveSaveButton.disabled = !configured || authorizing || !driveFolderDirty;
  }
  if (driveReconnectButton) driveReconnectButton.disabled = !configured || authorizing;
  if (driveCheckButton) driveCheckButton.disabled = !driveStatus.available;
  if (driveOpenFolder) {
    const folderUrl = String(driveStatus.folder_url || "");
    driveOpenFolder.hidden = !folderUrl;
    driveOpenFolder.href = folderUrl || "#";
  }
  updateDrivePreview();
  scheduleDrivePoll();
}

async function loadDriveSettings() {
  const system = await requestJson("/api/system");
  renderDriveSettings(system.google_drive_export || {});
  return system.google_drive_export || {};
}

async function saveDriveLocation() {
  const destination = String(driveFolderInput?.value || "").trim();
  if (!destination) {
    notify("Paste a Google Drive folder link or enter a folder path.", "error");
    driveFolderInput?.focus();
    return;
  }
  if (driveSaveButton) driveSaveButton.disabled = true;
  try {
    const data = await requestJson("/api/google-drive/settings", {
      method: "POST",
      body: JSON.stringify({ destination }),
    });
    driveFolderDirty = false;
    renderDriveSettings(data.google_drive_export || {});
    if (typeof window.loadSystem === "function") await window.loadSystem();
    notify("Video folder updated. New History exports will use this destination.", "success");
  } finally {
    if (driveSaveButton) driveSaveButton.disabled = !driveFolderDirty;
  }
}

async function reconnectDriveAccount() {
  const confirm = typeof window.confirmAction === "function"
    ? await window.confirmAction({
        title: "Reconnect Google Drive account",
        body: "A Google sign-in window will open on the training PC. Choose the account that should own new uploads. Existing Drive files will not move.",
        confirmLabel: "Choose Account",
      })
    : window.confirm("Reconnect the Google Drive account on the training PC?");
  if (!confirm) return;
  if (driveReconnectButton) driveReconnectButton.disabled = true;
  const data = await requestJson("/api/google-drive/reconnect", {
    method: "POST",
    body: JSON.stringify({}),
  });
  renderDriveSettings(data.google_drive_export || { reconnect: data.reconnect });
  notify(
    data.reconnect?.status === "authorizing"
      ? "Choose the Google account in the sign-in window on the training PC."
      : "Google account changed.",
    "success",
  );
}

async function copyReconnectCommand() {
  const command = document.querySelector("#drive-reconnect-command")?.textContent || "rclone config reconnect redrhex-drive:";
  if (typeof window.copyText === "function") await window.copyText(command);
  else await navigator.clipboard.writeText(command);
  notify("Drive reconnect command copied.", "success");
}

async function refreshSettings() {
  if (settingsRefreshButton) settingsRefreshButton.disabled = true;
  try {
    const tasks = [loadDriveSettings()];
    if (typeof window.loadRemoteStatus === "function") tasks.push(window.loadRemoteStatus());
    await Promise.all(tasks);
    notify("Settings status refreshed.", "success");
  } finally {
    if (settingsRefreshButton) settingsRefreshButton.disabled = false;
  }
}

sectionButtons.forEach((button, index) => {
  button.addEventListener("click", () => activateSettingsSection(button.dataset.settingsSection));
  button.addEventListener("keydown", (event) => {
    if (!['ArrowDown', 'ArrowUp', 'ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    let next = index;
    if (event.key === "Home") next = 0;
    else if (event.key === "End") next = sectionButtons.length - 1;
    else if (event.key === "ArrowDown" || event.key === "ArrowRight") next = (index + 1) % sectionButtons.length;
    else next = (index - 1 + sectionButtons.length) % sectionButtons.length;
    activateSettingsSection(sectionButtons[next].dataset.settingsSection, { focus: true });
  });
});

driveFolderInput?.addEventListener("input", () => {
  driveFolderDirty = String(driveFolderInput.value || "").trim() !== configuredDestinationValue();
  if (driveSaveButton) driveSaveButton.disabled = !driveStatus?.configured || !driveFolderDirty;
  updateDrivePreview();
});
driveSaveButton?.addEventListener("click", () => saveDriveLocation().catch((error) => notify(error.message, "error")));
driveReconnectButton?.addEventListener("click", () => reconnectDriveAccount().catch((error) => notify(error.message, "error")));
driveCheckButton?.addEventListener("click", () => loadDriveSettings()
  .then(() => notify("Google Drive connection checked.", "success"))
  .catch((error) => notify(error.message, "error")));
document.querySelector("#copy-drive-reconnect")?.addEventListener("click", () => copyReconnectCommand().catch((error) => notify(error.message, "error")));
settingsRefreshButton?.addEventListener("click", () => refreshSettings().catch((error) => notify(error.message, "error")));

activateSettingsSection(sessionStorage.getItem(SETTINGS_SECTION_KEY) || "remote");
loadDriveSettings().catch((error) => notify(error.message, "error"));
