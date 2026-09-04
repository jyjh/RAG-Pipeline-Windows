// Admin tab: ops dashboard, API key management, index maintenance.

import { confirmAction, els, escapeHtml, formatBytes, formatBrowserTimestamp, formatKeyTimestamp, formatKeyUsage, apiKeyHeaderObject, getApiKey, markIndexDirty, refreshOpsDashboard, requestJson, setStatus, showGeneratedKeyDialog, showToast, state, toastError } from "./core.js";
import { applyUpdate, refreshJobs, updateShutdownBanner } from "./status.js";
import { refreshPdfs } from "./library.js";
import { refreshCategories } from "./categories.js";

async function enqueueReingest() {
  if (els.reingestButton) {
    els.reingestButton.disabled = true;
  }
  const confirmed = await confirmAction(
    "Re-ingest all PDFs?",
    "This re-runs PDF extraction (ingestion) and indexing for every registered PDF. " +
      "Use it after changing OCR/parser/vision settings or when extracted text looks wrong across many documents. " +
      "It runs as a background job and may take a while.",
    "Re-ingest all",
    { danger: true, requireText: "REINGEST" },
  );
  if (!confirmed) {
    if (els.reingestButton) {
      els.reingestButton.disabled = false;
    }
    return;
  }
  setMaintenanceStatus("Queueing full re-ingest...");
  try {
    const job = await requestJson("/api/reingest", { method: "POST" });
    setMaintenanceStatus(`Queued re-ingest job ${job.id.slice(0, 8)}.`);
    showToast(`Full re-ingest queued (job ${job.id.slice(0, 8)}).`, { kind: "info" });
    state.jobsOffset = 0;
    markIndexDirty();
    await refreshJobs({ force: true });
    await refreshPdfs({ force: true });
  } catch (error) {
    setMaintenanceStatus(error.message, true);
  } finally {
    if (els.reingestButton) {
      els.reingestButton.disabled = false;
    }
  }
}


// Stops the whole server after a countdown. Only the local operator sees the
// button (localhost browser), and the endpoint independently enforces the
// loopback restriction server-side — an admin key from the LAN gets a 403.
async function shutdownServer() {
  const confirmed = await confirmAction(
    "Shut down the server?",
    "The server process stops after a countdown (60 s). Running jobs are asked to stop and are finalized so they do NOT resume automatically after the next start — re-run them manually if needed. " +
      "Every open page shows a shutdown warning at its next status poll, and the app is unreachable until the server is started again.",
    "Shut down server",
    { danger: true, requireText: "SHUTDOWN" },
  );
  if (!confirmed) {
    return;
  }
  setMaintenanceStatus("Shutdown requested...");
  try {
    const data = await requestJson("/api/server/shutdown", { method: "POST" });
    updateShutdownBanner(data);
    showToast("Server shutdown countdown started.", { kind: "info" });
    setMaintenanceStatus(`Server stops in ${Math.round(Number(data?.delay_seconds) || 60)}s. Start it again manually.`);
  } catch (error) {
    setMaintenanceStatus(error.message, true);
  }
}


function setMaintenanceStatus(text, isError = false) {
  if (els.maintenanceStatus) {
    setStatus(els.maintenanceStatus, text, isError);
  }
}


async function enqueueBackup() {
  if (els.backupIndexButton) {
    els.backupIndexButton.disabled = true;
  }
  setMaintenanceStatus("Queueing index backup...");
  try {
    const job = await requestJson("/api/index/backup", { method: "POST" });
    setMaintenanceStatus(`Queued index backup job ${job.id.slice(0, 8)}.`);
    state.jobsOffset = 0;
    await refreshJobs({ force: true });
    await loadIndexBackups();
  } catch (error) {
    setMaintenanceStatus(error.message, true);
  } finally {
    if (els.backupIndexButton) {
      els.backupIndexButton.disabled = false;
    }
  }
}


async function enqueueRebuild() {
  if (els.rebuildIndexButton) {
    els.rebuildIndexButton.disabled = true;
  }
  const confirmed = await confirmAction(
    "Re-build the LanceDB index?",
    "This drops the current index and rebuilds it from the processed Markdown. " +
      "It runs as a background job and the index stays queryable until the rebuild publishes. " +
      "Use this if the index is corrupted.",
    "Re-build index",
    { danger: true, requireText: "REBUILD" },
  );
  if (!confirmed) {
    if (els.rebuildIndexButton) {
      els.rebuildIndexButton.disabled = false;
    }
    return;
  }
  setMaintenanceStatus("Queueing index rebuild...");
  try {
    const job = await requestJson("/api/index/rebuild", { method: "POST" });
    setMaintenanceStatus(`Queued index rebuild job ${job.id.slice(0, 8)}.`);
    showToast(`Index rebuild queued (job ${job.id.slice(0, 8)}).`, { kind: "info" });
    state.jobsOffset = 0;
    markIndexDirty();
    await refreshJobs({ force: true });
    await refreshPdfs({ force: true });
  } catch (error) {
    setMaintenanceStatus(error.message, true);
  } finally {
    if (els.rebuildIndexButton) {
      els.rebuildIndexButton.disabled = false;
    }
  }
}


function toggleRestorePanel(forceOpen = null) {
  if (!els.restoreIndexPanel) {
    return;
  }
  const willOpen = forceOpen === null ? els.restoreIndexPanel.hidden : forceOpen;
  els.restoreIndexPanel.hidden = !willOpen;
  if (willOpen) {
    loadIndexBackups();
  }
}


function formatBackupTimestamp(value) {
  // Server writes "YYYY-MM-DD HH:MM:SS"; the space separator is invalid in
  // strict ISO parsing, so hand Date a "T" before delegating.
  const normalized = String(value || "").trim().replace(" ", "T");
  return formatBrowserTimestamp(normalized) || "Unknown date";
}


async function loadIndexBackups() {
  if (!els.restoreBackupsList) {
    return;
  }
  els.restoreBackupsList.innerHTML = '<p class="restore-empty">Loading backups…</p>';
  try {
    const data = await requestJson("/api/index/backups");
    renderIndexBackups(data.backups || [], data.keep);
  } catch (error) {
    els.restoreBackupsList.innerHTML = `<p class="restore-empty error">${escapeHtml(error.message)}</p>`;
  }
}


function renderIndexBackups(backups, keep) {
  if (!els.restoreBackupsList) {
    return;
  }
  if (!backups.length) {
    els.restoreBackupsList.innerHTML =
      '<p class="restore-empty">No index backups yet. Use “Backup index” to create one.</p>';
    return;
  }
  const rows = backups
    .map((backup) => {
      const name = escapeHtml(backup.name || "");
      const date = escapeHtml(formatBackupTimestamp(backup.created_at));
      const recordCount =
        backup.record_count === null || backup.record_count === undefined
          ? "—"
          : `${backup.record_count} records`;
      const size = escapeHtml(formatBytes(backup.size_bytes));
      const corrupt = backup.lancedb_present === false;
      const note = corrupt
        ? '<span class="restore-tag danger">missing LanceDB</span>'
        : "";
      const restoreDisabled = corrupt ? " disabled" : "";
      return (
        `<div class="backup-row" data-backup-name="${name}">` +
          `<div class="backup-row-main">` +
            `<div class="backup-row-title">${name}</div>` +
            `<div class="backup-row-meta">${date} · ${escapeHtml(recordCount)} · ${size} ${note}</div>` +
          `</div>` +
          `<div class="backup-row-actions">` +
            `<button type="button" class="danger" data-backup-action="restore" data-backup-name="${name}"${restoreDisabled}>Restore</button>` +
          `</div>` +
        `</div>`
      );
    })
    .join("");
  const footer = keep
    ? `<p class="restore-footnote">The ${keep} most recent backups are kept automatically.</p>`
    : "";
  els.restoreBackupsList.innerHTML = rows + footer;
}


async function handleBackupAction(event) {
  const button = event.target.closest("button[data-backup-action]");
  if (!button) {
    return;
  }
  const action = button.dataset.backupAction;
  const backupName = button.dataset.backupName || "";
  if (action !== "restore" || !backupName) {
    return;
  }
  const confirmed = await confirmAction(
    "Restore this index backup?",
    `The current live index will be swapped out for “${backupName}”. A safety backup of the current index is taken first so this is reversible.`,
    "Restore index",
    { danger: true, requireText: "RESTORE" },
  );
  if (!confirmed) {
    return;
  }
  setMaintenanceStatus("Queueing index restore...");
  try {
    const job = await requestJson("/api/index/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backup_name: backupName }),
    });
    setMaintenanceStatus(`Queued index restore job ${job.id.slice(0, 8)}.`);
    state.jobsOffset = 0;
    markIndexDirty();
    await refreshJobs({ force: true });
    await refreshPdfs({ force: true });
  } catch (error) {
    setMaintenanceStatus(error.message, true);
  }
}


function adminMetricRow(label, value) {
  return `
    <div class="admin-metric">
      <span class="admin-metric-label">${escapeHtml(label)}</span>
      <span class="admin-metric-value">${value}</span>
    </div>
  `;
}


function adminCard(title, bodyHtml, extraClass = "") {
  return `
    <div class="admin-card ${extraClass}">
      <h3>${escapeHtml(title)}</h3>
      ${bodyHtml}
    </div>
  `;
}


function adminStatusBadge(ok, okText, badText) {
  return `<span class="status-badge ${ok ? "status-good" : "status-bad"}">${escapeHtml(ok ? okText : badText)}</span>`;
}


function permSetOptionsHtml(selected, { includeEmpty = false } = {}) {
  const sets = state.adminPermSets || [];
  if (!sets.length) {
    return `<option value="${escapeHtml(selected || "user")}" selected>${escapeHtml(selected || "user")}</option>`;
  }
  const options = sets.map((set) => {
    const name = String(set.name || "");
    const label = String(set.label || name);
    const isSelected = name === selected ? " selected" : "";
    return `<option value="${escapeHtml(name)}"${isSelected}>${escapeHtml(label)}</option>`;
  });
  if (includeEmpty && !sets.some((set) => set.name === selected)) {
    options.unshift(`<option value="${escapeHtml(selected)}" selected>${escapeHtml(selected)} (missing)</option>`);
  }
  return options.join("");
}


function renderAdminApiKeys(keys, masterConfigured) {
  const body = els.adminKeysBody;
  if (!body) {
    return;
  }
  if (!keys.length) {
    body.innerHTML =
      '<tr class="admin-keys-empty"><td colspan="8">No API keys issued yet. Create one above — remote clients need a key; this machine is always trusted.</td></tr>';
    return;
  }
  body.innerHTML = keys
    .map((key) => {
      const prefix = String(key.prefix || "");
      const status = String(key.status || "active");
      const permissionSet = String(key.permission_set || key.role || "user");
      const expires = key.expires_at ? formatKeyTimestamp(key.expires_at) : "never";
      const usage = formatKeyUsage(key.usage);
      const lastUsed = key.usage && key.usage.last_used_at ? formatKeyTimestamp(key.usage.last_used_at) : "—";
      const statusToggleLabel = status === "active" ? "Disable" : "Enable";
      const statusToggleStatus = status === "active" ? "disabled" : "active";
      return `
        <tr data-key-prefix="${escapeHtml(prefix)}">
          <td class="admin-key-prefix">${escapeHtml(prefix)}</td>
          <td>${escapeHtml(String(key.label || ""))}</td>
          <td>
            <select data-admin-key-action="set-permset" class="admin-key-role-select" aria-label="Permission set for ${escapeHtml(prefix)}">
              ${permSetOptionsHtml(permissionSet, { includeEmpty: true })}
            </select>
          </td>
          <td><span class="status-badge ${status === "active" ? "status-good" : "status-bad"}">${escapeHtml(status)}</span></td>
          <td>${escapeHtml(expires)}</td>
          <td>${escapeHtml(usage)}</td>
          <td>${escapeHtml(lastUsed)}</td>
          <td>
            <div class="admin-key-actions">
              <button type="button" data-admin-key-action="toggle-status" data-status="${statusToggleStatus}">${statusToggleLabel}</button>
              <button type="button" class="danger" data-admin-key-action="delete">Delete</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}


async function refreshAdminApiKeys(options = {}) {
  const status = els.adminKeysStatus;
  try {
    const data = await requestJson("/api/admin/api-keys", { headers: apiKeyHeaderObject() });
    state.adminKeysAuthorized = true;
    if (status) {
      setStatus(status, "");
    }
    if (els.adminKeysHint) {
      els.adminKeysHint.hidden = false;
    }
    renderAdminApiKeys(data.keys || [], data.master_configured);
  } catch (error) {
    state.adminKeysAuthorized = false;
    renderAdminApiKeys([], false);
    if (status) {
      const hint = error.status === 401 || error.status === 403
        ? "Admin role required. Set an admin API key below-left (API key box on the Documents tab) or configure the master token."
        : error.message;
      setStatus(status, hint, true);
    }
  }
}


function refreshAdminPanel(options = {}) {
  refreshOpsDashboard(options);
  refreshAdminApiKeys(options);
  refreshPermissionSets(options);
  refreshCategories({ quiet: false });
  refreshUpdatePanel();
  scheduleAdminAutoRefresh();
}


async function createAdminApiKey() {
  const label = (els.adminKeyLabelInput?.value || "").trim();
  const permissionSet = els.adminKeyPermSetSelect?.value || "user";
  const expiresRaw = (els.adminKeyExpiresInput?.value || "").trim();
  const rateRaw = (els.adminKeyRateInput?.value || "").trim();
  const payload = { label, permission_set: permissionSet };
  if (expiresRaw) {
    payload.expires_in_days = Number(expiresRaw);
  }
  if (rateRaw) {
    payload.rate_limit_per_minute = Number(rateRaw);
  }
  if (expiresRaw) {
    payload.expires_in_days = Number(expiresRaw);
  }
  if (rateRaw) {
    payload.rate_limit_per_minute = Number(rateRaw);
  }
  if (els.adminKeyCreateButton) {
    els.adminKeyCreateButton.disabled = true;
  }
  try {
    const data = await requestJson("/api/admin/api-keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showToast(`API key created for “${label || "unlabeled"}”.`, { kind: "success" });
    if (els.adminKeyLabelInput) {
      els.adminKeyLabelInput.value = "";
    }
    if (els.adminKeyExpiresInput) {
      els.adminKeyExpiresInput.value = "";
    }
    if (els.adminKeyRateInput) {
      els.adminKeyRateInput.value = "";
    }
    showGeneratedKeyDialog(String(data.key || ""));
    await refreshAdminApiKeys();
  } catch (error) {
    toastError(error);
    setStatus(els.adminKeysStatus, error.message, true);
  } finally {
    if (els.adminKeyCreateButton) {
      els.adminKeyCreateButton.disabled = false;
    }
  }
}


// -- permission sets ----------------------------------------------------------
// A permission set scopes a key to a category allowlist and carries write/admin
// flags. The editor doubles as create ("Create permission set") and edit
// (opened per-row); categories render as checkboxes, none checked = all.


function permSetCategoriesText(set) {
  const categories = Array.isArray(set.categories) ? set.categories : [];
  if (!categories.length || categories.includes("*")) {
    return "All categories";
  }
  return categories.join(", ");
}


function renderPermSetCategoryChecks(selected) {
  const box = els.adminPermSetCatsBox;
  if (!box) {
    return;
  }
  const categories = (state.categoriesCache || []).filter(
    (entry) => entry && entry.key && entry.key !== "__all__",
  );
  if (!categories.length) {
    box.innerHTML = '<span class="permset-cats-empty">No categories defined yet — the set will apply to all of them.</span>';
    return;
  }
  const selectedSet = Array.isArray(selected) ? selected : [];
  const checksAll = !selectedSet.length || selectedSet.includes("*");
  box.innerHTML = categories
    .map((entry) => {
      const key = String(entry.key);
      const label = String(entry.label || key);
      const checked = checksAll || selectedSet.includes(key) ? " checked" : "";
      return (
        `<label class="permset-cat-check">` +
        `<input type="checkbox" value="${escapeHtml(key)}"${checked} /> ${escapeHtml(label)}` +
        `</label>`
      );
    })
    .join("");
}


function resetPermSetEditor() {
  state.adminPermSetEditing = null;
  if (els.adminPermSetEditorSummary) {
    els.adminPermSetEditorSummary.textContent = "Create permission set";
  }
  if (els.adminPermSetNameInput) {
    els.adminPermSetNameInput.value = "";
    els.adminPermSetNameInput.disabled = false;
  }
  if (els.adminPermSetLabelInput) {
    els.adminPermSetLabelInput.value = "";
  }
  if (els.adminPermSetWriteCheck) {
    els.adminPermSetWriteCheck.checked = true;
  }
  if (els.adminPermSetAdminCheck) {
    els.adminPermSetAdminCheck.checked = false;
  }
  if (els.adminPermSetSaveButton) {
    els.adminPermSetSaveButton.textContent = "Create set";
  }
  if (els.adminPermSetCancelButton) {
    els.adminPermSetCancelButton.hidden = true;
  }
  renderPermSetCategoryChecks([]);
}


function openPermSetEditor(set) {
  state.adminPermSetEditing = String(set?.name || "");
  if (els.adminPermSetEditor) {
    els.adminPermSetEditor.open = true;
  }
  if (els.adminPermSetEditorSummary) {
    els.adminPermSetEditorSummary.textContent = `Edit permission set: ${state.adminPermSetEditing}`;
  }
  if (els.adminPermSetNameInput) {
    els.adminPermSetNameInput.value = state.adminPermSetEditing;
    // The name is the primary key of a set (keys reference it); edits are
    // limited to label/categories/flags.
    els.adminPermSetNameInput.disabled = true;
  }
  if (els.adminPermSetLabelInput) {
    els.adminPermSetLabelInput.value = String(set?.label || "");
  }
  if (els.adminPermSetWriteCheck) {
    els.adminPermSetWriteCheck.checked = Boolean(set?.can_write);
  }
  if (els.adminPermSetAdminCheck) {
    els.adminPermSetAdminCheck.checked = Boolean(set?.admin);
    // The builtin admin set's powers are part of the bootstrap story.
    els.adminPermSetAdminCheck.disabled = set?.builtin && set?.name === "admin";
  } else if (els.adminPermSetAdminCheck) {
    els.adminPermSetAdminCheck.disabled = false;
  }
  if (els.adminPermSetSaveButton) {
    els.adminPermSetSaveButton.textContent = "Save changes";
  }
  if (els.adminPermSetCancelButton) {
    els.adminPermSetCancelButton.hidden = false;
  }
  renderPermSetCategoryChecks(Array.isArray(set?.categories) ? set.categories : []);
}


async function refreshPermissionSets(options = {}) {
  const status = els.adminPermSetsStatus;
  try {
    const data = await requestJson("/api/admin/permission-sets", { headers: apiKeyHeaderObject() });
    state.adminPermSets = data.permission_sets || [];
    if (status) {
      setStatus(status, "");
    }
    renderAdminPermissionSets();
    renderPermSetCategoryChecks(
      state.adminPermSetEditing
        ? state.adminPermSets.find((set) => set.name === state.adminPermSetEditing)?.categories
        : [],
    );
    // Keep the key-issue form's set options in sync with the registry.
    const select = els.adminKeyPermSetSelect;
    if (select) {
      const previous = select.value || "user";
      select.innerHTML = permSetOptionsHtml(previous);
    }
  } catch (error) {
    state.adminPermSets = [];
    renderAdminPermissionSets();
    if (status) {
      const hint = error.status === 401 || error.status === 403
        ? "Admin required. Use an admin key, the master token, or manage sets from this machine."
        : error.message;
      setStatus(status, hint, true);
    }
  }
}


function renderAdminPermissionSets() {
  const body = els.adminPermSetsBody;
  if (!body) {
    return;
  }
  const sets = state.adminPermSets || [];
  if (!sets.length) {
    body.innerHTML =
      '<tr class="admin-keys-empty"><td colspan="7">No permission sets available. API key auth may be disabled in config.</td></tr>';
    return;
  }
  body.innerHTML = sets
    .map((set) => {
      const name = String(set.name || "");
      const builtin = Boolean(set.builtin);
      const isAdmin = Boolean(set.admin);
      const canWrite = Boolean(set.can_write);
      const keyCount = Number(set.key_count || 0);
      const deleteDisabled = builtin || keyCount > 0 ? " disabled" : "";
      const deleteTitle = builtin
        ? "Built-in sets cannot be deleted"
        : keyCount > 0
          ? "Reassign its keys before deleting this set"
          : "";
      return `
        <tr data-permset-name="${escapeHtml(name)}">
          <td class="admin-key-prefix">${escapeHtml(name)}${builtin ? ' <span class="status-badge">built-in</span>' : ""}</td>
          <td>${escapeHtml(String(set.label || name))}</td>
          <td>${escapeHtml(permSetCategoriesText(set))}</td>
          <td><span class="status-badge ${canWrite ? "status-good" : ""}">${canWrite ? "write" : "read-only"}</span></td>
          <td><span class="status-badge ${isAdmin ? "status-good" : ""}">${isAdmin ? "admin" : "—"}</span></td>
          <td>${keyCount}</td>
          <td>
            <div class="admin-key-actions">
              <button type="button" data-permset-action="edit">Edit</button>
              <button type="button" class="danger" data-permset-action="delete"${deleteDisabled} title="${escapeHtml(deleteTitle)}">Delete</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}


async function saveAdminPermSet() {
  const editing = state.adminPermSetEditing;
  const name = (els.adminPermSetNameInput?.value || "").trim();
  const label = (els.adminPermSetLabelInput?.value || "").trim();
  const canWrite = Boolean(els.adminPermSetWriteCheck?.checked);
  const isAdmin = Boolean(els.adminPermSetAdminCheck?.checked);
  const categories = Array.from(
    els.adminPermSetCatsBox?.querySelectorAll("input[type=checkbox]:checked") || [],
  ).map((input) => input.value);
  if (!editing && !name) {
    setStatus(els.adminPermSetsStatus, "A permission set needs a name.", true);
    return;
  }
  if (els.adminPermSetSaveButton) {
    els.adminPermSetSaveButton.disabled = true;
  }
  try {
    if (editing) {
      await requestJson(`/api/admin/permission-sets/${encodeURIComponent(editing)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label, categories, can_write: canWrite, admin: isAdmin }),
      });
      showToast(`Permission set “${editing}” updated.`, { kind: "success" });
    } else {
      await requestJson("/api/admin/permission-sets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, label, categories, can_write: canWrite, admin: isAdmin }),
      });
      showToast(`Permission set “${name}” created.`, { kind: "success" });
    }
    resetPermSetEditor();
    await refreshPermissionSets();
    await refreshAdminApiKeys();
  } catch (error) {
    toastError(error);
    setStatus(els.adminPermSetsStatus, error.message, true);
  } finally {
    if (els.adminPermSetSaveButton) {
      els.adminPermSetSaveButton.disabled = false;
    }
  }
}


async function handleAdminPermSetAction(event) {
  const control = event.target.closest("[data-permset-action]");
  if (!control || control.disabled) {
    return;
  }
  const row = control.closest("tr[data-permset-name]");
  const name = row?.dataset.permsetName || "";
  if (!name) {
    return;
  }
  const action = control.dataset.permsetAction;
  if (action === "edit") {
    const set = (state.adminPermSets || []).find((entry) => entry.name === name);
    if (set) {
      await refreshCategories({ quiet: true }).catch(() => {});
      openPermSetEditor(set);
    }
    return;
  }
  if (action === "delete") {
    const confirmed = await confirmAction(
      "Delete this permission set?",
      `Set “${name}” disappears from the key-assignment options. Sets still referenced by keys cannot be deleted.`,
      "Delete set",
      { danger: true },
    );
    if (!confirmed) {
      return;
    }
    try {
      await requestJson(`/api/admin/permission-sets/${encodeURIComponent(name)}`, { method: "DELETE" });
      showToast(`Deleted permission set “${name}”.`, { kind: "success" });
      if (state.adminPermSetEditing === name) {
        resetPermSetEditor();
      }
    } catch (error) {
      toastError(error);
    }
    await refreshPermissionSets();
  }
}


async function handleAdminKeyAction(event) {
  const control = event.target.closest("[data-admin-key-action]");
  if (!control) {
    return;
  }
  // The body listener fires for both click and change; a <select> must only
  // act on change. On click it would POST the CURRENT role immediately and
  // the subsequent re-render would destroy the open dropdown.
  if (event.type === "click" && control.tagName === "SELECT") {
    return;
  }
  if (control.tagName === "BUTTON" && control.disabled) {
    return;
  }
  const action = control.dataset.adminKeyAction || "";
  const row = control.closest("tr[data-key-prefix]");
  const prefix = row?.dataset.keyPrefix || "";
  if (!prefix) {
    return;
  }
  if (action === "set-permset") {
    // Fires on the select's change event; re-render below refreshes options.
    try {
      await requestJson(`/api/admin/api-keys/${encodeURIComponent(prefix)}/permission-set`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ permission_set: control.value }),
      });
      showToast(`Permission set for ${prefix} set to ${control.value}.`, { kind: "success" });
    } catch (error) {
      toastError(error);
    }
    await refreshAdminApiKeys();
    return;
  }
  if (action === "toggle-status") {
    const nextStatus = control.dataset.status || "disabled";
    try {
      await requestJson(`/api/admin/api-keys/${encodeURIComponent(prefix)}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: nextStatus }),
      });
      showToast(`${prefix} is now ${nextStatus}.`, { kind: "success" });
    } catch (error) {
      toastError(error);
    }
    await refreshAdminApiKeys();
    return;
  }
  if (action === "delete") {
    const confirmed = await confirmAction(
      "Delete this API key?",
      `Key ${prefix} stops working immediately. Any browser or script holding this secret must be re-issued a new key.`,
      "Delete key",
      { danger: true },
    );
    if (!confirmed) {
      return;
    }
    try {
      await requestJson(`/api/admin/api-keys/${encodeURIComponent(prefix)}`, { method: "DELETE" });
      showToast(`Deleted key ${prefix}.`, { kind: "success" });
    } catch (error) {
      toastError(error);
    }
    await refreshAdminApiKeys();
  }
}

export {
  adminCard,
  adminMetricRow,
  adminStatusBadge,
  createAdminApiKey,
  enqueueBackup,
  enqueueRebuild,
  enqueueReingest,
  formatBackupTimestamp,
  handleAdminKeyAction,
  handleAdminPermSetAction,
  handleBackupAction,
  loadIndexBackups,
  refreshAdminApiKeys,
  refreshAdminPanel,
  refreshPermissionSets,
  renderAdminApiKeys,
  renderAdminPermissionSets,
  renderIndexBackups,
  resetPermSetEditor,
  saveAdminPermSet,
  setMaintenanceStatus,
  shutdownServer,
  toggleRestorePanel,
};

// -- updates panel + auto-refresh --------------------------------------------
// The header pill stays a compact status; this panel shows the full picture:
// current vs. target commit, the exact blocking reason, and branch state.

let adminAutoRefreshTimer = null;

function renderUpdatePanel(status) {
  const body = document.getElementById("updatePanelBody");
  if (!body) {
    return;
  }
  const state = String(status?.state || "unknown");
  const rows = [
    ["Branch", `${status?.current_branch || "?"} (target ${status?.target_remote || "?"}/${status?.target_branch || "?"})`],
    ["Current commit", (status?.current_sha || "?").slice(0, 9)],
    ["Latest on target", status?.latest_sha ? String(status.latest_sha).slice(0, 9) : "—"],
    ["State", `<span class="status-badge ${state === "current" ? "status-good" : state === "error" || state === "blocked" ? "status-bad" : ""}">${escapeHtml(state)}</span>`],
  ]
    .map(([label, value]) => `<div class="admin-metric"><span class="admin-metric-label">${label}</span><span class="admin-metric-value">${value}</span></div>`)
    .join("");
  const message = status?.message
    ? `<p class="hint" style="margin:8px 0 0">${escapeHtml(String(status.message))}</p>`
    : "";
  const action = status?.can_update
    ? '<button type="button" id="updateNowButton" style="margin-top:10px">Update now</button>'
    : "";
  body.innerHTML = `<div class="admin-metric-list">${rows}</div>${message}${action}`;
  body.querySelector("#updateNowButton")?.addEventListener("click", () => applyUpdate());
  const refreshedAt = document.getElementById("updatePanelRefreshedAt");
  if (refreshedAt) {
    refreshedAt.textContent = `Checked ${new Date().toLocaleTimeString()}`;
  }
}

export async function refreshUpdatePanel() {
  const body = document.getElementById("updatePanelBody");
  if (!body) {
    return;
  }
  try {
    const status = await requestJson("/api/update/status");
    renderUpdatePanel(status);
  } catch (error) {
    body.innerHTML = `<p class="admin-metric-error">${escapeHtml(error.message)}</p>`;
  }
}

export function scheduleAdminAutoRefresh() {
  cancelAdminAutoRefresh();
  adminAutoRefreshTimer = window.setInterval(() => {
    if (state.activeTab === "admin" && !document.hidden) {
      refreshOpsDashboard({ force: true });
      refreshUpdatePanel();
    }
  }, 30000);
}

export function cancelAdminAutoRefresh() {
  if (adminAutoRefreshTimer) {
    window.clearInterval(adminAutoRefreshTimer);
    adminAutoRefreshTimer = null;
  }
}

// Compaction reclaims space from tombstoned rows after incremental reindexes.
export async function enqueueCompact() {
  if (els.compactIndexButton) {
    els.compactIndexButton.disabled = true;
  }
  const confirmed = await confirmAction(
    "Compact the index?",
    "Merges LanceDB fragments and drops rows left behind by incremental reindexes. Queries pause briefly while compaction runs.",
    "Compact",
    { danger: false },
  );
  if (!confirmed) {
    if (els.compactIndexButton) {
      els.compactIndexButton.disabled = false;
    }
    return;
  }
  setMaintenanceStatus("Queueing index compaction...");
  try {
    const job = await requestJson("/api/index/compact", { method: "POST" });
    setMaintenanceStatus(`Queued compaction job ${String(job.id || "").slice(0, 8)}.`);
    showToast(`Index compaction queued (job ${String(job.id || "").slice(0, 8)}).`, { kind: "info" });
    await refreshJobs({ force: true });
  } catch (error) {
    setMaintenanceStatus(error.message, true);
  } finally {
    if (els.compactIndexButton) {
      els.compactIndexButton.disabled = false;
    }
  }
}

// Re-trains the ANN index (no re-embedding) after many incremental reindexes.
export async function enqueueRebuildVectorIndex() {
  if (els.rebuildVectorIndexButton) {
    els.rebuildVectorIndexButton.disabled = true;
  }
  const confirmed = await confirmAction(
    "Rebuild the vector index?",
    "Re-trains the ANN partitions without re-embedding. Cheaper than a full rebuild but briefly blocks queries.",
    "Rebuild vector index",
  );
  if (!confirmed) {
    if (els.rebuildVectorIndexButton) {
      els.rebuildVectorIndexButton.disabled = false;
    }
    return;
  }
  setMaintenanceStatus("Queueing vector index rebuild...");
  try {
    const job = await requestJson("/api/index/rebuild_vector_index", { method: "POST" });
    setMaintenanceStatus(`Queued vector index rebuild job ${String(job.id || "").slice(0, 8)}.`);
    showToast(`Vector index rebuild queued (job ${String(job.id || "").slice(0, 8)}).`, { kind: "info" });
    await refreshJobs({ force: true });
  } catch (error) {
    setMaintenanceStatus(error.message, true);
  } finally {
    if (els.rebuildVectorIndexButton) {
      els.rebuildVectorIndexButton.disabled = false;
    }
  }
}
