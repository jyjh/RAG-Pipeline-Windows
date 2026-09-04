// Server status: health/update polling, job queue table, jobs strip, notifications.

import { ANSWER_PRESET_LABELS, MIN_SERVER_POLL_INTERVAL_MS, confirmAction, els, escapeHtml, firstErrorText, formatEta, getCookie, markIndexDirty, numericSetting, requestJson, setCookie, setStatus, showToast, sleep, state, toastError, updatePageControls } from "./core.js";
import { refreshPdfs } from "./library.js";
import { renderJobRows } from "./chat.js";
import { SITE_VERSION_COOKIE, activateTab, showCachePrompt, welcomeTutorialPromptOpen } from "./shell.js";

const UPDATE_POLL_INTERVAL_MS = 5 * 60 * 1000;

const RESTART_POLL_INTERVAL_MS = 1000;

const RESTART_POLL_TIMEOUT_MS = 120000;

const JOBS_ACTIVE_POLL_INTERVAL_MS = 2000;
// Must match TERMINAL_JOB_STATUSES in rag_job_queue.py.

const TERMINAL_JOB_STATUSES = new Set(["done", "failed", "cancelled"]);

const ACTIVE_JOB_STATUSES = new Set(["queued", "running", "paused_for_queries"]);

const JOB_WATCH_LIMIT = 300;
// Lower bound for server-configured poll intervals (see positiveInterval).

async function refreshHealth() {
  try {
    const data = await requestJson("/api/health");
    if (data.notModified) {
      return;
    }
    updateShutdownBanner(data.shutting_down || null);
    applyServerConfig(data.server || {});
    applyChatConfig(data.chat || {});
    const queue = data.queue || {};
    els.statusLine.textContent =
      `${data.record_count} indexed chunks | ` +
      `${queue.active_query_count || 0} active queries | ` +
      `${queue.queued_count || 0} queued jobs`;
    notePollSuccess();
  } catch (error) {
    // If a shutdown was already announced, a failing health poll most likely
    // means the server has now stopped — switch the banner to its offline
    // form instead of the generic error line. (No-op when not latched.)
    noteShutdownOffline();
    notePollFailure();
    els.statusLine.textContent = `Health check failed: ${error.message}`;
  }
}


// -- server shutdown banner ---------------------------------------------------
// The banner is driven by /api/health (the one endpoint every open tab polls),
// so anyone using the app sees that the server is stopping. Once a shutdown is
// seen it is latched for the page's lifetime: a transient poll failure keeps
// the warning on screen instead of hiding it.

let shutdownLatched = false;

let shutdownCountdownTimer = null;


function stopShutdownCountdown() {
  if (shutdownCountdownTimer) {
    clearInterval(shutdownCountdownTimer);
    shutdownCountdownTimer = null;
  }
}


function updateShutdownBanner(shutdown) {
  const banner = els.shutdownBanner;
  if (!banner) {
    return;
  }
  if (shutdown && shutdown.active) {
    shutdownLatched = true;
    stopShutdownCountdown();
    banner.hidden = false;
    banner.classList.remove("offline");
    const deadline = Date.parse(String(shutdown.shutdown_at || ""));
    const text = els.shutdownBannerText;
    if (Number.isFinite(deadline)) {
      const render = () => {
        const remaining = Math.max(0, Math.round((deadline - Date.now()) / 1000));
        if (text) {
          text.textContent = remaining > 0
            ? `The server stops in ~${remaining}s. Active jobs are being finalized so they do not restart on the next boot.`
            : "The server is stopping now. Start it again manually to keep working.";
        }
        if (remaining <= 0) {
          stopShutdownCountdown();
        }
      };
      render();
      shutdownCountdownTimer = setInterval(render, 1000);
    } else if (text) {
      text.textContent =
        "The server is shutting down. Active jobs are being finalized so they do not restart on the next boot.";
    }
    return;
  }
  // A successful poll with no active shutdown: either nothing was ever
  // requested or a fresh server process answered after a restart — clear the
  // latch so the stale warning does not stick around forever.
  if (shutdownLatched) {
    shutdownLatched = false;
    stopShutdownCountdown();
    banner.hidden = true;
    banner.classList.remove("offline");
  }
}


function noteShutdownOffline() {
  const banner = els.shutdownBanner;
  if (!banner || !shutdownLatched) {
    return;
  }
  stopShutdownCountdown();
  banner.hidden = false;
  banner.classList.add("offline");
  if (els.shutdownBannerText) {
    els.shutdownBannerText.textContent =
      "The server is offline. Start it again manually to keep working; this page will reconnect once it is back.";
  }
}


function applyChatConfig(config) {
  if (config.context_window && !els.contextWindowInput.dataset.configApplied) {
    els.contextWindowInput.value = String(config.context_window);
    els.contextWindowInput.dataset.configApplied = "true";
  }
  if (config.llm_num_predict && !els.maxOutputInput.dataset.configApplied) {
    els.maxOutputInput.value = String(config.llm_num_predict);
    els.maxOutputInput.dataset.configApplied = "true";
  }
  if (config.retrieval_min_score !== undefined && !els.relevanceFloorInput.dataset.configApplied) {
    els.relevanceFloorInput.value = String(config.retrieval_min_score);
    els.relevanceFloorInput.dataset.configApplied = "true";
  }
  updateComposerSettingsSummary();
}

// -- answer-mode presets ------------------------------------------------------
// Friendly names over the raw sampler knobs. Manual edits flip the select to
// "custom" without overwriting the persisted preset, so a reload returns to
// the mode the user actually chose.


function updateComposerSettingsSummary() {
  if (!els.composerSettingsSummary) {
    return;
  }
  const presetId = els.answerPresetSelect.value;
  const label = ANSWER_PRESET_LABELS[presetId] || presetId;
  const temp = numericSetting(els.temperatureInput, 0.3, 0);
  const topK = Math.trunc(numericSetting(els.maxKInput, 40, 1));
  const web = els.webSearchInput.checked ? "web on" : "web off";
  // Category scope: General + custom indexes. A subset selection narrows what
  // an answer can see, so surface it next to the other sampler facts.
  const totalCategories = Array.isArray(state.categoriesCache)
    ? state.categoriesCache.filter((entry) => entry && entry.key !== "general").length + 1
    : 0;
  let scope = "";
  if (totalCategories > 1 && Array.isArray(state.chatSelectedCategories)) {
    scope = ` · ${state.chatSelectedCategories.length}/${totalCategories} categories`;
  }
  els.composerSettingsSummary.textContent = `${label} · temp ${temp} · top-k ${topK} · ${web}${scope}`;
}


function positiveInterval(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  // Floor server-supplied intervals: a config typo (e.g. 100) would otherwise
  // become a permanent 10 req/s poll per open tab.
  return Math.max(parsed, MIN_SERVER_POLL_INTERVAL_MS);
}


function applyServerConfig(config) {
  const nextHealth = positiveInterval(config.health_poll_interval_ms, state.healthPollIntervalMs);
  const nextJobs = positiveInterval(config.jobs_poll_interval_ms, state.jobsPollIntervalMs);

  if (nextHealth !== state.healthPollIntervalMs) {
    state.healthPollIntervalMs = nextHealth;
    scheduleHealthPolling();
  }
  if (nextJobs !== state.jobsPollIntervalMs) {
    state.jobsPollIntervalMs = nextJobs;
    scheduleJobsPolling();
  }
}


function setUpdateButton(kind, text, title, disabled = false) {
  els.updateButton.className = `update-button update-${kind}`;
  els.updateButton.textContent = text;
  els.updateButton.title = title || text;
  els.updateButton.disabled = disabled;
}


function shortSha(value) {
  return value ? String(value).slice(0, 7) : "";
}


function renderUpdateStatus(data) {
  const message = data.message || "Update status unavailable.";
  handleSiteVersionFromUpdateStatus(data);
  if (data.state === "current") {
    const sha = shortSha(data.current_sha);
    setUpdateButton("current", sha ? `Current ${sha}` : "Current", message, true);
    return;
  }
  if (data.state === "available" && data.can_update) {
    const latest = shortSha(data.latest_sha);
    setUpdateButton("available", latest ? `Update ${latest}` : "Update", message);
    return;
  }
  if (data.state === "blocked") {
    setUpdateButton("warning", "Blocked", message, true);
    return;
  }
  if (data.state === "error") {
    setUpdateButton("error", "Update error", message, true);
    return;
  }
  setUpdateButton("warning", "Update", message, true);
}


async function refreshUpdateStatus() {
  if (state.updateApplying) {
    return;
  }
  setUpdateButton("checking", "Checking", "Checking for updates", true);
  try {
    const data = await requestJson("/api/update/status");
    renderUpdateStatus(data);
  } catch (error) {
    setUpdateButton("error", "Update error", error.message, true);
  }
}


async function waitForRestart() {
  const startedAt = Date.now();
  let sawServerDown = false;
  while (Date.now() - startedAt < RESTART_POLL_TIMEOUT_MS) {
    await sleep(RESTART_POLL_INTERVAL_MS);
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      if (response.ok && (sawServerDown || Date.now() - startedAt > 3000)) {
        return;
      }
    } catch (_) {
      sawServerDown = true;
    }
  }
  throw new Error("Timed out waiting for the restarted server.");
}


async function applyUpdate() {
  if (state.updateApplying) {
    return;
  }
  state.updateApplying = true;
  setUpdateButton("restarting", "Updating", "Pulling latest commit and restarting", true);
  try {
    const data = await requestJson("/api/update/apply", { method: "POST" });
    setUpdateButton("restarting", "Restarting", data.message || "Restarting server", true);
    await waitForRestart();
    window.location.reload();
  } catch (error) {
    state.updateApplying = false;
    setUpdateButton("error", "Update error", error.message, true);
  }
}


function rememberJobLogOpenState() {
  const liveIds = new Set();
  els.jobsBody.querySelectorAll("details.job-log[data-job-id]").forEach((details) => {
    const jobId = details.dataset.jobId || "";
    if (!jobId) {
      return;
    }
    liveIds.add(jobId);
    if (details.open) {
      state.openJobLogIds.add(jobId);
    } else {
      state.openJobLogIds.delete(jobId);
    }
  });
  for (const jobId of Array.from(state.openJobLogIds)) {
    if (!liveIds.has(jobId)) {
      state.openJobLogIds.delete(jobId);
    }
  }
}


function formatJobProgress(job) {
  const p = job.progress;
  if (!p || typeof p !== "object") return "";
  const done = Number(p.done || 0);
  const total = Number(p.total || 0);
  const unit = escapeHtml(String(p.unit || "items"));
  const phase = escapeHtml(String(p.phase || ""));
  const rate = Number(p.rate_per_min || 0);
  const eta = Number(p.eta_seconds || 0);
  const pct = total > 0 ? Math.max(0, Math.min(100, Math.round((done / total) * 100))) : null;
  const parts = [];
  if (phase) parts.push(phase);
  if (total > 0) {
    parts.push(`${done.toLocaleString()}/${total.toLocaleString()} ${unit} (${pct}%)`);
  } else if (done > 0) {
    parts.push(`${done.toLocaleString()} ${unit}`);
  }
  if (rate > 0) parts.push(`${Math.round(rate).toLocaleString()}/${unit.split(" ")[0]}/min`);
  if (eta > 0) parts.push(`ETA ${formatEta(eta)}`);
  // Surface cumulative record counters from the indexer's `extra` payload so a
  // long indexing run shows records written/embedded alongside file progress.
  const extra = p.records_written != null ? ` | ${Number(p.records_written).toLocaleString()} recs` : "";
  const text = parts.join(" · ") + extra;
  if (!text.trim()) {
    return "";
  }
  const bar = pct === null
    ? '<div class="progress-track progress-indeterminate" aria-hidden="true"><div class="progress-fill"></div></div>'
    : `<div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}" aria-label="Job progress"><div class="progress-fill" style="width: ${pct}%"></div></div>`;
  return `<div class="job-progress">${bar}<div class="progress-text">${text}</div></div>`;
}

// Compact single-line variant for the global jobs strip (no HTML bar).

function formatJobProgressText(job) {
  const p = job && job.progress;
  if (!p || typeof p !== "object") {
    return "";
  }
  const done = Number(p.done || 0);
  const total = Number(p.total || 0);
  const unit = String(p.unit || "items");
  const eta = Number(p.eta_seconds || 0);
  const parts = [];
  if (total > 0) {
    const pct = Math.max(0, Math.min(100, Math.round((done / total) * 100)));
    parts.push(`${done.toLocaleString()}/${total.toLocaleString()} ${unit} (${pct}%)`);
  } else if (done > 0) {
    parts.push(`${done.toLocaleString()} ${unit}`);
  }
  if (eta > 0) parts.push(`ETA ${formatEta(eta)}`);
  return parts.join(" · ");
}

// -- job outcome notifications ----------------------------------------------
// Diff each poll's job statuses against the last seen state and toast (plus
// desktop-notify when the page is hidden) on transitions to a terminal state.


function trackJobTransitions(jobs) {
  for (const job of jobs || []) {
    const jobId = String(job.id || "");
    const status = String(job.status || "");
    if (!jobId || !status) {
      continue;
    }
    const previous = state.jobWatch.get(jobId);
    state.jobWatch.set(jobId, status);
    if (previous && previous !== status && TERMINAL_JOB_STATUSES.has(status) && !TERMINAL_JOB_STATUSES.has(previous)) {
      notifyJobOutcome(jobId, status, job);
    }
  }
  // Pagination means jobs rotate out of the fetched page; cap the map so it
  // cannot grow without bound across a long-lived tab.
  if (state.jobWatch.size > JOB_WATCH_LIMIT) {
    const excess = state.jobWatch.size - JOB_WATCH_LIMIT;
    let dropped = 0;
    for (const key of state.jobWatch.keys()) {
      if (dropped >= excess) {
        break;
      }
      state.jobWatch.delete(key);
      dropped += 1;
    }
  }
  // Keep the log-hydration timestamps bounded with the same discipline: one
  // entry per job ever rendered would otherwise live for the tab's lifetime.
  if (jobLogHydratedAt.size > JOB_WATCH_LIMIT) {
    const excess = jobLogHydratedAt.size - JOB_WATCH_LIMIT;
    let dropped = 0;
    for (const key of jobLogHydratedAt.keys()) {
      if (dropped >= excess) {
        break;
      }
      jobLogHydratedAt.delete(key);
      dropped += 1;
    }
  }
}


// A drained queue can fail dozens of recovered jobs within seconds; one
// full-width error toast per job buried the whole UI. Failures arriving in
// quick succession are batched into a single summary toast instead.
const FAILED_TOAST_COALESCE_MS = 1500;
const FAILED_TOAST_ERROR_CHARS = 90;
let failedToastTimer = null;
const pendingFailedToasts = [];


function flushFailedToasts() {
  failedToastTimer = null;
  const batch = pendingFailedToasts.splice(0, pendingFailedToasts.length);
  if (!batch.length) {
    return;
  }
  if (batch.length === 1) {
    const { label, detail } = batch[0];
    showToast(`${label} failed${detail ? `: ${detail}` : ""}.`, {
      kind: "error",
      onClick: () => activateTab("upload"),
    });
    return;
  }
  showToast(`${batch.length} jobs failed — open Documents for the errors.`, {
    kind: "error",
    onClick: () => activateTab("upload"),
  });
}


function notifyJobOutcome(jobId, status, job) {
  const shortId = jobId.slice(0, 8);
  const names = Array.isArray(job.filenames) ? job.filenames.filter(Boolean) : [];
  const label = names.length ? names.slice(0, 2).join(", ") + (names.length > 2 ? ` +${names.length - 2}` : "") : `Job ${shortId}`;
  if (status === "done") {
    showToast(`${label} finished.`, { kind: "success", onClick: () => activateTab("upload") });
  } else if (status === "failed") {
    const firstErrorLine = String(job.error || "").split("\n")[0].trim().slice(0, FAILED_TOAST_ERROR_CHARS);
    pendingFailedToasts.push({ label, detail: firstErrorLine });
    if (!failedToastTimer) {
      failedToastTimer = setTimeout(flushFailedToasts, FAILED_TOAST_COALESCE_MS);
    }
  } else if (status === "cancelled") {
    showToast(`${label} was cancelled.`, { kind: "info" });
  }
  // Desktop notification only matters while the tab is in the background;
  // the toast above already covers the visible case.
  if (document.hidden && "Notification" in window && Notification.permission === "granted") {
    try {
      const title = status === "done" ? "Job finished" : status === "failed" ? "Job failed" : "Job cancelled";
      const bodyText = status === "failed" && firstErrorText(job) ? firstErrorText(job) : label;
      new Notification(`Local FSAE RAG — ${title}`, { body: bodyText });
    } catch (_) {
      // Notification construction can throw (e.g. closed permissions UI); ignore.
    }
  }
}


function updateJobsStrip(jobs, activeCount) {
  const strip = els.jobsStrip;
  if (!strip) {
    return;
  }
  const activeJobs = (jobs || []).filter((job) => ACTIVE_JOB_STATUSES.has(String(job.status || "")));
  if (!activeCount && !activeJobs.length) {
    strip.hidden = true;
    return;
  }
  const running =
    activeJobs.find((job) => String(job.status) === "running") ||
    activeJobs.find((job) => String(job.status) === "paused_for_queries") ||
    activeJobs[0];
  const phase = String((running && running.phase) || running.status || "working");
  const progressText = running ? formatJobProgressText(running) : "";
  const extraCount = Math.max(0, (Number(activeCount) || activeJobs.length) - 1);
  const extraLabel = extraCount > 0 ? ` · +${extraCount} more job${extraCount === 1 ? "" : "s"}` : "";
  els.jobsStripText.textContent = `${phase}${progressText ? ` — ${progressText}` : ""}${extraLabel}`;
  strip.hidden = false;
}


function createJobRow(job) {
  const row = document.createElement("tr");
  const jobId = String(job.id || "");
  const canCancel = ["queued", "running", "paused_for_queries"].includes(String(job.status || ""))
    && !job.cancel_requested;
  const cancelButton = canCancel
    ? `<button type="button" class="danger" data-job-action="cancel" data-job-id="${escapeHtml(jobId)}">Cancel</button>`
    : "";
  const logTail = String(job.log_tail || "").trim();
  const logLineCount = Number(job.log_line_count || 0);
  const logOpen = state.openJobLogIds.has(jobId) ? " open" : "";
  // The list response omits log_tail entirely (it made every 2s poll ship
  // every visible job's whole tail); hydrateJobLogs fills open panels from
  // GET /api/jobs/{id} instead.
  const logBlock = (logTail || logLineCount)
    ? `<details class="job-log" data-job-id="${escapeHtml(jobId)}"${logOpen}><summary>Log (${logLineCount} lines)</summary><pre>${escapeHtml(logTail)}</pre></details>`
    : "";
  const progressBlock = formatJobProgress(job);
  row.innerHTML = `
    <td>${escapeHtml(jobId.slice(0, 8))}</td>
    <td>${escapeHtml(job.status)}</td>
    <td>${escapeHtml(job.phase)}${progressBlock}</td>
    <td>${escapeHtml((job.filenames || []).join(", "))}</td>
    <td>${escapeHtml(job.error || "")}${logBlock}</td>
    <td><div class="job-actions">${cancelButton}</div></td>
  `;
  row.dataset.jobStatus = String(job.status || "");
  return row;
}


const jobLogHydratedAt = new Map();


async function hydrateJobLogs() {
  // Fill empty log panels for open details elements from the job detail
  // endpoint. Terminal jobs keep their fetched tail; active ones refresh at
  // most every 5s so an open panel still tails a running ingest.
  const now = Date.now();
  els.jobsBody.querySelectorAll("details.job-log[data-job-id]").forEach((details) => {
    if (!details.open || details.dataset.hydrating) {
      return;
    }
    const jobId = details.dataset.jobId || "";
    const pre = details.querySelector("pre");
    if (!jobId || !pre) {
      return;
    }
    const row = details.closest("tr");
    const statusActive = Boolean(row) && ["queued", "running", "paused_for_queries"].includes(String(row.dataset.jobStatus || ""));
    if (pre.textContent && !statusActive) {
      return;
    }
    const last = jobLogHydratedAt.get(jobId) || 0;
    if (pre.textContent && statusActive && now - last < 5000) {
      return;
    }
    details.dataset.hydrating = "1";
    requestJson(`/api/jobs/${encodeURIComponent(jobId)}`).then((job) => {
      pre.textContent = String((job && job.log_tail) || "").trim();
      const summary = details.querySelector("summary");
      if (summary) {
        summary.textContent = `Log (${Number((job && job.log_line_count) || 0)} lines)`;
      }
      jobLogHydratedAt.set(jobId, Date.now());
    }).catch(() => {}).finally(() => {
      delete details.dataset.hydrating;
    });
  });
}


async function refreshJobs(options = {}) {
  const onUpload = state.activeTab === "upload";
  // Off the Documents tab: stay quiet unless jobs are active — the 2s active
  // poll keeps the global jobs strip and completion toasts fresh from any tab.
  if (!onUpload && !options.force && !state.jobsActive) {
    state.uploadDataDirty = true;
    return;
  }
  const renderTable = onUpload || options.force;
  // Stale-response token: a newer call (rapid paging, forced refresh) must
  // keep an older slow response from overwriting the fresher table state.
  const fetchSeq = ++state.jobsFetchSeq;
  try {
    const isAll = state.jobsPageSize === "all";
    state.jobsLimit = isAll ? 0 : (Number(state.jobsPageSize) || 10);
    // Background polling always watches the newest page so completions are
    // detected regardless of which page the table was left on.
    const offset = renderTable ? state.jobsOffset : 0;
    const limit = renderTable ? state.jobsLimit : 10;
    const search = renderTable ? state.jobSearch : "";
    const params = new URLSearchParams({
      offset: String(offset),
      limit: String(limit),
      search,
    });
    const url = `/api/jobs?${params}`;
    const data = await requestJson(url);
    if (fetchSeq !== state.jobsFetchSeq) {
      return;
    }
    state.jobsLoaded = true;
    notePollSuccess();
    const wasActive = state.jobsActive;
    state.jobsActive = Number(data.active_count || 0) > 0;
    if (data.notModified && state.jobsRenderedUrl === url) {
      if (renderTable) {
        hydrateJobLogs();
      }
      return;
    }
    trackJobTransitions(data.jobs || []);
    updateJobsStrip(data.jobs || [], Number(data.active_count || 0));
    if (!renderTable) {
      // Table state is refetched when the user returns to Documents.
      state.uploadDataDirty = true;
      if (wasActive !== state.jobsActive) {
        scheduleJobsPolling();
      }
      return;
    }
    state.jobsTotal = data.total || 0;
    if (!isAll && state.jobsOffset >= state.jobsTotal && state.jobsOffset > 0) {
      state.jobsOffset = Math.max(0, Math.floor((state.jobsTotal - 1) / state.jobsLimit) * state.jobsLimit);
      return refreshJobs({ force: true });
    }
    renderJobRows(data.jobs || []);
    state.jobsRenderedUrl = url;
    hydrateJobLogs();
    if (isAll) {
      const shown = (data.jobs || []).length;
      els.jobsPageLabel.textContent = shown < state.jobsTotal
        ? `All ${shown} of ${state.jobsTotal} jobs`
        : `All ${state.jobsTotal} jobs`;
      els.prevJobsPageButton.disabled = true;
      els.nextJobsPageButton.disabled = true;
    } else {
      updatePageControls({
        total: state.jobsTotal,
        offset: state.jobsOffset,
        limit: state.jobsLimit,
        label: els.jobsPageLabel,
        prevButton: els.prevJobsPageButton,
        nextButton: els.nextJobsPageButton,
      });
    }
    if (wasActive && !state.jobsActive) {
      markIndexDirty();
      // Fire-and-forget: the full-corpus refetch must not serialize behind
      // (or delay) the poll loop that noticed the transition, and a burst of
      // finishing jobs would otherwise trigger it repeatedly back-to-back.
      if (state.jobTransitionRefreshTimer) {
        clearTimeout(state.jobTransitionRefreshTimer);
      }
      state.jobTransitionRefreshTimer = setTimeout(() => {
        state.jobTransitionRefreshTimer = null;
        refreshPdfs({ force: true }).then(refreshHealth).catch(() => {});
      }, 1500);
    }
    if (wasActive !== state.jobsActive) {
      scheduleJobsPolling();
    }
  } catch (error) {
    notePollFailure();
    setStatus(els.uploadStatus, error.message, true);
  }
}


async function handleJobAction(event) {
  const button = event.target.closest("[data-job-action]");
  if (!button) {
    return;
  }
  const action = button.dataset.jobAction || "";
  const jobId = button.dataset.jobId || "";
  if (action !== "cancel" || !jobId) {
    return;
  }
  const confirmed = await confirmAction(
    "Cancel this job?",
    "Queued work that has not started yet is removed; a running job is asked to stop at the next safe checkpoint.",
    "Cancel job",
    { danger: true },
  );
  if (!confirmed) {
    return;
  }
  button.disabled = true;
  try {
    const job = await requestJson(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    });
    showToast(`Cancelled job ${String(job.id || jobId).slice(0, 8)}.`, { kind: "success" });
    markIndexDirty();
    await refreshJobs({ force: true });
    await refreshPdfs({ force: true });
  } catch (error) {
    toastError(error);
  } finally {
    button.disabled = false;
  }
}


function pollBackoffMultiplier() {
  return Math.min(1 + state.pollFailureCount, 30);
}


function notePollFailure() {
  state.pollFailureCount += 1;
  scheduleHealthPolling();
  scheduleJobsPolling();
}


function notePollSuccess() {
  if (state.pollFailureCount) {
    state.pollFailureCount = 0;
    scheduleHealthPolling();
    scheduleJobsPolling();
  }
}


function scheduleHealthPolling() {
  if (state.healthTimer) {
    clearInterval(state.healthTimer);
  }
  const interval = state.healthPollIntervalMs * pollBackoffMultiplier();
  state.healthTimer = setInterval(refreshHealth, interval);
}


function scheduleJobsPolling() {
  const base = state.jobsActive ? JOBS_ACTIVE_POLL_INTERVAL_MS : state.jobsPollIntervalMs;
  const interval = base * pollBackoffMultiplier();
  if (state.jobsTimer && state.jobsTimerIntervalMs === interval) {
    return;
  }
  if (state.jobsTimer) {
    clearInterval(state.jobsTimer);
  }
  state.jobsTimerIntervalMs = interval;
  state.jobsTimer = setInterval(refreshJobs, interval);
}


function scheduleUpdatePolling() {
  if (state.updateTimer) {
    clearInterval(state.updateTimer);
  }
  state.updateTimer = setInterval(refreshUpdateStatus, UPDATE_POLL_INTERVAL_MS);
}


function handleVisibilityChange() {
  if (document.hidden) {
    // Tear the timers down entirely so a background tab stops network traffic
    // (previously the 2s active-job poll kept running for a whole ingest).
    if (state.healthTimer) {
      clearInterval(state.healthTimer);
      state.healthTimer = null;
    }
    if (state.jobsTimer) {
      clearInterval(state.jobsTimer);
      state.jobsTimer = null;
      state.jobsTimerIntervalMs = null;
    }
    if (state.updateTimer) {
      clearInterval(state.updateTimer);
      state.updateTimer = null;
    }
    return;
  }
  // Catch up immediately on return, then restore the normal cadence.
  refreshHealth();
  refreshJobs();
  refreshUpdateStatus();
  scheduleHealthPolling();
  scheduleUpdatePolling();
  scheduleJobsPolling();
}


function siteVersionFromStatus(data) {
  const current = String(data?.current_sha || "").trim();
  return current || "";
}


function handleSiteVersionFromUpdateStatus(data) {
  const version = siteVersionFromStatus(data);
  if (!version) {
    return;
  }
  const previous = getCookie(SITE_VERSION_COOKIE);
  if (!previous) {
    setCookie(SITE_VERSION_COOKIE, version);
    return;
  }
  if (previous === version || state.pendingSiteVersion === version) {
    return;
  }
  if (state.walkthroughIndex >= 0 || welcomeTutorialPromptOpen()) {
    state.pendingSiteVersion = version;
    state.pendingVersionPrompt = true;
    return;
  }
  showCachePrompt(version);
}

export {
  ACTIVE_JOB_STATUSES,
  JOBS_ACTIVE_POLL_INTERVAL_MS,
  JOB_WATCH_LIMIT,
  RESTART_POLL_INTERVAL_MS,
  RESTART_POLL_TIMEOUT_MS,
  TERMINAL_JOB_STATUSES,
  UPDATE_POLL_INTERVAL_MS,
  applyChatConfig,
  applyServerConfig,
  applyUpdate,
  createJobRow,
  formatJobProgress,
  formatJobProgressText,
  handleJobAction,
  handleSiteVersionFromUpdateStatus,
  handleVisibilityChange,
  hydrateJobLogs,
  jobLogHydratedAt,
  notePollFailure,
  notePollSuccess,
  noteShutdownOffline,
  notifyJobOutcome,
  pollBackoffMultiplier,
  positiveInterval,
  refreshHealth,
  refreshJobs,
  refreshUpdateStatus,
  rememberJobLogOpenState,
  renderUpdateStatus,
  scheduleHealthPolling,
  scheduleJobsPolling,
  scheduleUpdatePolling,
  setUpdateButton,
  shortSha,
  siteVersionFromStatus,
  trackJobTransitions,
  updateComposerSettingsSummary,
  updateJobsStrip,
  updateShutdownBanner,
  waitForRestart,
};
