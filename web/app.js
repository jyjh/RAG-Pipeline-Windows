const state = {
  offset: 0,
  limit: 20,
  indexPageSize: "20",
  total: 0,
  search: "",
  pdfSearch: "",
  chats: [],
  activeChatId: null,
  streamingChatId: null,
  chatSidebarCollapsed: false,
  healthPollIntervalMs: 60000,
  jobsPollIntervalMs: 60000,
  healthTimer: null,
  jobsTimer: null,
  updateTimer: null,
  updateApplying: false,
};

const LIVE_RENDER_INTERVAL_MS = 200;
const UPDATE_POLL_INTERVAL_MS = 5 * 60 * 1000;
const RESTART_POLL_INTERVAL_MS = 1000;
const RESTART_POLL_TIMEOUT_MS = 120000;
const CHAT_STORAGE_KEY = "rag.chatHistory.v1";
const CHAT_UI_STORAGE_KEY = "rag.chatUi.v1";

const els = {
  statusLine: document.getElementById("statusLine"),
  updateButton: document.getElementById("updateButton"),
  fileInput: document.getElementById("fileInput"),
  uploadButton: document.getElementById("uploadButton"),
  reindexButton: document.getElementById("reindexButton"),
  uploadStatus: document.getElementById("uploadStatus"),
  pdfSearchInput: document.getElementById("pdfSearchInput"),
  pdfSearchButton: document.getElementById("pdfSearchButton"),
  pdfsBody: document.getElementById("pdfsBody"),
  jobsBody: document.getElementById("jobsBody"),
  searchInput: document.getElementById("searchInput"),
  searchButton: document.getElementById("searchButton"),
  indexPageSizeSelect: document.getElementById("indexPageSizeSelect"),
  prevPageButton: document.getElementById("prevPageButton"),
  nextPageButton: document.getElementById("nextPageButton"),
  pageLabel: document.getElementById("pageLabel"),
  indexStatus: document.getElementById("indexStatus"),
  indexBody: document.getElementById("indexBody"),
  chatForm: document.getElementById("chatForm"),
  questionInput: document.getElementById("questionInput"),
  temperatureInput: document.getElementById("temperatureInput"),
  maxKInput: document.getElementById("maxKInput"),
  contextWindowInput: document.getElementById("contextWindowInput"),
  maxOutputInput: document.getElementById("maxOutputInput"),
  relevanceFloorInput: document.getElementById("relevanceFloorInput"),
  webSearchInput: document.getElementById("webSearchInput"),
  sendButton: document.getElementById("sendButton"),
  chatLayout: document.getElementById("chatLayout"),
  chatSidebar: document.getElementById("chatSidebar"),
  collapseChatSidebarButton: document.getElementById("collapseChatSidebarButton"),
  expandChatSidebarButton: document.getElementById("expandChatSidebarButton"),
  newChatButton: document.getElementById("newChatButton"),
  savedChatsList: document.getElementById("savedChatsList"),
  chatMessages: document.getElementById("chatMessages"),
};

function setStatus(element, text, isError = false) {
  element.textContent = text || "";
  element.classList.toggle("error", Boolean(isError));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = await response.text();
    try {
      const parsed = JSON.parse(detail);
      detail = parsed.detail || detail;
    } catch (_) {
      // Keep the raw response text.
    }
    const message =
      typeof detail === "string"
        ? detail
        : detail.message || `${response.status} ${response.statusText}`;
    const error = new Error(message);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return response.json();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function renderMarkdown(text) {
  if (!text) {
    return "";
  }
  const data = await requestJson("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return data.html || "";
}

function renderKeys(kind) {
  if (kind === "thinking") {
    return {
      element: "thinkingBody",
      raw: "rawThinking",
      version: "thinkingRenderVersion",
      timer: "thinkingRenderTimer",
      inFlight: "thinkingRenderInFlight",
      lastAt: "thinkingLastRenderAt",
    };
  }
  return {
    element: "body",
    raw: "rawAnswer",
    version: "answerRenderVersion",
    timer: "answerRenderTimer",
    inFlight: "answerRenderInFlight",
    lastAt: "answerLastRenderAt",
  };
}

function renderDelay(parts, keys) {
  return Math.max(0, LIVE_RENDER_INTERVAL_MS - (Date.now() - parts[keys.lastAt]));
}

function scheduleMarkdownRender(parts, kind, immediate = false) {
  if (parts.finalized) {
    return;
  }
  const keys = renderKeys(kind);
  if (parts[keys.timer] || parts[keys.inFlight]) {
    return;
  }
  parts[keys.timer] = setTimeout(
    () => runMarkdownRender(parts, kind),
    immediate ? 0 : renderDelay(parts, keys),
  );
}

function queueMarkdownRender(parts, kind) {
  const keys = renderKeys(kind);
  parts[keys.version] += 1;
  scheduleMarkdownRender(parts, kind);
}

function addPersistentNotice(parts, text) {
  if (!text) {
    return;
  }
  parts.persistentNotices.push(text);
  updateNotice(parts);
}

function addFormattingNotice(parts, error) {
  if (parts.formattingErrorShown) {
    return;
  }
  parts.formattingErrorShown = true;
  addPersistentNotice(parts, `Formatting failed: ${error.message}`);
}

async function runMarkdownRender(parts, kind) {
  const keys = renderKeys(kind);
  if (parts[keys.inFlight]) {
    return;
  }
  if (parts[keys.timer]) {
    clearTimeout(parts[keys.timer]);
    parts[keys.timer] = null;
  }

  const version = parts[keys.version];
  const text = parts[keys.raw];
  if (!text) {
    return;
  }

  parts[keys.inFlight] = true;
  try {
    const html = await renderMarkdown(text);
    if (!parts.finalized && parts[keys.version] === version) {
      parts[keys.element].innerHTML = html;
    }
  } catch (error) {
    addFormattingNotice(parts, error);
  } finally {
    parts[keys.inFlight] = false;
    parts[keys.lastAt] = Date.now();
    if (!parts.finalized && parts[keys.version] !== version) {
      scheduleMarkdownRender(parts, kind);
    }
  }
}

function numericSetting(input, fallback, minimum = 1) {
  const value = Number(input.value);
  if (!Number.isFinite(value) || value < minimum) {
    return fallback;
  }
  return value;
}

function nowIso() {
  return new Date().toISOString();
}

function newId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function firstFiveWords(text) {
  const words = String(text || "").trim().split(/\s+/).filter(Boolean).slice(0, 5);
  return words.join(" ") || "New chat";
}

function loadChatState() {
  try {
    const parsed = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) || "{}");
    state.chats = Array.isArray(parsed.chats)
      ? parsed.chats
          .filter((chat) => chat && typeof chat === "object")
          .map((chat) => ({
            id: String(chat.id || newId()),
            title: String(chat.title || "New chat"),
            customTitle: Boolean(chat.customTitle),
            createdAt: String(chat.createdAt || nowIso()),
            updatedAt: String(chat.updatedAt || nowIso()),
            messages: Array.isArray(chat.messages) ? chat.messages : [],
          }))
      : [];
    state.activeChatId = parsed.activeChatId || null;
  } catch (_) {
    state.chats = [];
    state.activeChatId = null;
  }

  try {
    const ui = JSON.parse(localStorage.getItem(CHAT_UI_STORAGE_KEY) || "{}");
    state.chatSidebarCollapsed = Boolean(ui.chatSidebarCollapsed);
  } catch (_) {
    state.chatSidebarCollapsed = false;
  }

  if (!state.chats.some((chat) => chat.id === state.activeChatId)) {
    state.activeChatId = state.chats[0]?.id || null;
  }
  if (!state.activeChatId) {
    createChat({ activate: true, persist: false });
  }
}

function persistChatState() {
  localStorage.setItem(
    CHAT_STORAGE_KEY,
    JSON.stringify({
      activeChatId: state.activeChatId,
      chats: state.chats,
    }),
  );
}

function persistChatUiState() {
  localStorage.setItem(
    CHAT_UI_STORAGE_KEY,
    JSON.stringify({ chatSidebarCollapsed: state.chatSidebarCollapsed }),
  );
}

function activeChat() {
  return state.chats.find((chat) => chat.id === state.activeChatId) || null;
}

function createChat({ activate = true, persist = true } = {}) {
  const chat = {
    id: newId(),
    title: "New chat",
    customTitle: false,
    createdAt: nowIso(),
    updatedAt: nowIso(),
    messages: [],
  };
  state.chats.unshift(chat);
  if (activate) {
    state.activeChatId = chat.id;
  }
  if (persist) {
    persistChatState();
    renderSavedChats();
    renderActiveChat();
  }
  return chat;
}

function touchChat(chat) {
  chat.updatedAt = nowIso();
  state.chats = [chat, ...state.chats.filter((item) => item.id !== chat.id)];
}

function refreshChatTitle(chat) {
  if (chat.customTitle) {
    return;
  }
  const firstUserMessage = chat.messages.find((message) => message.role === "user");
  chat.title = firstUserMessage ? firstFiveWords(firstUserMessage.text) : "New chat";
}

function setChatSidebarCollapsed(collapsed) {
  state.chatSidebarCollapsed = collapsed;
  els.chatLayout.classList.toggle("sidebar-collapsed", collapsed);
  els.expandChatSidebarButton.hidden = !collapsed;
  persistChatUiState();
}

function renderSavedChats() {
  els.savedChatsList.innerHTML = "";
  for (const chat of state.chats) {
    const row = document.createElement("div");
    row.className = "saved-chat-row";
    row.dataset.chatId = chat.id;

    const title = document.createElement("button");
    title.type = "button";
    title.className = "saved-chat-title";
    title.classList.toggle("active", chat.id === state.activeChatId);
    title.textContent = chat.title || "New chat";
    title.title = chat.title || "New chat";
    title.addEventListener("click", () => selectChat(chat.id));

    const actions = document.createElement("div");
    actions.className = "saved-chat-actions";

    const rename = document.createElement("button");
    rename.type = "button";
    rename.textContent = "Rename";
    rename.addEventListener("click", () => renameChat(chat.id));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Delete";
    remove.className = "danger";
    remove.addEventListener("click", () => deleteChat(chat.id));

    actions.append(rename, remove);
    row.append(title, actions);
    els.savedChatsList.appendChild(row);
  }
}

function selectChat(chatId) {
  if (state.streamingChatId) {
    return;
  }
  if (!state.chats.some((chat) => chat.id === chatId)) {
    return;
  }
  state.activeChatId = chatId;
  persistChatState();
  renderSavedChats();
  renderActiveChat();
}

function renameChat(chatId) {
  if (state.streamingChatId) {
    return;
  }
  const chat = state.chats.find((item) => item.id === chatId);
  if (!chat) {
    return;
  }
  const nextTitle = window.prompt("Rename chat", chat.title || "New chat");
  if (nextTitle === null) {
    return;
  }
  const trimmed = nextTitle.trim();
  if (!trimmed) {
    return;
  }
  chat.title = trimmed;
  chat.customTitle = true;
  touchChat(chat);
  persistChatState();
  renderSavedChats();
}

function deleteChat(chatId) {
  if (state.streamingChatId) {
    return;
  }
  const chat = state.chats.find((item) => item.id === chatId);
  if (!chat || !window.confirm(`Delete "${chat.title || "New chat"}"?`)) {
    return;
  }
  state.chats = state.chats.filter((item) => item.id !== chatId);
  if (state.activeChatId === chatId) {
    state.activeChatId = state.chats[0]?.id || null;
    if (!state.activeChatId) {
      createChat({ activate: true, persist: false });
    }
  }
  persistChatState();
  renderSavedChats();
  renderActiveChat();
}

function addUserMessageToChat(chat, text) {
  chat.messages.push({ role: "user", text, createdAt: nowIso() });
  refreshChatTitle(chat);
  touchChat(chat);
  persistChatState();
  renderSavedChats();
}

function addAssistantMessageToChat(chat, parts) {
  chat.messages.push({
    role: "assistant",
    text: parts.rawAnswer,
    thinking: parts.rawThinking,
    answerHtml: parts.body.innerHTML,
    thinkingHtml: parts.rawThinking ? parts.thinkingBody.innerHTML : "",
    sources: parts.sources,
    notice: parts.notice.textContent || "",
    createdAt: nowIso(),
  });
  touchChat(chat);
  persistChatState();
  renderSavedChats();
}

function renderActiveChat() {
  els.chatMessages.innerHTML = "";
  const chat = activeChat();
  if (!chat) {
    return;
  }
  for (const message of chat.messages) {
    if (message.role === "assistant") {
      addSavedAssistantMessage(message);
    } else {
      addMessage("You", message.text || "");
    }
  }
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

async function refreshHealth() {
  try {
    const data = await requestJson("/api/health");
    applyServerConfig(data.server || {});
    applyChatConfig(data.chat || {});
    const queue = data.queue || {};
    els.statusLine.textContent =
      `${data.record_count} indexed chunks | ` +
      `${queue.active_query_count || 0} active queries | ` +
      `${queue.queued_count || 0} queued jobs`;
  } catch (error) {
    els.statusLine.textContent = `Health check failed: ${error.message}`;
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
}

function positiveInterval(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
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

async function refreshJobs() {
  try {
    const data = await requestJson("/api/jobs");
    els.jobsBody.innerHTML = "";
    for (const job of data.jobs || []) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${escapeHtml(job.id.slice(0, 8))}</td>
        <td>${escapeHtml(job.status)}</td>
        <td>${escapeHtml(job.phase)}</td>
        <td>${escapeHtml((job.filenames || []).join(", "))}</td>
        <td>${escapeHtml(job.error || "")}</td>
      `;
      els.jobsBody.appendChild(row);
    }
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  }
}

async function refreshPdfs() {
  try {
    const params = new URLSearchParams({ search: state.pdfSearch });
    const data = await requestJson(`/api/pdfs?${params}`);
    els.pdfsBody.innerHTML = "";
    for (const item of data.pdfs || []) {
      const row = document.createElement("tr");
      const download = item.download_url
        ? `<a class="download-link" href="${escapeHtml(item.download_url)}">Download</a>`
        : escapeHtml(item.path_error || "");
      row.innerHTML = `
        <td>
          <strong>${escapeHtml(item.filename || item.hash)}</strong><br />
          ${escapeHtml(item.hash || "")}
        </td>
        <td>${escapeHtml(item.status || "")}</td>
        <td>${download}</td>
      `;
      els.pdfsBody.appendChild(row);
    }
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  }
}

function scheduleHealthPolling() {
  if (state.healthTimer) {
    clearInterval(state.healthTimer);
  }
  state.healthTimer = setInterval(refreshHealth, state.healthPollIntervalMs);
}

function scheduleJobsPolling() {
  if (state.jobsTimer) {
    clearInterval(state.jobsTimer);
  }
  state.jobsTimer = setInterval(refreshJobs, state.jobsPollIntervalMs);
}

function scheduleUpdatePolling() {
  if (state.updateTimer) {
    clearInterval(state.updateTimer);
  }
  state.updateTimer = setInterval(refreshUpdateStatus, UPDATE_POLL_INTERVAL_MS);
}

function duplicateUploadMessage(detail) {
  const duplicates = Array.isArray(detail?.duplicates) ? detail.duplicates : [];
  if (!duplicates.length) {
    return detail?.message || "Duplicate PDF upload detected.";
  }
  const names = duplicates
    .map((item) => {
      const existing = item.existing_filename ? ` matches ${item.existing_filename}` : "";
      return `${item.filename || item.hash}${existing}`;
    })
    .join("\n");
  return `${detail.message || "Duplicate PDF upload detected."}\n\n${names}`;
}

async function uploadFiles(forceDuplicates = false) {
  const files = Array.from(els.fileInput.files || []);
  if (!files.length) {
    setStatus(els.uploadStatus, "Choose one or more PDF files.", true);
    return;
  }

  const body = new FormData();
  for (const file of files) {
    body.append("files", file);
  }
  if (forceDuplicates) {
    body.append("force_duplicates", "true");
  }

  els.uploadButton.disabled = true;
  setStatus(els.uploadStatus, forceDuplicates ? "Queueing forced upload..." : "Queueing upload...");
  try {
    const job = await requestJson("/api/uploads", { method: "POST", body });
    els.fileInput.value = "";
    setStatus(els.uploadStatus, `Queued job ${job.id.slice(0, 8)}.`);
    await refreshJobs();
    await refreshPdfs();
  } catch (error) {
    if (
      !forceDuplicates &&
      error.status === 409 &&
      error.detail &&
      error.detail.can_force !== false
    ) {
      const confirmed = window.confirm(`${duplicateUploadMessage(error.detail)}\n\nForce re-upload?`);
      if (confirmed) {
        await uploadFiles(true);
        return;
      }
      setStatus(els.uploadStatus, "Duplicate upload cancelled.", true);
      return;
    }
    if (error.status === 409 && error.detail) {
      setStatus(els.uploadStatus, duplicateUploadMessage(error.detail), true);
      return;
    }
    setStatus(els.uploadStatus, error.message, true);
  } finally {
    els.uploadButton.disabled = false;
  }
}

async function enqueueReindex() {
  els.reindexButton.disabled = true;
  setStatus(els.uploadStatus, "Queueing reindex...");
  try {
    const job = await requestJson("/api/reindex", { method: "POST" });
    setStatus(els.uploadStatus, `Queued reindex job ${job.id.slice(0, 8)}.`);
    await refreshJobs();
    await refreshPdfs();
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  } finally {
    els.reindexButton.disabled = false;
  }
}

async function loadIndex() {
  if (state.indexPageSize === "all") {
    return loadAllIndexRows();
  }
  state.limit = Number(state.indexPageSize) || 20;
  const params = new URLSearchParams({
    offset: String(state.offset),
    limit: String(state.limit),
    search: state.search,
  });
  try {
    const data = await requestJson(`/api/index?${params}`);
    state.total = data.total || 0;
    renderIndexRows(data.rows || []);
    const start = state.total ? state.offset + 1 : 0;
    const end = Math.min(state.offset + state.limit, state.total);
    els.pageLabel.textContent = `${start}-${end} of ${state.total}`;
    els.prevPageButton.disabled = state.offset <= 0;
    els.nextPageButton.disabled = state.offset + state.limit >= state.total;
    setStatus(els.indexStatus, `Embedding model: ${data.embedding_model || "unknown"}`);
  } catch (error) {
    els.indexBody.innerHTML = "";
    els.pageLabel.textContent = "";
    setStatus(els.indexStatus, error.message, true);
  }
}

async function loadAllIndexRows() {
  const batchSize = 100;
  const rows = [];
  let offset = 0;
  let total = 0;
  let embeddingModel = "unknown";
  try {
    while (true) {
      const params = new URLSearchParams({
        offset: String(offset),
        limit: String(batchSize),
        search: state.search,
      });
      const data = await requestJson(`/api/index?${params}`);
      const batch = data.rows || [];
      if (offset === 0) {
        total = data.total || 0;
        embeddingModel = data.embedding_model || "unknown";
      }
      rows.push(...batch);
      offset += batch.length;
      if (!batch.length || rows.length >= total) {
        break;
      }
    }
    state.total = total;
    renderIndexRows(rows);
    els.pageLabel.textContent = `All ${rows.length} of ${total}`;
    els.prevPageButton.disabled = true;
    els.nextPageButton.disabled = true;
    setStatus(els.indexStatus, `Embedding model: ${embeddingModel}`);
  } catch (error) {
    els.indexBody.innerHTML = "";
    els.pageLabel.textContent = "";
    setStatus(els.indexStatus, error.message, true);
  }
}

function renderIndexRows(rows) {
  els.indexBody.innerHTML = "";
  for (const item of rows) {
    const row = document.createElement("tr");
    row.dataset.recordId = item.id;
    const download = item.source_hash
      ? `<br /><a class="download-link" href="/api/pdfs/${encodeURIComponent(item.source_hash)}/download">Download PDF</a>`
      : "";
    row.innerHTML = `
      <td class="source-cell">
        <strong>${escapeHtml(item.id)}</strong><br />
        ${escapeHtml(item.file_path)}<br />
        chunk ${escapeHtml(item.chunk_index)}
        ${download}
      </td>
      <td>
        <textarea class="content-edit" spellcheck="false"></textarea>
      </td>
      <td>
        <div class="row-actions">
          <button type="button" data-action="save">Save</button>
          <button type="button" class="danger" data-action="delete">Delete</button>
        </div>
      </td>
    `;
    row.querySelector("textarea").value = item.content || "";
    els.indexBody.appendChild(row);
  }
}

async function handleIndexAction(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) {
    return;
  }

  const row = button.closest("tr");
  const recordId = row.dataset.recordId;
  const action = button.dataset.action;
  const textarea = row.querySelector("textarea");
  button.disabled = true;

  try {
    if (action === "save") {
      await requestJson("/api/index/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ record_id: recordId, content: textarea.value }),
      });
      setStatus(els.indexStatus, `Saved ${recordId}.`);
    }

    if (action === "delete") {
      await requestJson("/api/index/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ record_ids: [recordId] }),
      });
      setStatus(els.indexStatus, `Deleted ${recordId}.`);
      await loadIndex();
    }
  } catch (error) {
    setStatus(els.indexStatus, error.message, true);
  } finally {
    button.disabled = false;
  }
}

function addMessage(role, text = "") {
  const message = document.createElement("div");
  message.className = "message";
  const roleLabel = document.createElement("span");
  roleLabel.className = "role";
  roleLabel.textContent = role;
  const body = document.createElement("span");
  body.className = "body";
  body.textContent = text;
  message.append(roleLabel, body);
  els.chatMessages.appendChild(message);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  return body;
}

function addAssistantMessage() {
  const message = document.createElement("div");
  message.className = "message assistant-message";

  const roleLabel = document.createElement("span");
  roleLabel.className = "role";
  roleLabel.textContent = "Assistant";

  const thinking = document.createElement("details");
  thinking.className = "thinking-block";
  thinking.hidden = true;

  const summary = document.createElement("summary");
  summary.textContent = "Model thinking";

  const thinkingBody = document.createElement("div");
  thinkingBody.className = "thinking-body rendered";
  thinking.append(summary, thinkingBody);

  const sources = document.createElement("details");
  sources.className = "sources-block";
  sources.hidden = true;

  const sourcesSummary = document.createElement("summary");
  sourcesSummary.textContent = "Sources";

  const sourcesBody = document.createElement("div");
  sourcesBody.className = "sources-body";
  sources.append(sourcesSummary, sourcesBody);

  const notice = document.createElement("div");
  notice.className = "stream-notice";
  notice.hidden = true;

  const body = document.createElement("div");
  body.className = "body rendered";

  message.append(roleLabel, thinking, sources, notice, body);
  els.chatMessages.appendChild(message);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  return {
    body,
    thinking,
    thinkingBody,
    sourcesPanel: sources,
    sourcesBody,
    notice,
    rawAnswer: "",
    rawThinking: "",
    sourcesData: [],
    get sources() {
      return this.sourcesData;
    },
    set sources(value) {
      this.sourcesData = Array.isArray(value) ? value : [];
    },
    transientNotices: [],
    persistentNotices: [],
    gemmaResponseStarted: false,
    formattingErrorShown: false,
    finalized: false,
    answerRenderVersion: 0,
    answerRenderTimer: null,
    answerRenderInFlight: false,
    answerLastRenderAt: 0,
    thinkingRenderVersion: 0,
    thinkingRenderTimer: null,
    thinkingRenderInFlight: false,
    thinkingLastRenderAt: 0,
  };
}

function sourceTitle(source) {
  if (source.kind === "web") {
    return source.title || source.url || source.label;
  }
  return source.source_pdf_name || source.file_path || source.chunk_id || source.label;
}

function sourceLocation(source) {
  if (source.kind === "web") {
    return source.provider || "";
  }
  return [source.section_path, source.page_label].filter(Boolean).join(" | ");
}

function renderSourcePanel(parts) {
  const sources = Array.isArray(parts.sources) ? parts.sources : [];
  parts.sourcesPanel.hidden = sources.length === 0;
  parts.sourcesPanel.open = sources.length > 0;
  parts.sourcesBody.innerHTML = "";
  for (const source of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    const links = [];
    if (source.kind === "web" && source.url) {
      links.push(`<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">Open</a>`);
    }
    if (source.kind === "local" && source.download_url) {
      links.push(`<a href="${escapeHtml(source.download_url)}">Download PDF</a>`);
    }
    const score = source.kind === "local" && Number.isFinite(Number(source.score))
      ? `score ${Number(source.score).toFixed(3)}`
      : "";
    item.innerHTML = `
      <strong>${escapeHtml(source.label || source.id || "")} ${escapeHtml(sourceTitle(source))}</strong>
      <span>${escapeHtml([sourceLocation(source), score].filter(Boolean).join(" | "))}</span>
      <span>${escapeHtml(source.snippet || "")}</span>
      <span class="source-links">${links.join("")}</span>
    `;
    parts.sourcesBody.appendChild(item);
  }
}

function addSavedAssistantMessage(saved) {
  const parts = addAssistantMessage();
  parts.finalized = true;
  parts.rawAnswer = saved.text || "";
  parts.rawThinking = saved.thinking || "";
  parts.sources = saved.sources || [];
  if (parts.rawThinking) {
    parts.thinking.hidden = false;
    parts.thinking.open = false;
    parts.thinkingBody.innerHTML = saved.thinkingHtml || escapeHtml(parts.rawThinking);
  }
  renderSourcePanel(parts);
  if (saved.notice) {
    parts.notice.textContent = saved.notice;
    parts.notice.hidden = false;
  }
  if (saved.answerHtml) {
    parts.body.innerHTML = saved.answerHtml;
  } else {
    parts.body.textContent = parts.rawAnswer;
  }
  return parts;
}

function isTransientNotice(text) {
  return (
    text === "Embedding query and retrieving context..." ||
    text === "Planning retrieval tool calls..." ||
    /^Running .+\.\.\.$/.test(text) ||
    /^Retrieved \d+ .+\(s\)\.?$/.test(text) ||
    /^Retrieved \d+ context chunk\(s\)\. Requesting answer from .+\.\.\.$/.test(text)
  );
}

function updateNotice(parts) {
  const notices = [
    ...(parts.gemmaResponseStarted ? [] : parts.transientNotices),
    ...parts.persistentNotices,
  ];
  parts.notice.textContent = notices.join("\n");
  parts.notice.hidden = notices.length === 0;
}

function markGemmaResponseStarted(parts) {
  if (!parts.gemmaResponseStarted) {
    parts.gemmaResponseStarted = true;
    updateNotice(parts);
  }
}

function appendStreamEvent(parts, event) {
  const type = event.type || "answer";
  const text = event.text || "";
  if (!text) {
    return;
  }

  if (type === "thinking") {
    markGemmaResponseStarted(parts);
    parts.thinking.hidden = false;
    parts.thinking.open = true;
    parts.rawThinking += text;
    parts.thinkingBody.textContent = parts.rawThinking;
    queueMarkdownRender(parts, "thinking");
    return;
  }

  if (type === "error") {
    addPersistentNotice(parts, `[Error] ${text}`);
    return;
  }

  if (type === "sources") {
    parts.sources = Array.isArray(event.sources) ? event.sources : [];
    renderSourcePanel(parts);
    return;
  }

  if (type === "tool_call" || type === "tool_result") {
    if (isTransientNotice(text)) {
      parts.transientNotices.push(text);
    } else {
      parts.persistentNotices.push(text);
    }
    updateNotice(parts);
    return;
  }

  if (type === "notice") {
    if (isTransientNotice(text)) {
      parts.transientNotices.push(text);
    } else {
      parts.persistentNotices.push(text);
    }
    updateNotice(parts);
    return;
  }

  markGemmaResponseStarted(parts);
  parts.rawAnswer += text;
  parts.body.textContent = parts.rawAnswer;
  queueMarkdownRender(parts, "answer");
}

async function formatAssistantMessage(parts) {
  parts.finalized = true;
  for (const kind of ["answer", "thinking"]) {
    const keys = renderKeys(kind);
    if (parts[keys.timer]) {
      clearTimeout(parts[keys.timer]);
      parts[keys.timer] = null;
    }
    parts[keys.version] += 1;
  }

  try {
    const [thinkingHtml, answerHtml] = await Promise.all([
      parts.rawThinking ? renderMarkdown(parts.rawThinking) : "",
      parts.rawAnswer ? renderMarkdown(parts.rawAnswer) : "",
    ]);
    if (parts.rawThinking) {
      parts.thinkingBody.innerHTML = thinkingHtml;
    }
    if (parts.rawAnswer) {
      parts.body.innerHTML = answerHtml;
    }
  } catch (error) {
    addFormattingNotice(parts, error);
  }
}

async function sendQuestion(event) {
  event.preventDefault();
  if (state.streamingChatId) {
    return;
  }
  const question = els.questionInput.value.trim();
  if (!question) {
    return;
  }

  const chat = activeChat() || createChat({ activate: true });
  state.activeChatId = chat.id;
  state.streamingChatId = chat.id;
  addUserMessageToChat(chat, question);
  addMessage("You", question);
  const assistantParts = addAssistantMessage();
  els.questionInput.value = "";
  els.sendButton.disabled = true;
  renderSavedChats();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        temperature: numericSetting(els.temperatureInput, 0.9, 0),
        max_k: Math.trunc(numericSetting(els.maxKInput, 40, 1)),
        context_window: Math.trunc(numericSetting(els.contextWindowInput, 8192, 1)),
        llm_num_predict: Math.trunc(numericSetting(els.maxOutputInput, 4096, 1)),
        retrieval_min_score: numericSetting(els.relevanceFloorInput, 0.5, 0),
        web_search_enabled: Boolean(els.webSearchInput.checked),
      }),
    });
    if (!response.ok || !response.body) {
      throw new Error(await response.text());
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const processLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return;
      }
      try {
        appendStreamEvent(assistantParts, JSON.parse(trimmed));
      } catch (_) {
        appendStreamEvent(assistantParts, { type: "answer", text: line });
      }
      els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        processLine(line);
      }
    }
    buffer += decoder.decode();
    processLine(buffer);
  } catch (error) {
    appendStreamEvent(assistantParts, { type: "error", text: error.message });
  } finally {
    await formatAssistantMessage(assistantParts);
    addAssistantMessageToChat(chat, assistantParts);
    state.streamingChatId = null;
    els.sendButton.disabled = false;
    renderSavedChats();
    await refreshHealth();
  }
}

document.querySelectorAll("[data-tab-target]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(button.dataset.tabTarget).classList.add("active");
    if (button.dataset.tabTarget === "index") {
      loadIndex();
    }
  });
});

els.updateButton.addEventListener("click", applyUpdate);
els.uploadButton.addEventListener("click", uploadFiles);
els.reindexButton.addEventListener("click", enqueueReindex);
els.pdfSearchButton.addEventListener("click", () => {
  state.pdfSearch = els.pdfSearchInput.value.trim();
  refreshPdfs();
});
els.pdfSearchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    state.pdfSearch = els.pdfSearchInput.value.trim();
    refreshPdfs();
  }
});
els.searchButton.addEventListener("click", () => {
  state.offset = 0;
  state.search = els.searchInput.value.trim();
  loadIndex();
});
els.searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    state.offset = 0;
    state.search = els.searchInput.value.trim();
    loadIndex();
  }
});
els.indexPageSizeSelect.addEventListener("change", () => {
  state.indexPageSize = els.indexPageSizeSelect.value;
  state.offset = 0;
  loadIndex();
});
els.prevPageButton.addEventListener("click", () => {
  if (state.indexPageSize === "all") {
    return;
  }
  state.offset = Math.max(0, state.offset - state.limit);
  loadIndex();
});
els.nextPageButton.addEventListener("click", () => {
  if (state.indexPageSize === "all") {
    return;
  }
  state.offset += state.limit;
  loadIndex();
});
els.indexBody.addEventListener("click", handleIndexAction);
els.chatForm.addEventListener("submit", sendQuestion);
els.newChatButton.addEventListener("click", () => {
  if (state.streamingChatId) {
    return;
  }
  createChat({ activate: true });
});
els.collapseChatSidebarButton.addEventListener("click", () => setChatSidebarCollapsed(true));
els.expandChatSidebarButton.addEventListener("click", () => setChatSidebarCollapsed(false));

loadChatState();
setChatSidebarCollapsed(state.chatSidebarCollapsed);
renderSavedChats();
renderActiveChat();
persistChatState();
refreshHealth();
refreshUpdateStatus();
refreshJobs();
refreshPdfs();
scheduleHealthPolling();
scheduleUpdatePolling();
scheduleJobsPolling();
