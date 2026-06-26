const state = {
  offset: 0,
  limit: 20,
  indexPageSize: "20",
  total: 0,
  search: "",
  indexMode: "standard",
  vectorSearch: "",
  vectorRelevanceFloor: 0.7,
  pdfSearch: "",
  pdfOffset: 0,
  pdfLimit: 10,
  pdfTotal: 0,
  jobsOffset: 0,
  jobsLimit: 10,
  jobsTotal: 0,
  chats: [],
  activeChatId: null,
  streamingChatId: null,
  chatAbortController: null,
  chatSidebarCollapsed: false,
  healthPollIntervalMs: 60000,
  jobsPollIntervalMs: 60000,
  healthTimer: null,
  jobsTimer: null,
  updateTimer: null,
  updateApplying: false,
  indexAbortController: null,
  indexLoadToken: 0,
  uploadDragDepth: 0,
  pendingForceUploadToken: "",
};

const LIVE_RENDER_INTERVAL_MS = 200;
const STREAM_TAIL_HOLD_CHARS = 700;
const STREAM_TAIL_MAX_CHARS = 2200;
const UPDATE_POLL_INTERVAL_MS = 5 * 60 * 1000;
const RESTART_POLL_INTERVAL_MS = 1000;
const RESTART_POLL_TIMEOUT_MS = 120000;
const INDEX_STREAM_BATCH_SIZE = 250;
const INDEX_CHILD_BATCH_SIZE = 100;
const CHAT_STORAGE_KEY = "rag.chatHistory.v1";
const CHAT_UI_STORAGE_KEY = "rag.chatUi.v1";

const els = {
  statusLine: document.getElementById("statusLine"),
  updateButton: document.getElementById("updateButton"),
  uploadDropZone: document.getElementById("uploadDropZone"),
  fileInput: document.getElementById("fileInput"),
  selectedFilesLabel: document.getElementById("selectedFilesLabel"),
  uploadButton: document.getElementById("uploadButton"),
  reindexButton: document.getElementById("reindexButton"),
  uploadStatus: document.getElementById("uploadStatus"),
  duplicatePrompt: document.getElementById("duplicatePrompt"),
  duplicatePromptText: document.getElementById("duplicatePromptText"),
  forceUploadButton: document.getElementById("forceUploadButton"),
  pdfSearchInput: document.getElementById("pdfSearchInput"),
  pdfSearchButton: document.getElementById("pdfSearchButton"),
  prevPdfPageButton: document.getElementById("prevPdfPageButton"),
  pdfPageLabel: document.getElementById("pdfPageLabel"),
  nextPdfPageButton: document.getElementById("nextPdfPageButton"),
  pdfsBody: document.getElementById("pdfsBody"),
  prevJobsPageButton: document.getElementById("prevJobsPageButton"),
  jobsPageLabel: document.getElementById("jobsPageLabel"),
  nextJobsPageButton: document.getElementById("nextJobsPageButton"),
  jobsBody: document.getElementById("jobsBody"),
  searchInput: document.getElementById("searchInput"),
  searchButton: document.getElementById("searchButton"),
  vectorSearchInput: document.getElementById("vectorSearchInput"),
  vectorRelevanceFloorInput: document.getElementById("vectorRelevanceFloorInput"),
  vectorSearchButton: document.getElementById("vectorSearchButton"),
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

async function errorFromResponse(response) {
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
  return error;
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  return response.json();
}

function isAbortError(error) {
  return error && error.name === "AbortError";
}

function startIndexLoad() {
  if (state.indexAbortController) {
    state.indexAbortController.abort();
  }
  const abortController = new AbortController();
  state.indexAbortController = abortController;
  state.indexLoadToken += 1;
  return { abortController, token: state.indexLoadToken };
}

function isActiveIndexLoad(load) {
  return (
    state.indexLoadToken === load.token &&
    state.indexAbortController === load.abortController &&
    !load.abortController.signal.aborted
  );
}

function finishIndexLoad(load) {
  if (state.indexAbortController === load.abortController) {
    state.indexAbortController = null;
  }
}

function abortIndexLoad() {
  if (state.indexAbortController) {
    state.indexAbortController.abort();
    state.indexAbortController = null;
  }
}

async function readNdjson(response, onEvent) {
  if (!response.body) {
    for (const line of (await response.text()).split(/\r?\n/)) {
      const trimmed = line.trim();
      if (trimmed) {
        await onEvent(JSON.parse(trimmed));
      }
    }
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex !== -1) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) {
        await onEvent(JSON.parse(line));
      }
      newlineIndex = buffer.indexOf("\n");
    }
  }

  buffer += decoder.decode();
  const trimmed = buffer.trim();
  if (trimmed) {
    await onEvent(JSON.parse(trimmed));
  }
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
      stableElement: "thinkingStable",
      tailElement: "thinkingTail",
      raw: "rawThinking",
      committedLength: "thinkingCommittedLength",
      version: "thinkingRenderVersion",
      timer: "thinkingRenderTimer",
      inFlight: "thinkingRenderInFlight",
      lastAt: "thinkingLastRenderAt",
    };
  }
  return {
    stableElement: "answerStable",
    tailElement: "answerTail",
    raw: "rawAnswer",
    committedLength: "answerCommittedLength",
    version: "answerRenderVersion",
    timer: "answerRenderTimer",
    inFlight: "answerRenderInFlight",
    lastAt: "answerLastRenderAt",
  };
}

function setStreamTailRaw(parts, keys, text) {
  const tail = parts[keys.tailElement];
  tail.className = "stream-tail raw-tail";
  tail.textContent = text;
}

function setStreamTailHtml(parts, keys, html) {
  const tail = parts[keys.tailElement];
  tail.className = "stream-tail rendered";
  tail.innerHTML = html || "";
}

function appendStreamStableHtml(parts, keys, html) {
  if (html) {
    parts[keys.stableElement].insertAdjacentHTML("beforeend", html);
  }
}

function replaceStreamHtml(parts, kind, html) {
  const keys = renderKeys(kind);
  parts[keys.stableElement].innerHTML = html || "";
  setStreamTailRaw(parts, keys, "");
  parts[keys.committedLength] = parts[keys.raw].length;
}

function updateStreamTailRaw(parts, kind) {
  const keys = renderKeys(kind);
  setStreamTailRaw(parts, keys, parts[keys.raw].slice(parts[keys.committedLength]));
}

function countUnescapedMarker(text, marker) {
  let count = 0;
  let index = 0;
  while ((index = text.indexOf(marker, index)) !== -1) {
    let slashCount = 0;
    for (let i = index - 1; i >= 0 && text[i] === "\\"; i -= 1) {
      slashCount += 1;
    }
    if (slashCount % 2 === 0) {
      count += 1;
    }
    index += marker.length;
  }
  return count;
}

function hasOpenFencedCodeBlock(text) {
  const fencePattern = /^[ \t]*(`{3,}|~{3,})/gm;
  let openFence = null;
  let match = fencePattern.exec(text);
  while (match) {
    const fence = match[1];
    if (!openFence) {
      openFence = fence;
    } else if (fence[0] === openFence[0] && fence.length >= openFence.length) {
      openFence = null;
    }
    match = fencePattern.exec(text);
  }
  return Boolean(openFence);
}

function isSafeMarkdownCommit(text) {
  return (
    !hasOpenFencedCodeBlock(text) &&
    countUnescapedMarker(text, "$$") % 2 === 0 &&
    countUnescapedMarker(text, "\\[") === countUnescapedMarker(text, "\\]") &&
    countUnescapedMarker(text, "\\(") === countUnescapedMarker(text, "\\)")
  );
}

function blockBoundariesBefore(text, limit) {
  const boundaries = [];
  const boundaryPattern = /\n[ \t]*\n/g;
  let match = boundaryPattern.exec(text);
  while (match) {
    if (boundaryPattern.lastIndex > limit) {
      break;
    }
    boundaries.push(boundaryPattern.lastIndex);
    match = boundaryPattern.exec(text);
  }
  return boundaries;
}

function softBoundariesBefore(text, limit) {
  const boundaries = [];
  const newline = text.lastIndexOf("\n", limit);
  if (newline > 0) {
    boundaries.push(newline + 1);
  }

  const sentencePattern = /[.!?][)"'\]]?\s+/g;
  let match = sentencePattern.exec(text);
  while (match) {
    if (sentencePattern.lastIndex > limit) {
      break;
    }
    boundaries.push(sentencePattern.lastIndex);
    match = sentencePattern.exec(text);
  }

  const space = text.lastIndexOf(" ", limit);
  if (space > 0) {
    boundaries.push(space + 1);
  }
  return boundaries;
}

function streamingStableCutoff(text, committedLength) {
  const target = text.length - STREAM_TAIL_HOLD_CHARS;
  if (target <= committedLength) {
    return committedLength;
  }

  let candidates = blockBoundariesBefore(text, target);
  if (text.length - committedLength > STREAM_TAIL_MAX_CHARS) {
    candidates = candidates.concat(softBoundariesBefore(text, target));
  }

  const uniqueCandidates = [...new Set(candidates)]
    .filter((candidate) => candidate > committedLength && candidate <= target)
    .sort((left, right) => right - left);

  for (const candidate of uniqueCandidates) {
    if (isSafeMarkdownCommit(text.slice(0, candidate))) {
      return candidate;
    }
  }
  return committedLength;
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

  const committedLength = parts[keys.committedLength];
  const cutoff = streamingStableCutoff(text, committedLength);
  const stableDelta = text.slice(committedLength, cutoff);
  const tailText = text.slice(cutoff);

  parts[keys.inFlight] = true;
  try {
    const [stableHtml, tailHtml] = await Promise.all([
      stableDelta ? renderMarkdown(stableDelta) : "",
      tailText ? renderMarkdown(tailText) : "",
    ]);
    if (parts.finalized) {
      return;
    }

    const rawStillStartsWithRenderedText = parts[keys.raw].startsWith(text);
    if (
      stableDelta &&
      parts[keys.committedLength] === committedLength &&
      rawStillStartsWithRenderedText
    ) {
      appendStreamStableHtml(parts, keys, stableHtml);
      parts[keys.committedLength] = cutoff;
    }

    if (parts[keys.version] === version && parts[keys.raw] === text) {
      setStreamTailHtml(parts, keys, tailHtml);
    } else {
      updateStreamTailRaw(parts, kind);
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
    answerHtml: parts.answerStable.innerHTML,
    thinkingHtml: parts.rawThinking ? parts.thinkingStable.innerHTML : "",
    sources: parts.sources,
    toolResults: parts.toolResults,
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

function updatePageControls({ total, offset, limit, label, prevButton, nextButton }) {
  const start = total ? offset + 1 : 0;
  const end = Math.min(offset + limit, total);
  label.textContent = `${start}-${end} of ${total}`;
  prevButton.disabled = offset <= 0;
  nextButton.disabled = offset + limit >= total;
}

function qualityTitle(value) {
  if (value === "ready") {
    return "Ready";
  }
  if (value === "not_ready") {
    return "Not ready";
  }
  return "Needs review";
}

function qualityWarnings(warnings) {
  const labels = {
    low_extracted_text: "low text",
    missing_index_manifest: "index details missing",
    missing_markdown: "missing Markdown",
    no_chunks: "no chunks",
    not_indexed: "not indexed",
    marked_stale: "marked stale",
    rejected_source: "rejected",
    review_expired: "review expired",
    unreviewed_source: "unreviewed",
  };
  return (Array.isArray(warnings) ? warnings : []).map((warning) => labels[warning] || warning);
}

function trustTitle(value) {
  const labels = {
    approved: "approved",
    rejected: "rejected",
    stale: "stale",
    unreviewed: "unreviewed",
  };
  return labels[value] || "unreviewed";
}

function sourceTypeTitle(value) {
  return String(value || "unknown").replaceAll("_", " ");
}

function renderQualityCell(item) {
  const quality = item.quality;
  const trust = item.trust && typeof item.trust === "object" ? item.trust : {};
  const data = quality && typeof quality === "object" ? quality : {};
  const label = data.label || "review";
  const warningText = qualityWarnings(data.warnings).join(", ");
  const trustNotes = String(trust.notes || "").trim();
  const metrics = [
    Number(data.chunk_count || 0) ? `${Number(data.chunk_count)} chunks` : "",
    Number(data.markdown_char_count || 0) ? `${Number(data.markdown_char_count)} chars` : "",
    Number(data.enrichment_markers || 0) ? `${Number(data.enrichment_markers)} enriched` : "",
  ].filter(Boolean);
  const reprocessDisabled = item.can_download ? "" : " disabled";
  return `
    <span class="quality-badge quality-${escapeHtml(label)}">${escapeHtml(qualityTitle(label))}</span>
    <span class="quality-detail">${escapeHtml(metrics.join(" | "))}</span>
    <span class="quality-detail">Trust: ${escapeHtml(trustTitle(trust.review_status))} | ${escapeHtml(sourceTypeTitle(trust.source_type))}</span>
    <span class="quality-warning">${escapeHtml(warningText)}</span>
    ${trustNotes ? `<span class="quality-note">Note: ${escapeHtml(trustNotes)}</span>` : ""}
    <span class="quality-actions">
      <button type="button" data-pdf-action="approve" data-source-hash="${escapeHtml(item.hash || "")}">Approve</button>
      <button type="button" data-pdf-action="stale" data-source-hash="${escapeHtml(item.hash || "")}">Flag stale</button>
      <button type="button" data-pdf-action="reprocess" data-source-hash="${escapeHtml(item.hash || "")}"${reprocessDisabled}>Re-run</button>
    </span>
  `;
}

async function refreshJobs() {
  try {
    const params = new URLSearchParams({
      offset: String(state.jobsOffset),
      limit: String(state.jobsLimit),
    });
    const data = await requestJson(`/api/jobs?${params}`);
    state.jobsTotal = data.total || 0;
    if (state.jobsOffset >= state.jobsTotal && state.jobsOffset > 0) {
      state.jobsOffset = Math.max(0, Math.floor((state.jobsTotal - 1) / state.jobsLimit) * state.jobsLimit);
      return refreshJobs();
    }
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
    updatePageControls({
      total: state.jobsTotal,
      offset: state.jobsOffset,
      limit: state.jobsLimit,
      label: els.jobsPageLabel,
      prevButton: els.prevJobsPageButton,
      nextButton: els.nextJobsPageButton,
    });
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  }
}

async function refreshPdfs() {
  try {
    const params = new URLSearchParams({
      offset: String(state.pdfOffset),
      limit: String(state.pdfLimit),
      search: state.pdfSearch,
    });
    const data = await requestJson(`/api/pdfs?${params}`);
    state.pdfTotal = data.total || 0;
    if (state.pdfOffset >= state.pdfTotal && state.pdfOffset > 0) {
      state.pdfOffset = Math.max(0, Math.floor((state.pdfTotal - 1) / state.pdfLimit) * state.pdfLimit);
      return refreshPdfs();
    }
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
        <td>${renderQualityCell(item)}</td>
        <td>${download}</td>
      `;
      els.pdfsBody.appendChild(row);
    }
    updatePageControls({
      total: state.pdfTotal,
      offset: state.pdfOffset,
      limit: state.pdfLimit,
      label: els.pdfPageLabel,
      prevButton: els.prevPdfPageButton,
      nextButton: els.nextPdfPageButton,
    });
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  }
}

async function handlePdfAction(event) {
  const button = event.target.closest("[data-pdf-action]");
  if (!button) {
    return;
  }
  const sourceHash = button.dataset.sourceHash || "";
  const action = button.dataset.pdfAction || "";
  if (!sourceHash) {
    return;
  }
  const body = {};
  if (action === "approve") {
    body.review_status = "approved";
  } else if (action === "stale") {
    body.review_status = "stale";
    body.notes = window.prompt("Why is this source stale?", "") || "";
  } else if (action === "reprocess") {
    if (!window.confirm("Re-run ingestion and indexing for this source?")) {
      return;
    }
  } else {
    return;
  }
  button.disabled = true;
  try {
    if (action === "reprocess") {
      const job = await requestJson(`/api/pdfs/${encodeURIComponent(sourceHash)}/reprocess`, {
        method: "POST",
      });
      setStatus(els.uploadStatus, `Queued re-run job ${String(job.id || "").slice(0, 8)}.`);
      await refreshJobs();
    } else {
      await requestJson(`/api/pdfs/${encodeURIComponent(sourceHash)}/trust`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }
    await refreshPdfs();
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  } finally {
    button.disabled = false;
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

function clearDuplicatePrompt() {
  state.pendingForceUploadToken = "";
  els.duplicatePrompt.hidden = true;
  els.duplicatePromptText.textContent = "";
  els.forceUploadButton.disabled = true;
}

function showDuplicatePrompt(detail) {
  const forceToken = detail?.force_token || "";
  state.pendingForceUploadToken = forceToken;
  const actionText = forceToken
    ? "Use Force upload to queue the duplicate file(s) anyway."
    : "The server did not issue a force token for this upload.";
  els.duplicatePromptText.textContent = `${duplicateUploadMessage(detail)}\n\n${actionText}`;
  els.duplicatePrompt.hidden = false;
  els.forceUploadButton.disabled = !forceToken;
}

function pdfFilesFromList(files) {
  const accepted = [];
  const rejected = [];
  for (const file of Array.from(files || [])) {
    const name = file.name || "";
    const isPdf = file.type === "application/pdf" || name.toLowerCase().endsWith(".pdf");
    if (isPdf) {
      accepted.push(file);
    } else {
      rejected.push(name || "unnamed file");
    }
  }
  return { accepted, rejected };
}

function updateSelectedFilesLabel() {
  const files = Array.from(els.fileInput.files || []);
  if (!files.length) {
    els.selectedFilesLabel.textContent = "Drop PDFs";
    return;
  }
  const names = files.map((file) => file.name || "unnamed.pdf");
  const shown = names.slice(0, 3).join(", ");
  const extra = names.length > 3 ? ` +${names.length - 3} more` : "";
  els.selectedFilesLabel.textContent = `${names.length} selected: ${shown}${extra}`;
}

function setSelectedUploadFiles(files) {
  clearDuplicatePrompt();
  const { accepted, rejected } = pdfFilesFromList(files);
  if (!accepted.length) {
    setStatus(els.uploadStatus, "Drop one or more PDF files.", true);
    updateSelectedFilesLabel();
    return;
  }

  const transfer = new DataTransfer();
  for (const file of accepted) {
    transfer.items.add(file);
  }
  els.fileInput.files = transfer.files;
  updateSelectedFilesLabel();
  if (rejected.length) {
    setStatus(els.uploadStatus, `Skipped non-PDF file(s): ${rejected.join(", ")}`, true);
  } else {
    setStatus(els.uploadStatus, "");
  }
}

function uploadDragHasFiles(event) {
  return Array.from(event.dataTransfer?.types || []).includes("Files");
}

function handleUploadDrag(event) {
  if (!uploadDragHasFiles(event)) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  if (event.type === "dragenter") {
    state.uploadDragDepth += 1;
  }
  els.uploadDropZone.classList.add("drag-over");
}

function clearUploadDrag(event) {
  if (!uploadDragHasFiles(event)) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  state.uploadDragDepth = Math.max(0, state.uploadDragDepth - 1);
  if (!state.uploadDragDepth) {
    els.uploadDropZone.classList.remove("drag-over");
  }
}

function handleUploadDrop(event) {
  event.preventDefault();
  event.stopPropagation();
  state.uploadDragDepth = 0;
  els.uploadDropZone.classList.remove("drag-over");
  setSelectedUploadFiles(event.dataTransfer?.files || []);
}

async function uploadFiles(forceDuplicates = false, forceToken = "") {
  const isForced = forceDuplicates === true;
  const files = Array.from(els.fileInput.files || []);
  if (!files.length) {
    setStatus(els.uploadStatus, "Choose one or more PDF files.", true);
    return;
  }
  if (!isForced) {
    clearDuplicatePrompt();
  }

  const body = new FormData();
  for (const file of files) {
    body.append("files", file);
  }
  if (isForced) {
    body.append("force_duplicates", "true");
    if (forceToken) {
      body.append("force_token", forceToken);
    }
  }

  els.uploadButton.disabled = true;
  els.forceUploadButton.disabled = true;
  setStatus(els.uploadStatus, isForced ? "Queueing forced upload..." : "Queueing upload...");
  try {
    const result = await requestJson("/api/uploads", { method: "POST", body });
    const jobs = Array.isArray(result.jobs) && result.jobs.length ? result.jobs : [result];
    els.fileInput.value = "";
    updateSelectedFilesLabel();
    clearDuplicatePrompt();
    state.jobsOffset = 0;
    state.pdfOffset = 0;
    if (jobs.length === 1) {
      setStatus(els.uploadStatus, `Queued job ${jobs[0].id.slice(0, 8)}.`);
    } else {
      setStatus(els.uploadStatus, `Queued ${jobs.length} jobs.`);
    }
    await refreshJobs();
    await refreshPdfs();
  } catch (error) {
    if (
      error.status === 409 &&
      error.detail &&
      error.detail.can_force !== false &&
      error.detail.force_token
    ) {
      showDuplicatePrompt(error.detail);
      setStatus(
        els.uploadStatus,
        isForced ? "Forced upload was blocked. Review the warning below and retry." : "Duplicate upload blocked.",
        true,
      );
      return;
    }
    if (error.status === 409 && error.detail) {
      clearDuplicatePrompt();
      setStatus(els.uploadStatus, duplicateUploadMessage(error.detail), true);
      return;
    }
    clearDuplicatePrompt();
    setStatus(els.uploadStatus, error.message, true);
  } finally {
    els.uploadButton.disabled = false;
    els.forceUploadButton.disabled = !state.pendingForceUploadToken;
  }
}

async function enqueueReindex() {
  els.reindexButton.disabled = true;
  setStatus(els.uploadStatus, "Queueing reindex...");
  try {
    const job = await requestJson("/api/reindex", { method: "POST" });
    setStatus(els.uploadStatus, `Queued reindex job ${job.id.slice(0, 8)}.`);
    state.jobsOffset = 0;
    await refreshJobs();
    await refreshPdfs();
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  } finally {
    els.reindexButton.disabled = false;
  }
}

async function loadIndex() {
  const load = startIndexLoad();
  state.indexMode = "standard";
  if (state.indexPageSize === "all") {
    return loadAllIndexSummaries(load);
  }
  state.limit = Number(state.indexPageSize) || 20;
  const params = new URLSearchParams({
    offset: String(state.offset),
    limit: String(state.limit),
    search: state.search,
  });
  try {
    const data = await requestJson(`/api/index/summaries?${params}`, {
      signal: load.abortController.signal,
    });
    if (!isActiveIndexLoad(load)) {
      return;
    }
    state.total = data.total || 0;
    renderIndexRows(data.rows || []);
    const start = state.total ? state.offset + 1 : 0;
    const end = Math.min(state.offset + state.limit, state.total);
    els.pageLabel.textContent = `${start}-${end} of ${state.total} summaries`;
    els.prevPageButton.disabled = state.offset <= 0;
    els.nextPageButton.disabled = state.offset + state.limit >= state.total;
    setStatus(els.indexStatus, `Embedding model: ${data.embedding_model || "unknown"}`);
  } catch (error) {
    if (isAbortError(error) || !isActiveIndexLoad(load)) {
      return;
    }
    els.indexBody.innerHTML = "";
    els.pageLabel.textContent = "";
    setStatus(els.indexStatus, error.message, true);
  } finally {
    finishIndexLoad(load);
  }
}

async function loadAllIndexSummaries(load) {
  const params = new URLSearchParams({
    offset: "0",
    limit: "0",
    search: state.search,
  });

  els.indexBody.innerHTML = "";
  els.pageLabel.textContent = "Loading summaries...";
  els.prevPageButton.disabled = true;
  els.nextPageButton.disabled = true;
  setStatus(els.indexStatus, "Loading summary chunks...");

  try {
    const data = await requestJson(`/api/index/summaries?${params}`, {
      signal: load.abortController.signal,
    });
    if (!isActiveIndexLoad(load)) {
      return;
    }
    const rows = data.rows || [];
    state.total = data.total || rows.length;
    renderIndexRows(rows);
    els.pageLabel.textContent = `All ${rows.length} of ${state.total} summaries`;
    els.prevPageButton.disabled = true;
    els.nextPageButton.disabled = true;
    setStatus(els.indexStatus, `Embedding model: ${data.embedding_model || "unknown"}`);
  } catch (error) {
    if (isAbortError(error) || !isActiveIndexLoad(load)) {
      return;
    }
    els.indexBody.innerHTML = "";
    els.pageLabel.textContent = "";
    setStatus(els.indexStatus, error.message, true);
  } finally {
    finishIndexLoad(load);
  }
}

function indexNodeLabel(item) {
  const nodeType = item.node_type || "chunk";
  if (nodeType === "document_summary") {
    return "Document summary";
  }
  if (nodeType === "section_summary") {
    return "Section summary";
  }
  return `Detail chunk ${item.chunk_index}`;
}

function indexPageRange(item) {
  const start = Number(item.page_start || 0);
  const end = Number(item.page_end || 0);
  if (!start && !end) {
    return "";
  }
  if (start && end && start !== end) {
    return `pages ${start}-${end}`;
  }
  return `page ${start || end}`;
}

function indexChildCountLabel(item) {
  const detailCount = Number(item.detail_count || 0);
  const summaryCount = Number(item.summary_count || 0);
  if (!detailCount && !summaryCount) {
    return "";
  }
  const parts = [];
  if (summaryCount) {
    parts.push(`${summaryCount} summaries`);
  }
  if (detailCount) {
    parts.push(`${detailCount} details`);
  }
  return parts.join(", ");
}

function indexScoreLabel(item) {
  const score = Number(item.score || 0);
  if (!score) {
    return "";
  }
  return `score ${score.toFixed(3)}`;
}

function createIndexRow(item, options = {}) {
  const row = document.createElement("tr");
  row.dataset.recordId = item.id;
  row.dataset.nodeType = item.node_type || "chunk";
  row.classList.add("index-node-row");
  const level = Number(options.level ?? item.node_level ?? 0);
  row.style.setProperty("--index-level", String(Math.max(0, level)));
  if ((item.node_type || "") === "document_summary" || (item.node_type || "") === "section_summary") {
    row.classList.add("index-summary-row");
  } else {
    row.classList.add("index-detail-row");
  }
  if (options.parentId) {
    row.dataset.parentId = options.parentId;
  }

  const download = item.source_hash
    ? `<br /><a class="download-link" href="/api/pdfs/${encodeURIComponent(item.source_hash)}/download">Download PDF</a>`
    : "";
  const hasChildren = Number(item.child_count || 0) > 0;
  const toggle = hasChildren && options.allowToggle
    ? `<button type="button" class="tree-toggle" data-action="toggle-children" aria-expanded="false" title="Show details" aria-label="Show details">+</button>`
    : `<span class="tree-spacer"></span>`;
  const sourceName = item.source_pdf_name || item.file_path || item.title || item.id;
  const pageRange = indexPageRange(item);
  const childCount = indexChildCountLabel(item);
  const meta = [indexNodeLabel(item), indexScoreLabel(item), pageRange, childCount]
    .filter(Boolean)
    .join(" | ");
  row.innerHTML = `
      <td class="source-cell">
        <div class="index-source-node">
          ${toggle}
          <div>
            <strong>${escapeHtml(item.title || item.id)}</strong><br />
            <span class="index-node-meta">${escapeHtml(meta)}</span><br />
            ${escapeHtml(sourceName)}<br />
            <span class="index-record-id">${escapeHtml(item.id)}</span>
            ${download}
          </div>
        </div>
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
  return row;
}

function createIndexLoadMoreRow(parentId, nextOffset, total) {
  const row = document.createElement("tr");
  row.className = "index-load-more-row";
  row.dataset.parentId = parentId;
  const remaining = Math.max(0, total - nextOffset);
  row.innerHTML = `
    <td colspan="3">
      <button type="button" data-action="load-more-children" data-next-offset="${nextOffset}">
        Load ${Math.min(INDEX_CHILD_BATCH_SIZE, remaining)} more detail rows
      </button>
    </td>
  `;
  return row;
}

function createIndexEmptyChildRow(parentId) {
  const row = document.createElement("tr");
  row.className = "index-empty-child-row";
  row.dataset.parentId = parentId;
  row.innerHTML = `<td colspan="3">No matching detail rows.</td>`;
  return row;
}

function appendIndexRows(rows) {
  const fragment = document.createDocumentFragment();
  for (const item of rows) {
    fragment.appendChild(createIndexRow(item, { allowToggle: true }));
  }
  els.indexBody.appendChild(fragment);
}

function renderIndexRows(rows) {
  els.indexBody.innerHTML = "";
  appendIndexRows(rows);
}

function renderVectorIndexRows(rows) {
  els.indexBody.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const item of rows) {
    const row = createIndexRow(item, { allowToggle: false });
    row.classList.add("index-vector-result-row");
    fragment.appendChild(row);
  }
  els.indexBody.appendChild(fragment);
}

async function runIndexVectorSearch() {
  const query = els.vectorSearchInput.value.trim();
  if (!query) {
    setStatus(els.indexStatus, "Enter a vector search query.", true);
    return;
  }

  const relevanceFloor = numericSetting(els.vectorRelevanceFloorInput, 0.7, 0);
  state.indexMode = "vector";
  state.vectorSearch = query;
  state.vectorRelevanceFloor = relevanceFloor;
  const load = startIndexLoad();
  els.vectorSearchButton.disabled = true;
  els.prevPageButton.disabled = true;
  els.nextPageButton.disabled = true;
  els.pageLabel.textContent = "Vector search running...";
  setStatus(els.indexStatus, "Vector search is querying embeddings and may take longer.");

  try {
    const data = await requestJson("/api/index/vector-search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: load.abortController.signal,
      body: JSON.stringify({
        query,
        relevance_floor: relevanceFloor,
      }),
    });
    if (!isActiveIndexLoad(load)) {
      return;
    }
    const rows = data.rows || [];
    renderVectorIndexRows(rows);
    state.total = data.total || rows.length;
    els.pageLabel.textContent = `${rows.length} vector result${rows.length === 1 ? "" : "s"}`;
    setStatus(
      els.indexStatus,
      `Vector search complete. Relevance floor: ${Number(data.relevance_floor || relevanceFloor).toFixed(2)}`
    );
  } catch (error) {
    if (isAbortError(error) || !isActiveIndexLoad(load)) {
      return;
    }
    els.indexBody.innerHTML = "";
    els.pageLabel.textContent = "";
    setStatus(els.indexStatus, error.message, true);
  } finally {
    els.vectorSearchButton.disabled = false;
    finishIndexLoad(load);
  }
}

function indexRowsForParent(parentId) {
  return Array.from(els.indexBody.querySelectorAll("tr")).filter(
    (row) => row.dataset.parentId === parentId
  );
}

function indexParentRow(parentId) {
  return Array.from(els.indexBody.querySelectorAll("tr")).find(
    (row) => row.dataset.recordId === parentId && !row.dataset.parentId
  );
}

function removeIndexLoadMoreRows(parentId) {
  for (const row of indexRowsForParent(parentId)) {
    if (row.classList.contains("index-load-more-row")) {
      row.remove();
    }
  }
}

function setIndexChildrenVisible(parentId, visible) {
  for (const row of indexRowsForParent(parentId)) {
    row.hidden = !visible;
  }
}

function setIndexToggle(button, expanded) {
  button.textContent = expanded ? "-" : "+";
  button.setAttribute("aria-expanded", expanded ? "true" : "false");
  button.title = expanded ? "Hide details" : "Show details";
  button.setAttribute("aria-label", button.title);
}

function insertIndexRowsAfter(anchor, rows) {
  const fragment = document.createDocumentFragment();
  for (const row of rows) {
    fragment.appendChild(row);
  }
  anchor.after(fragment);
}

async function loadIndexChildren(parentRow, offset = 0, toggleButton = null) {
  const parentId = parentRow.dataset.recordId;
  const params = new URLSearchParams({
    parent_id: parentId,
    offset: String(offset),
    limit: String(INDEX_CHILD_BATCH_SIZE),
    search: state.search,
  });
  const existingRows = indexRowsForParent(parentId).filter(
    (row) => !row.classList.contains("index-load-more-row")
  );
  const anchor = existingRows.length ? existingRows[existingRows.length - 1] : parentRow;
  removeIndexLoadMoreRows(parentId);

  const data = await requestJson(`/api/index/children?${params}`);
  const rows = data.rows || [];
  const nodes = rows.map((item) =>
    createIndexRow(item, {
      level: item.node_level || 1,
      parentId,
      allowToggle: false,
    })
  );
  if (!rows.length && offset === 0) {
    nodes.push(createIndexEmptyChildRow(parentId));
  }

  const nextOffset = Number(data.offset || 0) + rows.length;
  if (nextOffset < Number(data.total || 0)) {
    nodes.push(createIndexLoadMoreRow(parentId, nextOffset, Number(data.total || 0)));
  }
  insertIndexRowsAfter(anchor, nodes);
  if (toggleButton) {
    setIndexToggle(toggleButton, true);
  }
}

async function toggleIndexChildren(parentRow, button) {
  const parentId = parentRow.dataset.recordId;
  const expanded = button.getAttribute("aria-expanded") === "true";
  if (expanded) {
    setIndexChildrenVisible(parentId, false);
    setIndexToggle(button, false);
    return;
  }

  const existingRows = indexRowsForParent(parentId);
  if (existingRows.length) {
    setIndexChildrenVisible(parentId, true);
    setIndexToggle(button, true);
    return;
  }

  button.disabled = true;
  try {
    await loadIndexChildren(parentRow, 0, button);
  } catch (error) {
    setStatus(els.indexStatus, error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function loadMoreIndexChildren(button) {
  const row = button.closest("tr");
  const parentId = row.dataset.parentId;
  const parentRow = indexParentRow(parentId);
  if (!parentRow) {
    return;
  }
  button.disabled = true;
  try {
    await loadIndexChildren(parentRow, Number(button.dataset.nextOffset || 0));
  } catch (error) {
    setStatus(els.indexStatus, error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function handleIndexAction(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) {
    return;
  }

  const row = button.closest("tr");
  const action = button.dataset.action;
  if (action === "toggle-children") {
    await toggleIndexChildren(row, button);
    return;
  }
  if (action === "load-more-children") {
    await loadMoreIndexChildren(button);
    return;
  }

  const recordId = row.dataset.recordId;
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
      if (state.indexMode === "vector") {
        await runIndexVectorSearch();
      } else {
        await loadIndex();
      }
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
  thinkingBody.className = "thinking-body stream-body";
  const thinkingStable = document.createElement("div");
  thinkingStable.className = "stream-stable rendered";
  const thinkingTail = document.createElement("div");
  thinkingTail.className = "stream-tail raw-tail";
  thinkingBody.append(thinkingStable, thinkingTail);
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
  body.className = "body stream-body";
  const answerStable = document.createElement("div");
  answerStable.className = "stream-stable rendered";
  const answerTail = document.createElement("div");
  answerTail.className = "stream-tail raw-tail";
  body.append(answerStable, answerTail);

  const toolResultsPanel = document.createElement("details");
  toolResultsPanel.className = "tool-results-block";
  toolResultsPanel.hidden = true;

  const toolResultsSummary = document.createElement("summary");
  toolResultsSummary.textContent = "Tool results";

  const toolResultsBody = document.createElement("div");
  toolResultsBody.className = "tool-results-body";
  toolResultsPanel.append(toolResultsSummary, toolResultsBody);

  message.append(roleLabel, thinking, sources, notice, body, toolResultsPanel);
  els.chatMessages.appendChild(message);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  return {
    body,
    answerStable,
    answerTail,
    thinking,
    thinkingBody,
    thinkingStable,
    thinkingTail,
    sourcesPanel: sources,
    sourcesBody,
    toolResultsPanel,
    toolResultsBody,
    notice,
    rawAnswer: "",
    rawThinking: "",
    answerCommittedLength: 0,
    thinkingCommittedLength: 0,
    sourcesData: [],
    toolResults: [],
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
    if (source.kind === "local" && source.open_url) {
      links.push(`<a href="${escapeHtml(source.open_url)}" target="_blank">Open page</a>`);
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

function normalizeToolResultEvent(event) {
  const result = event.result && typeof event.result === "object" ? event.result : null;
  return {
    tool: String(event.tool || result?.tool || "tool"),
    text: String(event.text || ""),
    result,
    content: String(event.content || (result ? JSON.stringify(result) : "")),
  };
}

function toolResultJson(entry) {
  if (entry.result && typeof entry.result === "object") {
    return JSON.stringify(entry.result, null, 2);
  }
  if (entry.content) {
    try {
      return JSON.stringify(JSON.parse(entry.content), null, 2);
    } catch (_) {
      return entry.content;
    }
  }
  return "";
}

function toolResultItemTitle(item) {
  return item.location || item.title || item.url || item.chunk_id || item.source_id || "";
}

function toolResultItemText(item) {
  return item.content || item.snippet || item.error || "";
}

function sourceAsToolResultItem(source) {
  if (source.kind === "web") {
    return {
      source_id: source.id || "",
      citation: source.label || source.id || "",
      title: source.title || source.url || "",
      url: source.url || "",
      snippet: source.snippet || "",
      provider: source.provider || "",
    };
  }
  return {
    source_id: source.id || "",
    citation: source.label || source.id || "",
    chunk_id: source.chunk_id || "",
    score: source.score,
    location: [source.source_pdf_name, source.section_path, source.page_label].filter(Boolean).join(" :: "),
    snippet: source.snippet || "",
  };
}

function fallbackToolResultEntries(parts) {
  if (Array.isArray(parts.toolResults) && parts.toolResults.length) {
    return parts.toolResults;
  }

  const sources = Array.isArray(parts.sources) ? parts.sources : [];
  const localSources = sources.filter((source) => source.kind !== "web");
  const webSources = sources.filter((source) => source.kind === "web");
  const entries = [];

  if (localSources.length) {
    const result = {
      tool: "search_local_context",
      result_count: localSources.length,
      results: localSources.map(sourceAsToolResultItem),
    };
    entries.push({
      tool: result.tool,
      text: `Retrieved ${localSources.length} local source chunk(s).`,
      result,
      content: JSON.stringify(result),
      fromSources: true,
    });
  }

  if (webSources.length) {
    const result = {
      tool: "web_search",
      result_count: webSources.length,
      results: webSources.map(sourceAsToolResultItem),
    };
    entries.push({
      tool: result.tool,
      text: `Retrieved ${webSources.length} web result(s).`,
      result,
      content: JSON.stringify(result),
      fromSources: true,
    });
  }

  return entries;
}

function renderToolResultsPanel(parts) {
  const entries = fallbackToolResultEntries(parts);
  parts.toolResultsPanel.hidden = entries.length === 0;
  parts.toolResultsPanel.open = entries.length > 0;
  parts.toolResultsBody.innerHTML = "";

  for (const [index, entry] of entries.entries()) {
    const result = entry.result && typeof entry.result === "object" ? entry.result : {};
    const toolName = entry.tool || result.tool || `tool_${index + 1}`;
    const query = result.query ? ` query: ${result.query}` : "";
    const item = document.createElement("div");
    item.className = "tool-result-item";

    const heading = document.createElement("div");
    heading.className = "tool-result-heading";
    const resultCount = Number.isFinite(Number(result.result_count))
      ? `${Number(result.result_count)} result(s)`
      : "";
    heading.innerHTML = `
      <strong>${escapeHtml(toolName)}</strong>
      <span>${escapeHtml([query, resultCount, result.provider || ""].filter(Boolean).join(" | "))}</span>
    `;
    item.appendChild(heading);

    if (entry.text || result.error) {
      const status = document.createElement("div");
      status.className = "tool-result-status";
      status.textContent = result.error || entry.text;
      item.appendChild(status);
    }

    const rows = Array.isArray(result.results) ? result.results : [];
    if (rows.length) {
      const list = document.createElement("div");
      list.className = "tool-result-list";
      for (const row of rows) {
        const rowItem = document.createElement("div");
        rowItem.className = "tool-result-row";

        const rowHeading = document.createElement("div");
        rowHeading.className = "tool-result-row-heading";
        const citation = row.citation || row.source_id || "";
        rowHeading.innerHTML = `
          <strong>${escapeHtml(citation)}</strong>
          <span>${escapeHtml(toolResultItemTitle(row))}</span>
        `;

        const rowText = document.createElement("pre");
        rowText.textContent = toolResultItemText(row);
        rowItem.append(rowHeading, rowText);
        list.appendChild(rowItem);
      }
      item.appendChild(list);
    }

    const rawJson = toolResultJson(entry);
    if (rawJson) {
      const raw = document.createElement("details");
      raw.className = "tool-result-json";
      const summary = document.createElement("summary");
      summary.textContent = entry.fromSources ? "Source details" : "JSON sent to model";
      const pre = document.createElement("pre");
      pre.textContent = rawJson;
      raw.append(summary, pre);
      item.appendChild(raw);
    }

    parts.toolResultsBody.appendChild(item);
  }
}

function addSavedAssistantMessage(saved) {
  const parts = addAssistantMessage();
  parts.finalized = true;
  parts.rawAnswer = saved.text || "";
  parts.rawThinking = saved.thinking || "";
  parts.sources = saved.sources || [];
  parts.toolResults = Array.isArray(saved.toolResults) ? saved.toolResults : [];
  if (parts.rawThinking) {
    parts.thinking.hidden = false;
    parts.thinking.open = false;
    if (saved.thinkingHtml) {
      replaceStreamHtml(parts, "thinking", saved.thinkingHtml);
    } else {
      parts.thinkingStable.className = "stream-stable raw-tail";
      parts.thinkingStable.textContent = parts.rawThinking;
      parts.thinkingCommittedLength = parts.rawThinking.length;
    }
  }
  renderSourcePanel(parts);
  if (saved.notice) {
    parts.notice.textContent = saved.notice;
    parts.notice.hidden = false;
  }
  if (saved.answerHtml) {
    replaceStreamHtml(parts, "answer", saved.answerHtml);
  } else {
    parts.answerStable.className = "stream-stable raw-tail";
    parts.answerStable.textContent = parts.rawAnswer;
    parts.answerCommittedLength = parts.rawAnswer.length;
  }
  renderToolResultsPanel(parts);
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

function setSendButtonStreaming(streaming) {
  els.sendButton.disabled = false;
  els.sendButton.textContent = streaming ? "Stop" : "Send";
  els.sendButton.classList.toggle("stop-button", streaming);
  els.sendButton.title = streaming ? "Stop generation" : "";
  els.sendButton.setAttribute("aria-label", streaming ? "Stop generation" : "Send");
}

function stopGeneration() {
  if (!state.streamingChatId || !state.chatAbortController) {
    return;
  }
  state.chatAbortController.abort();
}

function appendStreamEvent(parts, event) {
  const type = event.type || "answer";
  const text = event.text || "";

  if (type === "thinking") {
    if (!text) {
      return;
    }
    markGemmaResponseStarted(parts);
    parts.thinking.hidden = false;
    parts.thinking.open = true;
    parts.rawThinking += text;
    updateStreamTailRaw(parts, "thinking");
    queueMarkdownRender(parts, "thinking");
    return;
  }

  if (type === "error") {
    if (!text) {
      return;
    }
    addPersistentNotice(parts, `[Error] ${text}`);
    return;
  }

  if (type === "sources") {
    parts.sources = Array.isArray(event.sources) ? event.sources : [];
    renderSourcePanel(parts);
    return;
  }

  if (type === "tool_result") {
    if (event.result || event.content) {
      parts.toolResults.push(normalizeToolResultEvent(event));
    }
    if (!text) {
      return;
    }
    if (isTransientNotice(text)) {
      parts.transientNotices.push(text);
    } else {
      parts.persistentNotices.push(text);
    }
    updateNotice(parts);
    return;
  }

  if (type === "tool_call") {
    if (!text) {
      return;
    }
    if (isTransientNotice(text)) {
      parts.transientNotices.push(text);
    } else {
      parts.persistentNotices.push(text);
    }
    updateNotice(parts);
    return;
  }

  if (type === "notice") {
    if (!text) {
      return;
    }
    if (isTransientNotice(text)) {
      parts.transientNotices.push(text);
    } else {
      parts.persistentNotices.push(text);
    }
    updateNotice(parts);
    return;
  }

  if (!text) {
    return;
  }
  markGemmaResponseStarted(parts);
  parts.rawAnswer += text;
  updateStreamTailRaw(parts, "answer");
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
      replaceStreamHtml(parts, "thinking", thinkingHtml);
    }
    if (parts.rawAnswer) {
      replaceStreamHtml(parts, "answer", answerHtml);
    }
  } catch (error) {
    addFormattingNotice(parts, error);
  } finally {
    renderToolResultsPanel(parts);
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
  const abortController = new AbortController();
  state.chatAbortController = abortController;
  addUserMessageToChat(chat, question);
  addMessage("You", question);
  const assistantParts = addAssistantMessage();
  els.questionInput.value = "";
  setSendButtonStreaming(true);
  renderSavedChats();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: abortController.signal,
      body: JSON.stringify({
        question,
        temperature: numericSetting(els.temperatureInput, 0.3, 0),
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
    if (error.name === "AbortError") {
      appendStreamEvent(assistantParts, { type: "notice", text: "Generation stopped." });
    } else {
      appendStreamEvent(assistantParts, { type: "error", text: error.message });
    }
  } finally {
    await formatAssistantMessage(assistantParts);
    addAssistantMessageToChat(chat, assistantParts);
    state.streamingChatId = null;
    if (state.chatAbortController === abortController) {
      state.chatAbortController = null;
    }
    setSendButtonStreaming(false);
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
    } else {
      abortIndexLoad();
    }
  });
});

els.updateButton.addEventListener("click", applyUpdate);
els.fileInput.addEventListener("change", () => {
  clearDuplicatePrompt();
  updateSelectedFilesLabel();
});
els.uploadDropZone.addEventListener("dragenter", handleUploadDrag);
els.uploadDropZone.addEventListener("dragover", handleUploadDrag);
els.uploadDropZone.addEventListener("dragleave", clearUploadDrag);
els.uploadDropZone.addEventListener("drop", handleUploadDrop);
els.uploadButton.addEventListener("click", () => uploadFiles());
els.forceUploadButton.addEventListener("click", () => {
  if (!state.pendingForceUploadToken) {
    return;
  }
  uploadFiles(true, state.pendingForceUploadToken);
});
els.reindexButton.addEventListener("click", enqueueReindex);
els.pdfSearchButton.addEventListener("click", () => {
  state.pdfSearch = els.pdfSearchInput.value.trim();
  state.pdfOffset = 0;
  refreshPdfs();
});
els.pdfSearchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    state.pdfSearch = els.pdfSearchInput.value.trim();
    state.pdfOffset = 0;
    refreshPdfs();
  }
});
els.pdfsBody.addEventListener("click", handlePdfAction);
els.prevPdfPageButton.addEventListener("click", () => {
  state.pdfOffset = Math.max(0, state.pdfOffset - state.pdfLimit);
  refreshPdfs();
});
els.nextPdfPageButton.addEventListener("click", () => {
  state.pdfOffset += state.pdfLimit;
  refreshPdfs();
});
els.prevJobsPageButton.addEventListener("click", () => {
  state.jobsOffset = Math.max(0, state.jobsOffset - state.jobsLimit);
  refreshJobs();
});
els.nextJobsPageButton.addEventListener("click", () => {
  state.jobsOffset += state.jobsLimit;
  refreshJobs();
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
els.vectorSearchButton.addEventListener("click", runIndexVectorSearch);
els.vectorSearchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runIndexVectorSearch();
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
els.sendButton.addEventListener("click", (event) => {
  if (!state.streamingChatId) {
    return;
  }
  event.preventDefault();
  stopGeneration();
});
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
