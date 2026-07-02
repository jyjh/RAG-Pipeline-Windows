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
  jobsActive: false,
  chats: [],
  activeChatId: null,
  streamingChatId: null,
  chatAbortController: null,
  chatSidebarCollapsed: false,
  healthPollIntervalMs: 60000,
  jobsPollIntervalMs: 60000,
  healthTimer: null,
  jobsTimer: null,
  jobsTimerIntervalMs: 0,
  updateTimer: null,
  updateApplying: false,
  indexAbortController: null,
  indexLoadToken: 0,
  uploadDragDepth: 0,
  pendingForceUploadToken: "",
  sourceGroupPromptResolver: null,
  selectedPdfHashes: new Set(),
  walkthroughFakePdfPinned: false,
  walkthroughFakePdfVisible: false,
  walkthroughIndex: -1,
  pendingSiteVersion: "",
  pendingVersionPrompt: false,
};

const LIVE_RENDER_INTERVAL_MS = 200;
const STREAM_TAIL_HOLD_CHARS = 700;
const STREAM_TAIL_MAX_CHARS = 2200;
const UPDATE_POLL_INTERVAL_MS = 5 * 60 * 1000;
const RESTART_POLL_INTERVAL_MS = 1000;
const RESTART_POLL_TIMEOUT_MS = 120000;
const JOBS_ACTIVE_POLL_INTERVAL_MS = 2000;
const INDEX_STREAM_BATCH_SIZE = 250;
const INDEX_CHILD_BATCH_SIZE = 100;
const CHAT_STORAGE_KEY = "rag.chatHistory.v1";
const CHAT_UI_STORAGE_KEY = "rag.chatUi.v1";
const TUTORIAL_SEEN_COOKIE = "rag_tutorial_seen";
const SITE_VERSION_COOKIE = "rag_site_version";
const REVIEWER_NAME_COOKIE = "rag_reviewer_name";
const COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;
const WALKTHROUGH_FAKE_PDF_HASH = "__walkthrough_fake_untagged_pdf__";
const SOURCE_GROUP_LABELS = {
  official: "Official",
  student_research: "Student Research",
  unofficial: "Unofficial",
  ungrouped: "Ungrouped",
};
const SOURCE_GROUP_WEIGHTS = {
  official: 1.0,
  student_research: 0.9,
  unofficial: 0.8,
  ungrouped: 0.1,
};

const els = {
  statusLine: document.getElementById("statusLine"),
  updateButton: document.getElementById("updateButton"),
  uploadDropZone: document.getElementById("uploadDropZone"),
  fileInput: document.getElementById("fileInput"),
  selectedFilesLabel: document.getElementById("selectedFilesLabel"),
  uploadGroupsPanel: document.getElementById("uploadGroupsPanel"),
  uploadButton: document.getElementById("uploadButton"),
  reindexButton: document.getElementById("reindexButton"),
  uploadStatus: document.getElementById("uploadStatus"),
  duplicatePrompt: document.getElementById("duplicatePrompt"),
  duplicatePromptText: document.getElementById("duplicatePromptText"),
  forceUploadButton: document.getElementById("forceUploadButton"),
  pdfSearchInput: document.getElementById("pdfSearchInput"),
  pdfSearchButton: document.getElementById("pdfSearchButton"),
  reviewerNameInput: document.getElementById("reviewerNameInput"),
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
  startGuideButton: document.getElementById("startGuideButton"),
  walkthroughOverlay: document.getElementById("walkthroughOverlay"),
  walkthroughStepLabel: document.getElementById("walkthroughStepLabel"),
  walkthroughTitle: document.getElementById("walkthroughTitle"),
  walkthroughText: document.getElementById("walkthroughText"),
  walkthroughPrevButton: document.getElementById("walkthroughPrevButton"),
  walkthroughNextButton: document.getElementById("walkthroughNextButton"),
  walkthroughCloseButton: document.getElementById("walkthroughCloseButton"),
  welcomeTutorialOverlay: document.getElementById("welcomeTutorialOverlay"),
  welcomeTutorialStartButton: document.getElementById("welcomeTutorialStartButton"),
  welcomeTutorialSkipButton: document.getElementById("welcomeTutorialSkipButton"),
  cachePromptOverlay: document.getElementById("cachePromptOverlay"),
  cachePromptText: document.getElementById("cachePromptText"),
  cachePromptReloadButton: document.getElementById("cachePromptReloadButton"),
  cachePromptDoneButton: document.getElementById("cachePromptDoneButton"),
  sourceGroupPromptOverlay: document.getElementById("sourceGroupPromptOverlay"),
  sourceGroupPromptCancelButton: document.getElementById("sourceGroupPromptCancelButton"),
  pdfSelectAllCheckbox: document.getElementById("pdfSelectAllCheckbox"),
  pdfBulkActionBar: document.getElementById("pdfBulkActionBar"),
  pdfBulkCountLabel: document.getElementById("pdfBulkCountLabel"),
  pdfBulkTagButton: document.getElementById("pdfBulkTagButton"),
  pdfBulkClearButton: document.getElementById("pdfBulkClearButton"),
};

const walkthroughSteps = [
  {
    tab: "upload",
    target: "#uploadDropZone",
    title: "Add source PDFs",
    text: "Drop PDFs here or use the file picker. Each PDF is queued as a background job that extracts Markdown, images, formulas, tables, and retrieval chunks.",
  },
  {
    tab: "upload",
    target: "#jobsTable",
    title: "Track ingestion and indexing",
    text: "The Jobs table shows queued, running, paused, completed, and failed work. Long jobs continue in the background while the published index remains available.",
  },
  {
    tab: "upload",
    target: "#walkthroughFakePdfRow [data-pdf-action='tag-group']",
    title: "Tag source reliability",
    text: "Untagged PDFs are highlighted at the top of the table. Use Tag group to mark each source as Official, Student Research, or Unofficial before relying on retrieval ranking.",
    fakePdf: true,
  },
  {
    tab: "index",
    target: "#indexTable",
    title: "Inspect indexed content",
    text: "Review summaries and detail chunks before relying on them. Expand summary rows, search text, and edit records after indexing has settled.",
  },
  {
    tab: "index",
    target: "#vectorSearchInput",
    title: "Test retrieval",
    text: "Advanced retrieval search embeds a query and shows likely source chunks. Use it to diagnose whether the index can find the evidence you expect.",
  },
  {
    tab: "chat",
    target: "#questionInput",
    title: "Ask cited questions",
    text: "Ask focused engineering questions here. The assistant retrieves local context first, streams the answer, and exposes Sources and Tool results for inspection.",
  },
  {
    tab: "chat",
    target: "#savedChatsList",
    title: "Keep investigation threads",
    text: "Saved chats stay in this browser. Use separate chats for separate design questions, source audits, or debugging sessions.",
  },
];

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

function getCookie(name) {
  const prefix = `${encodeURIComponent(name)}=`;
  for (const part of document.cookie.split(";")) {
    const trimmed = part.trim();
    if (trimmed.startsWith(prefix)) {
      return decodeURIComponent(trimmed.slice(prefix.length));
    }
  }
  return "";
}

function setCookie(name, value, maxAgeSeconds = COOKIE_MAX_AGE_SECONDS) {
  const encodedName = encodeURIComponent(name);
  const encodedValue = encodeURIComponent(value);
  document.cookie = `${encodedName}=${encodedValue}; Max-Age=${maxAgeSeconds}; Path=/; SameSite=Lax`;
}

function normalizeReviewerName(value) {
  return String(value || "").trim().replace(/\s+/g, " ").slice(0, 80);
}

function saveReviewerName(value) {
  const reviewer = normalizeReviewerName(value);
  els.reviewerNameInput.value = reviewer;
  if (reviewer) {
    setCookie(REVIEWER_NAME_COOKIE, reviewer);
  } else {
    setCookie(REVIEWER_NAME_COOKIE, "", 0);
  }
  return reviewer;
}

function loadReviewerName() {
  saveReviewerName(getCookie(REVIEWER_NAME_COOKIE));
}

function ensureReviewerName() {
  let reviewer = saveReviewerName(els.reviewerNameInput.value || getCookie(REVIEWER_NAME_COOKIE));
  if (reviewer) {
    return reviewer;
  }
  const prompted = window.prompt("Enter your name to record who approved or flagged this source:", "");
  if (prompted === null) {
    return "";
  }
  reviewer = saveReviewerName(prompted);
  if (!reviewer) {
    setStatus(els.uploadStatus, "Enter a reviewer name before approving or flagging sources.", true);
  }
  return reviewer;
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
    job_interrupted: "job interrupted",
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

function sourceGroupTitle(value) {
  return SOURCE_GROUP_LABELS[String(value || "ungrouped")] || SOURCE_GROUP_LABELS.ungrouped;
}

function sourceGroupWeight(value) {
  return Number(SOURCE_GROUP_WEIGHTS[String(value || "ungrouped")] || SOURCE_GROUP_WEIGHTS.ungrouped);
}

function parseSourceGroupInput(value) {
  const text = String(value || "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  if (!text) {
    return "";
  }
  if (text === "student" || text === "studentresearch") {
    return "student_research";
  }
  if (text in SOURCE_GROUP_LABELS && text !== "ungrouped") {
    return text;
  }
  return "";
}

function closeSourceGroupPrompt(value = "") {
  if (els.sourceGroupPromptOverlay) {
    els.sourceGroupPromptOverlay.hidden = true;
  }
  const resolve = state.sourceGroupPromptResolver;
  state.sourceGroupPromptResolver = null;
  if (resolve) {
    resolve(value);
  }
}

function chooseSourceGroup() {
  if (!els.sourceGroupPromptOverlay) {
    return Promise.resolve("");
  }
  if (state.sourceGroupPromptResolver) {
    closeSourceGroupPrompt("");
  }
  els.sourceGroupPromptOverlay.hidden = false;
  return new Promise((resolve) => {
    state.sourceGroupPromptResolver = resolve;
  });
}

function togglePdfSelection(hash, checked) {
  if (!hash) {
    return;
  }
  if (checked) {
    state.selectedPdfHashes.add(hash);
  } else {
    state.selectedPdfHashes.delete(hash);
  }
}

function visibleUntaggedRowCheckboxes() {
  if (!els.pdfsBody) {
    return [];
  }
  return Array.from(els.pdfsBody.querySelectorAll("input.pdf-row-select[data-pdf-select]"));
}

function selectAllUntaggedPdfs(checked) {
  const checkboxes = visibleUntaggedRowCheckboxes();
  for (const checkbox of checkboxes) {
    const hash = checkbox.dataset.pdfSelect || "";
    checkbox.checked = checked;
    togglePdfSelection(hash, checked);
    const row = checkbox.closest("tr");
    if (row) {
      row.classList.toggle("pdf-row-selected", checked);
    }
  }
  updatePdfBulkBar();
  syncPdfSelectAllState();
}

function clearPdfSelection() {
  state.selectedPdfHashes.clear();
  visibleUntaggedRowCheckboxes().forEach((checkbox) => {
    checkbox.checked = false;
    const row = checkbox.closest("tr");
    if (row) {
      row.classList.remove("pdf-row-selected");
    }
  });
  updatePdfBulkBar();
  syncPdfSelectAllState();
}

function syncPdfSelectAllState() {
  const headerCheckbox = els.pdfSelectAllCheckbox;
  if (!headerCheckbox) {
    return;
  }
  const checkboxes = visibleUntaggedRowCheckboxes();
  const total = checkboxes.length;
  const checked = checkboxes.filter((checkbox) => checkbox.checked).length;
  headerCheckbox.checked = total > 0 && checked === total;
  headerCheckbox.indeterminate = checked > 0 && checked < total;
}

function updatePdfBulkBar() {
  const bar = els.pdfBulkActionBar;
  const label = els.pdfBulkCountLabel;
  const count = state.selectedPdfHashes.size;
  if (bar) {
    bar.hidden = count === 0;
  }
  if (label) {
    label.textContent = `${count} selected`;
  }
  if (els.pdfBulkTagButton) {
    els.pdfBulkTagButton.disabled = count === 0;
  }
}

function syncPdfSelectionAfterRender() {
  const liveHashes = new Set(
    visibleUntaggedRowCheckboxes().map((checkbox) => checkbox.dataset.pdfSelect || "")
  );
  for (const hash of Array.from(state.selectedPdfHashes)) {
    if (!liveHashes.has(hash)) {
      state.selectedPdfHashes.delete(hash);
    }
  }
  syncPdfSelectAllState();
  updatePdfBulkBar();
}

async function applyBulkTagGroup() {
  const hashes = Array.from(state.selectedPdfHashes);
  if (!hashes.length) {
    return;
  }
  const sourceGroup = await chooseSourceGroup();
  if (!sourceGroup) {
    return;
  }
  let successCount = 0;
  const failures = [];
  for (const hash of hashes) {
    try {
      await requestJson(`/api/pdfs/${encodeURIComponent(hash)}/trust`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_group: sourceGroup }),
      });
      successCount += 1;
    } catch (error) {
      failures.push(`${hash.slice(0, 8)}: ${error.message}`);
    }
  }
  clearPdfSelection();
  const label = SOURCE_GROUP_LABELS[sourceGroup] || sourceGroup;
  const summary = `Tagged ${successCount} PDF${successCount === 1 ? "" : "s"} as ${label}.`;
  if (failures.length) {
    setStatus(els.uploadStatus, `${summary} ${failures.length} failed: ${failures.join("; ")}`, true);
  } else {
    setStatus(els.uploadStatus, summary);
  }
  await refreshPdfs();
}

function formatBrowserTimestamp(value) {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) {
    return text;
  }
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
  }).format(date);
}

function renderQualityCell(item) {
  const quality = item.quality;
  const trust = item.trust && typeof item.trust === "object" ? item.trust : {};
  const data = quality && typeof quality === "object" ? quality : {};
  const label = data.label || "review";
  const warningText = qualityWarnings(data.warnings).join(", ");
  const trustNotes = String(trust.notes || "").trim();
  const reviewedBy = String(trust.reviewed_by || "").trim();
  const reviewedAt = formatBrowserTimestamp(trust.reviewed_at);
  const sourceGroup = String(trust.source_group || "ungrouped");
  const reliabilityWeight = Number(trust.reliability_weight || sourceGroupWeight(sourceGroup));
  const metrics = [
    Number(data.chunk_count || 0) ? `${Number(data.chunk_count)} chunks` : "",
    Number(data.markdown_char_count || 0) ? `${Number(data.markdown_char_count)} chars` : "",
    Number(data.enrichment_markers || 0) ? `${Number(data.enrichment_markers)} enriched` : "",
  ].filter(Boolean);
  return `
    <span class="quality-badge quality-${escapeHtml(label)}">${escapeHtml(qualityTitle(label))}</span>
    ${sourceGroup === "ungrouped" ? '<span class="quality-badge quality-untagged">Untagged</span>' : ""}
    <span class="quality-detail">${escapeHtml(metrics.join(" | "))}</span>
    <span class="quality-detail">Trust: ${escapeHtml(trustTitle(trust.review_status))} | ${escapeHtml(sourceTypeTitle(trust.source_type))}</span>
    <span class="quality-detail">Group: ${escapeHtml(sourceGroupTitle(sourceGroup))} | weight ${escapeHtml(reliabilityWeight.toFixed(2))}</span>
    ${reviewedBy ? `<span class="quality-detail">Reviewed by: ${escapeHtml(reviewedBy)}${reviewedAt ? ` | ${escapeHtml(reviewedAt)}` : ""}</span>` : ""}
    <span class="quality-warning">${escapeHtml(warningText)}</span>
    ${trustNotes ? `<span class="quality-note">Note: ${escapeHtml(trustNotes)}</span>` : ""}
  `;
}

function renderPdfActions(item) {
  const trust = item.trust && typeof item.trust === "object" ? item.trust : {};
  const sourceGroup = String(trust.source_group || "ungrouped");
  const sourceHash = escapeHtml(item.hash || "");
  const warnings = Array.isArray(item.quality?.warnings) ? item.quality.warnings : [];
  const reprocessDisabled = item.can_download ? "" : " disabled";
  const reindexDisabled = warnings.includes("missing_markdown") ? " disabled" : "";
  const tagGroupButton = sourceGroup === "ungrouped"
    ? `<button type="button" data-pdf-action="tag-group" data-source-hash="${sourceHash}">Tag group</button>`
    : "";
  return `
    <div class="pdf-actions">
      ${tagGroupButton}
      <button type="button" data-pdf-action="approve" data-source-hash="${sourceHash}">Approve</button>
      <button type="button" data-pdf-action="stale" data-source-hash="${sourceHash}">Flag stale</button>
      <button type="button" data-pdf-action="reindex" data-source-hash="${sourceHash}" title="Rebuild this source's index without re-running ingestion"${reindexDisabled}>Re-index</button>
      <button type="button" data-pdf-action="reprocess" data-source-hash="${sourceHash}" title="Re-run ingestion and indexing"${reprocessDisabled}>Re-run</button>
      <button type="button" class="danger" data-pdf-action="delete" data-source-hash="${sourceHash}" title="Delete this PDF, its processed Markdown, assets, and index records">Delete</button>
    </div>
  `;
}

function renderPdfInterruptedBadge(item) {
  const interruptedAt = formatBrowserTimestamp(item.last_interrupted_at);
  if (!interruptedAt && item.status !== "interrupted") {
    return "";
  }
  const title = interruptedAt
    ? `Job interrupted ${interruptedAt}`
    : "Job interrupted";
  return `<span class="pdf-warning-badge" title="${escapeHtml(title)}">Interrupted</span>`;
}

function createPdfRow(item, options = {}) {
  const row = document.createElement("tr");
  const trust = item.trust && typeof item.trust === "object" ? item.trust : {};
  const isUntagged = String(trust.source_group || "ungrouped") === "ungrouped";
  const isFakeRow = Boolean(options.fake);
  const sourceHash = String(item.hash || "");
  row.classList.toggle("pdf-untagged-row", isUntagged);
  if (isFakeRow) {
    row.id = "walkthroughFakePdfRow";
    row.classList.add("walkthrough-fake-pdf-row");
  }
  const download = item.download_url
    ? `<a class="download-link" href="${escapeHtml(item.download_url)}">Download</a>`
    : escapeHtml(item.path_error || "");
  const selectCell = isUntagged && sourceHash && sourceHash !== WALKTHROUGH_FAKE_PDF_HASH
    ? `<td class="pdf-select-col"><input type="checkbox" class="pdf-row-select" data-pdf-select="${escapeHtml(sourceHash)}" title="Select this untagged PDF"${state.selectedPdfHashes.has(sourceHash) ? " checked" : ""} /></td>`
    : `<td class="pdf-select-col"></td>`;
  if (state.selectedPdfHashes.has(sourceHash) && isUntagged && !isFakeRow) {
    row.classList.add("pdf-row-selected");
  }
  row.innerHTML = `
    ${selectCell}
    <td>
      <div class="pdf-title-line">
        <strong>${escapeHtml(item.filename || item.hash)}</strong>
        ${renderPdfInterruptedBadge(item)}
      </div>
      <span class="pdf-hash">${escapeHtml(item.hash || "")}</span>
      ${renderPdfActions(item)}
    </td>
    <td>${escapeHtml(item.status || "")}</td>
    <td>${renderQualityCell(item)}</td>
    <td>${download}</td>
  `;
  return row;
}

async function refreshJobs() {
  try {
    const params = new URLSearchParams({
      offset: String(state.jobsOffset),
      limit: String(state.jobsLimit),
    });
    const data = await requestJson(`/api/jobs?${params}`);
    state.jobsTotal = data.total || 0;
    const wasActive = state.jobsActive;
    state.jobsActive = Number(data.active_count || 0) > 0;
    if (state.jobsOffset >= state.jobsTotal && state.jobsOffset > 0) {
      state.jobsOffset = Math.max(0, Math.floor((state.jobsTotal - 1) / state.jobsLimit) * state.jobsLimit);
      return refreshJobs();
    }
    els.jobsBody.innerHTML = "";
    for (const job of data.jobs || []) {
      const row = document.createElement("tr");
      const canCancel = ["queued", "running", "paused_for_queries"].includes(String(job.status || ""))
        && !job.cancel_requested;
      const cancelButton = canCancel
        ? `<button type="button" class="danger" data-job-action="cancel" data-job-id="${escapeHtml(job.id || "")}">Cancel</button>`
        : "";
      row.innerHTML = `
        <td>${escapeHtml(job.id.slice(0, 8))}</td>
        <td>${escapeHtml(job.status)}</td>
        <td>${escapeHtml(job.phase)}</td>
        <td>${escapeHtml((job.filenames || []).join(", "))}</td>
        <td>${escapeHtml(job.error || "")}</td>
        <td><div class="job-actions">${cancelButton}</div></td>
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
    if (wasActive && !state.jobsActive) {
      await refreshPdfs();
      await refreshHealth();
    }
    if (wasActive !== state.jobsActive) {
      scheduleJobsPolling();
    }
  } catch (error) {
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
  if (!window.confirm("Cancel this job?")) {
    return;
  }
  button.disabled = true;
  try {
    const job = await requestJson(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    });
    setStatus(els.uploadStatus, `Cancelled job ${String(job.id || jobId).slice(0, 8)}.`);
    await refreshJobs();
    await refreshPdfs();
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  } finally {
    button.disabled = false;
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
    renderPdfRows(data.pdfs || []);
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
  if (sourceHash === WALKTHROUGH_FAKE_PDF_HASH) {
    setStatus(els.uploadStatus, "This walkthrough row is a local preview and is removed after the step.");
    return;
  }
  const body = {};
  if (action === "approve") {
    body.review_status = "approved";
  } else if (action === "tag-group") {
    const sourceGroup = await chooseSourceGroup();
    if (!sourceGroup) {
      return;
    }
    body.source_group = sourceGroup;
  } else if (action === "stale") {
    body.review_status = "stale";
    body.notes = window.prompt("Why is this source stale?", "") || "";
  } else if (action === "reprocess") {
    if (!window.confirm("Re-run ingestion and indexing for this source?")) {
      return;
    }
  } else if (action === "reindex") {
    if (!window.confirm("Re-index this source without re-running ingestion?")) {
      return;
    }
  } else if (action === "delete") {
    if (!window.confirm("Delete this source, including its PDF, processed Markdown, assets, and index records?")) {
      return;
    }
  } else {
    return;
  }
  if (action === "approve" || action === "stale") {
    const reviewer = ensureReviewerName();
    if (!reviewer) {
      return;
    }
    body.reviewed_by = reviewer;
  }
  button.disabled = true;
  try {
    if (action === "reprocess") {
      const job = await requestJson(`/api/pdfs/${encodeURIComponent(sourceHash)}/reprocess`, {
        method: "POST",
      });
      setStatus(els.uploadStatus, `Queued re-run job ${String(job.id || "").slice(0, 8)}.`);
      await refreshJobs();
    } else if (action === "reindex") {
      const job = await requestJson(`/api/pdfs/${encodeURIComponent(sourceHash)}/reindex`, {
        method: "POST",
      });
      setStatus(els.uploadStatus, `Queued re-index job ${String(job.id || "").slice(0, 8)}.`);
      await refreshJobs();
    } else if (action === "delete") {
      const result = await requestJson(`/api/pdfs/${encodeURIComponent(sourceHash)}`, {
        method: "DELETE",
      });
      const deletedVectors = Number(result.vectors?.deleted || 0);
      setStatus(els.uploadStatus, `Deleted source ${sourceHash.slice(0, 8)} and ${deletedVectors} index record${deletedVectors === 1 ? "" : "s"}.`);
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
  const interval = state.jobsActive ? JOBS_ACTIVE_POLL_INTERVAL_MS : state.jobsPollIntervalMs;
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

function renderUploadGroupSelectors() {
  const files = Array.from(els.fileInput.files || []);
  els.uploadGroupsPanel.innerHTML = "";
  els.uploadGroupsPanel.hidden = files.length === 0;
  for (const file of files) {
    const row = document.createElement("label");
    row.className = "upload-group-row";
    const name = document.createElement("span");
    name.className = "upload-group-name";
    name.textContent = file.name || "unnamed.pdf";
    const select = document.createElement("select");
    select.className = "upload-group-select";
    select.dataset.uploadGroup = "true";
    select.innerHTML = `
      <option value="">Choose group</option>
      <option value="official">Official</option>
      <option value="student_research">Student Research</option>
      <option value="unofficial">Unofficial</option>
    `;
    row.append(name, select);
    els.uploadGroupsPanel.appendChild(row);
  }
}

function selectedUploadSourceGroups() {
  const selects = Array.from(els.uploadGroupsPanel.querySelectorAll("[data-upload-group='true']"));
  if (!selects.length) {
    return [];
  }
  const groups = [];
  for (const select of selects) {
    const value = parseSourceGroupInput(select.value);
    if (!value) {
      select.focus();
      setStatus(els.uploadStatus, "Choose a source group for each selected PDF.", true);
      return null;
    }
    groups.push(value);
  }
  return groups;
}

function setSelectedUploadFiles(files) {
  clearDuplicatePrompt();
  const { accepted, rejected } = pdfFilesFromList(files);
  if (!accepted.length) {
    setStatus(els.uploadStatus, "Drop one or more PDF files.", true);
    updateSelectedFilesLabel();
    renderUploadGroupSelectors();
    return;
  }

  const transfer = new DataTransfer();
  for (const file of accepted) {
    transfer.items.add(file);
  }
  els.fileInput.files = transfer.files;
  updateSelectedFilesLabel();
  renderUploadGroupSelectors();
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
  const sourceGroups = selectedUploadSourceGroups();
  if (sourceGroups === null) {
    return;
  }

  const body = new FormData();
  for (const [index, file] of files.entries()) {
    body.append("files", file);
    if (sourceGroups[index]) {
      body.append("source_groups", sourceGroups[index]);
    }
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
    renderUploadGroupSelectors();
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

function indexReliabilityLabel(item) {
  const sourceGroup = String(item.source_group || "");
  if (!sourceGroup) {
    return "";
  }
  const reliability = Number(item.reliability_modifier || sourceGroupWeight(sourceGroup));
  return `${sourceGroupTitle(sourceGroup)} x${reliability.toFixed(2)}`;
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
  const meta = [indexNodeLabel(item), indexScoreLabel(item), indexReliabilityLabel(item), pageRange, childCount]
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
      <td class="index-content-cell">
        <textarea class="content-edit" spellcheck="false"></textarea>
      </td>
      <td>
        <div class="row-actions">
          <button type="button" data-action="save">Save</button>
        </div>
      </td>
    `;
  row.querySelector("textarea").value = item.content || "";
  appendAssetPreviewGrid(row.querySelector(".index-content-cell"), item.assets, {
    className: "source-assets index-assets",
    itemClassName: "index-asset",
    fallbackAlt: "Extracted source image",
  });
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

function createAssetPreviewGrid(assets, options = {}) {
  const assetGrid = document.createElement("div");
  assetGrid.className = options.className || "source-assets";
  const itemClassName = options.itemClassName || "";
  const fallbackAlt = options.fallbackAlt || "Stored source image";
  const captionAction = options.captionAction || "Open image";
  for (const asset of Array.isArray(assets) ? assets : []) {
    if (!asset || !asset.url) {
      continue;
    }
    const link = document.createElement("a");
    link.className = ["source-asset", itemClassName].filter(Boolean).join(" ");
    link.href = asset.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    if (asset.description) {
      link.title = asset.description;
    }

    const image = document.createElement("img");
    image.src = asset.url;
    image.alt = asset.description || fallbackAlt;
    link.appendChild(image);

    const caption = document.createElement("span");
    caption.textContent = [asset.page_no ? `page ${asset.page_no}` : "", captionAction].filter(Boolean).join(" | ");
    link.appendChild(caption);
    assetGrid.appendChild(link);
  }
  return assetGrid;
}

function appendAssetPreviewGrid(container, assets, options = {}) {
  const assetGrid = createAssetPreviewGrid(assets, options);
  if (!assetGrid.childElementCount) {
    return false;
  }
  container.appendChild(assetGrid);
  return true;
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
    const reliability = source.kind === "local"
      ? `${sourceGroupTitle(source.source_group)} | weight ${Number(source.reliability_modifier || sourceGroupWeight(source.source_group)).toFixed(2)}`
      : "";
    item.innerHTML = `
      <strong>${escapeHtml(source.label || source.id || "")} ${escapeHtml(sourceTitle(source))}</strong>
      <span>${escapeHtml([sourceLocation(source), reliability, score].filter(Boolean).join(" | "))}</span>
      <span>${escapeHtml(source.snippet || "")}</span>
      <span class="source-links">${links.join("")}</span>
    `;
    const assets = Array.isArray(source.assets) ? source.assets : [];
    appendAssetPreviewGrid(item, assets);
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
    vector_score: source.vector_score,
    lexical_score: source.lexical_score,
    hybrid_score: source.hybrid_score,
    reliability_modifier: source.reliability_modifier,
    source_group: source.source_group || "ungrouped",
    location: [source.source_pdf_name, source.section_path, source.page_label].filter(Boolean).join(" :: "),
    snippet: source.snippet || "",
    assets: Array.isArray(source.assets)
      ? source.assets.map((asset) => ({
          asset_id: asset.asset_id || "",
          page_no: asset.page_no || 0,
          url: asset.url || "",
          description: asset.description || "",
          mime_type: asset.mime_type || "",
        }))
      : [],
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

function activateTab(tabTarget, options = {}) {
  const target = document.getElementById(tabTarget);
  const button = document.querySelector(`.tab[data-tab-target="${tabTarget}"]`);
  if (!target || !button) {
    return;
  }
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
  button.classList.add("active");
  target.classList.add("active");
  if (tabTarget === "index") {
    loadIndex();
  } else {
    abortIndexLoad();
  }
  if (tabTarget === "upload" && options.refreshUpload !== false) {
    refreshPdfs();
  }
}

function walkthroughFakePdfItem() {
  return {
    hash: WALKTHROUGH_FAKE_PDF_HASH,
    filename: "Example untagged source.pdf",
    status: "review",
    path_error: "walkthrough preview",
    trust: {
      review_status: "unreviewed",
      source_type: "unknown",
      source_group: "ungrouped",
      reliability_weight: 0.1,
    },
    quality: {
      label: "review",
      warnings: ["unreviewed_source"],
      chunk_count: 12,
      markdown_char_count: 18400,
      enrichment_markers: 2,
    },
  };
}

function removeWalkthroughFakePdf() {
  state.walkthroughFakePdfPinned = false;
  const row = document.getElementById("walkthroughFakePdfRow");
  if (row) {
    row.remove();
  }
  state.walkthroughFakePdfVisible = false;
}

function ensureWalkthroughFakePdf() {
  if (state.walkthroughFakePdfVisible && document.getElementById("walkthroughFakePdfRow")) {
    return false;
  }
  const row = createPdfRow(walkthroughFakePdfItem(), { fake: true });
  els.pdfsBody.prepend(row);
  state.walkthroughFakePdfVisible = true;
  return true;
}

function renderPdfRows(items) {
  const fragment = document.createDocumentFragment();
  state.walkthroughFakePdfVisible = false;
  if (state.walkthroughFakePdfPinned) {
    fragment.appendChild(createPdfRow(walkthroughFakePdfItem(), { fake: true }));
    state.walkthroughFakePdfVisible = true;
  }
  for (const item of items) {
    fragment.appendChild(createPdfRow(item));
  }
  els.pdfsBody.innerHTML = "";
  els.pdfsBody.appendChild(fragment);
  if (state.walkthroughFakePdfPinned) {
    const step = walkthroughSteps[state.walkthroughIndex];
    if (step) {
      window.requestAnimationFrame(() => highlightWalkthroughTarget(step.target));
    }
  }
  syncPdfSelectionAfterRender();
}

function clearWalkthroughHighlight() {
  document.querySelectorAll(".walkthrough-highlight").forEach((element) => {
    element.classList.remove("walkthrough-highlight");
  });
}

function highlightWalkthroughTarget(selector) {
  clearWalkthroughHighlight();
  const target = document.querySelector(selector);
  if (!target) {
    return;
  }
  const details = target.closest("details");
  if (details) {
    details.open = true;
  }
  target.classList.add("walkthrough-highlight");
  target.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
}

function renderWalkthroughStep() {
  const step = walkthroughSteps[state.walkthroughIndex];
  if (!step) {
    return;
  }
  const fakePdfStep = Boolean(step.fakePdf);
  state.walkthroughFakePdfPinned = fakePdfStep;
  if (!fakePdfStep) {
    removeWalkthroughFakePdf();
  }
  activateTab(step.tab, { refreshUpload: !fakePdfStep });
  if (fakePdfStep) {
    ensureWalkthroughFakePdf();
  }
  els.walkthroughOverlay.hidden = false;
  els.walkthroughStepLabel.textContent = `Step ${state.walkthroughIndex + 1} of ${walkthroughSteps.length}`;
  els.walkthroughTitle.textContent = step.title;
  els.walkthroughText.textContent = step.text;
  els.walkthroughPrevButton.disabled = state.walkthroughIndex === 0;
  els.walkthroughNextButton.textContent =
    state.walkthroughIndex === walkthroughSteps.length - 1 ? "Done" : "Next";
  window.requestAnimationFrame(() => highlightWalkthroughTarget(step.target));
  els.walkthroughNextButton.focus();
}

function startWalkthrough() {
  setCookie(TUTORIAL_SEEN_COOKIE, "1");
  state.walkthroughIndex = 0;
  renderWalkthroughStep();
}

function closeWalkthrough() {
  state.walkthroughIndex = -1;
  els.walkthroughOverlay.hidden = true;
  clearWalkthroughHighlight();
  removeWalkthroughFakePdf();
  if (state.pendingVersionPrompt) {
    showCachePrompt(state.pendingSiteVersion);
  }
}

function nextWalkthroughStep() {
  if (state.walkthroughIndex < 0) {
    return;
  }
  if (state.walkthroughIndex >= walkthroughSteps.length - 1) {
    closeWalkthrough();
    return;
  }
  state.walkthroughIndex += 1;
  renderWalkthroughStep();
}

function previousWalkthroughStep() {
  if (state.walkthroughIndex <= 0) {
    return;
  }
  state.walkthroughIndex -= 1;
  renderWalkthroughStep();
}

function maybeStartFirstVisitWalkthrough() {
  if (getCookie(TUTORIAL_SEEN_COOKIE) === "1") {
    return;
  }
  showWelcomeTutorialPrompt();
}

function showWelcomeTutorialPrompt() {
  els.welcomeTutorialOverlay.hidden = false;
  els.welcomeTutorialStartButton.focus();
}

function welcomeTutorialPromptOpen() {
  return !els.welcomeTutorialOverlay.hidden;
}

function closeWelcomeTutorialPrompt() {
  setCookie(TUTORIAL_SEEN_COOKIE, "1");
  els.welcomeTutorialOverlay.hidden = true;
  if (state.pendingVersionPrompt && state.walkthroughIndex < 0) {
    showCachePrompt(state.pendingSiteVersion);
  }
}

function acceptWelcomeTutorialPrompt() {
  closeWelcomeTutorialPrompt();
  startWalkthrough();
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

function showCachePrompt(version) {
  state.pendingSiteVersion = version;
  state.pendingVersionPrompt = false;
  els.cachePromptText.textContent =
    `A new app version is running (${shortSha(version)}). Empty your browser cache or use a hard reload so the latest interface files are loaded before continuing.`;
  els.cachePromptOverlay.hidden = false;
  els.cachePromptReloadButton.focus();
}

function closeCachePrompt({ acknowledgeVersion = true } = {}) {
  if (acknowledgeVersion && state.pendingSiteVersion) {
    setCookie(SITE_VERSION_COOKIE, state.pendingSiteVersion);
  }
  state.pendingSiteVersion = "";
  state.pendingVersionPrompt = false;
  els.cachePromptOverlay.hidden = true;
}

async function clearBrowserCaches() {
  if (!("caches" in window)) {
    return;
  }
  const names = await caches.keys();
  await Promise.all(names.map((name) => caches.delete(name)));
}

async function reloadAfterCacheClear() {
  const version = state.pendingSiteVersion;
  if (version) {
    setCookie(SITE_VERSION_COOKIE, version);
  }
  try {
    await clearBrowserCaches();
  } catch (_) {
    // Browser HTTP cache cannot be fully controlled from application JavaScript.
  }
  const url = new URL(window.location.href);
  url.searchParams.set("v", version || String(Date.now()));
  window.location.replace(url.toString());
}

document.querySelectorAll("[data-tab-target]").forEach((button) => {
  button.addEventListener("click", () => {
    activateTab(button.dataset.tabTarget);
    if (state.walkthroughIndex >= 0) {
      closeWalkthrough();
    }
  });
});

els.startGuideButton.addEventListener("click", startWalkthrough);
els.walkthroughPrevButton.addEventListener("click", previousWalkthroughStep);
els.walkthroughNextButton.addEventListener("click", nextWalkthroughStep);
els.walkthroughCloseButton.addEventListener("click", closeWalkthrough);
els.welcomeTutorialStartButton.addEventListener("click", acceptWelcomeTutorialPrompt);
els.welcomeTutorialSkipButton.addEventListener("click", closeWelcomeTutorialPrompt);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.walkthroughIndex >= 0) {
    closeWalkthrough();
    return;
  }
  if (event.key === "Escape" && welcomeTutorialPromptOpen()) {
    closeWelcomeTutorialPrompt();
    return;
  }
  if (event.key === "Escape" && !els.cachePromptOverlay.hidden) {
    closeCachePrompt();
    return;
  }
  if (event.key === "Escape" && els.sourceGroupPromptOverlay && !els.sourceGroupPromptOverlay.hidden) {
    closeSourceGroupPrompt("");
    return;
  }
  if (els.sourceGroupPromptOverlay && !els.sourceGroupPromptOverlay.hidden && (event.ctrlKey || event.metaKey)) {
    const HOTKEY_SOURCE_GROUPS = { "1": "official", "2": "student_research", "3": "unofficial" };
    const choice = HOTKEY_SOURCE_GROUPS[event.key];
    if (choice) {
      event.preventDefault();
      closeSourceGroupPrompt(parseSourceGroupInput(choice));
    }
  }
});
els.cachePromptReloadButton.addEventListener("click", reloadAfterCacheClear);
els.cachePromptDoneButton.addEventListener("click", () => closeCachePrompt());
els.sourceGroupPromptOverlay.addEventListener("click", (event) => {
  if (event.target === els.sourceGroupPromptOverlay) {
    closeSourceGroupPrompt("");
    return;
  }
  const button = event.target.closest("[data-source-group-choice]");
  if (!button) {
    return;
  }
  const sourceGroup = parseSourceGroupInput(button.dataset.sourceGroupChoice);
  closeSourceGroupPrompt(sourceGroup);
});
els.sourceGroupPromptCancelButton.addEventListener("click", () => closeSourceGroupPrompt(""));

els.updateButton.addEventListener("click", applyUpdate);
els.fileInput.addEventListener("change", () => {
  clearDuplicatePrompt();
  updateSelectedFilesLabel();
  renderUploadGroupSelectors();
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
els.reviewerNameInput.addEventListener("change", () => saveReviewerName(els.reviewerNameInput.value));
els.reviewerNameInput.addEventListener("blur", () => saveReviewerName(els.reviewerNameInput.value));
els.reviewerNameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    saveReviewerName(els.reviewerNameInput.value);
    els.reviewerNameInput.blur();
  }
});
els.pdfsBody.addEventListener("click", handlePdfAction);
els.pdfsBody.addEventListener("change", (event) => {
  const checkbox = event.target.closest("input.pdf-row-select[data-pdf-select]");
  if (!checkbox) {
    return;
  }
  const hash = checkbox.dataset.pdfSelect || "";
  togglePdfSelection(hash, checkbox.checked);
  const row = checkbox.closest("tr");
  if (row) {
    row.classList.toggle("pdf-row-selected", checkbox.checked);
  }
  updatePdfBulkBar();
  syncPdfSelectAllState();
});
if (els.pdfSelectAllCheckbox) {
  els.pdfSelectAllCheckbox.addEventListener("change", () => {
    selectAllUntaggedPdfs(els.pdfSelectAllCheckbox.checked);
  });
}
if (els.pdfBulkTagButton) {
  els.pdfBulkTagButton.addEventListener("click", applyBulkTagGroup);
}
if (els.pdfBulkClearButton) {
  els.pdfBulkClearButton.addEventListener("click", clearPdfSelection);
}
els.jobsBody.addEventListener("click", handleJobAction);
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

loadReviewerName();
loadChatState();
setChatSidebarCollapsed(state.chatSidebarCollapsed);
renderSavedChats();
renderActiveChat();
persistChatState();
maybeStartFirstVisitWalkthrough();
refreshHealth();
refreshUpdateStatus();
refreshJobs();
refreshPdfs();
scheduleHealthPolling();
scheduleUpdatePolling();
scheduleJobsPolling();
