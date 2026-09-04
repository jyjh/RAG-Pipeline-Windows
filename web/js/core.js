// Shared foundation: constants, state, DOM refs, fetch/API, toasts, dialogs.

import { syncPdfSelectAllState, togglePdfSelection, updatePdfBulkBar } from "./library.js";
import { renderOpsDashboard } from "./chat.js";
import { adminCard } from "./admin.js";
import { SITE_VERSION_COOKIE } from "./shell.js";

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
  pdfPageSize: "10",
  pdfTotal: 0,
  jobsOffset: 0,
  jobsLimit: 10,
  jobsPageSize: "10",
  jobsTotal: 0,
  jobSearch: "",
  jobsActive: false,
  activeTab: "chat",
  pdfsLoaded: false,
  jobsLoaded: false,
  uploadDataDirty: true,
  pdfsRenderedUrl: "",
  jobsRenderedUrl: "",
  pdfsFetchSeq: 0,
  jobsFetchSeq: 0,
  indexLoaded: false,
  indexDirty: true,
  indexRenderedUrl: "",
  chats: [],
  activeChatId: null,
  streamingChatId: null,
  chatAbortController: null,
  chatScrollFrame: 0,
  chatScrollForce: false,
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
  // Selection stash for the Force-upload retry (the file input is cleared
  // when the batch ends, even on a duplicate block).
  pendingDuplicateFiles: null,
  pendingDuplicateSourceGroups: null,
  pollFailureCount: 0,
  sourceGroupPromptResolver: null,
  selectedPdfHashes: new Set(),
  openJobLogIds: new Set(),
  walkthroughFakePdfPinned: false,
  walkthroughFakePdfVisible: false,
  walkthroughIndex: -1,
  pendingSiteVersion: "",
  pendingVersionPrompt: false,
  // id -> last seen status, used to detect job completions for toasts.
  jobWatch: new Map(),
  adminDashboardLoaded: false,
  adminKeysAuthorized: null,
  // Mirror of GET /api/admin/permission-sets for the keys + sets managers.
  adminPermSets: [],
  adminPermSetEditing: null,
  pdfGroupFilter: "all",
  pdfTrustFilter: "all",
  pdfStatusFilter: "all",
  // Category ("split databases") state. categoriesCache mirrors /api/categories;
  // chatSelectedCategories === null means "search all categories", an array is
  // an explicit subset (possibly empty = none, e.g. web-only answers).
  categoriesCache: [],
  categoriesLoaded: false,
  pdfCategoryFilter: "all",
  indexCategory: "general",
  chatSelectedCategories: null,
  chatSequenceKey: "",
  composerSettingsFrame: 0,
  appSidebarCollapsed: false,
  themePreference: "auto",
  pdfSort: "",
  chatSearch: "",
  reviewViewMode: "chunks",
};


const LIVE_RENDER_INTERVAL_MS = 200;

const STREAM_TAIL_HOLD_CHARS = 700;

const STREAM_TAIL_MAX_CHARS = 2200;

const MIN_SERVER_POLL_INTERVAL_MS = 2000;

const CHAT_AUTO_SCROLL_THRESHOLD = 120;

const CHAT_STORAGE_KEY = "rag.chatHistory.v1";

const CHAT_UI_STORAGE_KEY = "rag.chatUi.v1";
// API key for authenticated mutations. localStorage (not a cookie): it is a
// long-lived secret, and there is no reason to send it on every static GET.

const API_KEY_STORAGE_KEY = "rag.apiKey.v1";

const CHAT_HISTORY_LIMIT = 30;

const CHAT_MESSAGE_LIMIT = 120;

const REVIEWER_NAME_COOKIE = "rag_reviewer_name";

const DEBUG_MODE_COOKIE = "debug_mode";

const ANSWER_PRESET_STORAGE_KEY = "rag.answerPreset.v1";

const COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;
// Answer-mode presets: friendly names over the raw sampler knobs. "custom" is
// never applied — it only marks that the user edited a value by hand.

const ANSWER_PRESETS = {
  precise: { temperature: 0.2, max_k: 30, context_window: 8192, llm_num_predict: 4096, retrieval_min_score: 0.6, web_search_enabled: false },
  balanced: { temperature: 0.3, max_k: 40, context_window: 8192, llm_num_predict: 4096, retrieval_min_score: 0.5, web_search_enabled: true },
  deep: { temperature: 0.4, max_k: 80, context_window: 16384, llm_num_predict: 8192, retrieval_min_score: 0.35, web_search_enabled: true },
};
// Maps each assistant message DOM node to its streaming parts object, so
// citation clicks can resolve the matching source panel.

const getJsonCache = new Map();

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
  dropzoneButton: document.getElementById("dropzoneButton"),
  uploadStagingPanel: document.getElementById("uploadStagingPanel"),
  uploadStagingCount: document.getElementById("uploadStagingCount"),
  fileInput: document.getElementById("fileInput"),
  selectedFilesLabel: document.getElementById("selectedFilesLabel"),
  uploadGroupsPanel: document.getElementById("uploadGroupsPanel"),
  uploadCategorySelect: document.getElementById("uploadCategorySelect"),
  uploadButton: document.getElementById("uploadButton"),
  cancelUploadButton: document.getElementById("cancelUploadButton"),
  reindexButton: document.getElementById("reindexButton"),
  reingestButton: document.getElementById("reingestButton"),
  backupIndexButton: document.getElementById("backupIndexButton"),
  rebuildIndexButton: document.getElementById("rebuildIndexButton"),
  toggleRestorePanelButton: document.getElementById("toggleRestorePanelButton"),
  restoreIndexPanel: document.getElementById("restoreIndexPanel"),
  restoreBackupsList: document.getElementById("restoreBackupsList"),
  refreshBackupsButton: document.getElementById("refreshBackupsButton"),
  closeRestorePanelButton: document.getElementById("closeRestorePanelButton"),
  maintenanceStatus: document.getElementById("maintenanceStatus"),
  uploadStatus: document.getElementById("uploadStatus"),
  duplicatePrompt: document.getElementById("duplicatePrompt"),
  duplicatePromptText: document.getElementById("duplicatePromptText"),
  forceUploadButton: document.getElementById("forceUploadButton"),
  pdfSearchInput: document.getElementById("pdfSearchInput"),
  pdfSearchButton: document.getElementById("pdfSearchButton"),
  pdfGroupFilterSelect: document.getElementById("pdfGroupFilterSelect"),
  pdfTrustFilterSelect: document.getElementById("pdfTrustFilterSelect"),
  pdfStatusFilterSelect: document.getElementById("pdfStatusFilterSelect"),
  pdfCategoryFilterSelect: document.getElementById("pdfCategoryFilterSelect"),
  pdfAutoTagButton: document.getElementById("pdfAutoTagButton"),
  reviewerNameInput: document.getElementById("reviewerNameInput"),
  libraryStatus: document.getElementById("libraryStatus"),
  pdfPreviewOverlay: document.getElementById("pdfPreviewOverlay"),
  pdfPreviewTitle: document.getElementById("pdfPreviewTitle"),
  pdfPreviewFrame: document.getElementById("pdfPreviewFrame"),
  pdfPreviewFallback: document.getElementById("pdfPreviewFallback"),
  pdfPreviewDownloadLink: document.getElementById("pdfPreviewDownloadLink"),
  pdfPreviewCloseButton: document.getElementById("pdfPreviewCloseButton"),
  appSidebar: document.getElementById("appSidebar"),
  sidebarCollapseButton: document.getElementById("sidebarCollapseButton"),
  themeToggleButton: document.getElementById("themeToggleButton"),
  themeToggleLabel: document.getElementById("themeToggleLabel"),
  themeSelect: document.getElementById("themeSelect"),
  answerPresetSelect: document.getElementById("answerPresetSelect"),
  composerSettingsSummary: document.getElementById("composerSettingsSummary"),
  settingsButton: document.getElementById("settingsButton"),
  settingsOverlay: document.getElementById("settingsOverlay"),
  settingsCloseButton: document.getElementById("settingsCloseButton"),
  settingsStartTourButton: document.getElementById("settingsStartTourButton"),
  shortcutsOverlay: document.getElementById("shortcutsOverlay"),
  shortcutsCloseButton: document.getElementById("shortcutsCloseButton"),
  citationPopover: document.getElementById("citationPopover"),
  prevPdfPageButton: document.getElementById("prevPdfPageButton"),
  pdfPageLabel: document.getElementById("pdfPageLabel"),
  nextPdfPageButton: document.getElementById("nextPdfPageButton"),
  pdfPageSizeSelect: document.getElementById("pdfPageSizeSelect"),
  pdfsBody: document.getElementById("pdfsBody"),
  prevJobsPageButton: document.getElementById("prevJobsPageButton"),
  jobsPageLabel: document.getElementById("jobsPageLabel"),
  nextJobsPageButton: document.getElementById("nextJobsPageButton"),
  jobsPageSizeSelect: document.getElementById("jobsPageSizeSelect"),
  jobSearchInput: document.getElementById("jobSearchInput"),
  jobSearchButton: document.getElementById("jobSearchButton"),
  jobsBody: document.getElementById("jobsBody"),
  enableJobNotificationsButton: document.getElementById("enableJobNotificationsButton"),
  jobsStrip: document.getElementById("jobsStrip"),
  jobsStripText: document.getElementById("jobsStripText"),
  jobsStripViewButton: document.getElementById("jobsStripViewButton"),
  shutdownBanner: document.getElementById("shutdownBanner"),
  shutdownBannerText: document.getElementById("shutdownBannerText"),
  shutdownServerButton: document.getElementById("shutdownServerButton"),
  refreshAdminButton: document.getElementById("refreshAdminButton"),
  adminDashboard: document.getElementById("adminDashboard"),
  adminKeysStatus: document.getElementById("adminKeysStatus"),
  adminKeysHint: document.getElementById("adminKeysHint"),
  adminKeysBody: document.getElementById("adminKeysBody"),
  adminKeyLabelInput: document.getElementById("adminKeyLabelInput"),
  adminKeyRoleSelect: document.getElementById("adminKeyRoleSelect"),
  adminKeyPermSetSelect: document.getElementById("adminKeyPermSetSelect"),
  adminKeyExpiresInput: document.getElementById("adminKeyExpiresInput"),
  adminKeyRateInput: document.getElementById("adminKeyRateInput"),
  adminKeyCreateButton: document.getElementById("adminKeyCreateButton"),
  adminPermSetsStatus: document.getElementById("adminPermSetsStatus"),
  adminPermSetsHint: document.getElementById("adminPermSetsHint"),
  adminPermSetsBody: document.getElementById("adminPermSetsBody"),
  adminPermSetEditor: document.getElementById("adminPermSetEditor"),
  adminPermSetEditorSummary: document.getElementById("adminPermSetEditorSummary"),
  adminPermSetNameInput: document.getElementById("adminPermSetNameInput"),
  adminPermSetLabelInput: document.getElementById("adminPermSetLabelInput"),
  adminPermSetWriteCheck: document.getElementById("adminPermSetWriteCheck"),
  adminPermSetAdminCheck: document.getElementById("adminPermSetAdminCheck"),
  adminPermSetSaveButton: document.getElementById("adminPermSetSaveButton"),
  adminPermSetCancelButton: document.getElementById("adminPermSetCancelButton"),
  adminPermSetCatsBox: document.getElementById("adminPermSetCatsBox"),
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
  apiKeyPromptOverlay: document.getElementById("apiKeyPromptOverlay"),
  apiKeyPromptInput: document.getElementById("apiKeyPromptInput"),
  apiKeyPromptStatus: document.getElementById("apiKeyPromptStatus"),
  apiKeyPromptSaveButton: document.getElementById("apiKeyPromptSaveButton"),
  apiKeyPromptCancelButton: document.getElementById("apiKeyPromptCancelButton"),
  apiKeyInput: document.getElementById("apiKeyInput"),
  apiKeyStatus: document.getElementById("apiKeyStatus"),
  apiKeyClearButton: document.getElementById("apiKeyClearButton"),
  sourceGroupPromptOverlay: document.getElementById("sourceGroupPromptOverlay"),
  sourceGroupPromptCancelButton: document.getElementById("sourceGroupPromptCancelButton"),
  pdfSelectAllCheckbox: document.getElementById("pdfSelectAllCheckbox"),
  pdfBulkActionBar: document.getElementById("pdfBulkActionBar"),
  pdfBulkCountLabel: document.getElementById("pdfBulkCountLabel"),
  pdfBulkTagButton: document.getElementById("pdfBulkTagButton"),
  pdfBulkCategorySelect: document.getElementById("pdfBulkCategorySelect"),
  pdfBulkMoveButton: document.getElementById("pdfBulkMoveButton"),
  pdfBulkClearButton: document.getElementById("pdfBulkClearButton"),
  indexCategorySelect: document.getElementById("indexCategorySelect"),
  chatCategoryChips: document.getElementById("chatCategoryChips"),
  adminCategoriesStatus: document.getElementById("adminCategoriesStatus"),
  adminCategoryKeyInput: document.getElementById("adminCategoryKeyInput"),
  adminCategoryCreateButton: document.getElementById("adminCategoryCreateButton"),
  adminCategoriesRefreshButton: document.getElementById("adminCategoriesRefreshButton"),
  adminCategoriesBody: document.getElementById("adminCategoriesBody"),
  toastStack: document.getElementById("toastStack"),
  compactIndexButton: document.getElementById("compactIndexButton"),
  rebuildVectorIndexButton: document.getElementById("rebuildVectorIndexButton"),
  pdfBulkRerunButton: document.getElementById("pdfBulkRerunButton"),
  pdfBulkDeleteButton: document.getElementById("pdfBulkDeleteButton"),
  pdfPreviewModePdf: document.getElementById("pdfPreviewModePdf"),
  pdfPreviewModeText: document.getElementById("pdfPreviewModeText"),
  pdfPreviewText: document.getElementById("pdfPreviewText"),
  chatSearchInput: document.getElementById("chatSearchInput"),
};


function setStatus(element, text, isError = false) {
  element.textContent = text || "";
  element.classList.toggle("error", Boolean(isError));
}

// -- toasts -----------------------------------------------------------------
// Stacked transient notifications for action results. Inline .status lines stay
// for context that belongs to a specific panel; anything a user just DID (or
// that finished in the background) reports here so it is visible from any tab.


const TOAST_DEFAULT_TIMEOUT_MS = 6000;

const TOAST_MAX_VISIBLE = 6;


function showToast(text, { kind = "info", timeoutMs = null, onClick = null } = {}) {
  const stack = els.toastStack;
  if (!stack || !text) {
    return;
  }
  while (stack.children.length >= TOAST_MAX_VISIBLE) {
    stack.firstElementChild.remove();
  }
  const toast = document.createElement("div");
  toast.className = `toast toast-${kind}`;
  toast.setAttribute("role", kind === "error" ? "alert" : "status");
  const message = document.createElement("span");
  message.className = "toast-message";
  message.textContent = text;
  toast.appendChild(message);
  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "toast-close";
  closeButton.setAttribute("aria-label", "Dismiss notification");
  closeButton.textContent = "×";
  toast.appendChild(closeButton);
  const dismiss = () => {
    toast.classList.add("toast-leaving");
    setTimeout(() => toast.remove(), 180);
  };
  closeButton.addEventListener("click", dismiss);
  if (typeof onClick === "function") {
    toast.classList.add("toast-clickable");
    toast.addEventListener("click", (event) => {
      if (event.target === closeButton) {
        return;
      }
      onClick();
      dismiss();
    });
  }
  stack.appendChild(toast);
  const effectiveTimeout =
    timeoutMs === null ? (kind === "error" ? 0 : TOAST_DEFAULT_TIMEOUT_MS) : timeoutMs;
  if (effectiveTimeout > 0) {
    setTimeout(() => {
      if (toast.isConnected) {
        dismiss();
      }
    }, effectiveTimeout);
  }
}


function toastError(error) {
  showToast(error && error.message ? error.message : String(error), { kind: "error" });
}



function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}


function stableJson(value) {
  try {
    return JSON.stringify(value);
  } catch (_) {
    return String(value);
  }
}


function stableJsonHash(value) {
  // Compact change-detection key for row patching. The full stableJson(item)
  // string used to be stored in a data- attribute on every row; for index rows
  // it includes whole chunk contents, which inflates the DOM and costs an
  // O(payload) stringify per refresh. A length + double 32-bit FNV mix keeps
  // change detection at negligible collision risk.
  const text = stableJson(value);
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ code, 0x01000193) >>> 0;
    h2 = Math.imul(h2 + code, 0x85ebca6b) >>> 0;
  }
  return `${text.length}:${h1.toString(36)}:${h2.toString(36)}`;
}


async function errorFromResponse(response) {
  let detail = await response.text();
  return errorFromText(response.status, response.statusText, detail);
}


function errorFromText(status, statusText, text) {
  let detail = text;
  try {
    const parsed = JSON.parse(detail);
    detail = parsed.detail || detail;
  } catch (_) {
    // Keep the raw response text.
  }
  const message =
    typeof detail === "string"
      ? detail
      : detail.message || `${status} ${statusText}`;
  const error = new Error(message);
  error.status = status;
  error.detail = detail;
  return error;
}


async function requestJson(path, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const cacheable = method === "GET" && !options.body;
  const headers = new Headers(options.headers || {});
  applyApiKeyHeaders(headers, { method });
  if (cacheable) {
    const cached = getJsonCache.get(path);
    if (cached && cached.etag) {
      headers.set("If-None-Match", cached.etag);
    }
  }
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs || 30000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  // Honor a caller-supplied AbortSignal (e.g. "abort the superseded load").
  // Spreading `...options` BEFORE `signal` silently dropped it, so the
  // abort-previous-load machinery never aborted anything.
  let externalAborted = false;
  let response;
  const onExternalAbort = () => {
    externalAborted = true;
    controller.abort();
  };
  if (options.signal) {
    if (options.signal.aborted) {
      externalAborted = true;
      controller.abort();
    } else {
      options.signal.addEventListener("abort", onExternalAbort, { once: true });
    }
  }
  try {
    response = await fetch(path, { ...options, headers, signal: controller.signal });
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError") {
      // Distinguish a caller-initiated abort (rethrow so callers can ignore
      // it) from the internal timeout (surface as a timeout error).
      if (externalAborted) {
        throw err;
      }
      throw new Error(`Request to ${path} timed out after ${timeoutMs}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
    if (options.signal) {
      options.signal.removeEventListener("abort", onExternalAbort);
    }
  }
  if (response.status === 304 && cacheable) {
    const cached = getJsonCache.get(path);
    if (cached) {
      return { ...cached.data, notModified: true };
    }
    // The cache entry was evicted while the conditional request was in
    // flight (304 is not ok, and we have no body): refetch unconditionally.
    return requestJson(path, { ...options, headers: new Headers(options.headers) });
  }
  // A request failed auth: offer to (re-)enter the API key, then retry once.
  // Surfaced here so every requestJson caller benefits without per-call
  // handling. Sensitive GETs are gated server-side when auth is configured,
  // so they flow through the same prompt-and-retry path.
  if (response.status === 401 && !options.__apiKeyRetried) {
    const key = await promptForApiKey();
    if (key) {
      return requestJson(path, { ...options, __apiKeyRetried: true });
    }
  }
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  const data = await response.json();
  const etag = response.headers.get("ETag");
  if (cacheable && etag) {
    if (getJsonCache.size >= 100) {
      const oldest = getJsonCache.keys().next().value;
      getJsonCache.delete(oldest);
    }
    getJsonCache.set(path, { etag, data });
  }
  return data;
}


function isAbortError(error) {
  return error && error.name === "AbortError";
}

// Threshold above which a single file is uploaded via the chunked/resumable
// endpoint instead of a single multipart POST. Below this, the overhead of
// per-chunk requests isn't worth it; above it, resumability matters.

const CHUNKED_UPLOAD_THRESHOLD = 128 * 1024 * 1024; // 128 MiB

const CHUNKED_UPLOAD_CHUNK_SIZE = 16 * 1024 * 1024; // 16 MiB

// Server contract: upload_id must be a 32-char hex token (it becomes a path
// component under the staging dir, and anything else is rejected).

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


function isDebugMode() {
  const value = getCookie(DEBUG_MODE_COOKIE).trim().toLowerCase();
  return value === "1" || value === "true";
}


function setCookie(name, value, maxAgeSeconds = COOKIE_MAX_AGE_SECONDS) {
  const encodedName = encodeURIComponent(name);
  const encodedValue = encodeURIComponent(value);
  document.cookie = `${encodedName}=${encodedValue}; Max-Age=${maxAgeSeconds}; Path=/; SameSite=Lax`;
}

// The API key authenticates mutating requests (upload, reindex, delete, edit)
// when the deployment has keys configured. It is optional and inert on a fresh
// single-user deploy. Stored locally so the user enters it once; cleared from
// here to log out.

function getApiKey() {
  try {
    return localStorage.getItem(API_KEY_STORAGE_KEY) || "";
  } catch (_) {
    return "";
  }
}


function setApiKey(value) {
  const trimmed = String(value || "").trim();
  try {
    if (trimmed) {
      localStorage.setItem(API_KEY_STORAGE_KEY, trimmed);
    } else {
      localStorage.removeItem(API_KEY_STORAGE_KEY);
    }
  } catch (_) {
    // Private mode / disabled storage: fall back to carrying nothing; the
    // server will 401 and the user can re-enter the key in the prompt.
  }
  if (els.apiKeyInput) {
    els.apiKeyInput.value = trimmed;
  }
  if (els.apiKeyStatus) {
    els.apiKeyStatus.textContent = trimmed ? "API key saved" : "No API key set";
  }
  return trimmed;
}


function clearApiKey() {
  return setApiKey("");
}

// Apply the stored API key to a Headers object in place. Sent on ALL methods:
// sensitive GETs (/api/pdfs, /api/index, /api/jobs) are gated server-side when
// auth is configured, and fetch() can carry headers (unlike the <a>/<img>
// media routes, which intentionally stay open).

function applyApiKeyHeaders(headers, { method }) {
  const key = getApiKey();
  if (key) {
    headers.set("X-API-Token", key);
  }
  return headers;
}

// Object-flavored twin of applyApiKeyHeaders for requestJson's `headers`
// option. Single source of truth for the X-API-Token header.
function apiKeyHeaderObject() {
  const headers = {};
  const key = getApiKey();
  if (key) {
    headers["X-API-Token"] = key;
  }
  return headers;
}

// Media navigations (<a href> download/view, iframe src) cannot carry headers,
// and those routes are credential-gated for remote clients, so links carry the
// key as ?token= instead. The token goes before any #fragment (view links use
// "#page=N"). With no key stored (the normal local-operator case, where the
// server auto-authenticates loopback) URLs stay clean.
function mediaUrlWithToken(path) {
  const key = getApiKey();
  if (!key) {
    return path;
  }
  const [beforeHash, hash = ""] = String(path).split("#", 2);
  const separator = beforeHash.includes("?") ? "&" : "?";
  const tokenPart = `${separator}token=${encodeURIComponent(key)}`;
  return hash ? `${beforeHash}${tokenPart}#${hash}` : `${beforeHash}${tokenPart}`;
}

// Show the API-key prompt and resolve to the entered key (saved) or null
// (cancelled). Only one prompt is shown at a time; concurrent callers await the
// same in-flight promise so a burst of 401s produces a single dialog.

let apiKeyPromptInFlight = null;

function promptForApiKey() {
  if (apiKeyPromptInFlight) {
    return apiKeyPromptInFlight;
  }
  const overlay = els.apiKeyPromptOverlay;
  const input = els.apiKeyPromptInput;
  const status = els.apiKeyPromptStatus;
  if (!overlay || !input) {
    return Promise.resolve(null);
  }
  input.value = getApiKey();
  if (status) {
    status.textContent = "";
  }
  overlay.hidden = false;
  setTimeout(() => input.focus(), 0);
  apiKeyPromptInFlight = new Promise((resolve) => {
    const cleanup = () => {
      overlay.hidden = true;
      apiKeyPromptInFlight = null;
      els.apiKeyPromptSaveButton.removeEventListener("click", onSave);
      els.apiKeyPromptCancelButton.removeEventListener("click", onCancel);
      input.removeEventListener("keydown", onKey);
    };
    const onSave = () => {
      const value = String(input.value || "").trim();
      cleanup();
      setApiKey(value);
      resolve(value || null);
    };
    const onCancel = () => {
      cleanup();
      resolve(null);
    };
    const onKey = (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        onSave();
      } else if (event.key === "Escape") {
        event.preventDefault();
        onCancel();
      }
    };
    els.apiKeyPromptSaveButton.addEventListener("click", onSave);
    els.apiKeyPromptCancelButton.addEventListener("click", onCancel);
    input.addEventListener("keydown", onKey);
  });
  return apiKeyPromptInFlight;
}


function loadApiKey() {
  // Hydrate the settings field/status from storage without changing the value.
  const key = getApiKey();
  if (els.apiKeyInput) {
    els.apiKeyInput.value = key;
  }
  if (els.apiKeyStatus) {
    els.apiKeyStatus.textContent = key ? "API key saved" : "No API key set";
  }
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


const ANSWER_PRESET_LABELS = {
  precise: "Precise",
  balanced: "Balanced",
  deep: "Deep research",
  custom: "Custom",
};


function updatePageControls({ total, offset, limit, label, prevButton, nextButton }) {
  const start = total ? offset + 1 : 0;
  const end = Math.min(offset + limit, total);
  label.textContent = `${start}-${end} of ${total}`;
  prevButton.disabled = offset <= 0;
  nextButton.disabled = offset + limit >= total;
}


function patchTableRows(tbody, items, options) {
  const keyFor = options.keyFor;
  const createRow = options.createRow;
  const renderKeyFor = options.renderKeyFor || ((item) => stableJsonHash(item));
  const existing = new Map();
  Array.from(tbody.children).forEach((row) => {
    const key = row.dataset.patchKey;
    if (key) {
      existing.set(key, row);
    }
  });

  const currentRows = Array.from(tbody.children);
  const fragment = document.createDocumentFragment();
  const seen = new Set();
  let changed = currentRows.length !== items.length;
  items.forEach((item, index) => {
    const key = String(keyFor(item));
    const renderKey = renderKeyFor(item);
    let row = existing.get(key);
    if (!row || row.dataset.renderKey !== renderKey) {
      row = createRow(item);
      row.dataset.patchKey = key;
      row.dataset.renderKey = renderKey;
      changed = true;
    }
    if (row !== currentRows[index]) {
      changed = true;
    }
    seen.add(key);
    fragment.appendChild(row);
  });
  for (const key of existing.keys()) {
    if (!seen.has(key)) {
      changed = true;
    }
  }
  if (changed) {
    if (fragment.childElementCount > 80) {
      appendRowsProgressively(tbody, fragment, fragment.childElementCount);
    } else {
      tbody.replaceChildren(fragment);
    }
  }
  return changed;
}


function markUploadDataDirty() {
  state.uploadDataDirty = true;
  state.pdfsLoaded = false;
  state.jobsLoaded = false;
}


function markIndexDirty() {
  state.indexDirty = true;
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


function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const remS = s % 60;
  if (m < 60) return `${m}m ${remS}s`;
  const h = Math.floor(m / 60);
  const remM = m % 60;
  if (h < 48) return `${h}h ${remM}m`;
  const d = Math.floor(h / 24);
  const remH = h % 24;
  return `${d}d ${remH}h`;
}


function firstErrorText(job) {
  return String(job.error || "").split("\n")[0].slice(0, 160);
}

// -- global jobs strip -------------------------------------------------------
// Slim banner under the header that stays visible on every tab while any
// ingestion/indexing work is active, with a shortcut to the Documents tab.


const ZIP_IGNORED_PREFIXES = ["__macosx/"];

const ZIP_IGNORED_NAMES = new Set([".ds_store"]);

// Tracks which zip File each extracted PDF came from, so the source-group UI
// can render a zip-level "group for all" selector above its member PDFs.
// Files picked directly as PDFs are absent from this map (standalone rows).

const SOURCE_GROUP_OPTIONS_HTML = `
  <option value="">Choose group</option>
  <option value="official">Official</option>
  <option value="student_research">Student Research</option>
  <option value="unofficial">Unofficial</option>
`;


function confirmAction(title, body, confirmLabel = "Confirm", options = {}) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    overlay.hidden = false;
    const dialog = document.createElement("div");
    dialog.className = "modal-dialog";
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    const heading = document.createElement("h2");
    heading.textContent = title;
    dialog.appendChild(heading);
    if (body) {
      const paragraph = document.createElement("p");
      paragraph.textContent = body;
      dialog.appendChild(paragraph);
    }
    // requireText gates irreversible/corpus-wide actions: the confirm button
    // stays disabled until the operator types the exact token (e.g. REBUILD).
    const requireText = String(options.requireText || "").trim();
    let typedInput = null;
    if (requireText) {
      const label = document.createElement("label");
      label.className = "confirm-type-label";
      label.appendChild(document.createTextNode(`Type ${requireText} to confirm`));
      typedInput = document.createElement("input");
      typedInput.type = "text";
      typedInput.autocomplete = "off";
      typedInput.spellcheck = false;
      label.appendChild(typedInput);
      dialog.appendChild(label);
    }
    const actions = document.createElement("div");
    actions.className = "modal-actions";
    const confirmButton = document.createElement("button");
    confirmButton.type = "button";
    confirmButton.textContent = confirmLabel;
    if (options.danger || requireText) {
      confirmButton.className = "danger";
    }
    const cancelButton = document.createElement("button");
    cancelButton.type = "button";
    cancelButton.textContent = "Cancel";
    actions.appendChild(cancelButton);
    actions.appendChild(confirmButton);
    dialog.appendChild(actions);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    if (requireText) {
      confirmButton.disabled = true;
      typedInput.addEventListener("input", () => {
        confirmButton.disabled = typedInput.value.trim() !== requireText;
      });
    }

    const close = (result) => {
      overlay.remove();
      document.removeEventListener("keydown", onKey);
      resolve(result);
    };
    const onKey = (event) => {
      if (event.key === "Escape") {
        close(false);
      } else if (event.key === "Enter" && !confirmButton.disabled) {
        close(true);
      }
    };
    confirmButton.addEventListener("click", () => close(true));
    cancelButton.addEventListener("click", () => close(false));
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) {
        close(false);
      }
    });
    document.addEventListener("keydown", onKey);
    setTimeout(() => (typedInput || confirmButton).focus(), 0);
  });
}


function promptText(title, { body = "", placeholder = "", initialValue = "" } = {}) {
  // Styled replacement for window.prompt (used for chat renames and stale
  // notes). Resolves with the trimmed value, or null when cancelled.
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    overlay.hidden = false;
    const dialog = document.createElement("div");
    dialog.className = "modal-dialog";
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    const heading = document.createElement("h2");
    heading.textContent = title;
    dialog.appendChild(heading);
    if (body) {
      const paragraph = document.createElement("p");
      paragraph.textContent = body;
      dialog.appendChild(paragraph);
    }
    const input = document.createElement("input");
    input.type = "text";
    input.value = initialValue;
    input.placeholder = placeholder;
    input.autocomplete = "off";
    dialog.appendChild(input);
    const actions = document.createElement("div");
    actions.className = "modal-actions";
    const okButton = document.createElement("button");
    okButton.type = "button";
    okButton.textContent = "Save";
    const cancelButton = document.createElement("button");
    cancelButton.type = "button";
    cancelButton.textContent = "Cancel";
    actions.appendChild(cancelButton);
    actions.appendChild(okButton);
    dialog.appendChild(actions);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    const close = (result) => {
      overlay.remove();
      document.removeEventListener("keydown", onKey);
      resolve(result);
    };
    const submit = () => {
      const value = input.value.trim();
      close(value ? value : null);
    };
    const onKey = (event) => {
      if (event.key === "Escape") {
        close(null);
      } else if (event.key === "Enter") {
        event.preventDefault();
        submit();
      }
    };
    okButton.addEventListener("click", submit);
    cancelButton.addEventListener("click", () => close(null));
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) {
        close(null);
      }
    });
    document.addEventListener("keydown", onKey);
    setTimeout(() => {
      input.focus();
      input.select();
    }, 0);
  });
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


const indexRowContent = new WeakMap();

// -- in-place index row editing ----------------------------------------------
// Editing happens directly in the row: Edit swaps the read-only content cell
// for a textarea (with an optional Markdown preview toggle), Save/Cancel sit
// under it, and Esc cancels. No popup, no separate panel.


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


const CITATION_PATTERN = /\[([SW]\d+)\]/g;

// Walks the rendered answer and turns [S1]/[W1] tokens into clickable links
// pointing at the matching source in the Sources panel. Markdown is rendered
// server-side with HTML disabled, so the tokens always arrive as plain text
// inside text nodes; this post-process links them without touching the HTML.

function stopGeneration() {
  if (!state.streamingChatId || !state.chatAbortController) {
    return;
  }
  state.chatAbortController.abort();
}


async function copyTextToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard.", { kind: "success", timeoutMs: 2500 });
  } catch (_) {
    // Clipboard can be denied (non-secure context); fall back to a selection.
    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.appendChild(scratch);
    scratch.select();
    const copied = document.execCommand("copy");
    scratch.remove();
    if (copied) {
      showToast("Copied to clipboard.", { kind: "success", timeoutMs: 2500 });
    } else {
      showToast("Copy failed — select the text manually.", { kind: "error" });
    }
  }
}


function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value >= 1024 ** 3) {
    return `${(value / 1024 ** 3).toFixed(2)} GB`;
  }
  if (value >= 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (value >= 1024) {
    return `${(value / 1024).toFixed(0)} KB`;
  }
  return `${value} B`;
}


async function refreshOpsDashboard(options = {}) {
  const dashboard = els.adminDashboard;
  if (!dashboard) {
    return;
  }
  if (!options.force && state.adminDashboardLoaded) {
    return;
  }
  // /api/metrics is a sensitive GET: it is credential-gated when auth is
  // enabled, so attach the stored key (mutations get this automatically).
  const adminHeaders = apiKeyHeaderObject();
  const results = await Promise.allSettled([
    requestJson("/api/health", { headers: adminHeaders }),
    requestJson("/api/metrics", { headers: adminHeaders }),
    requestJson("/api/categories", { headers: adminHeaders }),
  ]);
  const health = results[0].status === "fulfilled" ? results[0].value : null;
  const metrics = results[1].status === "fulfilled" ? results[1].value : null;
  const categories = results[2].status === "fulfilled" ? results[2].value : null;
  if (!health && !metrics) {
    const reason = results[0].status === "rejected" ? results[0].reason : results[1].reason;
    dashboard.innerHTML = adminCard(
      "Dashboard unavailable",
      `<p class="admin-metric-error">${escapeHtml(reason && reason.message ? reason.message : String(reason))}</p>`,
    );
    return;
  }
  state.adminDashboardLoaded = true;
  renderOpsDashboard(health, metrics, categories);
}

// -- admin API key manager ---------------------------------------------------


function formatKeyTimestamp(value) {
  if (!value) {
    return "—";
  }
  return formatBrowserTimestamp(value);
}


function formatKeyUsage(usage) {
  const data = usage && typeof usage === "object" ? usage : {};
  const requests = Number(data.requests || 0);
  return `${requests.toLocaleString()} req`;
}


function showGeneratedKeyDialog(fullKey) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.hidden = false;
  const dialog = document.createElement("div");
  dialog.className = "modal-dialog";
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");
  const heading = document.createElement("h2");
  heading.textContent = "API key created";
  dialog.appendChild(heading);
  const warning = document.createElement("p");
  warning.textContent =
    "Copy this secret now — it is stored hashed and cannot be shown again after this dialog closes.";
  dialog.appendChild(warning);
  const row = document.createElement("div");
  row.className = "generated-key-row";
  const input = document.createElement("input");
  input.type = "text";
  input.readOnly = true;
  input.value = fullKey;
  input.setAttribute("aria-label", "New API key");
  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.textContent = "Copy";
  row.appendChild(input);
  row.appendChild(copyButton);
  dialog.appendChild(row);
  const actions = document.createElement("div");
  actions.className = "modal-actions";
  const doneButton = document.createElement("button");
  doneButton.type = "button";
  doneButton.textContent = "Done";
  actions.appendChild(doneButton);
  dialog.appendChild(actions);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  const close = () => {
    // Best effort: drop the plaintext from the input before removal.
    input.value = "";
    overlay.remove();
  };
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(fullKey);
      copyButton.textContent = "Copied";
    } catch (_) {
      // Clipboard can be denied (non-secure context); the input is selectable.
      input.select();
      copyButton.textContent = "Select + copy";
    }
    setTimeout(() => {
      copyButton.textContent = "Copy";
    }, 1500);
  });
  doneButton.addEventListener("click", close);
  setTimeout(() => {
    input.focus();
    input.select();
  }, 0);
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


const SHORTCUT_SEQUENCE_TIMEOUT_MS = 1200;


// Large pages (Library "All" is capped at 500 heavy rows) are appended in
// animation-frame batches so the main thread never stalls on one giant
// replaceChildren. Small pages keep the original single-swap path.
export function appendRowsProgressively(tbody, fragment, totalRows, batchSize = 80) {
  // Generation token: a re-render (poll tick, page change) swaps the tbody
  // contents while earlier batches are still queued; stale callbacks must
  // not append old rows into the fresh render.
  const generation = (Number(tbody.dataset.renderGeneration) || 0) + 1;
  tbody.dataset.renderGeneration = String(generation);
  if (totalRows <= batchSize) {
    tbody.replaceChildren(fragment);
    return;
  }
  const batches = [];
  let current = null;
  let count = 0;
  while (fragment.firstChild) {
    if (!current || count >= batchSize) {
      current = document.createDocumentFragment();
      batches.push(current);
      count = 0;
    }
    current.appendChild(fragment.firstChild);
    count += 1;
  }
  tbody.replaceChildren();
  let i = 0;
  const appendNext = () => {
    if (Number(tbody.dataset.renderGeneration) !== generation || !tbody.isConnected) {
      return;
    }
    const batch = batches[i++];
    if (!batch) {
      return;
    }
    tbody.appendChild(batch);
    requestAnimationFrame(appendNext);
  };
  requestAnimationFrame(appendNext);
}

export {
  ANSWER_PRESETS,
  ANSWER_PRESET_LABELS,
  ANSWER_PRESET_STORAGE_KEY,
  API_KEY_STORAGE_KEY,
  CHAT_AUTO_SCROLL_THRESHOLD,
  CHAT_HISTORY_LIMIT,
  CHAT_MESSAGE_LIMIT,
  CHAT_STORAGE_KEY,
  CHAT_UI_STORAGE_KEY,
  CHUNKED_UPLOAD_CHUNK_SIZE,
  CHUNKED_UPLOAD_THRESHOLD,
  CITATION_PATTERN,
  COOKIE_MAX_AGE_SECONDS,
  DEBUG_MODE_COOKIE,
  LIVE_RENDER_INTERVAL_MS,
  MIN_SERVER_POLL_INTERVAL_MS,
  REVIEWER_NAME_COOKIE,
  SHORTCUT_SEQUENCE_TIMEOUT_MS,
  SOURCE_GROUP_LABELS,
  SOURCE_GROUP_OPTIONS_HTML,
  SOURCE_GROUP_WEIGHTS,
  STREAM_TAIL_HOLD_CHARS,
  STREAM_TAIL_MAX_CHARS,
  TOAST_DEFAULT_TIMEOUT_MS,
  TOAST_MAX_VISIBLE,
  ZIP_IGNORED_NAMES,
  ZIP_IGNORED_PREFIXES,
  apiKeyPromptInFlight,
  apiKeyHeaderObject,
  applyApiKeyHeaders,
  blockBoundariesBefore,
  chooseSourceGroup,
  clearApiKey,
  clearBrowserCaches,
  closeSourceGroupPrompt,
  confirmAction,
  copyTextToClipboard,
  countUnescapedMarker,
  els,
  errorFromResponse,
  errorFromText,
  escapeHtml,
  firstErrorText,
  mediaUrlWithToken,
  formatBrowserTimestamp,
  formatBytes,
  formatEta,
  formatKeyTimestamp,
  formatKeyUsage,
  getApiKey,
  getCookie,
  getJsonCache,
  hasOpenFencedCodeBlock,
  indexChildCountLabel,
  indexNodeLabel,
  indexPageRange,
  indexParentRow,
  indexReliabilityLabel,
  indexRowContent,
  indexRowsForParent,
  indexScoreLabel,
  isAbortError,
  isDebugMode,
  isSafeMarkdownCommit,
  loadApiKey,
  markIndexDirty,
  markUploadDataDirty,
  newId,
  nowIso,
  numericSetting,
  parseSourceGroupInput,
  patchTableRows,
  promptForApiKey,
  promptText,
  readNdjson,
  refreshOpsDashboard,
  reloadAfterCacheClear,
  renderMarkdown,
  requestJson,
  selectAllUntaggedPdfs,
  setApiKey,
  setCookie,
  setStatus,
  showGeneratedKeyDialog,
  showToast,
  sleep,
  softBoundariesBefore,
  sourceGroupTitle,
  sourceGroupWeight,
  sourceTypeTitle,
  stableJson,
  stableJsonHash,
  state,
  stopGeneration,
  toastError,
  updatePageControls,
  visibleUntaggedRowCheckboxes,
};

// ---------------------------------------------------------------------------
// Overlay modality: focus trap + focus restore for every dialog in the app.
// Static overlays (hidden attribute) and dynamically appended ones (confirm,
// prompt, lightbox) both flow through the same observers, so individual open/
// close sites stay free of focus bookkeeping.
// ---------------------------------------------------------------------------
const overlayStack = [];
const overlayOpeners = new WeakMap();
const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

function isVisibleOverlay(el) {
  return !el.hidden && el.classList.contains("modal-overlay");
}

function overlayDialog(overlay) {
  return overlay.querySelector(".modal-dialog, .walkthrough-dialog") || overlay;
}

function overlayBecameVisible(overlay) {
  if (overlayOpeners.has(overlay)) {
    return;
  }
  overlayOpeners.set(overlay, document.activeElement);
  overlayStack.push(overlay);
  const dialog = overlayDialog(overlay);
  const target =
    dialog.querySelector("[data-autofocus]") ||
    dialog.querySelector(FOCUSABLE_SELECTOR) ||
    dialog;
  window.setTimeout(() => {
    if (typeof target.focus === "function") {
      target.focus();
    }
  }, 0);
}

function overlayBecameHidden(overlay) {
  if (!overlayOpeners.has(overlay)) {
    return;
  }
  const at = overlayStack.indexOf(overlay);
  if (at >= 0) {
    overlayStack.splice(at, 1);
  }
  const opener = overlayOpeners.get(overlay);
  overlayOpeners.delete(overlay);
  if (opener && document.contains(opener) && typeof opener.focus === "function") {
    opener.focus();
  }
}

function watchOverlay(overlay) {
  if (!overlay || overlay.__overlayWatched) {
    return;
  }
  overlay.__overlayWatched = true;
  if (isVisibleOverlay(overlay)) {
    overlayBecameVisible(overlay);
  }
  new MutationObserver(() => {
    if (isVisibleOverlay(overlay)) {
      overlayBecameVisible(overlay);
    } else {
      overlayBecameHidden(overlay);
    }
  }).observe(overlay, { attributes: true, attributeFilter: ["hidden", "class"] });
}

document.querySelectorAll(".modal-overlay, .walkthrough-overlay").forEach(watchOverlay);
new MutationObserver((mutations) => {
  for (const mutation of mutations) {
    for (const node of mutation.addedNodes) {
      if (node.nodeType === 1 && node.classList?.contains("modal-overlay")) {
        watchOverlay(node);
      }
    }
    for (const node of mutation.removedNodes) {
      if (node.nodeType === 1 && node.classList?.contains("modal-overlay")) {
        overlayBecameHidden(node);
      }
    }
  }
}).observe(document.body, { childList: true });

// Tab is confined to the topmost visible overlay.
document.addEventListener(
  "keydown",
  (event) => {
    if (event.key !== "Tab") {
      return;
    }
    const top = overlayStack[overlayStack.length - 1];
    if (!top || top.hidden) {
      return;
    }
    const dialog = overlayDialog(top);
    const focusables = [...dialog.querySelectorAll(FOCUSABLE_SELECTOR)].filter(
      (el) => el.offsetParent !== null || el === document.activeElement,
    );
    if (!focusables.length) {
      event.preventDefault();
      return;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (!dialog.contains(document.activeElement)) {
      event.preventDefault();
      first.focus();
    } else if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  },
  true,
);

// Inline anchored note prompt (replaces window.prompt for stale flags).
// Resolves with the trimmed note, or null when cancelled.
export function inlineNotePrompt(anchor, { title = "Notes for reviewers", placeholder = "" } = {}) {
  return new Promise((resolve) => {
    const popover = document.createElement("div");
    popover.className = "note-popover";
    const heading = document.createElement("strong");
    heading.textContent = title;
    const textarea = document.createElement("textarea");
    textarea.rows = 3;
    textarea.placeholder = placeholder;
    const actions = document.createElement("div");
    actions.className = "note-popover-actions";
    const finish = (value) => {
      popover.remove();
      document.removeEventListener("keydown", onKey, true);
      resolve(value);
    };
    const saveButton = document.createElement("button");
    saveButton.type = "button";
    saveButton.textContent = "Save";
    saveButton.addEventListener("click", () => finish(textarea.value.trim()));
    const cancelButton = document.createElement("button");
    cancelButton.type = "button";
    cancelButton.textContent = "Cancel";
    cancelButton.addEventListener("click", () => finish(null));
    actions.append(saveButton, cancelButton);
    popover.append(heading, textarea, actions);
    document.body.appendChild(popover);
    const rect = anchor.getBoundingClientRect();
    popover.style.left = `${Math.max(8, Math.min(rect.left + window.scrollX, window.scrollX + document.documentElement.clientWidth - popover.offsetWidth - 8))}px`;
    popover.style.top = `${Math.round(rect.bottom + window.scrollY + 6)}px`;
    const onKey = (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        finish(null);
      } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.stopPropagation();
        finish(textarea.value.trim());
      }
    };
    document.addEventListener("keydown", onKey, true);
    textarea.focus();
  });
}
