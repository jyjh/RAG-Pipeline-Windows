// App shell: sidebar, theme, shortcuts, settings, walkthrough, cache prompt.

import { SHORTCUT_SEQUENCE_TIMEOUT_MS, els, getCookie, requestJson, setCookie, state } from "./core.js";
import { refreshJobs, shortSha, updateComposerSettingsSummary } from "./status.js";
import { refreshPdfs, removeWalkthroughFakePdf } from "./library.js";
import { refreshCategories } from "./categories.js";
import { abortIndexLoad, loadIndex } from "./review.js";
import { createChat, renderWalkthroughStep } from "./chat.js";
import { refreshAdminPanel } from "./admin.js";
import { cancelAdminAutoRefresh } from "./admin.js";

const TUTORIAL_SEEN_COOKIE = "rag_tutorial_seen";

const SITE_VERSION_COOKIE = "rag_site_version";

const THEME_STORAGE_KEY = "rag.theme.v1";

const SIDEBAR_COLLAPSED_STORAGE_KEY = "rag.sidebarCollapsed.v1";

const WALKTHROUGH_FAKE_PDF_HASH = "__walkthrough_fake_untagged_pdf__";

const walkthroughSteps = [
  {
    tab: "upload",
    target: "#uploadDropZone",
    title: "Add source PDFs",
    text: "Drop PDFs here or use the file picker. Staged files ask for an index category and a per-file source group before uploading, and every PDF is queued as a background job that extracts Markdown, images, formulas, tables, and retrieval chunks.",
  },
  {
    tab: "upload",
    target: "#jobsTable",
    title: "Track ingestion and indexing",
    text: "The Jobs table shows queued, running, paused, completed, and failed work. Long jobs continue in the background while the published index remains available.",
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
  {
    tab: "library",
    target: "#walkthroughFakePdfRow [data-pdf-action='tag-group']",
    title: "Tag source reliability",
    text: "The Library tab lists every uploaded PDF. Untagged sources are highlighted at the top. Use Tag group to mark each source as Official, Student Research, or Unofficial before relying on retrieval ranking.",
    fakePdf: true,
  },
  {
    tab: "admin",
    target: "#adminDashboard",
    title: "Admin controls",
    text: "Administrators monitor the queue, index size, and LLM backend here, manage API keys, and run guarded maintenance actions such as backup, restore, and rebuild.",
  },
];


function markComposerSettingsCustom() {
  if (els.answerPresetSelect.value !== "custom") {
    els.answerPresetSelect.value = "custom";
  }
  updateComposerSettingsSummary();
}


function activateTab(tabTarget, options = {}) {
  const target = document.getElementById(tabTarget);
  const button = document.querySelector(`.tab[data-tab-target="${tabTarget}"]`);
  if (!target || !button) {
    return;
  }
  state.activeTab = tabTarget;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
  button.classList.add("active");
  target.classList.add("active");
  document.querySelectorAll(".sidebar-nav .tab").forEach((tab) => {
    if (tab.classList.contains("active")) {
      tab.setAttribute("aria-current", "page");
    } else {
      tab.removeAttribute("aria-current");
    }
  });
  if (tabTarget === "index") {
    if (options.forceRefresh || !state.indexLoaded || state.indexDirty) {
      loadIndex();
    }
  } else {
    abortIndexLoad();
  }
  if (tabTarget !== "admin") {
    cancelAdminAutoRefresh();
  }
  if (tabTarget === "upload" && options.refreshUpload !== false) {
    if (options.forceRefresh || state.uploadDataDirty || !state.jobsLoaded) {
      refreshJobs({ force: true });
    }
    state.uploadDataDirty = false;
  }
  if (tabTarget === "library" && options.refreshUpload !== false) {
    if (options.forceRefresh || state.uploadDataDirty || !state.pdfsLoaded) {
      refreshPdfs({ force: true });
    }
    state.uploadDataDirty = false;
  }
  if (tabTarget === "upload" || tabTarget === "chat") {
    // Keep category controls current where they are most visible; quiet on
    // failure so an offline registry never blocks tab use.
    refreshCategories();
  }
  if (tabTarget === "admin") {
    refreshAdminPanel({ force: options.forceRefresh === true });
  }
}

// -- admin tab ---------------------------------------------------------------

// The admin endpoints (key list, ops metrics) are credential-gated GETs; the
// default requestJson path never sends the key on GETs, so admin surfaces
// attach it explicitly.

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


function shortcutTargetIsEditable(event) {
  const target = event.target;
  if (!target) {
    return false;
  }
  const tag = String(target.tagName || "").toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
}


function handleGlobalShortcut(event) {
  if (event.ctrlKey || event.metaKey || event.altKey) {
    return;
  }
  const overlayOpen =
    (els.walkthroughOverlay && !els.walkthroughOverlay.hidden) ||
    (els.settingsOverlay && !els.settingsOverlay.hidden) ||
    (els.shortcutsOverlay && !els.shortcutsOverlay.hidden) ||
    (els.welcomeTutorialOverlay && !els.welcomeTutorialOverlay.hidden) ||
    (els.pdfPreviewOverlay && !els.pdfPreviewOverlay.hidden);
  if (overlayOpen) {
    return;
  }
  if (shortcutTargetIsEditable(event)) {
    state.chatSequenceKey = "";
    return;
  }
  if (event.key === "?") {
    event.preventDefault();
    els.shortcutsOverlay.hidden = !els.shortcutsOverlay.hidden;
    return;
  }
  if (state.chatSequenceKey === "g") {
    const targets = { d: "upload", l: "library", r: "index", a: "chat", m: "admin", g: "guide" };
    const target = targets[event.key.toLowerCase()];
    state.chatSequenceKey = "";
    if (target) {
      event.preventDefault();
      activateTab(target);
      return;
    }
  }
  if (event.key === "g" || event.key === "G") {
    state.chatSequenceKey = "g";
    setTimeout(() => {
      state.chatSequenceKey = "";
    }, SHORTCUT_SEQUENCE_TIMEOUT_MS);
    return;
  }
  state.chatSequenceKey = "";
  if (event.key === "/") {
    event.preventDefault();
    focusSearchForActiveTab();
    return;
  }
  if ((event.key === "n" || event.key === "N") && state.activeTab === "chat") {
    if (state.streamingChatId) {
      return;
    }
    event.preventDefault();
    createChat({ activate: true });
    els.questionInput.focus();
  }
}


function focusSearchForActiveTab() {
  const byTab = {
    upload: els.jobSearchInput,
    library: els.pdfSearchInput,
    index: els.searchInput,
    chat: els.questionInput,
  };
  const target = byTab[state.activeTab];
  if (target) {
    target.focus();
    target.select();
  }
}

// -- settings dialog -----------------------------------------------------------


function openSettingsDialog() {
  if (!els.settingsOverlay) {
    return;
  }
  if (els.themeSelect) {
    els.themeSelect.value = state.themePreference;
  }
  els.settingsOverlay.hidden = false;
}


function closeSettingsDialog() {
  if (els.settingsOverlay) {
    els.settingsOverlay.hidden = true;
  }
}

// -- theme (light / dark / system) ---------------------------------------------
// The resolved theme lands on <html data-theme> before first paint via a tiny
// inline script in index.html; this engine handles runtime changes.


function themeMediaDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}


function resolveTheme(pref) {
  if (pref === "dark" || pref === "light") {
    return pref;
  }
  return themeMediaDark() ? "dark" : "light";
}


function applyTheme() {
  document.documentElement.dataset.theme = resolveTheme(state.themePreference);
  syncThemeControls();
}


function setThemePreference(pref, { persist = true } = {}) {
  state.themePreference = pref === "dark" || pref === "light" ? pref : "auto";
  if (persist) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, state.themePreference);
    } catch (_) {
      // Private mode: theme still applies for this session.
    }
  }
  applyTheme();
}


function syncThemeControls() {
  const resolved = resolveTheme(state.themePreference);
  if (els.themeSelect) {
    els.themeSelect.value = state.themePreference;
  }
  if (els.themeToggleLabel) {
    // The toggle always offers the opposite of what is on screen.
    els.themeToggleLabel.textContent = resolved === "dark" ? "Light mode" : "Dark mode";
  }
  if (els.themeToggleButton) {
    els.themeToggleButton.title =
      resolved === "dark" ? "Switch to light mode" : "Switch to dark mode";
  }
}


function loadThemePreference() {
  let pref = "auto";
  try {
    pref = localStorage.getItem(THEME_STORAGE_KEY) || "auto";
  } catch (_) {
    pref = "auto";
  }
  state.themePreference = pref === "dark" || pref === "light" ? pref : "auto";
  applyTheme();
}

// -- sidebar collapse -------------------------------------------------------------


function setAppSidebarCollapsed(collapsed, { persist = true } = {}) {
  state.appSidebarCollapsed = Boolean(collapsed);
  document.body.classList.toggle("sidebar-collapsed", state.appSidebarCollapsed);
  if (els.sidebarCollapseButton) {
    els.sidebarCollapseButton.title = state.appSidebarCollapsed
      ? "Expand sidebar"
      : "Collapse sidebar";
  }
  if (persist) {
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_STORAGE_KEY, state.appSidebarCollapsed ? "1" : "0");
    } catch (_) {
      // Ignore storage failures; collapse is a visual preference only.
    }
  }
}


function loadAppSidebarCollapsed() {
  let collapsed = false;
  try {
    collapsed = localStorage.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY) === "1";
  } catch (_) {
    collapsed = false;
  }
  setAppSidebarCollapsed(collapsed, { persist: false });
}

export {
  SIDEBAR_COLLAPSED_STORAGE_KEY,
  SITE_VERSION_COOKIE,
  THEME_STORAGE_KEY,
  TUTORIAL_SEEN_COOKIE,
  WALKTHROUGH_FAKE_PDF_HASH,
  acceptWelcomeTutorialPrompt,
  activateTab,
  applyTheme,
  clearWalkthroughHighlight,
  closeCachePrompt,
  closeSettingsDialog,
  closeWalkthrough,
  closeWelcomeTutorialPrompt,
  focusSearchForActiveTab,
  handleGlobalShortcut,
  highlightWalkthroughTarget,
  loadAppSidebarCollapsed,
  loadThemePreference,
  markComposerSettingsCustom,
  maybeStartFirstVisitWalkthrough,
  nextWalkthroughStep,
  openSettingsDialog,
  previousWalkthroughStep,
  resolveTheme,
  setAppSidebarCollapsed,
  setThemePreference,
  shortcutTargetIsEditable,
  showCachePrompt,
  showWelcomeTutorialPrompt,
  startWalkthrough,
  syncThemeControls,
  themeMediaDark,
  walkthroughSteps,
  welcomeTutorialPromptOpen,
};

// Arrow-key navigation inside the sidebar nav: Up/Down move focus between
// the view buttons without leaving the keyboard trail.
export function handleSidebarKeydown(event) {
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") {
    return;
  }
  const tabs = [...document.querySelectorAll(".sidebar-nav .tab")];
  const current = tabs.indexOf(document.activeElement);
  if (current < 0) {
    return;
  }
  event.preventDefault();
  const next = event.key === "ArrowDown" ? current + 1 : current - 1;
  const target = tabs[(next + tabs.length) % tabs.length];
  target.focus();
}
