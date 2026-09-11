// App entry: wires modules to the DOM. All feature logic lives in web/js/*.
// Modules are ES modules (deferred), so the DOM is parsed before this runs.

import { ANSWER_PRESETS, clearApiKey, closeSourceGroupPrompt, els, loadApiKey, parseSourceGroupInput, reloadAfterCacheClear, selectAllUntaggedPdfs, setApiKey, state, stopGeneration, CHAT_UI_STORAGE_KEY } from "./js/core.js";
import { applyUpdate, handleJobAction, handleVisibilityChange, refreshHealth, refreshJobs, refreshUpdateStatus, scheduleHealthPolling, scheduleJobsPolling, scheduleUpdatePolling, updateComposerSettingsSummary } from "./js/status.js";
import { clearStagedUploadFiles, clearUploadDrag, enqueueReindex, fileToGroupSelect, handleUploadDrag, handleUploadDrop, renderUploadGroupSelectors, setSelectedUploadFiles, updateSelectedFilesLabel, uploadFiles } from "./js/upload.js";
import { applyBulkTagGroup, clearPdfSelection, closePdfPreview, handlePdfAction, loadReviewerName, refreshPdfs, runAutoTagSweep, saveReviewerName, syncPdfSelectAllState, togglePdfSelection, updatePdfBulkBar, handleLibrarySortClick, bulkDeleteSelected, bulkRerunSelected, setPdfPreviewMode } from "./js/library.js";
import { bulkMoveSelectedToCategory, createCategoryFromAdmin, deleteCategoryFromAdmin, refreshCategories, updateCategoryWeight } from "./js/categories.js";
import { endInlineEdit, handleIndexAction, loadIndex, runIndexVectorSearch, handleDocumentListClick, refreshDocumentList, setReviewViewMode, reviewViewMode } from "./js/review.js";
import { applyAnswerPreset, assistantMessageParts, cancelInlineMessageEdit, createChat, focusSourceForCitation, hideCitationPopover, loadChatState, persistChatState, renderActiveChat, renderSavedChats, restoreAnswerPreset, sendQuestion, setChatSidebarCollapsed, showCitationPopover } from "./js/chat.js";
import { createAdminApiKey, enqueueBackup, enqueueRebuild, enqueueReingest, handleAdminKeyAction, handleAdminPermSetAction, handleBackupAction, loadIndexBackups, refreshAdminPanel, resetPermSetEditor, saveAdminPermSet, toggleRestorePanel, refreshUpdatePanel, enqueueCompact, enqueueRebuildVectorIndex, shutdownServer } from "./js/admin.js";
import { acceptWelcomeTutorialPrompt, activateTab, applyTheme, closeCachePrompt, closeSettingsDialog, closeWalkthrough, closeWelcomeTutorialPrompt, handleGlobalShortcut, loadAppSidebarCollapsed, loadThemePreference, markComposerSettingsCustom, maybeStartFirstVisitWalkthrough, nextWalkthroughStep, openSettingsDialog, previousWalkthroughStep, resolveTheme, setAppSidebarCollapsed, setThemePreference, startWalkthrough, welcomeTutorialPromptOpen, handleSidebarKeydown } from "./js/shell.js";
import { closeTemplatesDialog, initUsabilityHelpers } from "./js/usability.js";

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
  if (event.key === "Escape" && els.shortcutsOverlay && !els.shortcutsOverlay.hidden) {
    els.shortcutsOverlay.hidden = true;
    return;
  }
  if (event.key === "Escape" && els.settingsOverlay && !els.settingsOverlay.hidden) {
    els.settingsOverlay.hidden = true;
    return;
  }
  if (event.key === "Escape" && els.pdfPreviewOverlay && !els.pdfPreviewOverlay.hidden) {
    closePdfPreview();
    return;
  }
  if (event.key === "Escape" && els.citationPopover && !els.citationPopover.hidden) {
    hideCitationPopover();
    return;
  }
  if (event.key === "Escape" && els.templatesOverlay && !els.templatesOverlay.hidden) {
    closeTemplatesDialog();
    return;
  }
  // An in-flight inline edit (review row or chat bubble) cancels on Esc.
  if (event.key === "Escape") {
    const editingRow = document.querySelector("#indexBody tr[data-editing='true']");
    if (editingRow) {
      endInlineEdit(editingRow);
      return;
    }
    const editingMessage = document.querySelector("#chatMessages .message[data-editing='true']");
    if (editingMessage) {
      cancelInlineMessageEdit(editingMessage);
      return;
    }
  }
  if (els.sourceGroupPromptOverlay && !els.sourceGroupPromptOverlay.hidden && (event.ctrlKey || event.metaKey)) {
    const HOTKEY_SOURCE_GROUPS = { "1": "official", "2": "student_research", "3": "unofficial" };
    const choice = HOTKEY_SOURCE_GROUPS[event.key];
    if (choice) {
      event.preventDefault();
      closeSourceGroupPrompt(parseSourceGroupInput(choice));
    }
    return;
  }
  handleGlobalShortcut(event);
});

// -- global keyboard shortcuts ------------------------------------------------
// "?", "/", "n", and two-key "g <tab>" sequences. Typing contexts (inputs,
// textareas, selects, contenteditable) are exempt so shortcuts never swallow
// user text; plain dialog keys above already returned before this runs.


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


els.updateButton.addEventListener("click", () => activateTab("admin"));

els.fileInput.addEventListener("change", () => {
  setSelectedUploadFiles(els.fileInput.files);
});

// The visually-hidden file input is driven by the big dropzone button.
if (els.dropzoneButton) {
  els.dropzoneButton.addEventListener("click", () => els.fileInput.click());
}

els.uploadCategorySelect.addEventListener("change", updateSelectedFilesLabel);

els.uploadDropZone.addEventListener("dragenter", handleUploadDrag);

els.uploadDropZone.addEventListener("dragover", handleUploadDrag);

els.uploadDropZone.addEventListener("dragleave", clearUploadDrag);

els.uploadDropZone.addEventListener("drop", handleUploadDrop);

els.uploadButton.addEventListener("click", () => uploadFiles());

if (els.cancelUploadButton) {
  els.cancelUploadButton.addEventListener("click", clearStagedUploadFiles);
}

els.forceUploadButton.addEventListener("click", () => {
  if (!state.pendingForceUploadToken) {
    return;
  }
  // uploadFiles clears the file input when the batch ends (even on a
  // duplicate block); restore the stashed selection -- and the source-group
  // picks -- before retrying, otherwise Force aborts with an empty input.
  if (state.pendingDuplicateFiles?.length) {
    const transfer = new DataTransfer();
    for (const file of state.pendingDuplicateFiles) {
      transfer.items.add(file);
    }
    els.fileInput.files = transfer.files;
    updateSelectedFilesLabel();
    renderUploadGroupSelectors();
    const groups = state.pendingDuplicateSourceGroups || [];
    state.pendingDuplicateFiles.forEach((file, index) => {
      const select = fileToGroupSelect.get(file);
      const group = groups[index];
      if (select && group && select.querySelector(`option[value="${group}"]`)) {
        select.value = group;
      }
    });
  }
  uploadFiles(true, state.pendingForceUploadToken);
});

els.reindexButton.addEventListener("click", enqueueReindex);

if (els.reingestButton) {
  els.reingestButton.addEventListener("click", enqueueReingest);
}

if (els.backupIndexButton) {
  els.backupIndexButton.addEventListener("click", enqueueBackup);
}

if (els.rebuildIndexButton) {
  els.rebuildIndexButton.addEventListener("click", enqueueRebuild);
}

if (els.toggleRestorePanelButton) {
  els.toggleRestorePanelButton.addEventListener("click", () => toggleRestorePanel());
}

if (els.closeRestorePanelButton) {
  els.closeRestorePanelButton.addEventListener("click", () => toggleRestorePanel(false));
}

if (els.refreshBackupsButton) {
  els.refreshBackupsButton.addEventListener("click", loadIndexBackups);
}

if (els.restoreBackupsList) {
  els.restoreBackupsList.addEventListener("click", handleBackupAction);
}

els.pdfSearchButton.addEventListener("click", () => {
  state.pdfSearch = els.pdfSearchInput.value.trim();
  state.pdfOffset = 0;
  refreshPdfs({ force: true });
});

els.pdfSearchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    state.pdfSearch = els.pdfSearchInput.value.trim();
    state.pdfOffset = 0;
    refreshPdfs({ force: true });
  }
});

if (els.pdfGroupFilterSelect) {
  els.pdfGroupFilterSelect.addEventListener("change", () => {
    state.pdfGroupFilter = els.pdfGroupFilterSelect.value || "all";
    state.pdfOffset = 0;
    refreshPdfs({ force: true });
  });
}

if (els.pdfTrustFilterSelect) {
  els.pdfTrustFilterSelect.addEventListener("change", () => {
    state.pdfTrustFilter = els.pdfTrustFilterSelect.value || "all";
    state.pdfOffset = 0;
    refreshPdfs({ force: true });
  });
}

if (els.pdfStatusFilterSelect) {
  els.pdfStatusFilterSelect.addEventListener("change", () => {
    state.pdfStatusFilter = els.pdfStatusFilterSelect.value || "all";
    state.pdfOffset = 0;
    refreshPdfs({ force: true });
  });
}

if (els.pdfCategoryFilterSelect) {
  els.pdfCategoryFilterSelect.addEventListener("change", () => {
    state.pdfCategoryFilter = els.pdfCategoryFilterSelect.value || "all";
    state.pdfOffset = 0;
    refreshPdfs({ force: true });
  });
}

if (els.indexCategorySelect) {
  els.indexCategorySelect.addEventListener("change", () => {
    state.indexCategory = els.indexCategorySelect.value || "general";
    state.offset = 0;
    loadIndex();
  });
}

if (els.pdfBulkMoveButton) {
  els.pdfBulkMoveButton.addEventListener("click", bulkMoveSelectedToCategory);
}

if (els.adminCategoryCreateButton) {
  els.adminCategoryCreateButton.addEventListener("click", createCategoryFromAdmin);
}

if (els.adminCategoriesRefreshButton) {
  els.adminCategoriesRefreshButton.addEventListener("click", () => refreshCategories());
}

if (els.adminCategoriesBody) {
  els.adminCategoriesBody.addEventListener("click", (event) => {
    const button = event.target.closest("[data-category-action]");
    if (!button) return;
    if (button.dataset.categoryAction === "delete") {
      deleteCategoryFromAdmin(button.dataset.categoryKey || "");
    }
  });
  els.adminCategoriesBody.addEventListener("change", (event) => {
    const input = event.target.closest("[data-category-weight-key]");
    if (input) updateCategoryWeight(input.dataset.categoryWeightKey || "", input.value);
  });
}

if (els.pdfPreviewCloseButton) {
  els.pdfPreviewCloseButton.addEventListener("click", closePdfPreview);
}

if (els.pdfPreviewOverlay) {
  els.pdfPreviewOverlay.addEventListener("click", (event) => {
    if (event.target === els.pdfPreviewOverlay) {
      closePdfPreview();
    }
  });
}

if (els.sidebarCollapseButton) {
  els.sidebarCollapseButton.addEventListener("click", () => {
    setAppSidebarCollapsed(!state.appSidebarCollapsed);
  });
}

if (els.themeToggleButton) {
  els.themeToggleButton.addEventListener("click", () => {
    setThemePreference(resolveTheme(state.themePreference) === "dark" ? "light" : "dark");
  });
}

if (els.themeSelect) {
  els.themeSelect.addEventListener("change", () => {
    setThemePreference(els.themeSelect.value);
  });
}
// Follow live OS theme changes while on "System".

if (window.matchMedia) {
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  const onSchemeChange = () => {
    if (state.themePreference === "auto") {
      applyTheme();
    }
  };
  if (darkQuery.addEventListener) {
    darkQuery.addEventListener("change", onSchemeChange);
  } else if (darkQuery.addListener) {
    darkQuery.addListener(onSchemeChange);
  }
}

if (els.settingsButton) {
  els.settingsButton.addEventListener("click", openSettingsDialog);
}

if (els.settingsCloseButton) {
  els.settingsCloseButton.addEventListener("click", closeSettingsDialog);
}

if (els.settingsOverlay) {
  els.settingsOverlay.addEventListener("click", (event) => {
    if (event.target === els.settingsOverlay) {
      closeSettingsDialog();
    }
  });
}

if (els.settingsStartTourButton) {
  els.settingsStartTourButton.addEventListener("click", () => {
    closeSettingsDialog();
    startWalkthrough();
  });
}

if (els.shortcutsCloseButton) {
  els.shortcutsCloseButton.addEventListener("click", () => {
    els.shortcutsOverlay.hidden = true;
  });
}

if (els.answerPresetSelect) {
  els.answerPresetSelect.addEventListener("change", () => {
    const presetId = els.answerPresetSelect.value;
    if (ANSWER_PRESETS[presetId]) {
      applyAnswerPreset(presetId);
    } else {
      updateComposerSettingsSummary();
    }
  });
}

for (const samplerInput of [
  els.temperatureInput,
  els.maxKInput,
  els.contextWindowInput,
  els.maxOutputInput,
  els.relevanceFloorInput,
  els.webSearchInput,
]) {
  if (samplerInput) {
    samplerInput.addEventListener("change", markComposerSettingsCustom);
  }
}
// "Find in Review" on a chat source jumps to the Review tab pre-filtered to
// that document, closing the answer-to-evidence loop inside the app.
els.chatMessages.addEventListener("click", (event) => {
  const link = event.target.closest("a.review-link");
  if (!link) {
    return;
  }
  event.preventDefault();
  const target = decodeURIComponent(link.dataset.reviewSearch || "");
  // The search toolbar only exists in chunks mode; Documents mode would
  // swallow the jump silently.
  if (reviewViewMode() !== "chunks") {
    setReviewViewMode("chunks");
  }
  activateTab("index");
  els.searchInput.value = target;
  els.searchButton.click();
});

// Citation hover previews: one delegated pair covers every rendered answer.

els.chatMessages.addEventListener("mouseover", (event) => {
  const link = event.target.closest(".citation-link");
  if (!link) {
    return;
  }
  const message = link.closest(".assistant-message");
  const parts = message ? assistantMessageParts.get(message) : null;
  showCitationPopover(link, parts);
});

els.chatMessages.addEventListener("mouseout", (event) => {
  if (event.target.closest(".citation-link")) {
    hideCitationPopover();
  }
});

if (els.citationPopover) {
  els.citationPopover.addEventListener("mouseleave", hideCitationPopover);
}

els.reviewerNameInput.addEventListener("change", () => saveReviewerName(els.reviewerNameInput.value));

els.reviewerNameInput.addEventListener("blur", () => saveReviewerName(els.reviewerNameInput.value));

els.reviewerNameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    saveReviewerName(els.reviewerNameInput.value);
    els.reviewerNameInput.blur();
  }
});

if (els.apiKeyInput) {
  const saveApiKeyField = () => {
    setApiKey(els.apiKeyInput.value);
  };
  els.apiKeyInput.addEventListener("change", saveApiKeyField);
  els.apiKeyInput.addEventListener("blur", saveApiKeyField);
  els.apiKeyInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      saveApiKeyField();
      els.apiKeyInput.blur();
    }
  });
}

if (els.apiKeyClearButton) {
  els.apiKeyClearButton.addEventListener("click", () => {
    clearApiKey();
  });
}

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


if (els.pdfAutoTagButton) {
  els.pdfAutoTagButton.addEventListener("click", runAutoTagSweep);
}

if (els.pdfBulkClearButton) {
  els.pdfBulkClearButton.addEventListener("click", clearPdfSelection);
}

els.jobsBody.addEventListener("click", handleJobAction);

els.jobsStripViewButton.addEventListener("click", () => activateTab("upload"));

if (els.enableJobNotificationsButton) {
  const syncNotificationButton = () => {
    const supported = "Notification" in window;
    const granted = supported && Notification.permission === "granted";
    els.enableJobNotificationsButton.hidden = !supported || granted;
  };
  syncNotificationButton();
  els.enableJobNotificationsButton.addEventListener("click", async () => {
    if (!("Notification" in window)) {
      return;
    }
    try {
      await Notification.requestPermission();
    } catch (_) {
      // Permission prompt can fail (e.g. dismissed); the button state refresh
      // below reflects whatever was decided.
    }
    syncNotificationButton();
  });
}

if (els.refreshAdminButton) {
  els.refreshAdminButton.addEventListener("click", () => refreshAdminPanel({ force: true }));
}

if (els.adminKeyCreateButton) {
  els.adminKeyCreateButton.addEventListener("click", createAdminApiKey);
}

if (els.adminKeysBody) {
  els.adminKeysBody.addEventListener("click", handleAdminKeyAction);
  els.adminKeysBody.addEventListener("change", handleAdminKeyAction);
}

if (els.adminPermSetSaveButton) {
  els.adminPermSetSaveButton.addEventListener("click", saveAdminPermSet);
}

if (els.adminPermSetCancelButton) {
  els.adminPermSetCancelButton.addEventListener("click", resetPermSetEditor);
}

if (els.adminPermSetsBody) {
  els.adminPermSetsBody.addEventListener("click", handleAdminPermSetAction);
}

els.jobSearchButton.addEventListener("click", () => {
  state.jobSearch = els.jobSearchInput.value.trim();
  state.jobsOffset = 0;
  refreshJobs({ force: true });
});

els.jobSearchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    state.jobSearch = els.jobSearchInput.value.trim();
    state.jobsOffset = 0;
    refreshJobs({ force: true });
  }
});

els.pdfPageSizeSelect.addEventListener("change", () => {
  state.pdfPageSize = els.pdfPageSizeSelect.value;
  state.pdfOffset = 0;
  refreshPdfs({ force: true });
});

els.jobsPageSizeSelect.addEventListener("change", () => {
  state.jobsPageSize = els.jobsPageSizeSelect.value;
  state.jobsOffset = 0;
  refreshJobs({ force: true });
});

els.prevPdfPageButton.addEventListener("click", () => {
  if (state.pdfPageSize === "all") {
    return;
  }
  state.pdfOffset = Math.max(0, state.pdfOffset - state.pdfLimit);
  refreshPdfs({ force: true });
});

els.nextPdfPageButton.addEventListener("click", () => {
  if (state.pdfPageSize === "all") {
    return;
  }
  state.pdfOffset += state.pdfLimit;
  refreshPdfs({ force: true });
});

els.prevJobsPageButton.addEventListener("click", () => {
  if (state.jobsPageSize === "all") {
    return;
  }
  state.jobsOffset = Math.max(0, state.jobsOffset - state.jobsLimit);
  refreshJobs({ force: true });
});

els.nextJobsPageButton.addEventListener("click", () => {
  if (state.jobsPageSize === "all") {
    return;
  }
  state.jobsOffset += state.jobsLimit;
  refreshJobs({ force: true });
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
// Composer: autosize to content (Claude/Gemini-style pill) and Enter-to-send
// (Shift+Enter keeps a newline; IME composition is respected).
const composerAutoGrow = () => {
  els.questionInput.style.height = "auto";
  els.questionInput.style.height = `${Math.min(els.questionInput.scrollHeight, 220)}px`;
};
els.questionInput.addEventListener("input", composerAutoGrow);
els.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (typeof els.chatForm.requestSubmit === "function") {
      els.chatForm.requestSubmit();
    } else {
      els.chatForm.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    }
  }
});

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
// Citation links ([S1]/[W1]) in the answer jump to the matching source.

els.chatMessages.addEventListener("click", (event) => {
  const link = event.target.closest("a.citation-link");
  if (!link) {
    return;
  }
  event.preventDefault();
  const message = link.closest(".assistant-message");
  if (!message) {
    return;
  }
  const parts = assistantMessageParts.get(message);
  if (!parts) {
    return;
  }
  focusSourceForCitation(parts, link.dataset.citation || "");
});


// Phones start with the chat sidebar collapsed to keep the conversation
// usable; an explicit expand (persisted) still wins. Storage can be blocked
// (e.g. "block all cookies"); this runs during bootstrap, so never throw.
let hasStoredChatUi = false;
try {
  hasStoredChatUi = Boolean(localStorage.getItem(CHAT_UI_STORAGE_KEY));
} catch (_) {
  // Storage unavailable: fall through with the default (collapsed) state.
}
if (window.innerWidth <= 760 && !hasStoredChatUi) {
  state.chatSidebarCollapsed = true;
}
loadReviewerName();

loadApiKey();

loadThemePreference();

loadAppSidebarCollapsed();

loadChatState();

restoreAnswerPreset();

setChatSidebarCollapsed(state.chatSidebarCollapsed);

renderSavedChats();

renderActiveChat();

persistChatState();

maybeStartFirstVisitWalkthrough();

refreshHealth();

refreshUpdateStatus();
// Hydrate job state immediately (from any tab) so the global jobs strip and
// completion tracking are correct on first paint instead of one poll late.

refreshJobs({ force: true });
// Category registry backs the upload target, Library facet, Review picker,
// Ask chips, and the Admin manager.

refreshCategories();

scheduleHealthPolling();

scheduleUpdatePolling();

scheduleJobsPolling();

// Chat sidebar search: filters saved chats by title and message text.
document.getElementById("chatSearchInput").addEventListener("input", (event) => {
  state.chatSearch = event.target.value;
  renderSavedChats();
});
// Library column sort (PDF/Status/Quality headers).
document.getElementById("pdfsTable").querySelector("thead").addEventListener("click", handleLibrarySortClick);
// Sidebar keyboard navigation (Up/Down between view buttons).
els.appSidebar.addEventListener("keydown", handleSidebarKeydown);
// Updates panel.
document.getElementById("updatePanelRefreshButton").addEventListener("click", () => refreshUpdatePanel());
// Review view switch: chunk list vs document-first browsing.
document.getElementById("reviewViewChunks").addEventListener("click", () => setReviewViewMode("chunks"));
document.getElementById("reviewViewDocuments").addEventListener("click", () => setReviewViewMode("documents"));
document.getElementById("docsBody").addEventListener("click", handleDocumentListClick);
if (els.compactIndexButton) {
  els.compactIndexButton.addEventListener("click", enqueueCompact);
}
if (els.rebuildVectorIndexButton) {
  els.rebuildVectorIndexButton.addEventListener("click", enqueueRebuildVectorIndex);
}
// "Shut down server" is a local-operator control: the endpoint enforces the
// loopback restriction server-side, so the button is only REVEALED when the
// browser itself is on the machine (a LAN admin never sees it at all).
if (els.shutdownServerButton) {
  const host = String(window.location.hostname || "").toLowerCase();
  const isLocalBrowser = host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "[::1]";
  els.shutdownServerButton.hidden = !isLocalBrowser;
  els.shutdownServerButton.addEventListener("click", shutdownServer);
}
if (els.pdfBulkRerunButton) {
  els.pdfBulkRerunButton.addEventListener("click", bulkRerunSelected);
}
if (els.pdfBulkDeleteButton) {
  els.pdfBulkDeleteButton.addEventListener("click", bulkDeleteSelected);
}
// Preview mode toggle: PDF page vs extracted Markdown text.
els.pdfPreviewModePdf.addEventListener("click", () => setPdfPreviewMode("pdf"));
els.pdfPreviewModeText.addEventListener("click", () => setPdfPreviewMode("text"));
// Templates dialog + "Ask about this" selection pill (see web/js/usability.js).
initUsabilityHelpers();
document.addEventListener("visibilitychange", handleVisibilityChange);
