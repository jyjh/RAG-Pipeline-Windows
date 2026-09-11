// Ask tab: saved chats, streaming answers, presets, sources/citations, message actions.

import { ANSWER_PRESETS, ANSWER_PRESET_STORAGE_KEY, CHAT_AUTO_SCROLL_THRESHOLD, CHAT_HISTORY_LIMIT, CHAT_MESSAGE_LIMIT, CHAT_STORAGE_KEY, CHAT_UI_STORAGE_KEY, CITATION_PATTERN, LIVE_RENDER_INTERVAL_MS, STREAM_TAIL_HOLD_CHARS, STREAM_TAIL_MAX_CHARS, applyApiKeyHeaders, blockBoundariesBefore, confirmAction, copyTextToClipboard, els, errorFromResponse, escapeHtml, formatBytes, inlineNotePrompt, isDebugMode, isSafeMarkdownCommit, mediaUrlWithToken, newId, nowIso, numericSetting, patchTableRows, promptForApiKey, promptText, renderMarkdown, requestJson, showToast, softBoundariesBefore, sourceGroupTitle, sourceGroupWeight, stableJson, state } from "./core.js";
import { createJobRow, refreshHealth, rememberJobLogOpenState, updateComposerSettingsSummary } from "./status.js";
import { ensureWalkthroughFakePdf, removeWalkthroughFakePdf } from "./library.js";
import { adminCard, adminMetricRow, adminStatusBadge } from "./admin.js";
import { activateTab, highlightWalkthroughTarget, walkthroughSteps } from "./shell.js";

const assistantMessageParts = new WeakMap();

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
    const stableHtml = stableDelta ? await renderMarkdown(stableDelta) : "";
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
      setStreamTailRaw(parts, keys, tailText);
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
            pinned: Boolean(chat.pinned),
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
  const active = activeChat();
  let chats = state.chats;
  if (active && !chats.slice(0, CHAT_HISTORY_LIMIT).some((chat) => chat.id === active.id)) {
    chats = [active, ...chats.filter((chat) => chat.id !== active.id)];
  }
  state.chats = chats.slice(0, CHAT_HISTORY_LIMIT);
  const buildPayload = (stripRenderedHtml) => JSON.stringify({
    activeChatId: state.activeChatId,
    chats: state.chats.map((chat) => ({
      ...chat,
      messages: Array.isArray(chat.messages)
        ? chat.messages.slice(-CHAT_MESSAGE_LIMIT).map((message) => {
            if (!stripRenderedHtml) {
              return message;
            }
            const { answerHtml, thinkingHtml, ...rest } = message;
            return rest;
          })
        : [],
    })),
  });
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, buildPayload(false));
  } catch (error) {
    // Persisted answers carry raw text AND rendered HTML (~2-3x per message),
    // which can exceed the ~5MB localStorage budget on long histories. Retry
    // once without the derived HTML (hydration falls back to plain text); a
    // second failure is swallowed because persistence must never break an
    // in-flight stream's cleanup path.
    try {
      localStorage.setItem(CHAT_STORAGE_KEY, buildPayload(true));
    } catch (retryError) {
      console.warn("Chat history not persisted (storage quota exceeded).", retryError);
    }
  }
}


function persistChatUiState() {
  // Guarded like persistChatState: in a storage-blocked context (e.g. Safari
  // private mode) an unguarded write throws, and this runs during init --
  // the rest of startup (render, polling) would never run.
  try {
    localStorage.setItem(
      CHAT_UI_STORAGE_KEY,
      JSON.stringify({ chatSidebarCollapsed: state.chatSidebarCollapsed }),
    );
  } catch (_) {
    // Storage unavailable; the preference is simply not persisted.
  }
}


function activeChat() {
  return state.chats.find((chat) => chat.id === state.activeChatId) || null;
}


function createChat({ activate = true, persist = true } = {}) {
  // "New chat" on an untouched chat would otherwise stack indistinguishable
  // empty "New chat" entries; reuse the empty one instead.
  const current = activeChat();
  if (current && !current.messages.length && !current.customTitle) {
    return current;
  }
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


function chatMatchesSearch(chat, needle) {
  if (!needle) {
    return true;
  }
  if ((chat.title || "").toLowerCase().includes(needle)) {
    return true;
  }
  return (chat.messages || []).some((message) =>
    String(message.text || "").toLowerCase().includes(needle),
  );
}

function renderSavedChats() {
  els.savedChatsList.innerHTML = "";
  const needle = state.chatSearch.trim().toLowerCase();
  const visible = state.chats
    .filter((chat) => chatMatchesSearch(chat, needle))
    .sort((a, b) => Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)));
  for (const chat of visible) {
    const row = document.createElement("div");
    row.className = "saved-chat-row";
    row.dataset.chatId = chat.id;

    const title = document.createElement("button");
    title.type = "button";
    title.className = "saved-chat-title";
    title.classList.toggle("active", chat.id === state.activeChatId);
    title.textContent = `${chat.pinned ? "★ " : ""}${chat.title || "New chat"}`;
    title.title = chat.title || "New chat";
    title.addEventListener("click", () => selectChat(chat.id));

    const actions = document.createElement("div");
    actions.className = "saved-chat-actions";

    const rename = document.createElement("button");
    rename.type = "button";
    rename.textContent = "Rename";
    rename.addEventListener("click", () => beginInlineChatRename(row, chat));

    const pin = document.createElement("button");
    pin.type = "button";
    pin.textContent = chat.pinned ? "Unpin" : "Pin";
    pin.title = "Pinned chats stay at the top of the list";
    pin.addEventListener("click", () => {
      chat.pinned = !chat.pinned;
      touchChat(chat);
      persistChatState();
      renderSavedChats();
    });

    const exportButton = document.createElement("button");
    exportButton.type = "button";
    exportButton.textContent = "Export";
    exportButton.title = "Download this chat as Markdown";
    exportButton.addEventListener("click", () => exportChatToMarkdown(chat));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Delete";
    remove.className = "danger";
    remove.addEventListener("click", () => deleteChat(chat.id));

    actions.append(rename, pin, exportButton, remove);
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


// Rename in place: the sidebar title becomes an input; Enter/blur saves,
// Esc cancels. No dialog.
function beginInlineChatRename(rowEl, chat) {
  const titleButton = rowEl?.querySelector(".saved-chat-title");
  if (!titleButton || rowEl.dataset.renaming === "true") {
    return;
  }
  rowEl.dataset.renaming = "true";
  const input = document.createElement("input");
  input.type = "text";
  input.className = "saved-chat-rename";
  input.value = chat.title || "";
  input.setAttribute("aria-label", "Chat name");
  const finish = (save) => {
    if (save) {
      const trimmed = input.value.trim();
      if (trimmed) {
        chat.title = trimmed;
        chat.customTitle = true;
        touchChat(chat);
      }
    }
    delete rowEl.dataset.renaming;
    persistChatState();
    renderSavedChats();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
    } else if (event.key === "Escape") {
      event.stopPropagation();
      finish(false);
    }
  });
  input.addEventListener("blur", () => finish(true));
  titleButton.replaceWith(input);
  input.focus();
  input.select();
}

async function renameChat(chatId) {
  const chat = state.chats.find((item) => item.id === chatId);
  const row = els.savedChatsList.querySelector(`.saved-chat-row[data-chat-id="${CSS.escape(chatId)}"]`);
  if (chat && row) {
    beginInlineChatRename(row, chat);
  }
}

// Export a chat as a Markdown download (messages + answer meta + sources).
function exportChatToMarkdown(chat) {
  const lines = [`# ${chat.title || "New chat"}`, ""];
  for (const message of chat.messages || []) {
    if (message.role === "user") {
      lines.push(`## Question`, "", message.text || "", "");
    } else {
      lines.push(`## Answer`, "", message.text || "", "");
      if (message.settingsSummary || message.durationSeconds) {
        lines.push(`_${[message.settingsSummary, message.durationSeconds ? `${Number(message.durationSeconds).toFixed(1)}s` : ""].filter(Boolean).join(" · ")}_`, "");
      }
      if (message.notice) {
      lines.push(`> ${String(message.notice).split(String.fromCharCode(10)).join(String.fromCharCode(10) + "> ")}`, "");
      }
      const sources = Array.isArray(message.sources) ? message.sources : [];
      if (sources.length) {
        lines.push(`### Sources`, "");
        for (const source of sources) {
          const title = source.kind === "web"
            ? source.title || source.url || source.label
            : source.source_pdf_name || source.file_path || source.label;
          lines.push(`- ${source.label || ""} ${title}`.trim());
        }
        lines.push("");
      }
    }
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = (chat.title || "chat").split(":").join("-").split("*").join("-").split("?").join("-").split("|").join("-").slice(0, 60) || "chat";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(link.href), 5000);
  showToast(`Exported “${chat.title || "chat"}”.`, { kind: "success", timeoutMs: 3000 });
}


async function deleteChat(chatId) {
  if (state.streamingChatId) {
    return;
  }
  const chat = state.chats.find((item) => item.id === chatId);
  if (!chat) {
    return;
  }
  const confirmed = await confirmAction(
    "Delete this chat?",
    `“${chat.title || "New chat"}” and its messages are removed from this browser. This cannot be undone.`,
    "Delete chat",
    { danger: true },
  );
  if (!confirmed) {
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
  const record = {
    role: "assistant",
    // Stable id so per-message actions (answer feedback) can find their
    // record again after reloads and re-renders.
    id: newId(),
    text: parts.rawAnswer,
    thinking: parts.rawThinking,
    answerHtml: parts.answerStable.innerHTML,
    thinkingHtml: parts.rawThinking ? parts.thinkingStable.innerHTML : "",
    sources: parts.sources,
    // Tool results are debug-only; skip persistence entirely when debug mode
    // is off so they don't accumulate in chat history.
    toolResults: isDebugMode() ? parts.toolResults : [],
    notice: parts.notice.textContent || "",
    failed: Boolean(parts.failed),
    settingsSummary: parts.settingsSummary || "",
    durationSeconds: parts.durationSeconds || 0,
    feedback: parts.feedback || "",
    followups: Array.isArray(parts.followups) ? parts.followups : [],
    createdAt: nowIso(),
  };
  parts.messageId = record.id;
  chat.messages.push(record);
  touchChat(chat);
  persistChatState();
  renderSavedChats();
}


// -- recent questions ----------------------------------------------------------
// Light-weight re-run history for the empty chat state: the last few questions
// asked in this browser, newest first, deduplicated case-insensitively.

const RECENT_QUESTIONS_STORAGE_KEY = "rag.recentQuestions.v1";
const RECENT_QUESTIONS_LIMIT = 8;

function loadRecentQuestions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(RECENT_QUESTIONS_STORAGE_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.filter((q) => typeof q === "string" && q.trim()) : [];
  } catch (_) {
    return [];
  }
}

function rememberQuestion(question) {
  const trimmed = String(question || "").trim();
  if (!trimmed) {
    return;
  }
  const recent = loadRecentQuestions().filter(
    (existing) => existing.toLowerCase() !== trimmed.toLowerCase(),
  );
  recent.unshift(trimmed);
  try {
    localStorage.setItem(
      RECENT_QUESTIONS_STORAGE_KEY,
      JSON.stringify(recent.slice(0, RECENT_QUESTIONS_LIMIT)),
    );
  } catch (_) {
    // Storage unavailable (private mode): the feature just stays session-less.
  }
}

const SUGGESTED_QUESTIONS = [
  "Summarise the braking system design guidance in the library.",
  "Which sources cover aerodynamic downforce?",
  "What rules apply to roll hoop design?",
  "Explain the trade-offs between upright materials.",
];

function renderEmptyChatState() {
  const wrap = document.createElement("div");
  wrap.className = "chat-empty";
  const heading = document.createElement("h2");
  heading.textContent = "What can I help you with?";
  const sub = document.createElement("p");
  sub.textContent = "Ask about anything in the local document library — answers cite their sources.";
  const chips = document.createElement("div");
  chips.className = "chat-suggestions";
  for (const question of SUGGESTED_QUESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chat-suggestion";
    chip.textContent = question;
    chip.addEventListener("click", () => {
      if (state.streamingChatId) {
        return;
      }
      els.questionInput.value = question;
      els.chatForm.requestSubmit();
    });
    chips.appendChild(chip);
  }
  wrap.append(heading, sub, chips);
  const recent = loadRecentQuestions();
  if (recent.length) {
    const recentHeading = document.createElement("h3");
    recentHeading.className = "chat-recent-heading";
    recentHeading.textContent = "Recent questions";
    const recentChips = document.createElement("div");
    recentChips.className = "chat-suggestions chat-recent";
    for (const question of recent) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chat-suggestion chat-recent-suggestion";
      chip.textContent = question;
      chip.title = "Ask this again";
      chip.addEventListener("click", () => {
        if (state.streamingChatId) {
          return;
        }
        els.questionInput.value = question;
        els.chatForm.requestSubmit();
      });
      recentChips.appendChild(chip);
    }
    wrap.append(recentHeading, recentChips);
  }
  return wrap;
}

function renderActiveChat() {
  els.chatMessages.innerHTML = "";
  const chat = activeChat();
  if (!chat) {
    return;
  }
  if (!chat.messages.length) {
    els.chatMessages.appendChild(renderEmptyChatState());
    return;
  }
  chat.messages.forEach((message, index) => {
    if (message.role === "assistant") {
      const parts = addSavedAssistantMessage(message);
      const messageEl = parts.body.closest(".message");
      if (messageEl) {
        attachAssistantMessageActions(messageEl, chat);
      }
    } else {
      const body = addMessage("You", message.text || "");
      attachUserMessageActions(body.closest(".message"), chat, index);
    }
  });
  scrollChatToBottom(true);
}


function applyAnswerPreset(presetId) {
  const preset = ANSWER_PRESETS[presetId];
  if (!preset) {
    return;
  }
  els.temperatureInput.value = String(preset.temperature);
  els.maxKInput.value = String(preset.max_k);
  els.contextWindowInput.value = String(preset.context_window);
  els.maxOutputInput.value = String(preset.llm_num_predict);
  els.relevanceFloorInput.value = String(preset.retrieval_min_score);
  els.webSearchInput.checked = Boolean(preset.web_search_enabled);
  // An explicitly chosen mode beats the server's one-time defaults; mark the
  // server-configurable inputs as applied so the health poll leaves them be.
  for (const input of [els.contextWindowInput, els.maxOutputInput, els.relevanceFloorInput]) {
    input.dataset.configApplied = "true";
  }
  try {
    localStorage.setItem(ANSWER_PRESET_STORAGE_KEY, presetId);
  } catch (_) {
    // Storage can be unavailable (private mode); the select still works live.
  }
  updateComposerSettingsSummary();
}


function restoreAnswerPreset() {
  let saved = "";
  try {
    saved = localStorage.getItem(ANSWER_PRESET_STORAGE_KEY) || "";
  } catch (_) {
    saved = "";
  }
  if (ANSWER_PRESETS[saved]) {
    els.answerPresetSelect.value = saved;
    applyAnswerPreset(saved);
    return;
  }
  updateComposerSettingsSummary();
}


function renderJobRows(jobs) {
  rememberJobLogOpenState();
  patchTableRows(els.jobsBody, jobs, {
    keyFor(job) {
      return `job:${String(job.id || "")}`;
    },
    createRow: createJobRow,
  });
}


function scrollChatToBottom(force = false) {
  const el = els.chatMessages;
  const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
  if (force || distanceFromBottom <= CHAT_AUTO_SCROLL_THRESHOLD) {
    el.scrollTop = el.scrollHeight;
  }
}


function scheduleChatScroll(force = false) {
  state.chatScrollForce = state.chatScrollForce || force;
  if (state.chatScrollFrame) {
    return;
  }
  state.chatScrollFrame = window.requestAnimationFrame(() => {
    const shouldForce = state.chatScrollForce;
    state.chatScrollFrame = 0;
    state.chatScrollForce = false;
    scrollChatToBottom(shouldForce);
  });
}


function addMessage(role, text = "") {
  els.chatMessages.querySelector(".chat-empty")?.remove();
  const message = document.createElement("div");
  message.className = "message";
  message.dataset.role = role === "You" ? "user" : "assistant";
  const roleLabel = document.createElement("span");
  roleLabel.className = "role";
  roleLabel.textContent = role;
  const body = document.createElement("span");
  body.className = "body";
  body.textContent = text;
  message.append(roleLabel, body);
  els.chatMessages.appendChild(message);
  scheduleChatScroll(true);
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

  // Sources and tool results sit below the answer body. The tool-results panel
  // is only attached (and populated) when debug mode is enabled via the
  // `debug_mode` cookie; it stays a detached stub otherwise so existing code
  // can still assign to it without error.
  message.append(roleLabel, thinking, notice, body, sources);
  if (isDebugMode()) {
    message.appendChild(toolResultsPanel);
  }
  els.chatMessages.appendChild(message);
  scheduleChatScroll(true);
  const parts = {
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
    sourcesSignature: "",
    toolResults: [],
    toolResultsSignature: "",
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
  assistantMessageParts.set(message, parts);
  return parts;
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
    // Lists can embed many extracted page images; defer fetch/decode until
    // near the viewport instead of loading every asset up front.
    image.loading = "lazy";
    image.decoding = "async";
    link.appendChild(image);

    const caption = document.createElement("span");
    caption.textContent = [asset.page_no ? `page ${asset.page_no}` : "", captionAction].filter(Boolean).join(" | ");
    link.appendChild(caption);
    // Lightbox keeps the user in-context instead of bouncing to a new tab;
    // the lightbox itself still offers an open-original link.
    link.addEventListener("click", (event) => {
      event.preventDefault();
      openImageLightbox(asset.url, asset.description || fallbackAlt);
    });
    assetGrid.appendChild(link);
  }
  return assetGrid;
}


function openImageLightbox(url, description) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay lightbox-overlay";
  overlay.hidden = false;
  const figure = document.createElement("figure");
  figure.className = "lightbox-figure";
  const image = document.createElement("img");
  image.src = url;
  image.alt = description || "Source image";
  const caption = document.createElement("figcaption");
  const text = document.createElement("span");
  text.textContent = description || "";
  const openOriginal = document.createElement("a");
  openOriginal.href = url;
  openOriginal.target = "_blank";
  openOriginal.rel = "noreferrer";
  openOriginal.textContent = "Open original";
  caption.append(text, openOriginal);
  figure.append(image, caption);
  overlay.appendChild(figure);
  document.body.appendChild(overlay);
  const close = () => {
    overlay.remove();
    document.removeEventListener("keydown", onKey);
  };
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) {
      close();
    }
  });
  const onKey = (event) => {
    if (event.key === "Escape") {
      close();
    }
  };
  document.addEventListener("keydown", onKey);
  figure.addEventListener("click", close);
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
  const signature = stableJson(sources);
  if (parts.sourcesSignature === signature) {
    return;
  }
  parts.sourcesSignature = signature;
  const wasOpen = parts.sourcesPanel.open;
  parts.sourcesPanel.hidden = sources.length === 0;
  // Sources are collapsed by default; preserve an explicit user expand.
  parts.sourcesPanel.open = wasOpen && sources.length > 0;
  parts.sourcesBody.innerHTML = "";
  const staleSources = sources.filter(
    (source) => source.kind === "local" && String(source.review_status || "") === "stale",
  );
  if (staleSources.length) {
    const warning = document.createElement("div");
    warning.className = "stale-citation-warning";
    warning.textContent =
      `${staleSources.length} cited source(s) are flagged stale — ` +
      `(${staleSources.map((source) => sourceTitle(source)).slice(0, 3).join(", ")}` +
      `${staleSources.length > 3 ? ", …" : ""}). Verify against a current source before relying on this answer.`;
    parts.sourcesBody.appendChild(warning);
  }
  for (const source of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    item.dataset.citation = String(source.label || source.id || "").trim();
    const links = [];
    if (source.kind === "web" && source.url) {
      // escapeHtml neutralizes markup but not the scheme; a javascript:/data:
      // URL scraped from remote results must not become an href.
      if (/^https?:\/\//i.test(String(source.url))) {
        links.push(`<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">Open</a>`);
      }
    }
    if (source.kind === "local" && source.open_url) {
      links.push(`<a href="${escapeHtml(mediaUrlWithToken(source.open_url))}" target="_blank">Open page</a>`);
    }
    if (source.kind === "local" && source.download_url) {
      links.push(`<a href="${escapeHtml(mediaUrlWithToken(source.download_url))}">Download PDF</a>`);
    }
    if (source.kind === "local" && (source.source_pdf_name || source.file_path)) {
      const searchTarget = escapeHtml(
        encodeURIComponent(source.source_pdf_name || source.file_path || ""),
      );
      links.push(`<a href="#" class="review-link" data-review-search="${searchTarget}">Find in Review</a>`);
    }
    const score = source.kind === "local" && Number.isFinite(Number(source.score))
      ? `score ${Number(source.score).toFixed(3)}`
      : "";
    const reliability = source.kind === "local"
      ? `${sourceGroupTitle(source.source_group)} | weight ${Number(source.reliability_modifier || sourceGroupWeight(source.source_group)).toFixed(2)}`
      : "";
    const groupKey = String(source.source_group || "ungrouped");
    const groupBadge = source.kind === "local"
      ? `<span class="source-group-badge group-${escapeHtml(groupKey)}">${escapeHtml(sourceGroupTitle(groupKey))}</span>`
      : "";
    item.innerHTML = `
      <strong>${escapeHtml(source.label || source.id || "")} ${escapeHtml(sourceTitle(source))} ${groupBadge}</strong>
      <span>${escapeHtml([sourceLocation(source), reliability, score].filter(Boolean).join(" | "))}</span>
      <span>${escapeHtml(source.snippet || "")}</span>
      <span class="source-links">${links.join("")}</span>
    `;
    const assets = Array.isArray(source.assets) ? source.assets : [];
    appendAssetPreviewGrid(item, assets);
    parts.sourcesBody.appendChild(item);
  }
}


function linkAnswerCitations(parts) {
  const root = parts.answerStable;
  if (!root) {
    return;
  }
  const validLabels = new Set(
    (Array.isArray(parts.sources) ? parts.sources : [])
      .map((source) => String(source.label || source.id || "").trim())
      .filter(Boolean),
  );
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      // Saved answers come back with citation anchors already in the stored
      // HTML; re-linking their text would nest <a> inside <a>.
      if (node.parentElement && node.parentElement.closest("a")) {
        return NodeFilter.FILTER_REJECT;
      }
      if (!node.nodeValue || !/\[[SW]\d+\]/.test(node.nodeValue)) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const targets = [];
  let current = walker.nextNode();
  while (current) {
    targets.push(current);
    current = walker.nextNode();
  }
  for (const textNode of targets) {
    const text = textNode.nodeValue;
    CITATION_PATTERN.lastIndex = 0;
    if (!CITATION_PATTERN.test(text)) {
      continue;
    }
    const fragment = document.createDocumentFragment();
    let lastIndex = 0;
    CITATION_PATTERN.lastIndex = 0;
    let match = CITATION_PATTERN.exec(text);
    while (match) {
      if (match.index > lastIndex) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }
      const label = match[0];
      const isValid = validLabels.size === 0 || validLabels.has(label);
      if (isValid) {
        const link = document.createElement("a");
        link.className = "citation-link";
        link.href = "#";
        link.dataset.citation = label;
        link.textContent = label;
        link.title = `Jump to source ${label}`;
        fragment.appendChild(link);
      } else {
        fragment.appendChild(document.createTextNode(label));
      }
      lastIndex = CITATION_PATTERN.lastIndex;
      match = CITATION_PATTERN.exec(text);
    }
    if (lastIndex < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
    textNode.parentNode.replaceChild(fragment, textNode);
  }
}


function focusSourceForCitation(parts, label) {
  const panel = parts.sourcesPanel;
  if (!panel || panel.hidden) {
    return;
  }
  panel.open = true;
  const selector = `.source-item[data-citation="${CSS.escape(label)}"]`;
  const target = parts.sourcesBody.querySelector(selector);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    target.classList.add("source-highlight");
    setTimeout(() => target.classList.remove("source-highlight"), 2000);
  } else {
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// Hover preview for [S1]/[W1] citation links: shows the cited chunk's snippet
// next to the reference so readers can weigh evidence without scrolling away.

function showCitationPopover(link, parts) {
  const popover = els.citationPopover;
  if (!popover || !parts) {
    return;
  }
  const label = String(link.dataset.citation || "").trim();
  const source = (Array.isArray(parts.sources) ? parts.sources : []).find(
    (item) => String(item.label || item.id || "").trim() === label,
  );
  if (!source) {
    return;
  }
  popover.innerHTML = "";
  const title = document.createElement("strong");
  title.textContent = `${label} ${sourceTitle(source)}`;
  const metaParts = [
    sourceLocation(source),
    source.kind === "local" && Number.isFinite(Number(source.score))
      ? `score ${Number(source.score).toFixed(3)}`
      : "",
    source.kind === "local" ? sourceGroupTitle(source.source_group || "ungrouped") : "",
  ].filter(Boolean);
  const meta = document.createElement("span");
  meta.className = "citation-popover-meta";
  meta.textContent = metaParts.join(" | ");
  const snippetText = String(source.snippet || "").trim();
  const snippet = document.createElement("div");
  snippet.className = "citation-popover-snippet";
  snippet.textContent = snippetText.length > 420 ? `${snippetText.slice(0, 420)}…` : snippetText || "(no preview text)";
  popover.append(title, meta, snippet);
  popover.hidden = false;
  const linkRect = link.getBoundingClientRect();
  const popRect = popover.getBoundingClientRect();
  const scrollX = window.scrollX;
  let left = linkRect.left + scrollX;
  const maxLeft = scrollX + document.documentElement.clientWidth - popRect.width - 12;
  left = Math.max(scrollX + 8, Math.min(left, Math.max(8, maxLeft)));
  popover.style.left = `${Math.round(left)}px`;
  popover.style.top = `${Math.round(linkRect.bottom + window.scrollY + 6)}px`;
}


function hideCitationPopover() {
  if (els.citationPopover) {
    els.citationPopover.hidden = true;
  }
}

// In-app PDF preview: renders the stored PDF in an iframe so reviewers can
// verify a source without leaving the Library or downloading the file.

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
  if (!isDebugMode()) {
    parts.toolResultsPanel.hidden = true;
    parts.toolResultsPanel.open = false;
    parts.toolResultsBody.innerHTML = "";
    parts.toolResultsSignature = "";
    return;
  }
  const entries = fallbackToolResultEntries(parts);
  const signature = stableJson(entries);
  if (parts.toolResultsSignature === signature) {
    return;
  }
  parts.toolResultsSignature = signature;
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
  parts.messageId = saved.id || "";
  parts.feedback = saved.feedback || "";
  parts.followups = Array.isArray(saved.followups) ? saved.followups : [];
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
  linkAnswerCitations(parts);
  renderToolResultsPanel(parts);
  attachAnswerMeta(parts);
  return parts;
}


function isTransientNotice(text) {
  return (
    text === "Embedding query and retrieving context..." ||
    text === "Planning retrieval tool calls..." ||
    text === "Searching local context..." ||
    /^Running .+\.\.\.$/.test(text) ||
    /^Retrieved \d+ .+\(s\)\.?$/.test(text) ||
    // Planner path: "Retrieved N local source chunk(s) from M query/queries."
    /^Retrieved \d+ .+\(s\) from \d+ quer(y|ies)\.$/.test(text) ||
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


function appendStreamEvent(parts, event) {
  const type = event.type || "answer";
  const text = event.text || "";

  if (type === "thinking") {
    if (!text) {
      return;
    }
    markGemmaResponseStarted(parts);
    parts.thinking.hidden = false;
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
    if (Array.isArray(parts.sources) && parts.sources.length) {
      renderSourcePanel(parts);
    }
    linkAnswerCitations(parts);
    renderToolResultsPanel(parts);
  }
}

// Streaming core shared by the composer, regenerate, and edit-and-resend.
// announceUser=false re-asks an existing history entry without duplicating the
// user bubble (regenerate) — the caller has already set up chat.messages.

async function runChatExchange(chat, question, {
  announceUser = true,
  isAutoRetry = false,
  precomputedHistory = null,
} = {}) {
  if (state.streamingChatId) {
    return;
  }
  state.activeChatId = chat.id;
  state.streamingChatId = chat.id;
  const abortController = new AbortController();
  state.chatAbortController = abortController;
  // Stall watchdog state lives at function scope: the catch/finally blocks
  // below read and clear it.
  let streamStalled = false;
  let stallTimer = 0;
  // One automatic retry for a hard failure that produced no answer text at
  // all (network blip, transient 5xx). A second failure leaves the manual
  // Retry action on the failed message. Auto-retries themselves never chain
  // a further retry — against a dead backend the retry loop would otherwise
  // spin forever, pinning Send as "Stop" and hammering the server.
  let retryRequested = false;
  // True while this exchange hands control back to a replacement exchange
  // (401 re-auth retry); the finally block must not persist the discarded
  // placeholder assistant message.
  let handedOff = false;
  // Conversation memory: the last few turns (before this question) so the
  // backend can resolve follow-up references during retrieval and prompting.
  // Captured BEFORE the question is pushed below — the server appends the
  // question itself, so sending it in the history too would duplicate it in
  // the prompt. (A hand-off retry reuses the captured turns verbatim: the
  // question is already in chat.messages by then.)
  const chatHistory = precomputedHistory || chat.messages
    .filter((message) => !message.failed && (message.text || "").trim())
    .slice(-6)
    .map((message) => ({ role: message.role, content: String(message.text).slice(0, 4000) }));
  if (announceUser) {
    addUserMessageToChat(chat, question);
    const userBody = addMessage("You", question);
    attachUserMessageActions(userBody.closest(".message"), chat, chat.messages.length - 1);
  }
  const assistantParts = addAssistantMessage();
  assistantParts.body
    .closest(".message")
    ?.classList.add("answer-streaming");
  const exchangeStartedAt = Date.now();
  const exchangeSettings = [
    "temp " + numericSetting(els.temperatureInput, 0.3, 0),
    "top-k " + Math.trunc(numericSetting(els.maxKInput, 40, 1)),
    els.webSearchInput.checked ? "web on" : "web off",
  ];
  setSendButtonStreaming(true);
  renderSavedChats();

  try {
    // applyApiKeyHeaders mutates a Headers instance with .set().
    // A plain object works for fetch(), but not for the shared auth helper.
    const chatHeaders = new Headers({ "Content-Type": "application/json" });
    applyApiKeyHeaders(chatHeaders, { method: "POST" });
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: chatHeaders,
      signal: abortController.signal,
      body: JSON.stringify({
        question,
        history: chatHistory,
        temperature: numericSetting(els.temperatureInput, 0.3, 0),
        max_k: Math.trunc(numericSetting(els.maxKInput, 40, 1)),
        context_window: Math.trunc(numericSetting(els.contextWindowInput, 8192, 1)),
        llm_num_predict: Math.trunc(numericSetting(els.maxOutputInput, 4096, 1)),
        retrieval_min_score: numericSetting(els.relevanceFloorInput, 0.5, 0),
        web_search_enabled: Boolean(els.webSearchInput.checked),
        // Explicit category subset; omitted entirely = search all categories.
        ...(Array.isArray(state.chatSelectedCategories)
          ? { categories: state.chatSelectedCategories }
          : {}),
      }),
    });
    // With API keys configured the chat POST can 401 like any other request;
    // run the same prompt-and-retry flow instead of dumping the raw JSON
    // body into the message's error notice.
    if (response.status === 401) {
      const key = await promptForApiKey();
      if (key) {
        // Hand off to a fresh exchange: drop the empty placeholder, and mark
        // the hand-off so the finally block skips persisting it. The user
        // bubble already exists from this exchange (announceUser), so the
        // retry must not announce again.
        handedOff = true;
        const placeholder = assistantParts.body?.closest(".message");
        if (placeholder) {
          placeholder.remove();
        }
        // Clear streaming state BEFORE recursing: runChatExchange's entry
        // guard reads state.streamingChatId synchronously, and this try
        // block's finally only runs after the recursive call has already
        // started (and bailed).
        state.streamingChatId = null;
        if (state.chatAbortController === abortController) {
          state.chatAbortController = null;
        }
        setSendButtonStreaming(false);
        // Reuse the history captured before the question was pushed, or the
        // retry's prompt would contain the question twice (once as the last
        // history turn, once as the live question).
        return runChatExchange(chat, question, {
          announceUser: false,
          precomputedHistory: chatHistory,
        });
      }
    }
    if (!response.ok || !response.body) {
      // Surface a readable message (the stream may still carry a JSON error
      // body) instead of dumping the raw payload into the notice.
      throw await errorFromResponse(response);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    // Abort when no bytes arrive for CHAT_STREAM_STALL_TIMEOUT_MS. Chosen
    // above the worst legitimate first-token latency (a cold local model
    // load, ~1-2 min) so only a genuinely hung stream trips it.
    const CHAT_STREAM_STALL_TIMEOUT_MS = 300000;
    stallTimer = setTimeout(() => {
      streamStalled = true;
      abortController.abort();
    }, CHAT_STREAM_STALL_TIMEOUT_MS);
    const resetStallTimer = () => {
      clearTimeout(stallTimer);
      stallTimer = setTimeout(() => {
        streamStalled = true;
        abortController.abort();
      }, CHAT_STREAM_STALL_TIMEOUT_MS);
    };
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
      scheduleChatScroll();
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      resetStallTimer();
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
      appendStreamEvent(
        assistantParts,
        {
          type: "notice",
          text: streamStalled
            ? "Generation timed out: the server stopped sending output. Click Send to retry."
            : "Generation stopped.",
        },
      );
    } else {
      assistantParts.failed = true;
      retryRequested = !assistantParts.rawAnswer && error.name !== "AbortError";
      appendStreamEvent(assistantParts, { type: "error", text: error.message });
    }
  } finally {
    clearTimeout(stallTimer);
    assistantParts.body
      .closest(".message")
      ?.classList.remove("answer-streaming");
    if (handedOff) {
      // The replacement exchange already cleared and re-acquired the
      // streaming state before this finally ran (its synchronous prefix
      // executes during `return runChatExchange(...)`); touching state here
      // would tear down the new exchange's ownership. Nothing to persist:
      // the placeholder was removed at the hand-off site.
      return;
    }
    await formatAssistantMessage(assistantParts);
    assistantParts.durationSeconds = (Date.now() - exchangeStartedAt) / 1000;
    assistantParts.settingsSummary = exchangeSettings.join(" · ");
    addAssistantMessageToChat(chat, assistantParts);
    attachAnswerMeta(assistantParts);
    if (retryRequested && !isAutoRetry) {
      // Release streaming state BEFORE delegating: regenerateLastAnswer
      // refuses to run while a chat is marked streaming, so returning here
      // without clearing would wedge Send/Stop/chat switching until reload.
      state.streamingChatId = null;
      if (state.chatAbortController === abortController) {
        state.chatAbortController = null;
      }
      setSendButtonStreaming(false);
      showToast("Chat stream failed once — retrying automatically.", { kind: "info" });
      regenerateLastAnswer(chat, { isAutoRetry: true });
      return;
    }
    state.streamingChatId = null;
    if (state.chatAbortController === abortController) {
      state.chatAbortController = null;
    }
    const messageEl = assistantParts.body.closest(".message");
    if (messageEl) {
      attachAssistantMessageActions(messageEl, chat);
    }
    setSendButtonStreaming(false);
    renderSavedChats();
    await refreshHealth();
    // Suggestions arrive after the answer is already usable, so the exchange
    // is not gated on them (fire-and-forget; failures leave no trace).
    if (!assistantParts.failed && assistantParts.rawAnswer && chat.id === state.activeChatId) {
      fetchFollowupSuggestions(chat, assistantParts, question, chatHistory);
    }
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
  els.questionInput.value = "";
  rememberQuestion(question);
  await runChatExchange(chat, question);
}

// -- per-message actions (copy / regenerate / edit-and-resend) ---------------


function buildMessageActions(actions) {
  const bar = document.createElement("div");
  bar.className = "message-actions";
  for (const action of actions) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = action.label;
    button.title = action.title || action.label;
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      action.onClick();
    });
    bar.appendChild(button);
  }
  return bar;
}


function attachAssistantMessageActions(messageEl, chat) {
  if (!messageEl || messageEl.querySelector(".message-actions")) {
    return;
  }
  const parts = assistantMessageParts.get(messageEl);
  const actions = [
    {
      label: "Copy",
      title: "Copy the answer as Markdown",
      onClick: () => copyTextToClipboard(parts ? parts.rawAnswer : ""),
    },
  ];
  if (parts?.failed) {
    actions.push({
      label: "Retry",
      title: "Ask this question again",
      onClick: () => regenerateLastAnswer(chat),
    });
  }
  // Regenerate re-asks the last exchange, so it only makes sense on the most
  // recent assistant message.
  const messages = chat.messages;
  if (messages.length && messages[messages.length - 1].role === "assistant") {
    actions.push({
      label: "Regenerate",
      title: "Re-ask the previous question",
      onClick: () => regenerateLastAnswer(chat),
    });
  }
  messageEl.appendChild(buildMessageActions(actions));
  attachFeedbackControls(messageEl, chat, parts);
  attachFollowupRow(messageEl, chat, parts);
}


// -- answer feedback + follow-up suggestions ----------------------------------
// Both turn a finished answer into the next action: a 👍/👎 rating that the
// team can review server-side, and clickable follow-up questions generated
// from the answer.

function attachFeedbackControls(messageEl, chat, parts) {
  if (!messageEl || messageEl.querySelector(".feedback-bar") || !parts || parts.failed) {
    return;
  }
  const bar = document.createElement("div");
  bar.className = "feedback-bar";
  const upButton = document.createElement("button");
  upButton.type = "button";
  upButton.className = "feedback-button";
  upButton.textContent = "👍 Helpful";
  const downButton = document.createElement("button");
  downButton.type = "button";
  downButton.className = "feedback-button";
  downButton.textContent = "👎 Not helpful";
  bar.append(upButton, downButton);

  const markVoted = (rating) => {
    bar.classList.add("feedback-voted");
    for (const [button, value] of [[upButton, "up"], [downButton, "down"]]) {
      button.disabled = true;
      button.classList.toggle("feedback-selected", value === rating);
    }
  };
  if (parts.feedback) {
    markVoted(parts.feedback);
  }

  const submitRating = async (rating, note = "") => {
    // The question for this answer is the nearest preceding user message.
    let question = "";
    const messageIndex = chat.messages.findIndex(
      (message) => message.role === "assistant" && message.id === parts.messageId,
    );
    for (let index = messageIndex >= 0 ? messageIndex : chat.messages.length - 1; index >= 0; index -= 1) {
      if (chat.messages[index].role === "user") {
        question = chat.messages[index].text || "";
        break;
      }
    }
    try {
      await requestJson("/api/feedback", {
        method: "POST",
        body: JSON.stringify({
          rating,
          question: question || "(unknown question)",
          answer_excerpt: (parts.rawAnswer || "").slice(0, 8000),
          note,
          chat_id: chat.id || "",
          message_id: parts.messageId || "",
          sources_count: Array.isArray(parts.sources) ? parts.sources.length : 0,
        }),
      });
    } catch (error) {
      // Feedback is best-effort telemetry: a failure must never suggest the
      // vote was recorded, but it also should not disrupt the conversation.
      showToast(`Feedback could not be saved: ${error.message}`, { kind: "error" });
      return;
    }
    parts.feedback = rating;
    const record = chat.messages.find(
      (message) => message.role === "assistant" && message.id === parts.messageId,
    );
    if (record) {
      record.feedback = rating;
      persistChatState();
    }
    markVoted(rating);
    showToast(rating === "up" ? "Thanks — glad it helped." : "Thanks — noted for review.", { kind: "success" });
  };

  upButton.addEventListener("click", () => {
    if (parts.feedback) {
      return;
    }
    submitRating("up");
  });
  downButton.addEventListener("click", async () => {
    if (parts.feedback) {
      return;
    }
    // Cancel (null) means "changed my mind": no vote is recorded.
    const note = await inlineNotePrompt(downButton, {
      title: "What was wrong? (optional)",
      placeholder: "e.g. wrong numbers, missed the rules section, bad source…",
    });
    if (note === null) {
      return;
    }
    submitRating("down", note);
  });
  messageEl.appendChild(bar);
}


function attachFollowupRow(messageEl, chat, parts) {
  if (!messageEl || messageEl.querySelector(".followup-row")) {
    return;
  }
  const suggestions = Array.isArray(parts.followups) ? parts.followups : [];
  if (!suggestions.length || parts.failed) {
    return;
  }
  const row = document.createElement("div");
  row.className = "followup-row";
  const label = document.createElement("span");
  label.className = "followup-label";
  label.textContent = "Follow-ups";
  row.appendChild(label);
  for (const suggestion of suggestions) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "followup-chip";
    chip.textContent = suggestion;
    chip.title = "Ask this follow-up";
    chip.addEventListener("click", () => {
      if (state.streamingChatId) {
        return;
      }
      rememberQuestion(suggestion);
      runChatExchange(chat, suggestion);
    });
    row.appendChild(chip);
  }
  messageEl.appendChild(row);
}


// Best-effort: one cheap non-streaming LLM call after a successful answer.
// Any failure (offline, rate limit, timeout) quietly leaves the answer as-is.
async function fetchFollowupSuggestions(chat, parts, question, history) {
  const messageEl = parts.body.closest(".message");
  if (!messageEl) {
    return;
  }
  try {
    const data = await requestJson("/api/chat/followups", {
      method: "POST",
      timeoutMs: 120000,
      body: JSON.stringify({
        question,
        answer: (parts.rawAnswer || "").slice(0, 20000),
        history: history.slice(-4),
        source_titles: (Array.isArray(parts.sources) ? parts.sources : [])
          .map((source) =>
            String(
              source.title || source.source_pdf_name || source.label || source.id || "",
            ).trim(),
          )
          .filter(Boolean)
          .slice(0, 8),
      }),
    });
    const suggestions = Array.isArray(data?.suggestions) ? data.suggestions : [];
    if (!suggestions.length) {
      return;
    }
    parts.followups = suggestions;
    const record = chat.messages.find(
      (message) => message.role === "assistant" && message.id === parts.messageId,
    );
    if (record) {
      record.followups = suggestions;
      persistChatState();
    }
    attachFollowupRow(messageEl, chat, parts);
  } catch (_) {
    // Suggestions are a convenience; silence is the right failure mode.
  }
}


function attachUserMessageActions(messageEl, chat, messageIndex) {
  if (!messageEl || messageEl.querySelector(".message-actions")) {
    return;
  }
  messageEl.appendChild(
    buildMessageActions([
      {
        label: "Copy",
        onClick: () => copyTextToClipboard(chat.messages[messageIndex]?.text || ""),
      },
      {
        label: "Edit",
        title: "Edit in place — everything after this message is removed on resend",
        onClick: () => beginInlineMessageEdit(messageEl, chat, messageIndex),
      },
    ]),
  );
}

// In-place edit of a sent message: the bubble swaps to a textarea with
// Cancel / Save & resend. Saving truncates the conversation at this message
// and re-asks with the new wording.

function beginInlineMessageEdit(messageEl, chat, messageIndex) {
  const body = messageEl.querySelector(".body");
  if (!body || messageEl.dataset.editing === "true") {
    return;
  }
  const original = chat.messages[messageIndex]?.text || "";
  messageEl.dataset.editing = "true";
  messageEl.classList.add("message-editing");

  const editor = document.createElement("div");
  editor.className = "message-inline-edit";
  const textarea = document.createElement("textarea");
  textarea.className = "message-inline-edit-textarea";
  textarea.value = original;
  textarea.rows = Math.min(10, original.split("\n").length + 2);
  const actions = document.createElement("div");
  actions.className = "message-inline-edit-actions";
  const cancelButton = document.createElement("button");
  cancelButton.type = "button";
  cancelButton.textContent = "Cancel";
  cancelButton.addEventListener("click", () => cancelInlineMessageEdit(messageEl));
  const resendButton = document.createElement("button");
  resendButton.type = "button";
  resendButton.textContent = "Save & resend";
  resendButton.addEventListener("click", () => {
    const edited = textarea.value.trim();
    if (!edited || edited === original) {
      cancelInlineMessageEdit(messageEl);
      return;
    }
    if (state.streamingChatId) {
      showToast("Wait for the current answer to finish first.", { kind: "info" });
      return;
    }
    editAndResendUserMessage(chat, messageIndex, edited);
  });
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      cancelInlineMessageEdit(messageEl);
    } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      resendButton.click();
    }
  });
  actions.append(cancelButton, resendButton);
  editor.append(textarea, actions);
  body.replaceWith(editor);
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}


function cancelInlineMessageEdit(messageEl) {
  if (!messageEl || messageEl.dataset.editing !== "true") {
    return;
  }
  // A full re-render is the cheapest consistent restore: state and DOM stay
  // in sync even if history shifted while editing.
  renderActiveChat();
}


function regenerateLastAnswer(chat, { isAutoRetry = false } = {}) {
  if (state.streamingChatId) {
    showToast("Wait for the current answer to finish first.", { kind: "info" });
    return;
  }
  const messages = chat.messages;
  if (!messages.length || messages[messages.length - 1].role !== "assistant") {
    return;
  }
  // Drop the answer being regenerated, then find the question before it.
  messages.pop();
  let question = "";
  while (messages.length && messages[messages.length - 1].role !== "user") {
    messages.pop();
  }
  if (messages.length && messages[messages.length - 1].role === "user") {
    question = messages[messages.length - 1].text || "";
    messages.pop();
  }
  if (!question) {
    return;
  }
  persistChatState();
  renderSavedChats();
  renderActiveChat();
  runChatExchange(chat, question, { isAutoRetry });
}


function editAndResendUserMessage(chat, messageIndex, editedText) {
  if (state.streamingChatId) {
    showToast("Wait for the current answer to finish first.", { kind: "info" });
    return;
  }
  if (messageIndex < 0 || messageIndex >= chat.messages.length) {
    return;
  }
  // Truncate the conversation at the edited message (inclusive) and re-ask.
  chat.messages = chat.messages.slice(0, messageIndex);
  persistChatState();
  renderSavedChats();
  renderActiveChat();
  runChatExchange(chat, editedText);
}


// Tiny dependency-free sparkline. values = numbers; the polyline is scaled
// to the observed min/max so quiet series still show shape.
function sparklineSvg(values, { width = 190, height = 34 } = {}) {
  const points = (Array.isArray(values) ? values : []).map((value) => Number(value) || 0);
  if (points.length < 2) {
    return "";
  }
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const step = width / (points.length - 1);
  const coords = points
    .map((value, index) => {
      const x = (index * step).toFixed(1);
      const y = (height - 3 - ((value - min) / span) * (height - 6)).toFixed(1);
      return `${x},${y}`;
    })
    .join(" ");
  const last = points[points.length - 1];
  const rising = last >= points[0];
  const color = rising ? "var(--good)" : "var(--danger)";
  return (
    `<svg class="sparkline" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" ` +
    `role="img" aria-label="trend over the last ${points.length} samples">` +
    `<polyline fill="none" stroke="${color}" stroke-width="2" points="${coords}" />` +
    "</svg>"
  );
}

function renderOpsDashboard(health, metrics, categories) {
  const dashboard = els.adminDashboard;
  if (!dashboard) {
    return;
  }
  // Security posture + startup repair notices render as full-width banners.
  const banners = [];
  if (health?.security && health.security.bind_all && !health.security.auth_enabled) {
    banners.push(
      '<div class="admin-banner admin-banner-warning">This server accepts unauthenticated changes from the network. Set <code>[server] api_token</code> or issue API keys in Admin.</div>',
    );
  }
  for (const notice of health?.startup_notices || []) {
    banners.push(`<div class="admin-banner">${escapeHtml(notice)}</div>`);
  }
  const queue = (metrics && metrics.queue) || (health && health.queue) || {};
  // llm_* keys live on /api/health; /api/metrics carries the ollama_* snapshot.
  const ollamaSnap = (metrics && metrics.ollama) || {};
  const llmBackend = String(health?.llm_backend ?? ollamaSnap.llm_backend ?? "unknown");
  const llmReachable = Boolean(health?.llm_reachable ?? ollamaSnap.reachable);
  const llmUrl = String(
    health?.llm_base_url ?? ollamaSnap.llm_base_url ?? ollamaSnap.ollama_active_host ?? "—",
  );
  const embedding = (metrics && metrics.embedding_config) || {};
  const embeddingRows = embedding.error
    ? adminMetricRow("config", `<span class="admin-metric-error">${escapeHtml(embedding.error)}</span>`)
    : [
        adminMetricRow("backend", escapeHtml(String(embedding.backend || "—"))),
        adminMetricRow("model", escapeHtml(String(embedding.model || "—"))),
        adminMetricRow("batch size", escapeHtml(String(embedding.batch_size ?? "—"))),
        adminMetricRow("replicas", escapeHtml(String(embedding.replica_count ?? "—"))),
        adminMetricRow("concurrency", escapeHtml(String(embedding.concurrency ?? "—"))),
      ].join("");
  const history = Array.isArray(metrics?.history) ? metrics.history : [];
  const recordsTrend = sparklineSvg(history.map((sample) => sample.records));
  const queuedTrend = sparklineSvg(history.map((sample) => sample.queued));
  const cards = [
    adminCard(
      "Index",
      [
        adminMetricRow("records", escapeHtml(Number(metrics?.record_count ?? health?.record_count ?? 0).toLocaleString())),
        adminMetricRow("documents", escapeHtml(Number(metrics?.document_count ?? 0).toLocaleString())),
        adminMetricRow("on disk", escapeHtml(formatBytes(metrics?.index_bytes || 0))),
        adminMetricRow(
          "embedded / reused",
          escapeHtml(
            `${Number(metrics?.embedded_records || 0).toLocaleString()} / ${Number(metrics?.reused_records || 0).toLocaleString()}`,
          ),
        ),
      ].join("") + (recordsTrend ? `<div class="sparkline-row"><span class="sparkline-caption">records / 24h</span>${recordsTrend}</div>` : ""),
    ),
    adminCard(
      "Job queue",
      [
        adminMetricRow("active jobs", escapeHtml(String(queue.active_job_count ?? queue.active_count ?? "—"))),
        adminMetricRow("queued", escapeHtml(String(queue.queued_count ?? "—"))),
        adminMetricRow("active queries", escapeHtml(String(queue.active_query_count ?? "—"))),
      ].join("") + (queuedTrend ? `<div class="sparkline-row"><span class="sparkline-caption">queued / 24h</span>${queuedTrend}</div>` : ""),
    ),
    adminCard(
      "LLM backend",
      [
        adminMetricRow("backend", escapeHtml(llmBackend)),
        adminMetricRow("status", adminStatusBadge(llmReachable, "reachable", "unreachable")),
        adminMetricRow("endpoint", escapeHtml(llmUrl)),
      ].join(""),
      llmReachable ? "" : "admin-card-warning",
    ),
    adminCard("Embeddings", embeddingRows),
  ];
  const disk = Array.isArray(metrics?.disk) ? metrics.disk : [];
  if (disk.length) {
    const diskRows = disk
      .map((volume) => {
        const free = Number(volume.free_bytes || 0);
        const total = Math.max(1, Number(volume.total_bytes || 0));
        const low = free < 10 * 1024 ** 3 || free / total < 0.05;
        return adminMetricRow(
          volume.label + " free",
          '<span class="' + (low ? "admin-metric-error" : "") + '">' + escapeHtml(formatBytes(free)) + "</span>",
        );
      })
      .join("");
    const anyLow = disk.some((volume) => Number(volume.free_bytes || 0) < 10 * 1024 ** 3);
    cards.push(adminCard("Disk", diskRows, anyLow ? "admin-card-warning" : ""));
  }
  if (categories && Array.isArray(categories.categories)) {
    const list = categories.categories;
    const totalSources = Number(categories.total_sources || 0);
    const covered = list.reduce((sum, cat) => sum + Number(cat.source_count || 0), 0);
    const coverage = totalSources ? Math.round((covered / totalSources) * 100) : 0;
    const catRows = [
      adminMetricRow(
        "coverage",
        escapeHtml(coverage + "% of " + totalSources.toLocaleString() + " sources"),
      ),
    ]
      .concat(
        list.slice(0, 6).map((cat) =>
          adminMetricRow(
            // adminMetricRow escapes its label itself; pre-escaping here
            // double-escaped "&" as "&amp;amp;" for labels like "Q&A docs".
            String(cat.label || cat.key || "?"),
            escapeHtml(Number(cat.source_count || 0).toLocaleString()) +
              " sources" +
              (cat.exists === false ? ' <span class="status-badge status-bad">no index</span>' : ""),
          ),
        ),
      )
      .join("");
    cards.push(adminCard("Categories", catRows, coverage < 100 ? "admin-card-warning" : ""));
  }
  dashboard.innerHTML = banners.join("") + cards.join("");
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

export {
  activeChat,
  addAssistantMessage,
  addAssistantMessageToChat,
  addFormattingNotice,
  addMessage,
  addPersistentNotice,
  addSavedAssistantMessage,
  addUserMessageToChat,
  appendAssetPreviewGrid,
  appendStreamEvent,
  appendStreamStableHtml,
  applyAnswerPreset,
  assistantMessageParts,
  attachAssistantMessageActions,
  attachUserMessageActions,
  beginInlineMessageEdit,
  buildMessageActions,
  cancelInlineMessageEdit,
  createAssetPreviewGrid,
  createChat,
  deleteChat,
  editAndResendUserMessage,
  fallbackToolResultEntries,
  firstFiveWords,
  focusSourceForCitation,
  formatAssistantMessage,
  hideCitationPopover,
  isTransientNotice,
  linkAnswerCitations,
  loadChatState,
  markGemmaResponseStarted,
  normalizeToolResultEvent,
  openImageLightbox,
  persistChatState,
  persistChatUiState,
  queueMarkdownRender,
  refreshChatTitle,
  regenerateLastAnswer,
  renameChat,
  renderActiveChat,
  renderDelay,
  renderJobRows,
  renderKeys,
  renderOpsDashboard,
  renderSavedChats,
  renderSourcePanel,
  renderToolResultsPanel,
  renderWalkthroughStep,
  replaceStreamHtml,
  restoreAnswerPreset,
  runChatExchange,
  runMarkdownRender,
  scheduleChatScroll,
  scheduleMarkdownRender,
  scrollChatToBottom,
  selectChat,
  sendQuestion,
  setChatSidebarCollapsed,
  setSendButtonStreaming,
  setStreamTailHtml,
  setStreamTailRaw,
  showCitationPopover,
  sourceAsToolResultItem,
  sourceLocation,
  sourceTitle,
  streamingStableCutoff,
  toolResultItemText,
  toolResultItemTitle,
  toolResultJson,
  touchChat,
  updateNotice,
  updateStreamTailRaw,
};
// Small muted line under an answer: the settings it used and how long the
// generation took. Persisted per message so old answers keep their meta.
export function attachAnswerMeta(parts) {
  const messageEl = parts.body?.closest(".message");
  if (!messageEl) {
    return;
  }
  let meta = messageEl.querySelector(".answer-meta");
  const text = [parts.settingsSummary, parts.durationSeconds ? `${Number(parts.durationSeconds).toFixed(1)}s` : ""]
    .filter(Boolean)
    .join(" · ");
  if (!text) {
    return;
  }
  if (!meta) {
    meta = document.createElement("div");
    meta.className = "answer-meta";
    parts.body.after(meta);
  }
  meta.textContent = text;
}
