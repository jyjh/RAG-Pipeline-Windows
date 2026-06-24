const state = {
  offset: 0,
  limit: 20,
  total: 0,
  search: "",
};

const els = {
  statusLine: document.getElementById("statusLine"),
  fileInput: document.getElementById("fileInput"),
  uploadButton: document.getElementById("uploadButton"),
  reindexButton: document.getElementById("reindexButton"),
  uploadStatus: document.getElementById("uploadStatus"),
  jobsBody: document.getElementById("jobsBody"),
  searchInput: document.getElementById("searchInput"),
  searchButton: document.getElementById("searchButton"),
  prevPageButton: document.getElementById("prevPageButton"),
  nextPageButton: document.getElementById("nextPageButton"),
  pageLabel: document.getElementById("pageLabel"),
  indexStatus: document.getElementById("indexStatus"),
  indexBody: document.getElementById("indexBody"),
  chatForm: document.getElementById("chatForm"),
  questionInput: document.getElementById("questionInput"),
  sendButton: document.getElementById("sendButton"),
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
      detail = JSON.parse(detail).detail || detail;
    } catch (_) {
      // Keep the raw response text.
    }
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function refreshHealth() {
  try {
    const data = await requestJson("/api/health");
    const queue = data.queue || {};
    els.statusLine.textContent =
      `${data.record_count} indexed chunks | ` +
      `${queue.active_query_count || 0} active queries | ` +
      `${queue.queued_count || 0} queued jobs`;
  } catch (error) {
    els.statusLine.textContent = `Health check failed: ${error.message}`;
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

async function uploadFiles() {
  const files = Array.from(els.fileInput.files || []);
  if (!files.length) {
    setStatus(els.uploadStatus, "Choose one or more PDF files.", true);
    return;
  }

  const body = new FormData();
  for (const file of files) {
    body.append("files", file);
  }

  els.uploadButton.disabled = true;
  setStatus(els.uploadStatus, "Queueing upload...");
  try {
    const job = await requestJson("/api/uploads", { method: "POST", body });
    els.fileInput.value = "";
    setStatus(els.uploadStatus, `Queued job ${job.id.slice(0, 8)}.`);
    await refreshJobs();
  } catch (error) {
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
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  } finally {
    els.reindexButton.disabled = false;
  }
}

async function loadIndex() {
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

function renderIndexRows(rows) {
  els.indexBody.innerHTML = "";
  for (const item of rows) {
    const row = document.createElement("tr");
    row.dataset.recordId = item.id;
    row.innerHTML = `
      <td class="source-cell">
        <strong>${escapeHtml(item.id)}</strong><br />
        ${escapeHtml(item.file_path)}<br />
        chunk ${escapeHtml(item.chunk_index)}
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

  const thinkingBody = document.createElement("pre");
  thinking.append(summary, thinkingBody);

  const body = document.createElement("span");
  body.className = "body";

  message.append(roleLabel, thinking, body);
  els.chatMessages.appendChild(message);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  return { body, thinking, thinkingBody };
}

function appendStreamEvent(parts, event) {
  const type = event.type || "answer";
  const text = event.text || "";
  if (!text) {
    return;
  }

  if (type === "thinking") {
    parts.thinking.hidden = false;
    parts.thinkingBody.textContent += text;
    return;
  }

  if (type === "error") {
    parts.body.textContent += `\n\n[Error] ${text}`;
    return;
  }

  parts.body.textContent += text;
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = els.questionInput.value.trim();
  if (!question) {
    return;
  }

  addMessage("You", question);
  const assistantParts = addAssistantMessage();
  els.questionInput.value = "";
  els.sendButton.disabled = true;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
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
    els.sendButton.disabled = false;
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

els.uploadButton.addEventListener("click", uploadFiles);
els.reindexButton.addEventListener("click", enqueueReindex);
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
els.prevPageButton.addEventListener("click", () => {
  state.offset = Math.max(0, state.offset - state.limit);
  loadIndex();
});
els.nextPageButton.addEventListener("click", () => {
  state.offset += state.limit;
  loadIndex();
});
els.indexBody.addEventListener("click", handleIndexAction);
els.chatForm.addEventListener("submit", sendQuestion);

refreshHealth();
refreshJobs();
setInterval(refreshHealth, 4000);
setInterval(refreshJobs, 3000);
