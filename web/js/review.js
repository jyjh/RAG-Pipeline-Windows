// Review tab: index browsing, search, in-place record editing.

import { els, escapeHtml, indexChildCountLabel, indexNodeLabel, indexPageRange, indexParentRow, indexReliabilityLabel, indexRowContent, indexRowsForParent, indexScoreLabel, isAbortError, markIndexDirty, numericSetting, patchTableRows, requestJson, setStatus, showToast, stableJsonHash, state, toastError, sourceGroupTitle } from "./core.js";
import { appendAssetPreviewGrid } from "./chat.js";

const INDEX_STREAM_BATCH_SIZE = 250;

const INDEX_CHILD_BATCH_SIZE = 100;

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
    category: state.indexCategory || "general",
  });
  const url = `/api/index/summaries?${params}`;
  try {
    const data = await requestJson(url, {
      signal: load.abortController.signal,
    });
    if (!isActiveIndexLoad(load)) {
      return;
    }
    state.indexLoaded = true;
    state.indexDirty = false;
    if (data.notModified && state.indexRenderedUrl === url) {
      return;
    }
    state.total = data.total || 0;
    renderIndexRows(data.rows || []);
    state.indexRenderedUrl = url;
    const start = state.total ? state.offset + 1 : 0;
    const end = Math.min(state.offset + state.limit, state.total);
    els.pageLabel.textContent = `${start}-${end} of ${state.total} summaries`;
    els.prevPageButton.disabled = state.offset <= 0;
    els.nextPageButton.disabled = state.offset + state.limit >= state.total;
    if (data.degraded) {
      setStatus(
        els.indexStatus,
        `Large index — showing a flat list. ${data.degraded_reason || ""}`,
        true,
      );
    } else {
      setStatus(els.indexStatus, `Embedding model: ${data.embedding_model || "unknown"}`);
    }
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
    category: state.indexCategory || "general",
  });
  const url = `/api/index/summaries?${params}`;

  els.indexBody.innerHTML = "";
  els.pageLabel.textContent = "Loading summaries...";
  els.prevPageButton.disabled = true;
  els.nextPageButton.disabled = true;
  setStatus(els.indexStatus, "Loading summary chunks...");

  try {
    const data = await requestJson(url, {
      signal: load.abortController.signal,
    });
    if (!isActiveIndexLoad(load)) {
      return;
    }
    state.indexLoaded = true;
    state.indexDirty = false;
    if (data.notModified && state.indexRenderedUrl === url) {
      return;
    }
    const rows = data.rows || [];
    state.total = data.total || rows.length;
    renderIndexRows(rows);
    state.indexRenderedUrl = url;
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
  const editedLabel = item.edited ? "Edited" : "";
  const meta = [indexNodeLabel(item), editedLabel, indexScoreLabel(item), indexReliabilityLabel(item), pageRange, childCount]
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
        <div class="index-content-text"></div>
        <div class="index-inline-editor">
          <textarea class="inline-edit-textarea" spellcheck="false" aria-label="Edit record content"></textarea>
          <div class="inline-edit-preview rendered" hidden></div>
          <div class="inline-edit-actions">
            <button type="button" data-action="preview-toggle" title="Render the Markdown to check formatting">Preview</button>
            <span class="inline-edit-status status"></span>
            <button type="button" data-action="cancel-edit">Cancel</button>
            <button type="button" data-action="save-edit">Save</button>
          </div>
        </div>
      </td>
      <td>
        <div class="row-actions">
          <button type="button" data-action="edit" title="Edit this record in place">Edit</button>
        </div>
      </td>
    `;
  const contentCell = row.querySelector(".index-content-text");
  contentCell.textContent = item.content || "";
  indexRowContent.set(row, item.content || "");
  appendAssetPreviewGrid(row.querySelector(".index-content-cell"), item.assets, {
    className: "source-assets index-assets",
    itemClassName: "index-asset",
    fallbackAlt: "Extracted source image",
  });
  return row;
}

// Row -> original content, so in-place editing can restore and diff against
// the last published text without pulling it back through the DOM.

function beginInlineEdit(row) {
  if (row.dataset.editing === "true") {
    return;
  }
  row.dataset.editing = "true";
  row.classList.add("index-row-editing");
  const textarea = row.querySelector(".inline-edit-textarea");
  textarea.value = indexRowContent.get(row) || "";
  const preview = row.querySelector(".inline-edit-preview");
  preview.hidden = true;
  preview.innerHTML = "";
  row.querySelector("[data-action='preview-toggle']").textContent = "Preview";
  textarea.focus();
}


function endInlineEdit(row) {
  delete row.dataset.editing;
  row.classList.remove("index-row-editing");
}


function toggleInlineEditPreview(row) {
  const textarea = row.querySelector(".inline-edit-textarea");
  const preview = row.querySelector(".inline-edit-preview");
  const button = row.querySelector("[data-action='preview-toggle']");
  if (!preview.hidden) {
    preview.hidden = true;
    textarea.hidden = false;
    button.textContent = "Preview";
    return;
  }
  button.disabled = true;
  requestJson("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: textarea.value }),
  })
    .then((data) => {
      preview.innerHTML = data.html || "";
    })
    .catch(() => {
      preview.textContent = textarea.value;
    })
    .finally(() => {
      preview.hidden = false;
      textarea.hidden = true;
      button.textContent = "Edit text";
      button.disabled = false;
    });
}


async function saveInlineEdit(row) {
  const recordId = row.dataset.recordId || "";
  const content = row.querySelector(".inline-edit-textarea").value;
  const saveButton = row.querySelector("[data-action='save-edit']");
  const status = row.querySelector(".inline-edit-status");
  saveButton.disabled = true;
  try {
    await requestJson("/api/index/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record_id: recordId, content }),
    });
    indexRowContent.set(row, content);
    row.querySelector(".index-content-text").textContent = content;
    showToast(`Saved record ${recordId}.`, { kind: "success" });
    endInlineEdit(row);
    markIndexDirty();
  } catch (error) {
    toastError(error);
    if (status) {
      setStatus(status, error.message, true);
    }
  } finally {
    saveButton.disabled = false;
  }
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


function renderIndexRows(rows) {
  const existingTopRows = new Map();
  const existingChildRows = new Map();
  Array.from(els.indexBody.children).forEach((row) => {
    const parentId = row.dataset.parentId;
    if (parentId) {
      const group = existingChildRows.get(parentId) || [];
      group.push(row);
      existingChildRows.set(parentId, group);
      return;
    }
    const recordId = row.dataset.recordId;
    if (recordId) {
      existingTopRows.set(recordId, row);
    }
  });

  const fragment = document.createDocumentFragment();
  rows.forEach((item) => {
    const recordId = String(item.id || "");
    const renderKey = stableJsonHash(item);
    let row = existingTopRows.get(recordId);
    let rowReused = Boolean(row && row.dataset.renderKey === renderKey);
    if (!row || row.dataset.renderKey !== renderKey) {
      row = createIndexRow(item, { allowToggle: true });
      row.dataset.renderKey = renderKey;
      rowReused = false;
    }
    row.dataset.patchKey = `index:${recordId}`;
    fragment.appendChild(row);

    const childRows = rowReused ? existingChildRows.get(recordId) || [] : [];
    if (childRows.length) {
      const firstVisible = !childRows[0].hidden;
      const toggle = row.querySelector("[data-action='toggle-children']");
      if (toggle) {
        setIndexToggle(toggle, firstVisible);
      }
      for (const childRow of childRows) {
        fragment.appendChild(childRow);
      }
    }
  });
  els.indexBody.replaceChildren(fragment);
}


function renderVectorIndexRows(rows) {
  patchTableRows(els.indexBody, rows, {
    keyFor(item) {
      return `vector:${String(item.id || "")}`;
    },
    createRow(item) {
      const row = createIndexRow(item, { allowToggle: false });
      row.classList.add("index-vector-result-row");
      return row;
    },
  });
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
        category: state.indexCategory || "general",
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
    category: state.indexCategory || "general",
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
  if (action === "edit") {
    beginInlineEdit(row);
    return;
  }
  if (action === "cancel-edit") {
    endInlineEdit(row);
    return;
  }
  if (action === "preview-toggle") {
    toggleInlineEditPreview(row);
    return;
  }
  if (action === "save-edit") {
    await saveInlineEdit(row);
    return;
  }
}

export {
  INDEX_CHILD_BATCH_SIZE,
  INDEX_STREAM_BATCH_SIZE,
  abortIndexLoad,
  beginInlineEdit,
  createIndexEmptyChildRow,
  createIndexLoadMoreRow,
  createIndexRow,
  endInlineEdit,
  finishIndexLoad,
  handleIndexAction,
  insertIndexRowsAfter,
  isActiveIndexLoad,
  loadAllIndexSummaries,
  loadIndex,
  loadIndexChildren,
  loadMoreIndexChildren,
  removeIndexLoadMoreRows,
  renderIndexRows,
  renderVectorIndexRows,
  runIndexVectorSearch,
  saveInlineEdit,
  setIndexChildrenVisible,
  setIndexToggle,
  startIndexLoad,
  toggleIndexChildren,
  toggleInlineEditPreview,
};

// -- document-first browsing --------------------------------------------------
// At corpus scale the flat chunk list is unwieldy. Documents mode lists one
// row per source document (from /api/pdfs, which already carries chunk
// counts); expanding fetches that document's records and renders them with
// the same inline editor as chunk mode.

export function reviewViewMode() {
  return state.reviewViewMode || "chunks";
}

export function setReviewViewMode(mode) {
  state.reviewViewMode = mode === "documents" ? "documents" : "chunks";
  document.getElementById("reviewViewChunks").classList.toggle("active", state.reviewViewMode === "chunks");
  document.getElementById("reviewViewDocuments").classList.toggle("active", state.reviewViewMode === "documents");
  document.getElementById("reviewSearchToolbar").hidden = state.reviewViewMode !== "chunks";
  document.getElementById("reviewVectorToolbar").hidden = state.reviewViewMode !== "chunks";
  document.getElementById("docsTable").hidden = state.reviewViewMode !== "documents";
  document.getElementById("indexTable").hidden = state.reviewViewMode !== "chunks";
  document.getElementById("indexStatus").hidden = state.reviewViewMode !== "chunks";
  if (state.reviewViewMode === "documents") {
    abortIndexLoad();
    refreshDocumentList({ force: true });
  } else if (state.indexDirty) {
    loadIndex();
  }
}

let documentsAbort = null;
// Monotonic load token: a superseded load (rapid chunks<->documents toggles)
// must not render its older response over the newer one.
let documentsLoadToken = 0;

export async function refreshDocumentList(options = {}) {
  const body = document.getElementById("docsBody");
  if (!body) {
    return;
  }
  if (!options.force && body.children.length) {
    return;
  }
  if (documentsAbort) {
    documentsAbort.abort();
  }
  documentsAbort = new AbortController();
  const load = { controller: documentsAbort, token: ++documentsLoadToken };
  body.innerHTML = '<tr><td colspan="4" class="library-empty-cell">Loading documents…</td></tr>';
  try {
    const data = await requestJson("/api/pdfs?offset=0&limit=500&sort=filename", {
      signal: documentsAbort.signal,
    });
    if (documentsLoadToken !== load.token || documentsAbort !== load.controller) {
      return; // superseded by a newer load
    }
    renderDocumentList(data.pdfs || [], Number(data.total || 0));
  } catch (error) {
    if (!isAbortError(error)) {
      if (documentsLoadToken !== load.token || documentsAbort !== load.controller) {
        return;
      }
      body.innerHTML = `<tr><td colspan="4" class="library-empty-cell">${escapeHtml(error.message)}</td></tr>`;
    }
  }
}

function renderDocumentList(docs, total) {
  const body = document.getElementById("docsBody");
  body.innerHTML = "";
  if (!docs.length) {
    body.innerHTML = '<tr><td colspan="4" class="library-empty-cell">No documents yet.</td></tr>';
    return;
  }
  const label = document.getElementById("docsPageLabel");
  if (label) {
    label.textContent = `${docs.length} of ${total.toLocaleString()} documents`;
  }
  for (const doc of docs) {
    const row = document.createElement("tr");
    row.className = "doc-row";
    row.dataset.sourceHash = doc.hash || "";
    // Remember the document's category: document_records resolves an empty
    // category to the General index, which would show "No index records" for
    // every document living in a split-database category.
    row.dataset.docCategory = String(doc.category || "general");
    const chunks = Number(doc.quality?.chunk_count || 0);
    const status = String(doc.status || "");
    const group = String(doc.trust?.source_group || "ungrouped");
    row.innerHTML = `
      <td class="doc-expand-cell">
        <button type="button" class="tree-toggle" data-action="toggle-doc" aria-expanded="false" title="Show this document's indexed records">+</button>
      </td>
      <td class="doc-title-cell">
        <strong>${escapeHtml(doc.filename || doc.hash)}</strong>
        <span class="index-node-meta">${escapeHtml(chunks ? chunks + " chunks" : "")}</span>
      </td>
      <td>${escapeHtml(status)}</td>
      <td><span class="source-group-badge group-${escapeHtml(group)}">${escapeHtml(sourceGroupTitle(group))}</span></td>
    `;
    body.appendChild(row);
  }
}

async function toggleDocumentRecords(row, button) {
  const sourceHash = row.dataset.sourceHash || "";
  if (!sourceHash) {
    return;
  }
  const existing = row.nextElementSibling;
  if (existing && existing.classList.contains("doc-records-row")) {
    existing.remove();
    button.textContent = "+";
    button.setAttribute("aria-expanded", "false");
    return;
  }
  button.textContent = "−";
  button.setAttribute("aria-expanded", "true");
  const recordsRow = document.createElement("tr");
  recordsRow.className = "doc-records-row";
  const cell = document.createElement("td");
  cell.colSpan = 4;
  cell.innerHTML = '<p class="hint" style="margin:6px 12px">Loading records…</p>';
  recordsRow.appendChild(cell);
  row.after(recordsRow);
  try {
    const data = await requestJson(
      `/api/index/document_records?source_hash=${encodeURIComponent(sourceHash)}` +
        `&category=${encodeURIComponent(row.dataset.docCategory || "general")}` +
        `&limit=200`,
    );
    cell.innerHTML = "";
    const table = document.createElement("table");
    table.className = "data-table doc-records-table";
    const tbody = document.createElement("tbody");
    for (const item of data.rows || []) {
      const recordRow = createIndexRow(item, { allowToggle: false });
      // Hide the expand (children) cell content — document mode shows a flat
      // record list per document.
      tbody.appendChild(recordRow);
    }
    table.appendChild(tbody);
    cell.appendChild(table);
    if (!(data.rows || []).length) {
      cell.innerHTML = '<p class="hint" style="margin:6px 12px">No index records for this document.</p>';
    } else if (Number(data.total || 0) > (data.rows || []).length) {
      const more = document.createElement("p");
      more.className = "hint";
      more.style.margin = "6px 12px";
      more.textContent = `Showing ${(data.rows || []).length} of ${Number(data.total).toLocaleString()} records.`;
      cell.appendChild(more);
    }
  } catch (error) {
    cell.innerHTML = `<p class="hint" style="margin:6px 12px">${escapeHtml(error.message)}</p>`;
  }
}

export async function handleDocumentListClick(event) {
  const toggle = event.target.closest('[data-action="toggle-doc"]');
  if (!toggle) {
    return;
  }
  const row = toggle.closest("tr.doc-row");
  if (row) {
    await toggleDocumentRecords(row, toggle);
  }
}
