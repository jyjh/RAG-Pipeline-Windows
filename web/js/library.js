// Library tab: PDF trust/review table, facets, preview modal, auto-tag.

import { REVIEWER_NAME_COOKIE, SOURCE_GROUP_LABELS, chooseSourceGroup, confirmAction, els, escapeHtml, inlineNotePrompt, formatBrowserTimestamp, getCookie, markIndexDirty, mediaUrlWithToken, patchTableRows, promptText, requestJson, setCookie, setStatus, showToast, sourceGroupTitle, sourceGroupWeight, sourceTypeTitle, stableJsonHash, state, toastError, updatePageControls, visibleUntaggedRowCheckboxes, renderMarkdown } from "./core.js";
import { _populateCategorySelect, categoryBadgeHtml } from "./categories.js";
import { refreshJobs } from "./status.js";
import { extractPdfsFromZip, isZipFile } from "./upload.js";
import { WALKTHROUGH_FAKE_PDF_HASH, highlightWalkthroughTarget, walkthroughSteps } from "./shell.js";

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


async function ensureReviewerName() {
  let reviewer = saveReviewerName(els.reviewerNameInput.value || getCookie(REVIEWER_NAME_COOKIE));
  if (reviewer) {
    return reviewer;
  }
  const prompted = await promptText("Reviewer name", {
    body: "Record who approved or flagged this source.",
    placeholder: "Your name",
  });
  if (prompted === null) {
    return "";
  }
  reviewer = saveReviewerName(prompted);
  if (!reviewer) {
    setStatus(els.libraryStatus, "Enter a reviewer name before approving or flagging sources.", true);
  }
  return reviewer;
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
    low_chunk_density: "few chunks for its size",
    low_extracted_text: "low text",
    missing_index_manifest: "index details missing",
    missing_markdown: "missing Markdown",
    no_chunks: "no chunks",
    not_indexed: "not indexed",
    job_interrupted: "job interrupted",
    marked_stale: "marked stale",
    rejected_source: "rejected",
    review_expired: "review expired",
    single_chunk: "single chunk",
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
  const checkboxes = visibleUntaggedRowCheckboxes();
  const liveHashes = new Set(checkboxes.map((checkbox) => checkbox.dataset.pdfSelect || ""));
  for (const hash of Array.from(state.selectedPdfHashes)) {
    if (!liveHashes.has(hash)) {
      state.selectedPdfHashes.delete(hash);
    }
  }
  for (const checkbox of checkboxes) {
    const hash = checkbox.dataset.pdfSelect || "";
    const checked = state.selectedPdfHashes.has(hash);
    checkbox.checked = checked;
    const row = checkbox.closest("tr");
    if (row) {
      row.classList.toggle("pdf-row-selected", checked);
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
  try {
    const result = await requestJson("/api/pdfs/trust/bulk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_hashes: hashes, source_group: sourceGroup }),
    });
    let patchedAny = false;
    for (const item of result.updated || []) {
      if (item.pdf) {
        patchedAny = patchPdfRow(item.pdf, { skipSelectionSync: true }) || patchedAny;
      }
    }
    if (patchedAny) {
      syncPdfSelectionAfterRender();
    }
    clearPdfSelection();
    const label = SOURCE_GROUP_LABELS[sourceGroup] || sourceGroup;
    const successCount = (result.updated || []).length;
    const failures = (result.failed || []).map((item) => `${String(item.source_hash || "").slice(0, 8)}: ${item.error}`);
    const summary = `Tagged ${successCount} PDF${successCount === 1 ? "" : "s"} as ${label}.`;
    if (failures.length) {
      showToast(`${summary} ${failures.length} failed: ${failures.join("; ")}`, { kind: "error" });
      setStatus(els.libraryStatus, `${summary} ${failures.length} failed: ${failures.join("; ")}`, true);
    } else {
      showToast(summary, { kind: "success" });
      setStatus(els.libraryStatus, summary);
    }
  } catch (error) {
    setStatus(els.libraryStatus, error.message, true);
  }
  await refreshPdfs({ force: true });
}


// One classification batch takes tens of seconds on a local model, so a
// 200-PDF sweep can run for the better part of an hour. Poll the run status
// far longer than any other UI request before giving up on the report (the
// background run itself is unaffected; this only bounds the button/status).
const AUTO_TAG_POLL_INTERVAL_MS = 3000;
const AUTO_TAG_POLL_MAX_MS = 2 * 60 * 60 * 1000;


async function pollAutoTagRun(queuedCount) {
  const deadline = Date.now() + AUTO_TAG_POLL_MAX_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, AUTO_TAG_POLL_INTERVAL_MS));
    let status;
    try {
      status = await requestJson("/api/pdfs/trust/auto-tag");
    } catch {
      continue; // transient poll failure; the server-side run is unaffected
    }
    if (!status || status.running !== true) {
      return status || null;
    }
    const tagged = Number(status.tagged) || 0;
    setStatus(
      els.libraryStatus,
      `Auto-tagging ${queuedCount} PDF${queuedCount === 1 ? "" : "s"}... ${tagged} tagged so far.`
    );
  }
  return null;
}


async function runAutoTagSweep() {
  if (els.pdfAutoTagButton) {
    els.pdfAutoTagButton.disabled = true;
  }
  try {
    const selected = Array.from(state.selectedPdfHashes);
    const body = selected.length ? { source_hashes: selected } : {};
    setStatus(els.libraryStatus, "Asking the LLM to sort ungrouped PDFs...");
    const result = await requestJson("/api/pdfs/trust/auto-tag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const queued = Array.isArray(result.queued) ? result.queued : [];
    if (!queued.length) {
      showToast(result.message || "No ungrouped PDFs to tag.", { kind: "info" });
      setStatus(els.libraryStatus, result.message || "No ungrouped PDFs to tag.");
      return;
    }
    // total_ungrouped > queued.length means this run hit the per-run cap
    // ([auto_tag].max_items_per_run): say "200 of 327" so the round number
    // reads as a limit, not the whole backlog.
    const totalUngrouped = Number(result.total_ungrouped) || queued.length;
    const scope = totalUngrouped > queued.length
      ? `${queued.length} of ${totalUngrouped} ungrouped PDFs`
      : `${queued.length} PDF${queued.length === 1 ? "" : "s"}`;
    const autoTagMessage = `Auto-tagging ${scope} with ${result.model || "the LLM"}. Rows update as decisions land.`;
    setStatus(els.libraryStatus, autoTagMessage);
    showToast(autoTagMessage, { kind: "info" });
    // The sweep is one background LLM run that can outlive many refreshes;
    // track it to completion so the outcome (or its failures) is visible.
    const finalStatus = await pollAutoTagRun(queued.length);
    await refreshPdfs({ force: true });
    if (!finalStatus) {
      showToast("Auto-tag is still running in the background; refresh later to see the rest.", { kind: "info", timeoutMs: 10000 });
      setStatus(els.libraryStatus, "Auto-tag run still in progress; refresh later for the rest.");
      return;
    }
    const tagged = Number(finalStatus.tagged) || 0;
    const failedBatches = Number(finalStatus.failed_batches) || 0;
    const errorText = String(finalStatus.last_error || "").trim();
    if (errorText) {
      setStatus(
        els.libraryStatus,
        `Auto-tag finished with failures (${tagged} tagged, ${failedBatches} batch${failedBatches === 1 ? "" : "es"} failed): ${errorText}`,
        true
      );
      showToast(`Auto-tag finished with failures: ${tagged} tagged, ${failedBatches} batches failed.`, { kind: "error" });
    } else if (!tagged) {
      setStatus(els.libraryStatus, "Auto-tag finished: no PDFs were tagged (replies were unparseable or below the confidence floor).");
      showToast("Auto-tag finished: nothing was tagged.", { kind: "info" });
    } else {
      const doneMessage = `Auto-tag finished: ${tagged} of ${queued.length} PDF${queued.length === 1 ? "" : "s"} tagged.`;
      setStatus(els.libraryStatus, doneMessage);
      showToast(doneMessage, { kind: "info" });
    }
  } catch (error) {
    setStatus(els.libraryStatus, error.message, true);
  } finally {
    if (els.pdfAutoTagButton) {
      els.pdfAutoTagButton.disabled = false;
    }
  }
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
  const autoTagged = Boolean(trust.auto_tagged);
  const autoTagModel = String(trust.auto_tag_model || "").trim();
  const autoConfidence =
    trust.auto_tag_confidence === null || trust.auto_tag_confidence === undefined || trust.auto_tag_confidence === ""
      ? ""
      : Number(trust.auto_tag_confidence).toFixed(2);
  const autoTagReason = String(trust.auto_tag_reason || "").trim();
  const metrics = [
    Number(data.chunk_count || 0) ? `${Number(data.chunk_count)} chunks` : "",
    Number(data.markdown_char_count || 0) ? `${Number(data.markdown_char_count)} chars` : "",
    Number(data.enrichment_markers || 0) ? `${Number(data.enrichment_markers)} enriched` : "",
  ].filter(Boolean);
  return `
    <span class="quality-badge quality-${escapeHtml(label)}">${escapeHtml(qualityTitle(label))}</span>
    ${sourceGroup === "ungrouped" ? '<span class="quality-badge quality-untagged">Untagged</span>' : ""}
    ${autoTagged ? `<span class="quality-badge quality-auto-tagged" title="Group chosen automatically by ${escapeHtml(autoTagModel || "the LLM")}; a manual tag overrides it">Auto-tagged</span>` : ""}
    <span class="quality-detail">${escapeHtml(metrics.join(" | "))}</span>
    <span class="quality-detail">Trust: ${escapeHtml(trustTitle(trust.review_status))} | ${escapeHtml(sourceTypeTitle(trust.source_type))}</span>
    <span class="quality-detail">Group: ${escapeHtml(sourceGroupTitle(sourceGroup))} | weight ${escapeHtml(reliabilityWeight.toFixed(2))}</span>
    ${autoTagged ? `<span class="quality-detail">Auto: ${escapeHtml(autoTagModel || "LLM")}${autoConfidence ? ` | confidence ${escapeHtml(autoConfidence)}` : ""}${autoTagReason ? ` | ${escapeHtml(autoTagReason)}` : ""}</span>` : ""}
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
      <button type="button" data-pdf-action="preview" data-source-hash="${sourceHash}" data-has-pdf="${item.can_download ? "1" : "0"}" title="Open an in-app preview"${item.can_download || item.markdown_available ? "" : " disabled"}>Preview</button>
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
    ? `<a class="download-link" href="${escapeHtml(mediaUrlWithToken(item.download_url))}">Download</a>`
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
    <td class="pdf-category-cell">${categoryBadgeHtml(item.category || "general")}</td>
    <td>${download}</td>
  `;
  return row;
}


function pdfRowPatchKey(item) {
  return `pdf:${String(item.hash || "")}`;
}


function pdfRowRenderKey(item, options = {}) {
  return stableJsonHash({ item, fake: Boolean(options.fake) });
}


function patchPdfRow(item, options = {}) {
  const key = pdfRowPatchKey(item);
  const existing = els.pdfsBody.querySelector(`tr[data-patch-key="${CSS.escape(key)}"]`);
  if (!existing) {
    return false;
  }
  const row = createPdfRow(item, options);
  row.dataset.patchKey = key;
  row.dataset.renderKey = pdfRowRenderKey(item, options);
  existing.replaceWith(row);
  // Callers patching many rows at once pass skipSelectionSync and run
  // syncPdfSelectionAfterRender() once afterwards; the per-row sync
  // re-queries every checkbox in the table (O(rows) per patch).
  if (!options.skipSelectionSync) {
    syncPdfSelectionAfterRender();
  }
  return true;
}


async function refreshPdfs(options = {}) {
  if (state.activeTab !== "library" && !options.force) {
    state.uploadDataDirty = true;
    return;
  }
  // Stale-response token (see refreshJobs).
  const fetchSeq = ++state.pdfsFetchSeq;
  try {
    const isAll = state.pdfPageSize === "all";
    state.pdfLimit = isAll ? 0 : (Number(state.pdfPageSize) || 10);
    const params = new URLSearchParams({
      offset: String(state.pdfOffset),
      limit: String(state.pdfLimit),
      search: state.pdfSearch,
      source_group: state.pdfGroupFilter || "all",
      trust_status: state.pdfTrustFilter || "all",
      status: state.pdfStatusFilter || "all",
      category: state.pdfCategoryFilter || "all",
      sort: state.pdfSort || "",
    });
    const url = `/api/pdfs?${params}`;
    const data = await requestJson(url);
    if (fetchSeq !== state.pdfsFetchSeq) {
      return;
    }
    state.pdfsLoaded = true;
    if (data.notModified && state.pdfsRenderedUrl === url) {
      return;
    }
    state.pdfTotal = data.total || 0;
    if (!isAll && state.pdfOffset >= state.pdfTotal && state.pdfOffset > 0) {
      state.pdfOffset = Math.max(0, Math.floor((state.pdfTotal - 1) / state.pdfLimit) * state.pdfLimit);
      return refreshPdfs({ force: true });
    }
    renderPdfRows(data.pdfs || []);
    state.pdfsRenderedUrl = url;
    if (isAll) {
      const shown = (data.pdfs || []).length;
      els.pdfPageLabel.textContent = shown < state.pdfTotal
        ? `All ${shown} of ${state.pdfTotal} PDFs`
        : `All ${state.pdfTotal} PDFs`;
      els.prevPdfPageButton.disabled = true;
      els.nextPdfPageButton.disabled = true;
    } else {
      updatePageControls({
        total: state.pdfTotal,
        offset: state.pdfOffset,
        limit: state.pdfLimit,
        label: els.pdfPageLabel,
        prevButton: els.prevPdfPageButton,
        nextButton: els.nextPdfPageButton,
      });
    }
  } catch (error) {
    setStatus(els.libraryStatus, error.message, true);
  }
}


async function handlePdfAction(event) {
  // Category badge in the row: a one-click facet filter for that category.
  const filterBadge = event.target.closest("[data-category-filter]");
  if (filterBadge) {
    const key = filterBadge.dataset.categoryFilter || "general";
    state.pdfCategoryFilter = state.pdfCategoryFilter === key ? "all" : key;
    state.pdfOffset = 0;
    _populateCategorySelect(els.pdfCategoryFilterSelect, {
      value: state.pdfCategoryFilter,
      includeAll: true,
      allLabel: "All categories",
    });
    refreshPdfs({ force: true });
    return;
  }
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
    showToast("This walkthrough row is a local preview and is removed after the step.");
    return;
  }
  if (action === "preview") {
    const row = button.closest("tr");
    const filename = row?.querySelector("strong")?.textContent || sourceHash.slice(0, 12);
    openPdfPreview(sourceHash, filename, { preferText: button.dataset.hasPdf !== "1" });
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
    body.notes = (await inlineNotePrompt(button, {
      title: "Why is this source stale?",
      placeholder: "Notes for future reviewers",
    })) || "";
  } else if (action === "reprocess") {
    const ok = await confirmAction(
      "Re-run this source?",
      "Re-runs ingestion (PDF extraction) and then indexing for this one source as a background job.",
      "Re-run",
    );
    if (!ok) {
      return;
    }
  } else if (action === "reindex") {
    const ok = await confirmAction(
      "Re-index this source?",
      "Re-runs indexing without re-running ingestion (reuses the extracted Markdown).",
      "Re-index",
    );
    if (!ok) {
      return;
    }
  } else if (action === "delete") {
    const ok = await confirmAction(
      "Delete this source?",
      "Permanently removes the uploaded PDF, its processed Markdown, extracted assets, and all of its index records. This cannot be undone.",
      "Delete source",
      { danger: true, requireText: "DELETE" },
    );
    if (!ok) {
      return;
    }
  } else {
    return;
  }
  if (action === "approve" || action === "stale") {
    const reviewer = await ensureReviewerName();
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
      showToast(`Re-run queued (job ${String(job.id || "").slice(0, 8)}).`, { kind: "success" });
      markIndexDirty();
      await refreshJobs({ force: true });
      await refreshPdfs({ force: true });
    } else if (action === "reindex") {
      const job = await requestJson(`/api/pdfs/${encodeURIComponent(sourceHash)}/reindex`, {
        method: "POST",
      });
      showToast(`Re-index queued (job ${String(job.id || "").slice(0, 8)}).`, { kind: "success" });
      markIndexDirty();
      await refreshJobs({ force: true });
      await refreshPdfs({ force: true });
    } else if (action === "delete") {
      const result = await requestJson(`/api/pdfs/${encodeURIComponent(sourceHash)}`, {
        method: "DELETE",
      });
      const deletedVectors = Number(result.vectors?.deleted || 0);
      showToast(
        `Deleted source ${sourceHash.slice(0, 8)} and ${deletedVectors} index record${deletedVectors === 1 ? "" : "s"}.`,
        { kind: "success" },
      );
      markIndexDirty();
      await refreshJobs({ force: true });
      await refreshPdfs({ force: true });
    } else {
      const result = await requestJson(`/api/pdfs/${encodeURIComponent(sourceHash)}/trust`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (result.pdf) {
        patchPdfRow(result.pdf);
      }
      if (action === "approve") {
        showToast(`Marked “${result.pdf?.filename || sourceHash.slice(0, 8)}” as approved.`, { kind: "success" });
      } else if (action === "stale") {
        showToast(`Flagged “${result.pdf?.filename || sourceHash.slice(0, 8)}” as stale.`, { kind: "success" });
      }
      if (action === "tag-group") {
        await refreshPdfs({ force: true });
      }
    }
  } catch (error) {
    toastError(error);
    setStatus(els.libraryStatus, error.message, true);
  } finally {
    button.disabled = false;
  }
}

// Poll backoff: consecutive failed polls stretch the interval so a down or
// restarting server isn't hammered every 2s indefinitely; the first success
// restores the normal cadence.

async function pdfFilesFromList(files) {
  const accepted = [];
  const rejected = [];
  const errors = [];
  for (const file of Array.from(files || [])) {
    const name = file.name || "";
    const isPdf = file.type === "application/pdf" || name.toLowerCase().endsWith(".pdf");
    if (isPdf) {
      accepted.push(file);
    } else if (isZipFile(file)) {
      const result = await extractPdfsFromZip(file);
      accepted.push(...result.accepted);
      rejected.push(...result.rejected);
      errors.push(...result.errors);
    } else {
      rejected.push(name || "unnamed file");
    }
  }
  return { accepted, rejected, errors };
}


let pdfPreviewActive = { sourceHash: "", mode: "pdf" };

export function setPdfPreviewMode(mode) {
  pdfPreviewActive.mode = mode;
  const showPdf = mode === "pdf";
  els.pdfPreviewFrame.hidden = !showPdf;
  els.pdfPreviewText.hidden = showPdf;
  els.pdfPreviewModePdf.classList.toggle("active", showPdf);
  els.pdfPreviewModeText.classList.toggle("active", !showPdf);
  if (!showPdf && pdfPreviewActive.sourceHash && !els.pdfPreviewText.dataset.loaded) {
    // Capture the hash: a slow fetch must never render document A's text
    // under document B's title after the preview target changed.
    const requestedHash = pdfPreviewActive.sourceHash;
    els.pdfPreviewText.textContent = "Loading extracted text…";
    requestJson(`/api/pdfs/${encodeURIComponent(requestedHash)}/markdown`)
      .then(async (payload) => {
        if (pdfPreviewActive.sourceHash !== requestedHash || !els.pdfPreviewOverlay || els.pdfPreviewOverlay.hidden) {
          return;
        }
        if (payload && typeof payload.markdown === "string") {
          els.pdfPreviewText.innerHTML = await renderMarkdown(payload.markdown);
        } else {
          els.pdfPreviewText.textContent = "";
        }
        els.pdfPreviewText.dataset.loaded = "1";
      })
      .catch((error) => {
        if (pdfPreviewActive.sourceHash !== requestedHash || !els.pdfPreviewOverlay || els.pdfPreviewOverlay.hidden) {
          return;
        }
        els.pdfPreviewText.textContent = error.message;
      });
  }
}

function openPdfPreview(sourceHash, filename, { preferText = false } = {}) {
  if (!els.pdfPreviewOverlay) {
    return;
  }
  pdfPreviewActive = { sourceHash, mode: preferText ? "text" : "pdf" };
  els.pdfPreviewTitle.textContent = `Preview — ${filename}`;
  els.pdfPreviewDownloadLink.href = mediaUrlWithToken(`/api/pdfs/${encodeURIComponent(sourceHash)}/download`);
  els.pdfPreviewFrame.src = preferText ? "about:blank" : mediaUrlWithToken(`/api/pdfs/${encodeURIComponent(sourceHash)}/view`);
  delete els.pdfPreviewText.dataset.loaded;
  els.pdfPreviewText.innerHTML = "";
  setPdfPreviewMode(preferText ? "text" : "pdf");
  els.pdfPreviewFallback.hidden = true;
  els.pdfPreviewOverlay.hidden = false;
}

function closePdfPreview() {
  if (!els.pdfPreviewOverlay || els.pdfPreviewOverlay.hidden) {
    return;
  }
  // Drop the document from the iframe so a hidden multi-MB PDF is not kept.
  els.pdfPreviewFrame.src = "about:blank";
  els.pdfPreviewFrame.hidden = true;
  els.pdfPreviewText.innerHTML = "";
  delete els.pdfPreviewText.dataset.loaded;
  els.pdfPreviewOverlay.hidden = true;
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
  const rows = [];
  state.walkthroughFakePdfVisible = false;
  if (state.walkthroughFakePdfPinned) {
    rows.push({ item: walkthroughFakePdfItem(), fake: true });
    state.walkthroughFakePdfVisible = true;
  }
  for (const item of items) {
    rows.push({ item, fake: false });
  }
  if (!rows.length) {
    const emptyRow = document.createElement("tr");
    emptyRow.className = "library-empty-state";
    const detail = state.pdfSearch || state.pdfGroupFilter !== "all" || state.pdfTrustFilter !== "all" || state.pdfStatusFilter !== "all" || state.pdfCategoryFilter !== "all"
      ? "No PDFs match the current search or filters."
      : "No PDFs yet — upload some on the Documents tab.";
    emptyRow.innerHTML = `<td colspan="6" class="library-empty-cell">${escapeHtml(detail)}</td>`;
    els.pdfsBody.replaceChildren(emptyRow);
    return;
  }
  patchTableRows(els.pdfsBody, rows, {
    keyFor(entry) {
      return pdfRowPatchKey(entry.item);
    },
    renderKeyFor(entry) {
      return pdfRowRenderKey(entry.item, { fake: entry.fake });
    },
    createRow(entry) {
      return createPdfRow(entry.item, { fake: entry.fake });
    },
  });
  if (state.walkthroughFakePdfPinned) {
    const step = walkthroughSteps[state.walkthroughIndex];
    if (step) {
      window.requestAnimationFrame(() => highlightWalkthroughTarget(step.target));
    }
  }
  syncPdfSelectionAfterRender();
}

export {
  applyBulkTagGroup,
  clearPdfSelection,
  closePdfPreview,
  createPdfRow,
  ensureReviewerName,
  ensureWalkthroughFakePdf,
  handlePdfAction,
  loadReviewerName,
  normalizeReviewerName,
  openPdfPreview,
  patchPdfRow,
  pdfFilesFromList,
  pdfRowPatchKey,
  pdfRowRenderKey,
  qualityTitle,
  qualityWarnings,
  refreshPdfs,
  removeWalkthroughFakePdf,
  renderPdfActions,
  renderPdfInterruptedBadge,
  renderPdfRows,
  renderQualityCell,
  runAutoTagSweep,
  saveReviewerName,
  syncPdfSelectAllState,
  syncPdfSelectionAfterRender,
  togglePdfSelection,
  trustTitle,
  updatePdfBulkBar,
  walkthroughFakePdfItem,
};

// -- library column sort ------------------------------------------------------
// PDF/Status/Quality headers toggle filename/trust/group sorts. The third
// click clears back to the default (ungrouped-first) order.

function updateSortIndicators() {
  document.querySelectorAll("#pdfsTable .th-sort").forEach((button) => {
    const key = button.dataset.sortKey || "";
    const indicator = button.querySelector(".sort-indicator");
    const active = state.pdfSort === key || state.pdfSort === `-${key}`;
    button.classList.toggle("sort-active", active);
    if (!indicator) {
      return;
    }
    indicator.textContent = active ? (state.pdfSort.startsWith("-") ? "↓" : "↑") : "";
  });
}

export function handleLibrarySortClick(event) {
  const button = event.target.closest(".th-sort[data-sort-key]");
  if (!button) {
    return;
  }
  const key = button.dataset.sortKey;
  if (state.pdfSort === key) {
    state.pdfSort = `-${key}`;
  } else if (state.pdfSort === `-${key}`) {
    state.pdfSort = "";
  } else {
    state.pdfSort = key;
  }
  updateSortIndicators();
  state.pdfOffset = 0;
  refreshPdfs({ force: true });
}

// -- bulk destructive/long operations ----------------------------------------
// Client-side sequential loops over the per-source endpoints: bounded by the
// selection size, each step reports progress, and the whole run is guarded by
// a typed confirmation.

async function bulkPdfOperation(hashes, runOne, label) {
  let ok = 0;
  const failures = [];
  for (const hash of hashes) {
    try {
      await runOne(hash);
      ok += 1;
      showToast(`${label} ${ok}/${hashes.length} done.`, { kind: "info", timeoutMs: 2500 });
    } catch (error) {
      failures.push(`${hash.slice(0, 8)}: ${error.message}`);
    }
  }
  if (failures.length) {
    showToast(`${label} finished with ${failures.length} failure(s): ${failures.slice(0, 3).join("; ")}`, { kind: "error" });
  } else {
    showToast(`${label} finished (${ok}).`, { kind: "success" });
  }
  clearPdfSelection();
  await refreshPdfs({ force: true });
}

export async function bulkDeleteSelected() {
  const hashes = Array.from(state.selectedPdfHashes);
  if (!hashes.length) {
    return;
  }
  const confirmed = await confirmAction(
    `Delete ${hashes.length} source(s)?`,
    "Permanently removes each selected PDF with its processed Markdown, extracted assets, and index records. This cannot be undone.",
    "Delete all",
    { danger: true, requireText: "DELETE" },
  );
  if (!confirmed) {
    return;
  }
  await bulkPdfOperation(
    hashes,
    (hash) => requestJson(`/api/pdfs/${encodeURIComponent(hash)}`, { method: "DELETE" }),
    "Delete",
  );
  markIndexDirty();
  await refreshJobs({ force: true });
}

export async function bulkRerunSelected() {
  const hashes = Array.from(state.selectedPdfHashes);
  if (!hashes.length) {
    return;
  }
  const confirmed = await confirmAction(
    `Re-run ${hashes.length} source(s)?`,
    "Re-runs ingestion and indexing for each selected source as sequential background jobs.",
    "Re-run all",
  );
  if (!confirmed) {
    return;
  }
  await bulkPdfOperation(
    hashes,
    (hash) => requestJson(`/api/pdfs/${encodeURIComponent(hash)}/reprocess`, { method: "POST" }),
    "Re-run",
  );
  markIndexDirty();
  await refreshJobs({ force: true });
}
