// Documents tab: drop zone, zip handling, source-group staging, uploads.

import { CHUNKED_UPLOAD_CHUNK_SIZE, CHUNKED_UPLOAD_THRESHOLD, SOURCE_GROUP_OPTIONS_HTML, ZIP_IGNORED_NAMES, ZIP_IGNORED_PREFIXES, els, errorFromText, getApiKey, markIndexDirty, parseSourceGroupInput, promptForApiKey, requestJson, setStatus, state } from "./core.js";
import { categoryLabel } from "./categories.js";
import { refreshJobs } from "./status.js";
import { pdfFilesFromList, refreshPdfs } from "./library.js";

function uploadFormData(path, body, options = {}) {
  const onProgress = typeof options.onProgress === "function" ? options.onProgress : null;
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", path);
    const apiKey = getApiKey();
    if (apiKey) {
      request.setRequestHeader("X-API-Token", apiKey);
    }
    request.addEventListener("load", () => {
      const text = request.responseText || "";
      if (request.status === 401 && !options.__apiKeyRetried) {
        promptForApiKey().then((key) => {
          if (key) {
            uploadFormData(path, body, { ...options, __apiKeyRetried: true }).then(resolve, reject);
          } else {
            reject(errorFromText(request.status, request.statusText, text));
          }
        });
        return;
      }
      if (request.status >= 200 && request.status < 300) {
        try {
          resolve(JSON.parse(text || "{}"));
        } catch (_) {
          reject(new Error("Upload response was not valid JSON."));
        }
        return;
      }
      reject(errorFromText(request.status, request.statusText, text));
    });
    request.addEventListener("error", () => {
      reject(new Error("Upload failed. Check the server connection."));
    });
    request.addEventListener("abort", () => {
      const error = new Error("Upload cancelled.");
      error.name = "AbortError";
      reject(error);
    });
    if (onProgress) {
      request.upload.addEventListener("progress", (event) => {
        const total = Number(event.total || 0);
        const loaded = Number(event.loaded || 0);
        const percent = event.lengthComputable && total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : null;
        onProgress({ loaded, total, percent });
      });
    }
    request.send(body);
  });
}


function newUploadId() {
  const bytes = new Uint8Array(16);
  if (window.crypto && crypto.getRandomValues) {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i += 1) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

// Upload a single large file via the chunked/resumable protocol. Splits the
// file into CHUNKED_UPLOAD_CHUNK_SIZE pieces, POSTs each to /api/uploads/chunk,
// then finalizes with /api/uploads/complete. On a 409 offset-mismatch (server
// has more/fewer bytes than expected), re-syncs to the server's offset and
// resumes. Returns the completion response.

async function uploadFileChunked(
  file,
  { sourceGroup = "", category = "", forceDuplicates = false, forceToken = "", onProgress = null } = {},
) {
  const totalSize = file.size;
  const chunkSize = CHUNKED_UPLOAD_CHUNK_SIZE;
  const uploadId = newUploadId();
  let offset = 0;

  while (offset < totalSize) {
    const end = Math.min(offset + chunkSize, totalSize);
    const blob = file.slice(offset, end);
    const body = new FormData();
    body.append("upload_id", uploadId);
    body.append("filename", file.name);
    body.append("offset", String(offset));
    body.append("total_size", String(totalSize));
    body.append("chunk", blob, file.name);
    try {
      const result = await uploadFormData("/api/uploads/chunk", body, {
        onProgress(progress) {
          if (!onProgress || progress.percent === null) return;
          const overall = Math.round(((offset + (progress.loaded || 0)) / totalSize) * 100);
          onProgress({ percent: Math.min(99, overall) });
        },
      });
      offset = Number(result.offset || end);
    } catch (err) {
      if (err && err.status === 409 && err.detail && typeof err.detail.expected_offset === "number") {
        // Re-sync to the server's offset and continue.
        offset = err.detail.expected_offset;
        continue;
      }
      throw err;
    }
  }

  const completeBody = { upload_id: uploadId, filename: file.name };
  if (sourceGroup) completeBody.source_groups = sourceGroup;
  if (category && category !== "general") completeBody.category = category;
  if (forceDuplicates) {
    // Mirrors the single-POST path: without this a forced duplicate of a
    // large file uploads every byte and then fails with 409 at complete.
    completeBody.force_duplicates = "true";
    if (forceToken) completeBody.force_token = forceToken;
  }
  const completeResult = await requestJson("/api/uploads/complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(completeBody),
  });
  if (onProgress) onProgress({ percent: 100 });
  return completeResult;
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
  state.pendingDuplicateFiles = null;
  state.pendingDuplicateSourceGroups = null;
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

// Zip-entry prefixes/names that are never useful PDFs (macOS metadata, etc.).

const fileZipOrigin = new WeakMap();
// Per-File selects, keyed by File identity (not name). Survives basename
// collisions (e.g. two zips each containing report.pdf) and decouples DOM
// render order from the staged FileList order consumed at submit time.

const fileToGroupSelect = new WeakMap();

const zipToDefaultSelect = new WeakMap();

const selectZipOrigin = new WeakMap();


function isIgnoredZipEntry(path) {
  const lower = String(path || "").toLowerCase();
  if (ZIP_IGNORED_NAMES.has(lower)) return true;
  return ZIP_IGNORED_PREFIXES.some((prefix) => lower.startsWith(prefix));
}


function isZipFile(file) {
  const name = (file.name || "").toLowerCase();
  return (
    name.endsWith(".zip") ||
    file.type === "application/zip" ||
    file.type === "application/x-zip-compressed"
  );
}

// Zip files are now sent to the server verbatim and extracted server-side.
// Previously this decompressed the entire archive in browser memory via
// fflate.unzipSync, which OOM'd on multi-GB zips. The server streams the zip to
// disk and extracts PDF entries one at a time (bounded memory). We keep the
// File as-is so the server-side path handles it.

async function extractPdfsFromZip(file) {
  const accepted = [];
  const rejected = [];
  const errors = [];
  const displayName = file.name || "archive.zip";
  accepted.push(file);
  return { accepted, rejected, errors };
}

// Split a selection into PDFs + zip-expanded PDFs. Non-PDF, non-zip files fall
// through to `rejected` exactly as the legacy path did.

function updateSelectedFilesLabel() {
  // The staging card (category + per-file source groups) only makes sense once
  // files exist, so its visibility is driven here where every selection change
  // already flows through.
  const files = Array.from(els.fileInput.files || []);
  if (els.uploadStagingPanel) {
    els.uploadStagingPanel.hidden = files.length === 0;
  }
  const target = categoryLabel(
    (els.uploadCategorySelect && els.uploadCategorySelect.value) || "general",
  );
  if (!files.length) {
    if (els.uploadStagingCount) {
      els.uploadStagingCount.textContent = "";
    }
    els.selectedFilesLabel.textContent = "";
    return;
  }
  const names = files.map((file) => file.name || "unnamed.pdf");
  const shown = names.slice(0, 3).join(", ");
  const extra = names.length > 3 ? ` +${names.length - 3} more` : "";
  els.selectedFilesLabel.textContent = `${names.length} staged: ${shown}${extra}`;
  if (els.uploadStagingCount) {
    els.uploadStagingCount.textContent =
      `${files.length} file${files.length === 1 ? "" : "s"} ready — indexing into “${target}”`;
  }
}

// Option list used by both standalone-PDF and zip-level selectors.

function buildGroupSelect({ inheritDefault = false } = {}) {
  const select = document.createElement("select");
  select.className = "upload-group-select";
  if (inheritDefault) {
    // First option = "inherit from zip". Its concrete value is resolved at
    // submit time via the zip-level selector, so it stays empty here.
    select.innerHTML = `
      <option value="" data-inherit="true">Use group default</option>
      <option value="official" data-override="official">Official</option>
      <option value="student_research" data-override="student_research">Student Research</option>
      <option value="unofficial" data-override="unofficial">Unofficial</option>
    `;
  } else {
    select.innerHTML = SOURCE_GROUP_OPTIONS_HTML;
  }
  return select;
}


function renderUploadGroupSelectors() {
  const files = Array.from(els.fileInput.files || []);
  els.uploadGroupsPanel.innerHTML = "";
  els.uploadGroupsPanel.hidden = files.length === 0;
  if (!files.length) return;

  // Group members by their source zip (preserve selection order). Standalone
  // PDFs land in the `standalone` bucket in input order.
  const standalone = [];
  const zipGroups = new Map(); // zip File -> File[]
  for (const file of files) {
    const zip = fileZipOrigin.get(file);
    if (zip) {
      if (!zipGroups.has(zip)) zipGroups.set(zip, []);
      zipGroups.get(zip).push(file);
    } else {
      standalone.push(file);
    }
  }

  // Render standalone PDFs first (unchanged layout).
  for (const file of standalone) {
    const row = document.createElement("label");
    row.className = "upload-group-row";
    const name = document.createElement("span");
    name.className = "upload-group-name";
    name.textContent = file.name || "unnamed.pdf";
    const select = buildGroupSelect();
    select.dataset.uploadGroup = "true";
    fileToGroupSelect.set(file, select);
    row.append(name, select);
    els.uploadGroupsPanel.appendChild(row);
  }

  // Then one section per zip: a header row ("group for all PDFs in <zip>") and
  // indented member rows whose default option inherits the zip-level choice.
  for (const [zip, members] of zipGroups) {
    const header = document.createElement("div");
    header.className = "upload-zip-header";
    const label = document.createElement("span");
    label.className = "upload-zip-label";
    const count = members.length;
    label.textContent = `${zip.name || "archive.zip"} · ${count} PDF${count === 1 ? "" : "s"}`;
    const defaultSelect = buildGroupSelect();
    defaultSelect.dataset.zipGroupDefault = "true";
    defaultSelect.dataset.zipName = zip.name || "archive.zip";
    zipToDefaultSelect.set(zip, defaultSelect);
    const defaultLabel = document.createElement("span");
    defaultLabel.className = "upload-zip-default-label";
    defaultLabel.textContent = "Group for all:";
    header.append(label, defaultLabel, defaultSelect);
    els.uploadGroupsPanel.appendChild(header);

    let lastRow = header;
    for (const file of members) {
      const row = document.createElement("label");
      row.className = "upload-group-row upload-group-row--member";
      const name = document.createElement("span");
      name.className = "upload-group-name";
      name.textContent = file.name || "unnamed.pdf";
      const select = buildGroupSelect({ inheritDefault: true });
      select.dataset.uploadGroup = "true";
      select.dataset.inheritFrom = "zip";
      fileToGroupSelect.set(file, select);
      // Remember which zip this member inherits from, by identity.
      selectZipOrigin.set(select, zip);
      row.append(name, select);
      lastRow.after(row); // keep members directly under the header, in order
      lastRow = row;
    }
  }
}


function selectedUploadSourceGroups() {
  const files = Array.from(els.fileInput.files || []);
  if (!files.length || !fileToGroupSelect.has(files[0])) {
    return [];
  }

  // Resolve in the exact order files are appended to FormData. Each file maps
  // to its own select by identity; member rows whose value is the "inherit"
  // option resolve their group from the owning zip's header select.
  const groups = [];
  for (const file of files) {
    const select = fileToGroupSelect.get(file);
    if (!select) {
      setStatus(els.uploadStatus, `No source-group selector for ${file.name || "file"}.`, true);
      return null;
    }
    let value = parseSourceGroupInput(select.value);
    let labelSelect = select;
    if (!value && select.dataset.inheritFrom === "zip") {
      const zip = selectZipOrigin.get(select);
      const zipSelect = zip ? zipToDefaultSelect.get(zip) : null;
      if (zipSelect) {
        value = parseSourceGroupInput(zipSelect.value);
        labelSelect = zipSelect;
      }
    }
    if (!value) {
      labelSelect.focus();
      const isMember = select.dataset.inheritFrom === "zip";
      setStatus(
        els.uploadStatus,
        isMember
          ? `Choose a group for ${file.name || "this PDF"}'s archive (or override it individually).`
          : `Choose a source group for ${file.name || "this PDF"}.`,
        true,
      );
      return null;
    }
    groups.push(value);
  }
  return groups;
}


async function setSelectedUploadFiles(files) {
  clearDuplicatePrompt();
  const { accepted, rejected, errors } = await pdfFilesFromList(files);
  if (!accepted.length) {
    const messages = errors.slice();
    if (rejected.length) {
      messages.push(`Skipped non-PDF file(s): ${rejected.join(", ")}`);
    }
    setStatus(
      els.uploadStatus,
      messages.length ? messages.join("\n") : "Drop one or more PDF files.",
      true,
    );
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
  // The staging card (category + source groups) just appeared below the drop
  // zone; bring it into view so the next step of the flow is obvious.
  if (els.uploadStagingPanel && !els.uploadStagingPanel.hidden) {
    els.uploadStagingPanel.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  const messages = errors.slice();
  if (rejected.length) {
    messages.push(`Skipped non-PDF file(s): ${rejected.join(", ")}`);
  }
  if (messages.length) {
    setStatus(els.uploadStatus, messages.join("\n"), true);
  } else {
    setStatus(els.uploadStatus, "");
  }
}


// Discard a staged selection (files picked but not yet sent): clears the file
// input, hides the staging card, and resets the batch target so the next
// upload starts fresh.
function clearStagedUploadFiles() {
  clearDuplicatePrompt();
  els.fileInput.value = "";
  if (els.uploadCategorySelect) {
    els.uploadCategorySelect.value = "general";
  }
  updateSelectedFilesLabel();
  renderUploadGroupSelectors();
  setStatus(els.uploadStatus, "");
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
  // Whole batch targets one category ("split databases"); General is "".
  const uploadCategory =
    (els.uploadCategorySelect && els.uploadCategorySelect.value) || "general";

  els.uploadButton.disabled = true;
  els.forceUploadButton.disabled = true;
  if (els.cancelUploadButton) {
    els.cancelUploadButton.disabled = true;
  }

  // Upload each file as its own request so a network drop only loses the
  // current file, not the whole batch. The server dedupes per-file via the
  // registry, so completed files are not re-uploaded on retry. Each file
  // becomes its own ingest/index job.
  const allJobs = [];
  const errors = [];
  let firstDuplicateDetail = null;
  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    const body = new FormData();
    body.append("files", file);
    if (sourceGroups[index]) {
      body.append("source_groups", sourceGroups[index]);
    }
    if (uploadCategory && uploadCategory !== "general") {
      body.append("category", uploadCategory);
    }
    if (isForced) {
      body.append("force_duplicates", "true");
      if (forceToken) {
        body.append("force_token", forceToken);
      }
    }
    const displayName = file.name || `file ${index + 1}`;
    setStatus(
      els.uploadStatus,
      `Uploading ${index + 1}/${files.length}: ${displayName}...`,
    );
    try {
      let result;
      // Large files use the chunked/resumable path so a network drop only
      // loses the current 16 MiB chunk, not the whole file.
      if (file.size >= CHUNKED_UPLOAD_THRESHOLD) {
        result = await uploadFileChunked(file, {
          sourceGroup: sourceGroups[index] || "",
          category: uploadCategory,
          forceDuplicates: isForced,
          forceToken,
          onProgress(progress) {
            if (progress.percent === null) {
              setStatus(
                els.uploadStatus,
                `Uploading ${index + 1}/${files.length}: ${displayName}...`,
              );
              return;
            }
            setStatus(
              els.uploadStatus,
              `Uploading ${index + 1}/${files.length}: ${displayName} ${progress.percent}%...`,
            );
          },
        });
      } else {
        result = await uploadFormData("/api/uploads", body, {
          onProgress(progress) {
            if (progress.percent === null) {
              setStatus(
                els.uploadStatus,
                `Uploading ${index + 1}/${files.length}: ${displayName}...`,
              );
              return;
            }
            setStatus(
              els.uploadStatus,
              `Uploading ${index + 1}/${files.length}: ${displayName} ${progress.percent}%...`,
            );
          },
        });
      }
      const jobs = Array.isArray(result.jobs) && result.jobs.length ? result.jobs : [result];
      allJobs.push(...jobs);
    } catch (error) {
      if (
        error.status === 409 &&
        error.detail &&
        error.detail.can_force !== false &&
        error.detail.force_token &&
        !firstDuplicateDetail
      ) {
        firstDuplicateDetail = error.detail;
      }
      errors.push(`${displayName}: ${error.message || error.detail || "upload failed"}`);
    }
  }

  els.fileInput.value = "";
  updateSelectedFilesLabel();
  renderUploadGroupSelectors();
  state.jobsOffset = 0;
  state.pdfOffset = 0;

  if (firstDuplicateDetail && allJobs.length === 0) {
    // Keep the selection for the Force button: the input was just cleared and
    // Force re-reads it, so without this stash the button always aborted with
    // "Choose one or more PDF files."
    state.pendingDuplicateFiles = files;
    state.pendingDuplicateSourceGroups = sourceGroups;
    showDuplicatePrompt(firstDuplicateDetail);
    setStatus(
      els.uploadStatus,
      isForced ? "Forced upload was blocked. Review the warning below and retry." : "Duplicate upload blocked.",
      true,
    );
  } else if (errors.length && allJobs.length === 0) {
    clearDuplicatePrompt();
    setStatus(els.uploadStatus, errors.join("; "), true);
  } else {
    clearDuplicatePrompt();
    let msg = `Queued ${allJobs.length} job(s).`;
    if (errors.length) {
      msg += ` ${errors.length} file(s) failed: ${errors.join("; ")}`;
    }
    setStatus(els.uploadStatus, msg, errors.length ? true : false);
  }
  markIndexDirty();
  await refreshJobs({ force: true });
  await refreshPdfs({ force: true });
  els.uploadButton.disabled = false;
  els.forceUploadButton.disabled = !state.pendingForceUploadToken;
  if (els.cancelUploadButton) {
    els.cancelUploadButton.disabled = false;
  }
}


async function enqueueReindex() {
  els.reindexButton.disabled = true;
  setStatus(els.uploadStatus, "Queueing reindex...");
  try {
    const job = await requestJson("/api/reindex", { method: "POST" });
    setStatus(els.uploadStatus, `Queued reindex job ${job.id.slice(0, 8)}.`);
    state.jobsOffset = 0;
    markIndexDirty();
    await refreshJobs({ force: true });
    await refreshPdfs({ force: true });
  } catch (error) {
    setStatus(els.uploadStatus, error.message, true);
  } finally {
    els.reindexButton.disabled = false;
  }
}

export {
  buildGroupSelect,
  clearDuplicatePrompt,
  clearStagedUploadFiles,
  clearUploadDrag,
  duplicateUploadMessage,
  enqueueReindex,
  extractPdfsFromZip,
  fileToGroupSelect,
  fileZipOrigin,
  handleUploadDrag,
  handleUploadDrop,
  isIgnoredZipEntry,
  isZipFile,
  newUploadId,
  renderUploadGroupSelectors,
  selectZipOrigin,
  selectedUploadSourceGroups,
  setSelectedUploadFiles,
  showDuplicatePrompt,
  updateSelectedFilesLabel,
  uploadDragHasFiles,
  uploadFileChunked,
  uploadFiles,
  uploadFormData,
  zipToDefaultSelect,
};
