// Source categories: selects, chat chips, admin management (user feature work).

import { confirmAction, els, escapeHtml, requestJson, setStatus, showToast, state } from "./core.js";
import { refreshJobs, updateComposerSettingsSummary } from "./status.js";
import { clearPdfSelection, refreshPdfs } from "./library.js";
import { updateSelectedFilesLabel } from "./upload.js";

async function refreshCategories(options = {}) {
  // Lightweight registry fetch backing every category control (upload target,
  // Library facet, Review picker, Ask chips, Admin manager). Failures leave
  // the current controls alone — category state is an enhancement, never a
  // hard dependency for the rest of the UI.
  try {
    const data = await requestJson("/api/categories");
    state.categoriesCache = Array.isArray(data.categories) ? data.categories : [];
    state.categoriesLoaded = true;
    renderCategoryControls();
    renderAdminCategories();
    relabelCategoryBadges();
    if (options.onLoaded) options.onLoaded();
  } catch (error) {
    if (options.quiet !== false) return;
    setStatus(els.adminCategoriesStatus, error.message, true);
  }
}


function categoryLabel(key) {
  const entry = state.categoriesCache.find((item) => item.key === key);
  return entry ? entry.label || entry.key : key;
}


function customCategoryEntries() {
  return state.categoriesCache.filter((entry) => entry.key !== "general");
}


function categoryHue(key) {
  // Deterministic hue per category key so badges stay recognizable across
  // sessions without storing colors anywhere.
  let hash = 0;
  const text = String(key || "");
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) % 360;
  }
  return hash;
}


function categoryBadgeHtml(key, { interactive = true } = {}) {
  const normalized = String(key || "general");
  const isGeneral = normalized === "general";
  const label = categoryLabel(normalized);
  const attrs = ` class="category-badge${isGeneral ? " category-badge-general" : " category-badge-custom"}"`
    + ` style="--cat-hue:${categoryHue(normalized)}"`
    + ` data-category-key="${escapeHtml(normalized)}"`
    + ` title="${escapeHtml(badgeTitle(normalized, label, interactive))}"`;
  if (!interactive) {
    return `<span${attrs}>${escapeHtml(label)}</span>`;
  }
  return `<button type="button"${attrs} data-category-filter="${escapeHtml(normalized)}">${escapeHtml(label)}</button>`;
}


function badgeTitle(key, label, clickable) {
  if (String(key || "general") === "general") {
    return `Category: ${label} (default index)`;
  }
  return `Category: ${label}${clickable ? " — click to show only this category" : ""}`;
}


// Rows can render before /api/categories resolves (cold boot race), at which
// point badges fall back to the raw key; re-label them once the cache lands.
function relabelCategoryBadges() {
  document.querySelectorAll(".category-badge[data-category-key]").forEach((badge) => {
    const key = badge.dataset.categoryKey || "general";
    const label = categoryLabel(key);
    badge.textContent = label;
    badge.title = badgeTitle(key, label, badge.tagName === "BUTTON");
  });
}


function _populateCategorySelect(select, { value, includeAll = false, allLabel }) {
  if (!select) return;
  const previous = value !== undefined ? value : select.value;
  const options = [];
  if (includeAll) {
    options.push({ value: "all", label: allLabel || "All categories" });
  }
  options.push({ value: "general", label: "General" });
  for (const entry of customCategoryEntries()) {
    options.push({ value: entry.key, label: entry.label || entry.key });
  }
  select.replaceChildren(
    ...options.map((option) => {
      const node = document.createElement("option");
      node.value = option.value;
      node.textContent = option.label;
      return node;
    })
  );
  const values = new Set(options.map((option) => option.value));
  select.value = values.has(previous) ? previous : (includeAll ? "all" : "general");
}


function renderCategoryControls() {
  _populateCategorySelect(els.uploadCategorySelect, {});
  _populateCategorySelect(els.pdfCategoryFilterSelect, {
    value: state.pdfCategoryFilter,
    includeAll: true,
    allLabel: "All categories",
  });
  // Preserve the user's pending "Move to" choice across re-renders instead
  // of snapping back to General before they can click Move.
  _populateCategorySelect(els.pdfBulkCategorySelect, {
    value: (els.pdfBulkCategorySelect && els.pdfBulkCategorySelect.value) || "general",
  });
  _populateCategorySelect(els.indexCategorySelect, { value: state.indexCategory });
  renderChatCategoryChips();
  // Repopulation can change the upload target (e.g. a deleted category
  // falling back to General); keep the drop-zone label honest.
  updateSelectedFilesLabel();
}


function toggleChatCategory(key) {
  const current = new Set(Array.isArray(state.chatSelectedCategories) ? state.chatSelectedCategories : []);
  if (current.has(key)) {
    current.delete(key);
  } else {
    current.add(key);
  }
  // An empty explicit set would search nothing local; treat "untick
  // the last one" as back to All instead of a surprising no-context answer.
  state.chatSelectedCategories = current.size ? Array.from(current) : null;
  renderChatCategoryChips();
  updateComposerSettingsSummary();
}


function renderChatCategoryChips() {
  const container = els.chatCategoryChips;
  if (!container) return;
  const custom = customCategoryEntries();
  const selected = state.chatSelectedCategories;
  const chip = (label, title, active, onClick) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `category-chip${active ? " category-chip-active" : ""}`;
    button.textContent = label;
    button.title = title;
    button.addEventListener("click", onClick);
    return button;
  };
  if (!custom.length) {
    // Only the General index exists, so there is nothing to choose — but
    // showing the scope keeps categorization visible where answers happen.
    container.hidden = false;
    const hint = document.createElement("span");
    hint.className = "category-chip category-chip-static";
    hint.textContent = "Searching: General (all documents)";
    container.replaceChildren(hint);
    return;
  }
  container.hidden = false;
  const allActive = !Array.isArray(selected);
  const nodes = [
    chip(
      allActive ? "All categories" : "All",
      "Search every category (default)",
      allActive,
      () => {
        state.chatSelectedCategories = null;
        renderChatCategoryChips();
        updateComposerSettingsSummary();
      },
    ),
  ];
  // General is a real index like any other: offer it as a chip so a subset
  // selection can keep searching the default corpus (the server searches
  // only the keys the request sends).
  const generalActive = Array.isArray(selected) && selected.includes("general");
  nodes.push(
    chip(
      `${generalActive ? "✓ " : ""}General`,
      'Toggle searching "General" (the default index)',
      generalActive,
      () => toggleChatCategory("general"),
    ),
  );
  for (const entry of custom) {
    const active = Array.isArray(selected) && selected.includes(entry.key);
    nodes.push(
      chip(
        `${active ? "✓ " : ""}${entry.label || entry.key}`,
        `Toggle searching "${entry.label || entry.key}"`,
        active,
        () => toggleChatCategory(entry.key),
      ),
    );
  }
  container.replaceChildren(...nodes);
}


async function bulkMoveSelectedToCategory() {
  const hashes = Array.from(state.selectedPdfHashes);
  const target = (els.pdfBulkCategorySelect && els.pdfBulkCategorySelect.value) || "general";
  if (!hashes.length) {
    setStatus(els.libraryStatus, "Select documents to move first.", true);
    return;
  }
  const ok = await confirmAction(
    `Move ${hashes.length} document(s) to "${categoryLabel(target)}"?`,
    "Vectors are reused, so moving does not re-embed. The documents become searchable under the new category (and no longer under the old one).",
    "Move",
  );
  if (!ok) {
    return;
  }
  try {
    els.pdfBulkMoveButton.disabled = true;
    await requestJson("/api/pdfs/categories/bulk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_hashes: hashes, category: target }),
    });
    showToast(`Moving ${hashes.length} document(s) to ${categoryLabel(target)}.`);
    clearPdfSelection();
    state.jobsOffset = 0;
    await refreshJobs({ force: true });
    await refreshPdfs({ force: true });
    await refreshCategories();
  } catch (error) {
    setStatus(els.libraryStatus, error.message, true);
  } finally {
    els.pdfBulkMoveButton.disabled = false;
  }
}


function renderAdminCategories() {
  const body = els.adminCategoriesBody;
  if (!body) return;
  const entries = state.categoriesCache.length
    ? state.categoriesCache
    : [{ key: "general", label: "General", source_count: 0, record_count: 0, exists: false }];
  body.replaceChildren(
    ...entries.map((entry) => {
      const row = document.createElement("tr");
      const isGeneral = entry.key === "general";
      const embedding = entry.embedding_model
        ? `${escapeHtml(String(entry.embedding_model))} (${Number(entry.embedding_dim || 0)}d)`
        : entry.exists
          ? "unknown"
          : "—";
      row.innerHTML = `
        <td><code>${escapeHtml(entry.key)}</code></td>
        <td>${escapeHtml(entry.label || entry.key)}</td>
        <td>${Number(entry.source_count || 0)}</td>
        <td>${Number(entry.record_count || 0)}</td>
        <td>${embedding}</td>
        <td><input type="number" min="0.01" max="100" step="0.01" value="${Number(entry.weight ?? 1)}" data-category-weight-key="${escapeHtml(entry.key)}" aria-label="Retrieval weight for ${escapeHtml(entry.label || entry.key)}"${isGeneral ? " disabled" : ""} /></td>
        <td>${
          isGeneral
            ? '<span class="hint">default index</span>'
            : `<button type="button" data-category-action="delete" data-category-key="${escapeHtml(entry.key)}">Delete</button>`
        }</td>
      `;
      return row;
    })
  );
}


async function updateCategoryWeight(key, value) {
  try {
    await requestJson(`/api/categories/${encodeURIComponent(key)}/weight`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weight: Number(value) }),
    });
    setStatus(els.adminCategoriesStatus, `Updated retrieval weight for "${categoryLabel(key)}".`);
    await refreshCategories();
  } catch (error) {
    setStatus(els.adminCategoriesStatus, error.message, true);
    await refreshCategories();
  }
}


async function createCategoryFromAdmin() {
  const input = els.adminCategoryKeyInput;
  const name = (input && input.value || "").trim();
  if (!name) {
    setStatus(els.adminCategoriesStatus, "Enter a category name first.", true);
    return;
  }
  try {
    els.adminCategoryCreateButton.disabled = true;
    await requestJson("/api/categories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: name, label: name }),
    });
    input.value = "";
    setStatus(els.adminCategoriesStatus, `Created category "${name}".`);
    await refreshCategories();
  } catch (error) {
    setStatus(els.adminCategoriesStatus, error.message, true);
  } finally {
    els.adminCategoryCreateButton.disabled = false;
  }
}


async function deleteCategoryFromAdmin(key) {
  const ok = await confirmAction(
    `Delete category "${categoryLabel(key)}"?`,
    "Only empty categories can be deleted. Move its documents back to General first (Library → select → Move to). The category's index directory is removed too.",
    "Delete category",
    { danger: true, requireText: "DELETE" },
  );
  if (!ok) {
    return;
  }
  try {
    await requestJson(`/api/categories/${encodeURIComponent(key)}`, { method: "DELETE" });
    setStatus(els.adminCategoriesStatus, `Deleted category "${key}".`);
    await refreshCategories();
  } catch (error) {
    setStatus(els.adminCategoriesStatus, error.message, true);
  }
}

export {
  _populateCategorySelect,
  bulkMoveSelectedToCategory,
  categoryBadgeHtml,
  categoryLabel,
  createCategoryFromAdmin,
  customCategoryEntries,
  deleteCategoryFromAdmin,
  refreshCategories,
  renderAdminCategories,
  renderCategoryControls,
  renderChatCategoryChips,
  updateCategoryWeight,
};
