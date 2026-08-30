// Source categories: selects, chat chips, admin management (user feature work).

import { confirmAction, els, escapeHtml, requestJson, setStatus, showToast, state } from "./core.js";
import { refreshJobs } from "./status.js";
import { clearPdfSelection, refreshPdfs } from "./library.js";

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
  _populateCategorySelect(els.pdfBulkCategorySelect, { value: "general" });
  _populateCategorySelect(els.indexCategorySelect, { value: state.indexCategory });
  renderChatCategoryChips();
}


function renderChatCategoryChips() {
  const container = els.chatCategoryChips;
  if (!container) return;
  const custom = customCategoryEntries();
  if (!custom.length) {
    container.hidden = true;
    container.replaceChildren();
    return;
  }
  container.hidden = false;
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
  const allActive = !Array.isArray(selected);
  const nodes = [
    chip(
      allActive ? "All categories" : "All",
      "Search every category (default)",
      allActive,
      () => {
        state.chatSelectedCategories = null;
        renderChatCategoryChips();
      },
    ),
  ];
  for (const entry of custom) {
    const active = Array.isArray(selected) && selected.includes(entry.key);
    nodes.push(
      chip(
        `${active ? "✓ " : ""}${entry.label || entry.key}`,
        `Toggle searching "${entry.label || entry.key}"`,
        active,
        () => {
          const current = new Set(Array.isArray(state.chatSelectedCategories) ? state.chatSelectedCategories : []);
          if (current.has(entry.key)) {
            current.delete(entry.key);
          } else {
            current.add(entry.key);
          }
          // An empty explicit set would search nothing local; treat "untick
          // the last one" as back to All instead of a surprising no-context answer.
          state.chatSelectedCategories = current.size ? Array.from(current) : null;
          renderChatCategoryChips();
        },
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
  categoryLabel,
  createCategoryFromAdmin,
  customCategoryEntries,
  deleteCategoryFromAdmin,
  refreshCategories,
  renderAdminCategories,
  renderCategoryControls,
  renderChatCategoryChips,
};
