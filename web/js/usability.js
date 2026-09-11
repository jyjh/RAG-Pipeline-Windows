// Engineer usability helpers, kept out of the per-view modules:
//   * Prompt template library — reusable question scaffolds ({{Placeholder}}
//     filling) seeded with design-engineering starters.
//   * "Ask about this" — select text in any answer, source panel, Review row,
//     or Library text preview and quote it straight into the composer.

import { confirmAction, els, promptText, showToast } from "./core.js";
import { activateTab } from "./shell.js";

const PROMPT_TEMPLATES_STORAGE_KEY = "rag.promptTemplates.v1";

// Seed templates cover the recurring shapes of design work against a
// corpus of rules, lecture notes, and research papers. Seeds are re-added
// (once) whenever the stored list is empty, and deletions always win: a
// seed removed by the user is remembered in deletedSeeds and not re-added.
const SEED_TEMPLATES = [
  {
    id: "seed-compare",
    name: "Compare design options",
    text:
      "Compare {{Option A}} and {{Option B}} for our FSAE EV design. Use only what the library sources support: " +
      "list the trade-offs with numbers where available, note which source says what, and end with a recommendation for our use case.",
  },
  {
    id: "seed-requirements",
    name: "Design requirements digest",
    text:
      "Summarise every design requirement the library imposes on {{component or system}}. Group them into rules " +
      "requirements, physics/handbook guidance, and team good practice, and cite a source for each item.",
  },
  {
    id: "seed-rules",
    name: "Rules check",
    text:
      "What do the rulebook documents in the library say about {{topic}}? List each relevant requirement with its " +
      "specific numbers, cite the document and section it comes from, and flag anything that may come from an outdated edition.",
  },
  {
    id: "seed-concept",
    name: "Explain with the math",
    text:
      "Explain {{concept}} in two passes: first a plain-language intuition for a new team member, then the engineering " +
      "treatment with the key equations (define every symbol) and a worked example with realistic FSAE numbers if the sources contain one.",
  },
  {
    id: "seed-failures",
    name: "Failure modes",
    text:
      "List the failure modes of {{component}} described in the library, their root causes, and the design mitigations " +
      "the sources recommend. Present them as a table with citations.",
  },
  {
    id: "seed-specs",
    name: "Extract specs table",
    text:
      "Extract every numerical specification, formula, and sizing guideline the library gives for {{topic}} into a " +
      "Markdown table (value | condition | source). Note where sources disagree.",
  },
  {
    id: "seed-material",
    name: "Material selection",
    text:
      "I am choosing a material for {{part}}. From the library, compare candidate materials on stiffness, strength, " +
      "weight, cost, and manufacturability, then recommend one with justification and citations.",
  },
];

let templates = [];
let deletedSeedIds = [];
let templatesLoaded = false;

function loadTemplates() {
  if (templatesLoaded) {
    return;
  }
  templatesLoaded = true;
  try {
    const raw = JSON.parse(localStorage.getItem(PROMPT_TEMPLATES_STORAGE_KEY) || "{}");
    templates = Array.isArray(raw.templates) ? raw.templates : [];
    deletedSeedIds = Array.isArray(raw.deletedSeeds) ? raw.deletedSeeds : [];
  } catch (_) {
    templates = [];
    deletedSeedIds = [];
  }
  if (!templates.length) {
    templates = SEED_TEMPLATES.filter((seed) => !deletedSeedIds.includes(seed.id)).map((seed) => ({ ...seed }));
    persistTemplates();
  }
}

function persistTemplates() {
  try {
    localStorage.setItem(
      PROMPT_TEMPLATES_STORAGE_KEY,
      JSON.stringify({ templates, deletedSeeds: deletedSeedIds }),
    );
  } catch (_) {
    // Storage can be unavailable (private mode); the dialog still works live.
  }
}

const PLACEHOLDER_PATTERN = /\{\{([^{}]+)\}\}/g;

function uniquePlaceholders(text) {
  const found = [];
  const seen = new Set();
  for (const match of String(text || "").matchAll(PLACEHOLDER_PATTERN)) {
    const name = match[1].trim();
    if (name && !seen.has(name.toLowerCase())) {
      seen.add(name.toLowerCase());
      found.push(name);
    }
  }
  return found;
}

function fillTemplate(text, values) {
  return String(text || "").replace(PLACEHOLDER_PATTERN, (whole, name) => {
    const value = values[String(name).trim()] ?? "";
    return String(value).trim() || whole;
  });
}

function composeQuestion(text) {
  closeTemplatesDialog();
  activateTab("chat");
  els.questionInput.value = text;
  els.questionInput.dispatchEvent(new Event("input", { bubbles: true }));
  // The dialog's focus trap hands focus back to its opener on close (as a
  // microtask), so claim composer focus once that has settled.
  window.setTimeout(() => {
    els.questionInput.focus();
    const end = els.questionInput.value.length;
    els.questionInput.setSelectionRange(end, end);
  }, 0);
}

// -- templates dialog ---------------------------------------------------------

function openTemplatesDialog() {
  loadTemplates();
  renderTemplatesList();
  els.templatesOverlay.hidden = false;
}

function closeTemplatesDialog() {
  if (!els.templatesOverlay || els.templatesOverlay.hidden) {
    return;
  }
  els.templatesOverlay.hidden = true;
}

function templateRow(template) {
  const row = document.createElement("div");
  row.className = "template-row";
  row.dataset.templateId = template.id;

  const header = document.createElement("div");
  header.className = "template-row-header";
  const name = document.createElement("strong");
  name.textContent = template.name;
  const actions = document.createElement("div");
  actions.className = "template-row-actions";
  const useButton = document.createElement("button");
  useButton.type = "button";
  useButton.textContent = "Use";
  useButton.title = "Put this template into the Ask composer";
  useButton.addEventListener("click", () => useTemplate(template, row));
  const editButton = document.createElement("button");
  editButton.type = "button";
  editButton.textContent = "Edit";
  editButton.addEventListener("click", () => beginTemplateEditor(row, template));
  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.textContent = "Delete";
  deleteButton.addEventListener("click", async () => {
    const ok = await confirmAction("Delete template", `Delete the "${template.name}" template?`, "Delete");
    if (!ok) {
      return;
    }
    templates = templates.filter((entry) => entry.id !== template.id);
    if (template.seedId) {
      deletedSeedIds.push(template.seedId);
    }
    persistTemplates();
    renderTemplatesList();
  });
  actions.append(useButton, editButton, deleteButton);
  header.append(name, actions);

  const preview = document.createElement("p");
  preview.className = "template-preview";
  preview.textContent = template.text;

  row.append(header, preview);
  return row;
}

function useTemplate(template, row) {
  const placeholders = uniquePlaceholders(template.text);
  if (!placeholders.length) {
    composeQuestion(template.text);
    return;
  }
  // Swap the list row for a tiny fill-in form: one input per unique
  // {{placeholder}}, then compose on Insert. Left blank, a placeholder is
  // kept verbatim so the engineer can fill it in the composer.
  let form = row.querySelector(".template-fill");
  if (form) {
    form.remove();
    return;
  }
  form = document.createElement("div");
  form.className = "template-fill";
  const values = {};
  for (const name of placeholders) {
    const field = document.createElement("label");
    field.className = "template-fill-field";
    const labelSpan = document.createElement("span");
    labelSpan.textContent = name;
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = name;
    input.addEventListener("input", () => {
      values[name] = input.value;
    });
    field.append(labelSpan, input);
    form.appendChild(field);
    if (placeholders.indexOf(name) === 0) {
      input.setAttribute("data-autofocus", "true");
    }
  }
  const formActions = document.createElement("div");
  formActions.className = "template-fill-actions";
  const insertButton = document.createElement("button");
  insertButton.type = "button";
  insertButton.className = "primary";
  insertButton.textContent = "Insert into composer";
  insertButton.addEventListener("click", () => composeQuestion(fillTemplate(template.text, values)));
  formActions.appendChild(insertButton);
  form.appendChild(formActions);
  row.appendChild(form);
  const firstInput = form.querySelector("input");
  if (firstInput) {
    firstInput.focus();
  }
}

// Inline editor used for both "Edit" and "New template". Save writes back to
// the store; Cancel just re-renders the list.
function beginTemplateEditor(row, template) {
  const isNew = !template;
  const editing = template || { id: `tpl-${Date.now().toString(36)}`, name: "", text: "" };
  row.classList.add("template-row-editing");
  row.innerHTML = "";

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "template-name-input";
  nameInput.placeholder = "Template name (e.g. Cooling comparison)";
  nameInput.maxLength = 80;
  nameInput.value = editing.name;
  nameInput.setAttribute("data-autofocus", "true");

  const textInput = document.createElement("textarea");
  textInput.className = "template-text-input";
  textInput.rows = 5;
  textInput.placeholder =
    "Question template. Use {{Placeholder}} for parts you fill in each time.";
  textInput.value = editing.text;

  const actions = document.createElement("div");
  actions.className = "template-row-actions";
  const saveButton = document.createElement("button");
  saveButton.type = "button";
  saveButton.className = "primary";
  saveButton.textContent = "Save";
  saveButton.addEventListener("click", () => {
    const name = nameInput.value.trim();
    const text = textInput.value.trim();
    if (!name || !text) {
      showToast("Give the template a name and some text.", { kind: "info" });
      return;
    }
    editing.name = name;
    editing.text = text;
    if (isNew) {
      templates.unshift(editing);
    }
    persistTemplates();
    renderTemplatesList();
    showToast(isNew ? "Template saved." : "Template updated.", { kind: "success" });
  });
  const cancelButton = document.createElement("button");
  cancelButton.type = "button";
  cancelButton.textContent = "Cancel";
  cancelButton.addEventListener("click", () => renderTemplatesList());
  actions.append(saveButton, cancelButton);

  const hint = document.createElement("p");
  hint.className = "template-hint";
  hint.textContent = "Tip: {{Placeholder}} becomes a fill-in field when the template is used.";

  row.append(nameInput, textInput, actions, hint);
  nameInput.focus();
}

function renderTemplatesList() {
  els.templatesList.innerHTML = "";
  // The "New template" editor lives outside the list; any save/cancel path
  // re-renders, so this is also where the stale editor form gets cleared.
  if (els.templateEditorHost) {
    els.templateEditorHost.innerHTML = "";
  }
  if (!templates.length) {
    const empty = document.createElement("p");
    empty.className = "template-hint";
    empty.textContent = "No templates yet — add one below or save the current composer text.";
    els.templatesList.appendChild(empty);
  }
  for (const template of templates) {
    els.templatesList.appendChild(templateRow(template));
  }
}

async function addTemplateFromComposer() {
  const text = els.questionInput.value.trim();
  if (!text) {
    showToast("Write the question in the composer first, then save it as a template.", { kind: "info" });
    return;
  }
  const name = await promptText("Save as template", {
    body: "Name this template so you recognise it later.",
    placeholder: "e.g. Cooling comparison",
    initialValue: "",
  });
  if (name === null) {
    return;
  }
  const trimmedName = String(name).trim();
  if (!trimmedName) {
    showToast("Template name cannot be empty.", { kind: "info" });
    return;
  }
  loadTemplates();
  templates.unshift({ id: `tpl-${Date.now().toString(36)}`, name: trimmedName.slice(0, 80), text });
  persistTemplates();
  showToast("Template saved — find it under Templates.", { kind: "success" });
}

// -- "Ask about this" selection pill ------------------------------------------

const ASK_SELECTION_MAX_CHARS = 1200;
// Containers whose text selections can be quoted into the composer: chat
// answers + source panels, the Library preview's extracted-text mode, and the
// Review chunk/document tables. (PDF-page selections live inside an <iframe>
// and are invisible to the parent document — a documented limitation.)
const ASK_SELECTION_CONTAINERS = ".chat-messages, #pdfPreviewText, #indexBody, #docsBody";

let askSelection = null;

function hideAskSelectionPill() {
  askSelection = null;
  if (els.askSelectionPill && !els.askSelectionPill.hidden) {
    els.askSelectionPill.hidden = true;
  }
}

function showAskSelectionPill(selection, rect) {
  const pill = els.askSelectionPill;
  askSelection = selection;
  pill.hidden = false;
  const width = pill.offsetWidth || 140;
  const height = pill.offsetHeight || 34;
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
  const top = rect.top - height - 8 > 8 ? rect.top - height - 8 : rect.bottom + 8;
  pill.style.left = `${Math.round(left)}px`;
  pill.style.top = `${Math.round(Math.max(8, Math.min(top, window.innerHeight - height - 8)))}px`;
}

function selectionAskContext() {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return null;
  }
  const range = selection.getRangeAt(0);
  const container = range.commonAncestorContainer;
  const element = container.nodeType === Node.TEXT_NODE ? container.parentElement : container;
  if (!element || !element.closest) {
    return null;
  }
  const root = element.closest(ASK_SELECTION_CONTAINERS);
  if (!root) {
    return null;
  }
  const text = selection.toString().replace(/\s+/g, " ").trim();
  if (text.length < 3) {
    return null;
  }
  // A source-item selection can name the document; everything else falls
  // back to the view the selection came from.
  const sourceTitle = element.closest(".source-item")?.querySelector("strong")?.textContent?.trim();
  const contextLabel = sourceTitle || root.closest(".modal-overlay")?.querySelector("strong")?.textContent?.trim() || "the library";
  return { text, rect: range.getBoundingClientRect(), contextLabel };
}

function askAboutSelection() {
  if (!askSelection) {
    return;
  }
  const { text, contextLabel } = askSelection;
  const excerpt = text.length > ASK_SELECTION_MAX_CHARS
    ? `${text.slice(0, ASK_SELECTION_MAX_CHARS)}…`
    : text;
  const quoted = excerpt.split("\n").map((line) => `> ${line}`).join("\n");
  composeQuestion(`From ${contextLabel}, selected excerpt:\n${quoted}\n\n`);
  hideAskSelectionPill();
  try {
    window.getSelection()?.removeAllRanges();
  } catch (_) {
    // Selection already gone; the composer is primed either way.
  }
}

export function initUsabilityHelpers() {
  if (els.templatesButton) {
    els.templatesButton.addEventListener("click", openTemplatesDialog);
  }
  if (els.templatesCloseButton) {
    els.templatesCloseButton.addEventListener("click", closeTemplatesDialog);
  }
  if (els.templatesOverlay) {
    els.templatesOverlay.addEventListener("click", (event) => {
      if (event.target === els.templatesOverlay) {
        closeTemplatesDialog();
      }
    });
  }
  if (els.templateNewButton) {
    els.templateNewButton.addEventListener("click", () => {
      beginTemplateEditor(els.templateEditorHost, null);
    });
  }
  if (els.templateSaveCurrentButton) {
    els.templateSaveCurrentButton.addEventListener("click", addTemplateFromComposer);
  }
  if (els.askSelectionPill) {
    // pointerdown (not click) so pressing the pill never collapses the
    // selection it was spawned from.
    els.askSelectionPill.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      askAboutSelection();
    });
  }
  document.addEventListener("pointerup", () => {
    // Let click-completion settle first, then decide whether the current
    // selection deserves the pill.
    window.setTimeout(() => {
      const context = selectionAskContext();
      if (context) {
        showAskSelectionPill(context, context.rect);
      } else {
        hideAskSelectionPill();
      }
    }, 0);
  });
  document.addEventListener("pointerdown", (event) => {
    if (els.askSelectionPill && !els.askSelectionPill.hidden && !els.askSelectionPill.contains(event.target)) {
      hideAskSelectionPill();
    }
  }, true);
  window.addEventListener("scroll", hideAskSelectionPill, true);
  window.addEventListener("resize", hideAskSelectionPill);
}

export { closeTemplatesDialog, openTemplatesDialog };

export const PROMPT_TEMPLATES_KEY = PROMPT_TEMPLATES_STORAGE_KEY;
