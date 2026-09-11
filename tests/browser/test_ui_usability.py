"""Regression tests for chat, toast, category UI, and saved-chat fixes.

Read-only against the server: state changes happen only in the throwaway
browser context (seeded localStorage, in-page module calls, and network
stubs scoped to the test's own context). Run via `python -m pytest tests/browser -q`.
"""
from __future__ import annotations

import json
import re

import pytest

from helpers import activate_tab  # noqa: E402  (tests/browser is a sys.path root)

pytestmark = pytest.mark.browser

CATEGORY_ENTRIES = [
    {"key": "general", "label": "General", "source_count": 2, "record_count": 10},
    {"key": "design-docs", "label": "Design Docs", "source_count": 1, "record_count": 4},
]


def _pdf_row(hash_: str, filename: str, category: str) -> dict:
    return {
        "hash": hash_,
        "filename": filename,
        "status": "indexed",
        "category": category,
        "download_url": f"/api/pdfs/{hash_}/download",
        "trust": {"source_group": "official", "review_status": "unreviewed"},
        "quality": {
            "label": "ready",
            "chunk_count": 3,
            "markdown_char_count": 100,
            "enrichment_markers": 0,
            "warnings": [],
        },
    }


def _stub_categories(page, entries) -> None:
    page.route(
        re.compile(r"/api/categories/?$"),
        lambda route: route.fulfill(
            content_type="application/json",
            json={"categories": entries, "total_sources": sum(e.get("source_count", 0) for e in entries)},
        ),
    )


def _stub_pdfs(page, rows) -> None:
    def fulfill(route):
        route.fulfill(
            content_type="application/json",
            json={"pdfs": rows, "total": len(rows), "offset": 0, "limit": 10},
        )

    page.route(re.compile(r"/api/pdfs\?"), fulfill)


def _seed_chat_with_saved_answer(page) -> None:
    """Seed a saved chat whose stored answerHtml already contains a linked
    citation (exactly what addAssistantMessageToChat persists)."""
    payload = {
        "activeChatId": "test-chat-1",
        "chats": [
            {
                "id": "test-chat-1",
                "title": "Citation test",
                "createdAt": "2026-08-31T00:00:00Z",
                "updatedAt": "2026-08-31T00:00:00Z",
                "messages": [
                    {"role": "user", "text": "What is downforce?"},
                    {
                        "role": "assistant",
                        "text": "lift [S5] and drag [S7].",
                        "answerHtml": (
                            '<p>lift <a class="citation-link" href="#" data-citation="[S5]"'
                            ' title="Jump to source [S5]">[S5]</a> and drag [S7].</p>'
                        ),
                        "sources": [],
                    },
                ],
            }
        ],
    }
    page.evaluate(
        "payload => localStorage.setItem('rag.chatHistory.v1', JSON.stringify(payload))",
        payload,
    )
    page.reload()
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(400)


def test_saved_chat_citations_do_not_nest(page):
    _seed_chat_with_saved_answer(page)
    links = page.locator("#chatMessages a.citation-link")
    # [S5] was already linked in the stored HTML; [S7] must be linked exactly
    # once by linkAnswerCitations; neither may end up inside another anchor.
    assert links.count() == 2, "expected one link per citation label"
    assert page.locator("#chatMessages a.citation-link a").count() == 0
    assert page.locator('#chatMessages a.citation-link[data-citation="[S5]"]').count() == 1
    assert page.locator('#chatMessages a.citation-link[data-citation="[S7]"]').count() == 1


def test_job_failure_burst_coalesces_into_one_toast(page):
    page.evaluate(
        """
        async () => {
          const status = await import('/static/js/status.js');
          const err = (
            'Background pipeline command failed with exit code 1: '
            + 'D:\\\\x\\\\.venv\\\\Scripts\\\\python.exe D:\\\\x\\\\main.py --mode index'
          );
          for (let i = 0; i < 5; i++) {
            status.notifyJobOutcome(`deadbeef${i}000000`, 'failed', {
              filenames: [`report-${i}.pdf`],
              error: err,
            });
          }
        }
        """
    )
    page.wait_for_timeout(2000)  # > FAILED_TOAST_COALESCE_MS
    toasts = page.locator("#toastStack .toast-error")
    assert toasts.count() == 1, "a failure burst must produce one summary toast"
    assert "5 jobs failed" in toasts.first.inner_text()


def test_single_job_failure_toast_keeps_short_detail(page):
    page.evaluate(
        """
        async () => {
          const status = await import('/static/js/status.js');
          status.notifyJobOutcome('cafe123400000000', 'failed', {
            filenames: ['Welding of Aluminum - Mathers.pdf'],
            error: 'Ollama embedding preflight failed: ' + 'x'.repeat(400),
          });
        }
        """
    )
    page.wait_for_timeout(2000)
    toasts = page.locator("#toastStack .toast-error")
    assert toasts.count() == 1
    text = toasts.first.inner_text()
    assert text.startswith("Welding of Aluminum - Mathers.pdf failed:")
    assert len(text) < 200, "failure toasts must not ship the whole error line"


def test_new_chat_reuses_empty_active_chat(page):
    activate_tab(page, "chat")
    list_rows = page.locator("#savedChatsList .saved-chat-row")
    new_chat = page.locator("#newChatButton")

    # The fresh context starts with a single empty chat; clicking again must
    # reuse it instead of stacking another empty "New chat" entry.
    new_chat.click()
    page.wait_for_timeout(200)
    assert list_rows.count() == 1
    new_chat.click()
    page.wait_for_timeout(200)
    assert list_rows.count() == 1

    # Once the chat has content, a new chat is created for real. The message
    # is injected into the in-memory state so the test never calls the LLM.
    page.evaluate(
        """
        async () => {
          const core = await import('/static/js/core.js');
          const active = core.state.chats.find((c) => c.id === core.state.activeChatId);
          active.messages.push({ role: 'user', text: 'seed message' });
        }
        """
    )
    new_chat.click()
    page.wait_for_timeout(300)
    assert list_rows.count() == 2


def test_library_category_column_badges_sort_and_click_filter(page):
    _stub_categories(page, CATEGORY_ENTRIES)
    _stub_pdfs(
        page,
        [
            _pdf_row("a" * 64, "Design Brief.pdf", "design-docs"),
            _pdf_row("b" * 64, "Textbook.pdf", "general"),
        ],
    )
    # Reload so the boot-time /api/categories fetch runs against the stub and
    # races the first Library render, exactly like a cold start.
    page.reload()
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(400)
    activate_tab(page, "library")
    page.wait_for_selector("#pdfsBody .category-badge")

    # Sortable Category column exists.
    assert page.locator("#pdfsTable .th-sort[data-sort-key='category']").count() == 1

    # Badges may initially show the raw key until /api/categories lands; the
    # relabel pass must resolve them to their labels.
    page.wait_for_function(
        "() => document.querySelector('#pdfsBody .category-badge-custom')?.textContent === 'Design Docs'"
    )
    badges = page.locator("#pdfsBody .category-badge")
    assert badges.count() == 2
    custom = page.locator("#pdfsBody .category-badge-custom")
    assert custom.count() == 1
    assert custom.first.inner_text() == "Design Docs"
    general = page.locator("#pdfsBody .category-badge-general")
    assert general.count() == 1
    assert general.first.inner_text() == "General"

    # Clicking a badge facets the table to that category and syncs the select.
    custom.first.click()
    page.wait_for_timeout(400)
    assert page.locator("#pdfCategoryFilterSelect").input_value() == "design-docs"


def test_ask_category_chips_offer_general_and_summary_scope(page):
    _stub_categories(page, CATEGORY_ENTRIES)
    activate_tab(page, "chat")
    # Boot may render the single-category hint first; wait for the stubbed
    # category set to land.
    page.wait_for_function(
        "() => document.querySelectorAll('#chatCategoryChips .category-chip').length === 3"
    )

    labels = [
        page.locator("#chatCategoryChips .category-chip").nth(i).inner_text()
        for i in range(page.locator("#chatCategoryChips .category-chip").count())
    ]
    assert labels == ["All categories", "General", "Design Docs"]

    # Selecting a subset narrows the search and is reflected in the summary.
    page.locator("#chatCategoryChips .category-chip").filter(has_text="Design Docs").click()
    page.wait_for_timeout(200)
    assert page.locator(
        "#chatCategoryChips .category-chip"
    ).filter(has_text="✓ Design Docs").count() == 1
    assert page.locator("#composerSettingsSummary").inner_text().endswith("1/2 categories")

    # Back to everything: no scope suffix. (The All chip shortens to "All"
    # while a subset is active, so target it by its stable title.)
    page.locator(
        "#chatCategoryChips .category-chip[title='Search every category (default)']"
    ).click()
    page.wait_for_timeout(200)
    assert not page.locator("#composerSettingsSummary").inner_text().endswith("categories")


def test_ask_chips_show_scope_when_only_general_exists(page):
    # Single-category deployments hide nothing: the composer states its scope.
    _stub_categories(page, [CATEGORY_ENTRIES[0]])
    activate_tab(page, "chat")
    hint = page.locator("#chatCategoryChips .category-chip-static")
    hint.wait_for(state="visible")
    assert "General" in hint.inner_text()


def test_upload_cancel_button_clears_staged_files(page):
    """The staging card's Cancel discards a picked-but-not-sent selection
    without a page refresh (previously refreshing the tab was the only way
    out of a mistaken pick)."""
    activate_tab(page, "upload")
    staging = page.locator("#uploadStagingPanel")
    assert staging.is_hidden()

    page.set_input_files(
        "#fileInput",
        [
            {"name": "mistake.pdf", "mimeType": "application/pdf", "buffer": b"%PDF-1.4\n"},
            {"name": "keep.pdf", "mimeType": "application/pdf", "buffer": b"%PDF-1.4\n"},
        ],
    )
    staging.wait_for(state="visible")
    assert "2 staged" in page.locator("#selectedFilesLabel").inner_text()

    page.locator("#cancelUploadButton").click()

    staging.wait_for(state="hidden")
    assert page.locator("#selectedFilesLabel").inner_text().strip() == ""
    assert page.locator("#uploadGroupsPanel").is_hidden()
    assert page.eval_on_selector("#fileInput", "el => el.files.length") == 0
    # The batch target resets so the next upload starts fresh.
    assert page.locator("#uploadCategorySelect").input_value() == "general"


# ---------------------------------------------------------------------------
# Engineer usability: templates, follow-up chips, feedback, recent questions,
# and ask-about-selection. All server calls are stubbed per context.
# ---------------------------------------------------------------------------


def test_question_templates_dialog_fills_composer(page):
    activate_tab(page, "chat")
    page.locator("#promptTemplatesButton").click()
    page.wait_for_selector("#templatesOverlay .template-row", state="visible")
    assert page.locator("#templatesList .template-row").count() >= 7, "seed templates must be listed"

    # "Compare design options" carries two {{...}} placeholders, so Use first
    # opens the fill-in form and Insert composes the final prompt.
    row = page.locator("#templatesList .template-row", has_text="Compare design options")
    row.get_by_role("button", name="Use").click()
    fill_form = row.locator(".template-fill")
    fill_form.wait_for(state="visible")
    fill_form.get_by_role("textbox").nth(0).fill("steel")
    fill_form.get_by_role("textbox").nth(1).fill("aluminium")
    fill_form.get_by_role("button", name="Insert into composer").click()

    value = page.locator("#questionInput").input_value()
    assert "Compare steel and aluminium" in value, "placeholders must be replaced"
    assert "{{" not in value
    assert page.locator("#templatesOverlay").is_hidden(), "dialog closes on insert"
    # The chat tab is the ask surface; it must be front and the composer focused.
    assert page.locator("#chat.panel").is_visible()
    assert page.evaluate("() => document.activeElement && document.activeElement.id") == "questionInput"


def test_template_new_and_delete_roundtrip(page):
    activate_tab(page, "chat")
    page.locator("#promptTemplatesButton").click()
    page.wait_for_selector("#templatesOverlay .template-row", state="visible")

    page.locator("#templateNewButton").click()
    name_input = page.locator("#templateEditorHost .template-name-input")
    name_input.wait_for(state="visible")
    name_input.fill("Upright check")
    page.locator("#templateEditorHost .template-text-input").fill("Summarise upright design guidance.")
    page.locator("#templateEditorHost").get_by_role("button", name="Save").click()

    saved_row = page.locator("#templatesList .template-row", has_text="Upright check")
    saved_row.wait_for(state="visible")
    # Persisted in localStorage, so the dialog reopens with it still present.
    page.locator("#templatesCloseButton").click()
    page.locator("#promptTemplatesButton").click()
    page.wait_for_selector("#templatesOverlay .template-row", state="visible")
    assert page.locator("#templatesList .template-row", has_text="Upright check").count() == 1

    # Deletion goes through the in-DOM confirmAction dialog (not a native one).
    saved_row.get_by_role("button", name="Delete").click()
    confirm = page.locator(".modal-actions button", has_text="Delete")
    confirm.wait_for(state="visible")
    confirm.click()
    page.wait_for_timeout(300)
    assert page.locator("#templatesList .template-row", has_text="Upright check").count() == 0


def _stub_ask_pipeline(page) -> dict:
    """Stub chat streaming, followups, and feedback; capture POST bodies."""
    captured: dict = {"feedback": [], "questions": []}

    def fulfill_stream(route):
        body = (
            json.dumps(
                {
                    "type": "sources",
                    "sources": [
                        {
                            "id": "S1",
                            "label": "[S1]",
                            "kind": "local",
                            "title": "Aero notes",
                            "source_group": "official",
                            "snippet": "Drag rises with the square of speed.",
                        }
                    ],
                }
            )
            + "\n"
            + json.dumps({"type": "answer", "text": "Downforce rises roughly with the square of speed [S1]."})
            + "\n"
        )
        route.fulfill(content_type="application/x-ndjson", body=body)

    def capture_feedback(route):
        captured["feedback"].append(route.request.post_data or "")
        route.fulfill(content_type="application/json", json={"ok": True, "record": {}})

    page.route(re.compile(r"/api/chat/stream$"), fulfill_stream)
    page.route(
        re.compile(r"/api/chat/followups$"),
        lambda route: route.fulfill(
            content_type="application/json",
            json={"suggestions": ["How does wing angle change the coefficient?", "What corner speed does the library assume?"]},
        ),
    )
    page.route(re.compile(r"/api/feedback$"), capture_feedback)
    return captured


def test_followup_chips_and_feedback_flow(page):
    captured = _stub_ask_pipeline(page)
    activate_tab(page, "chat")
    page.locator("#questionInput").fill("How does downforce scale with speed?")
    page.locator("#sendButton").click()

    # Follow-ups arrive after the answer and render as clickable chips.
    page.wait_for_selector(".followup-chip", state="visible")
    assert page.locator(".followup-chip").count() == 2

    # Feedback: a 👍 records the rating and marks the message voted.
    page.locator(".feedback-button", has_text="Helpful").first.click()
    page.wait_for_selector(".feedback-bar.feedback-voted", state="visible")
    page.wait_for_timeout(300)
    assert captured["feedback"], "the vote must reach POST /api/feedback"
    assert json.loads(captured["feedback"][0])["rating"] == "up"

    # A follow-up chip asks the next question through the same pipeline.
    page.locator(".followup-chip").first.click()
    page.wait_for_timeout(600)
    user_messages = page.locator("#chatMessages .message[data-role='user']")
    assert user_messages.count() == 2
    assert "wing angle" in user_messages.nth(1).inner_text()


def test_recent_questions_render_and_rerun(page):
    _stub_ask_pipeline(page)
    page.evaluate(
        "qs => localStorage.setItem('rag.recentQuestions.v1', JSON.stringify(qs))",
        ["What battery should we run?", "How stiff should the uprights be?"],
    )
    page.reload()
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(400)
    activate_tab(page, "chat")

    recent = page.locator(".chat-recent .chat-recent-suggestion")
    assert recent.count() == 2

    recent.first.click()
    page.wait_for_timeout(600)
    # Clicking re-runs the question through the (stubbed) ask pipeline.
    assert page.locator("#chatMessages .message[data-role='user']").count() == 1
    assert page.locator(".assistant-message").count() == 1


def test_ask_about_selection_quotes_excerpt_into_composer(page):
    _seed_chat_with_saved_answer(page)
    assert page.locator("#askSelectionPill").is_hidden()

    # Select the saved answer's text (triple-click selects the paragraph).
    # The answer stable sits in the body; the thinking stable above it stays
    # empty/hidden for a seeded answer without reasoning text.
    paragraph = page.locator("#chatMessages .assistant-message .body .stream-stable").first
    paragraph.click(click_count=3)

    pill = page.locator("#askSelectionPill")
    pill.wait_for(state="visible")
    pill.click()

    value = page.locator("#questionInput").input_value()
    assert "selected excerpt" in value
    assert "lift [S5] and drag [S7]." in value
    assert page.locator("#askSelectionPill").is_hidden()
    # The selection is quoted; the caret sits at the end ready for the question.
    page.wait_for_function("() => document.activeElement && document.activeElement.id === 'questionInput'")
