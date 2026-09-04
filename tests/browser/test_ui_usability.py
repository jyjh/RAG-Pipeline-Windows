"""Regression tests for chat, toast, category UI, and saved-chat fixes.

Read-only against the server: state changes happen only in the throwaway
browser context (seeded localStorage, in-page module calls, and network
stubs scoped to the test's own context). Run via `python -m pytest tests/browser -q`.
"""
from __future__ import annotations

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
