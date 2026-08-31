"""Regression tests for chat, toast, and saved-chat usability fixes.

Read-only against the server: state changes happen only in the throwaway
browser context (seeded localStorage / in-page module calls). Run via
`python -m pytest tests/browser -q`.
"""
from __future__ import annotations

import pytest

from helpers import activate_tab  # noqa: E402  (tests/browser is a sys.path root)

pytestmark = pytest.mark.browser


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
