"""Read-only UI smoke tests: the app boots from ES modules and key flows work.

Every test here is safe to run against a real data directory: nothing POSTs,
DELETEs, or otherwise mutates server state (dialogs are opened and cancelled
only). Run via `python -m pytest tests/browser -q`.
"""
from __future__ import annotations

import pytest

from helpers import activate_tab  # noqa: E402  (tests/browser is a sys.path root)

pytestmark = pytest.mark.browser


def test_app_boots_with_all_modules_and_tabs(page):
    # Boot evidence: health poll replaced the placeholder status line.
    page.wait_for_function(
        "() => !document.getElementById('statusLine').textContent.startsWith('Checking')"
    )
    modules = page.evaluate(
        "() => performance.getEntriesByType('resource')"
        ".filter(r => r.name.includes('/js/')).map(r => r.name.split('/js/')[1])"
    )
    assert set(modules) == {
        "core.js", "status.js", "upload.js", "categories.js", "library.js",
        "review.js", "chat.js", "admin.js", "shell.js", "usability.js",
    }
    for target in ["upload", "library", "index", "chat", "admin", "guide"]:
        activate_tab(page, target)


def test_library_table_renders_and_filter_wiring_fires(page):
    activate_tab(page, "library")
    page.wait_for_selector("#pdfsBody tr", state="visible")
    assert page.locator("#pdfsBody tr").count() > 0

    requests: list[str] = []
    page.on(
        "request",
        lambda request: requests.append(request.url)
        if "/api/pdfs?" in request.url
        else None,
    )
    page.select_option("#pdfGroupFilterSelect", "official")
    page.wait_for_timeout(1200)
    assert any("source_group=official" in url for url in requests), requests
    page.select_option("#pdfGroupFilterSelect", "all")


def test_review_inline_edit_opens_and_cancels(page):
    activate_tab(page, "index")
    page.wait_for_selector('#indexBody button[data-action="edit"]', state="visible")
    page.locator('#indexBody button[data-action="edit"]').first.click()
    editing_row = page.locator('#indexBody tr[data-editing="true"]')
    editing_row.wait_for(state="visible")
    assert editing_row.locator(".inline-edit-textarea").is_visible()
    editing_row.locator("[data-action='cancel-edit']").click()
    page.wait_for_selector('#indexBody tr[data-editing="true"]', state="detached")


def test_delete_dialog_requires_typed_confirmation_and_cancels(page):
    activate_tab(page, "library")
    page.wait_for_selector('#pdfsBody button[data-pdf-action="delete"]', state="visible")
    page.locator('#pdfsBody button[data-pdf-action="delete"]').first.click()
    dialog = page.locator("body > .modal-overlay:visible").last
    dialog.wait_for(state="visible")
    confirm_button = dialog.locator(".modal-actions button.danger")
    assert not confirm_button.is_enabled()
    dialog.locator(".confirm-type-label input").fill("DELETE")
    assert confirm_button.is_enabled()
    dialog.locator(".modal-actions button:not(.danger)").click()
    page.wait_for_selector("body > .modal-overlay:visible", state="detached")


def test_theme_toggle_flips_and_persists_across_reload(page):
    page.evaluate("document.getElementById('themeToggleButton').click()")
    after_toggle = page.evaluate("document.documentElement.dataset.theme")
    assert after_toggle in {"dark", "light"}
    assert page.evaluate("localStorage.getItem('rag.theme.v1')") == after_toggle
    page.reload()
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(800)
    assert page.evaluate("document.documentElement.dataset.theme") == after_toggle
    # Restore the light default so later tests are deterministic.
    page.evaluate(
        "document.documentElement.dataset.theme = 'light';"
        "localStorage.setItem('rag.theme.v1', 'auto');"
    )


def test_sidebar_collapse_is_persistent_and_hides_labels(page):
    page.evaluate("document.getElementById('sidebarCollapseButton').click()")
    assert page.evaluate("document.body.classList.contains('sidebar-collapsed')")
    assert page.evaluate("localStorage.getItem('rag.sidebarCollapsed.v1')") == "1"
    page.reload()
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(600)
    assert page.evaluate("document.body.classList.contains('sidebar-collapsed')")
    page.evaluate("document.getElementById('sidebarCollapseButton').click()")


def test_answer_presets_apply_and_manual_edit_flips_to_custom(page):
    summary = page.locator("#composerSettingsSummary")
    select = page.locator("#answerPresetSelect")
    select.select_option("deep")
    page.wait_for_timeout(200)
    assert "Deep research" in summary.inner_text()
    assert page.locator("#temperatureInput").input_value() == "0.4"
    select.select_option("precise")
    page.wait_for_timeout(200)
    assert page.locator("#temperatureInput").input_value() == "0.2"
    assert not page.locator("#webSearchInput").is_checked()
    # Manual edits happen inside the advanced disclosure; open it first.
    page.locator("summary", has_text="Advanced answer settings").click()
    page.locator("#temperatureInput").wait_for(state="visible")
    page.locator("#temperatureInput").fill("0.9")
    page.dispatch_event("#temperatureInput", "change")
    page.wait_for_timeout(200)
    assert select.input_value() == "custom"
    assert "Custom" in summary.inner_text()
    select.select_option("balanced")


def test_keyboard_shortcuts_help_and_g_jump(page):
    page.keyboard.press("?")
    assert page.locator("#shortcutsOverlay").is_visible()
    page.locator("#shortcutsCloseButton").click()
    assert not page.locator("#shortcutsOverlay").is_visible()
    page.keyboard.press("g")
    page.keyboard.press("l")
    page.locator("#library.panel.active").wait_for(state="visible")
    page.keyboard.press("/")
    page.wait_for_timeout(200)
    focused = page.evaluate(
        "document.activeElement === document.getElementById('pdfSearchInput')"
    )
    assert focused


def test_admin_dashboard_renders_cards_and_key_table(page):
    activate_tab(page, "admin")
    page.wait_for_selector(".admin-card", state="visible")
    assert page.locator(".admin-card").count() >= 3
    assert page.locator("#adminKeysTable").is_visible()


def test_library_sort_header_fires_sort_request(page):
    activate_tab(page, "library")
    page.wait_for_selector("#pdfsBody tr", state="visible")
    requests: list[str] = []
    page.on(
        "request",
        lambda request: requests.append(request.url)
        if "/api/pdfs?" in request.url
        else None,
    )
    page.locator('#pdfsTable .th-sort[data-sort-key="filename"]').click()
    page.wait_for_timeout(1200)
    assert any("sort=filename" in url for url in requests), requests
    page.locator('#pdfsTable .th-sort[data-sort-key="filename"]').click()
    page.wait_for_timeout(1200)
    assert any("sort=-filename" in url for url in requests), requests


def test_chat_search_filters_saved_chats(page):
    sidebar = page.locator("#savedChatsList")
    page.wait_for_selector(".saved-chat-row", state="visible")
    before = sidebar.locator(".saved-chat-row").count()
    assert before >= 1
    page.fill("#chatSearchInput", "zz-no-such-chat-zz")
    page.wait_for_timeout(300)
    assert sidebar.locator(".saved-chat-row").count() == 0
    page.fill("#chatSearchInput", "")
    page.wait_for_timeout(300)
    assert sidebar.locator(".saved-chat-row").count() == before


def test_chat_pin_persists_across_reload(page):
    page.wait_for_selector(".saved-chat-row", state="visible")
    row = page.locator(".saved-chat-row").first
    row.locator("button", has_text="Pin").first.click()
    page.wait_for_timeout(300)
    first_title = page.evaluate(
        "document.querySelector('.saved-chat-row .saved-chat-title')?.textContent"
    )
    assert first_title.startswith("\u2605")
    page.reload()
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(1200)
    title_after = page.evaluate(
        "document.querySelector('.saved-chat-row .saved-chat-title')?.textContent"
    )
    assert title_after.startswith("\u2605")
    # Unpin to leave a clean state.
    page.locator(".saved-chat-row").first.locator("button", has_text="Unpin").first.click()


def test_skip_link_and_focus_are_wired(page):
    assert page.locator(".skip-link").count() == 1
    page.keyboard.press("Tab")
    focused = page.evaluate("document.activeElement?.className")
    assert focused == "skip-link"


def test_update_panel_renders_in_admin(page):
    activate_tab(page, "admin")
    page.wait_for_selector("#updatePanelBody", state="visible")
    page.wait_for_function(
        "() => document.getElementById('updatePanelBody').innerText.length > 10"
    )


def test_review_documents_view_lists_and_expands(page):
    activate_tab(page, "index")
    page.locator("#reviewViewDocuments").click()
    page.wait_for_selector("#docsTable:not([hidden])", state="attached")
    page.wait_for_selector("#docsBody tr.doc-row", state="visible")
    assert page.locator("#docsBody tr.doc-row").count() > 0
    # Expand the first document; its records sub-table loads real records.
    page.locator('#docsBody button[data-action="toggle-doc"]').first.click()
    page.wait_for_selector(".doc-records-table tr", state="visible")
    assert page.locator(".doc-records-table tr").count() > 0
    # Back to chunks view restores the chunk table.
    page.locator("#reviewViewChunks").click()
    page.wait_for_selector("#indexTable:not([hidden])", state="attached")
