"""Browser smoke tests (Playwright + system Edge/Chrome).

These tests start the real FastAPI app in-process on a random port and drive
it with a headless system browser. Startup job recovery is stubbed out so the
suite never enqueues work against the shared data directory; every test is
READ-ONLY (no uploads, deletes, trust writes, or index edits).

Run explicitly (excluded from the default `pytest` run via pytest.ini):

    python -m pytest tests/browser -q

Requires the `playwright` package and a local Edge or Chrome install; tests
skip cleanly when neither is available.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import urllib.request

import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "browser: end-to-end UI smoke tests (Playwright + system Edge/Chrome)"
    )


# ---------------------------------------------------------------------------
# Browser availability: skip the whole directory cleanly when Playwright or a
# system browser is missing.
# ---------------------------------------------------------------------------
playwright_api = None
_launch_channel = None

try:  # pragma: no cover - environment probe
    from playwright.sync_api import sync_playwright as _sync_playwright

    playwright_api = _sync_playwright
except ImportError:  # pragma: no cover
    playwright_api = None


def _probe_channel(pw) -> str | None:
    for channel in ("msedge", "chrome"):
        try:
            browser = pw.chromium.launch(channel=channel, headless=True)
            browser.close()
            return channel
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# App server fixture (in-process uvicorn, recovery stubbed, random port).
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def app_server():
    # The test server intentionally runs alongside the developer's instance,
    # so it opts out of the single-instance data-dir guard.
    os.environ.setdefault("RAG_ALLOW_MULTIPLE_WEB_INSTANCES", "1")
    import src.web_app as web_app

    # Never let a test run enqueue or recover real work against the shared
    # data directory: the suite is read-only by construction.
    web_app.recover_pending_upload_jobs_on_startup = lambda: {"recovered": []}

    port = _free_port()
    import uvicorn

    config = uvicorn.Config(web_app.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 30
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/health", timeout=2
            ) as response:
                if response.status == 200:
                    break
        except Exception as exc:  # noqa: PERF203 - probe loop
            last_error = exc
            time.sleep(0.3)
    else:
        pytest.fail(f"test app server never became healthy: {last_error}")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=15)


@pytest.fixture(scope="session")
def browser_instance():
    if playwright_api is None:
        pytest.skip("playwright is not installed")
    with playwright_api() as pw:
        channel = _probe_channel(pw)
        if channel is None:
            pytest.skip("no system Edge/Chrome available for Playwright")
        browser = pw.chromium.launch(channel=channel, headless=True)
        yield browser
        browser.close()


@pytest.fixture()
def page(browser_instance, app_server):
    context = browser_instance.new_context()
    page = context.new_page()
    page.set_default_timeout(15000)
    page.goto(app_server + "/")
    page.wait_for_load_state("domcontentloaded")
    _dismiss_first_run_overlays(page)
    yield page
    context.close()


def _dismiss_first_run_overlays(page) -> None:
    """A fresh browser context triggers the first-run prompts; close them."""
    page.wait_for_timeout(600)
    try:
        if page.locator("#welcomeTutorialOverlay").is_visible():
            page.locator("#welcomeTutorialSkipButton").click()
    except Exception:
        pass
    try:
        if page.locator("#cachePromptOverlay").is_visible():
            page.locator("#cachePromptDoneButton").click()
    except Exception:
        pass
    page.wait_for_timeout(200)
