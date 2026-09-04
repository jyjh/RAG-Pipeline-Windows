import gc
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import pytest


_cleanup_log = logging.getLogger(__name__)

def _rmtree_with_retry(path, *, attempts=4, delay=0.5):
    """Remove a directory tree, retrying on Windows file-lock errors."""
    for attempt in range(1, attempts + 1):
        gc.collect()
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, OSError):
            if attempt >= attempts:
                _cleanup_log.warning("Could not remove temp dir %s after %d attempts", path, attempts)
                return
            time.sleep(delay * attempt)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def safe_tmp_path():
    # RAG_TEST_TMP_DIR moves the scratch root when the default directory is
    # unusable (e.g. a leftover run left it with a broken ACL that only an
    # elevated shell can delete).
    root = ROOT / os.environ.get("RAG_TEST_TMP_DIR", ".tmp_test_pytest_safe")
    root.mkdir(exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        _rmtree_with_retry(path)


@pytest.fixture(autouse=True)
def _hermetic_pipeline_config(monkeypatch):
    """Point config discovery at a known test config.

    Module-level helpers (src.llm_api, src.local_rag, src.embeddings) resolve
    config.toml via ``src.config.default_config_path()``. Without this fixture
    they would read the developer's repo config.toml, and backend-dependent
    tests would pass or fail depending on that file's ``[llm_api].backend``.
    Tests that need a specific config set ``RAG_PIPELINE_CONFIG`` themselves
    (a test's own monkeypatch.setenv overrides this fixture's value).
    """
    monkeypatch.setenv(
        "RAG_PIPELINE_CONFIG",
        str(ROOT / "tests" / "fixtures" / "hermetic_config.toml"),
    )


@pytest.fixture(autouse=True)
def _inert_llm_auto_tag(monkeypatch):
    """Keep LLM source-group auto-tagging network-inert during tests.

    Upload paths schedule background auto-tag runs; with a reachable local
    Ollama those would issue real chat requests and write trust entries into
    test workspaces mid-test. Stub the classifier suite-wide (returns no
    decisions); tests that exercise auto-tagging override this with their
    own fakes.
    """
    from src import auto_tag

    monkeypatch.setattr(auto_tag, "classify_documents", lambda items, **kwargs: {})


@pytest.fixture(autouse=True)
def _identity_serving_model_resolution(monkeypatch):
    """Keep auto-tag model-name resolution hermetic in endpoint tests.

    web_app resolves the configured auto-tag model onto the model that would
    actually serve it (``llm_api.resolve_local_model``) for status messages
    and provenance stamps. Left live, endpoint tests would report
    machine-dependent names on a developer box with a running Ollama, and pay
    a /api/tags network timeout on every cold cache when it is down. Tests
    for the resolution itself re-install the real helper (saved at import
    time in their module) and stub ``llm_api`` instead.
    """
    import src.web_app as web_app

    monkeypatch.setattr(web_app, "_resolve_serving_model", lambda model: model)


@pytest.fixture(autouse=True)
def _isolated_local_model_cache():
    """Start every test with a cold /api/tags cache.

    A developer machine often runs a live Ollama: any test that resolves a
    model name without stubbing the tags fetch performs a real lookup and
    populates the process-wide TTL cache with that server's model sizes.
    Later tests stub ``_ollama_tags`` (names) but then read REAL sizes from
    the leaked cache, so size-aware substitution gates on models that exist
    only on this machine. Clearing the cache per test keeps results
    deterministic regardless of test ordering and of whether a local Ollama
    is running.
    """
    from src import llm_api

    llm_api.reset_local_model_cache()
    yield
    llm_api.reset_local_model_cache()


@pytest.fixture(autouse=True)
def _local_operator(request, monkeypatch):
    """Run server tests as the trusted local operator by default.

    The server is always in an authentication posture: loopback clients are
    auto-authenticated as full admin, and every other client needs a
    credential. TestClient requests arrive with the non-loopback host
    "testclient", so by default this fixture treats them as loopback (exactly
    what uvicorn on 127.0.0.1 does for the real UI) and every handler test
    passes auth the way the local app does. Tests marked ``remote_client``
    (individual tests, or whole modules via ``pytestmark``) opt out and see
    the strict network posture: gated requests without a credential get 401.
    """
    import src.web_app as web_app

    monkeypatch.setattr(web_app, "_LOCALHOST_AUTO_AUTH", True)
    if request.node.get_closest_marker("remote_client"):
        return
    real_loopback = web_app._is_loopback_host
    monkeypatch.setattr(
        web_app,
        "_is_loopback_host",
        lambda host: True if str(host or "") == "testclient" else real_loopback(host),
    )
