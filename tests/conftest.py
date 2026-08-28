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
