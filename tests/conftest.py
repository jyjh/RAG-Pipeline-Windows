import sys
import shutil
import uuid
from pathlib import Path

import gc
import logging
import time

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
    root = ROOT / ".tmp_test_pytest_safe"
    root.mkdir(exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        _rmtree_with_retry(path)
