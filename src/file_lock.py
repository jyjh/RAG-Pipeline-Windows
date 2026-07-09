"""Cross-process file locks.

Indexing and ingestion run as separate OS processes (``main.py --mode
index|ingest`` spawned by the server), so the in-process ``threading.RLock``
guards in ``pdf_registry``/``web_app`` cannot coordinate them. A direct CLI
``main.py --mode index`` run and the server can otherwise clobber the same
live ``db/`` simultaneously, or the ingest subprocess can write ``.source_map.json``
while the server mutates the registry.

These helpers wrap :mod:`portalocker` (``msvcrt`` on Windows, ``fcntl`` on
POSIX) as context managers around the few live-resource mutation entry points.
A short acquire timeout turns a stuck holder into a clear error instead of an
infinite hang.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Iterator

import portalocker


# Lock files are dotfiles placed alongside the resource they protect.
INDEX_LOCK_FILENAME = ".index.lock"
REGISTRY_LOCK_FILENAME = ".registry.lock"

# How long to wait for a contended lock before giving up. A long-running
# indexing subprocess legitimately holds the index lock for the whole build;
# acquisition from the *server* side (publish/backup/restore) should wait out
# a normal build but not hang forever on a wedged holder. The default is
# generous; callers that need fail-fast behavior pass a smaller value.
DEFAULT_LOCK_TIMEOUT = 300.0

_LOCK_RETRY_INTERVAL = 0.2


def _lock_path(resource_dir: str | Path, filename: str) -> Path:
    base = Path(resource_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


@contextlib.contextmanager
def _acquire(
    lock_path: Path,
    *,
    timeout: float,
    poll_interval: float = _LOCK_RETRY_INTERVAL,
) -> Iterator[object]:
    """Acquire an exclusive cross-process lock on ``lock_path``.

    portalocker's lock handle is returned to the caller's ``with`` block but
    is opaque; callers only need the mutual-exclusion guarantee. The lock file
    itself is left on disk between runs (it is a 0-byte marker) -- that is
    normal for portalocker and harmless.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    last_error: Exception | None = None
    while True:
        try:
            handle = portalocker.Lock(
                str(lock_path),
                timeout=0 if deadline is None else max(0.0, deadline - time.monotonic()),
            )
            handle.acquire()
            try:
                yield handle
            finally:
                try:
                    handle.release()
                except Exception:
                    pass
            return
        except portalocker.exceptions.LockException as exc:
            last_error = exc
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(poll_interval)
    raise TimeoutError(
        f"Could not acquire file lock {lock_path} within {timeout:g}s. "
        f"Another ingestion/indexing process may be holding it. "
        f"Original error: {last_error}"
    )


@contextlib.contextmanager
def acquire_index_lock(
    db_dir: str | Path,
    *,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> Iterator[object]:
    """Exclusive lock protecting the live ``db/`` index directory."""
    lock_path = _lock_path(db_dir, INDEX_LOCK_FILENAME)
    with _acquire(lock_path, timeout=timeout) as handle:
        yield handle


@contextlib.contextmanager
def acquire_registry_lock(
    data_dir: str | Path,
    *,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> Iterator[object]:
    """Exclusive lock protecting registry / source-map mutations under ``data_dir``."""
    lock_path = _lock_path(data_dir, REGISTRY_LOCK_FILENAME)
    with _acquire(lock_path, timeout=timeout) as handle:
        yield handle
