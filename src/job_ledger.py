"""Durable ledger for non-upload background jobs.

Upload jobs are already crash-recoverable via the PDF registry (their per-file
status survives a restart). Non-upload jobs (``reindex``, ``reindex_source``,
``rebuild``, ``backup``, ``restore``) lived only in the in-memory
``RagJobQueue._jobs`` dict, so a server crash mid-reindex silently lost the job
and the user had to re-trigger it manually.

This module persists just enough of a non-upload job to re-enqueue it on the
next startup: its kind and the options needed to reconstruct it. The ledger is
written atomically (``write_json_atomic`` + cross-process lock, same pattern as
the PDF registry). Entries are added on enqueue and removed when the job reaches
a terminal state (done/failed/cancelled), so on restart the ledger contains
only jobs that were interrupted.
"""

from __future__ import annotations

import contextlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.atomic_io import write_json_atomic

LEDGER_FILENAME = ".job_ledger.json"
LEDGER_VERSION = 1

# Job kinds that are NOT tracked by the PDF registry and therefore need the
# ledger for crash recovery. Upload jobs are recovered via the registry.
LEDGER_TRACKED_KINDS = {"reindex", "reindex_source", "rebuild", "backup", "restore", "rebuild_vector_index"}

_LOCK = threading.RLock()
_JSON_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    """Read and JSON-parse the ledger, cached on its ``(mtime_ns, size)`` signature.

    Same memoization strategy as ``pdf_registry._load_json``. A missing file or
    non-dict payload yields an empty ledger. ``json.JSONDecodeError`` is allowed
    to propagate so corruption surfaces rather than silently masking as empty
    (atomic writes mean a decode error is real damage).
    """
    cache_key = str(path)
    try:
        stat = path.stat()
    except OSError:
        _JSON_CACHE.pop(cache_key, None)
        return {"version": LEDGER_VERSION, "jobs": {}}
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _JSON_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    payload = path.read_text(encoding="utf-8")
    import json

    value = json.loads(payload) if payload else {"version": LEDGER_VERSION, "jobs": {}}
    if not isinstance(value, dict):
        value = {"version": LEDGER_VERSION, "jobs": {}}
    value.setdefault("version", LEDGER_VERSION)
    value.setdefault("jobs", {})
    if not isinstance(value.get("jobs"), dict):
        value["jobs"] = {}
    _JSON_CACHE[cache_key] = (signature, value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)
    _JSON_CACHE.pop(str(path), None)


class JobLedger:
    """Persistent record of tracked (non-upload) jobs that have not finished.

    A job is recorded when enqueued and removed when it reaches a terminal
    status. On startup, ``pending_entries`` returns whatever remains, so the
    queue can re-enqueue jobs that were interrupted by a crash.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        with _LOCK:
            return _load_json(self.path)

    def record(self, job_id: str, *, kind: str, **fields: Any) -> None:
        """Add or update a job entry in the ledger.

        Only kinds in :data:`LEDGER_TRACKED_KINDS` are recorded; others are a
        no-op so callers can call this unconditionally for every enqueue.
        """
        if kind not in LEDGER_TRACKED_KINDS:
            return
        with _LOCK:
            payload = self._load()
            jobs = payload.setdefault("jobs", {})
            entry: dict[str, Any] = {"kind": kind, "recorded_at": utcnow()}
            entry.update(fields)
            jobs[str(job_id)] = entry
            _write_json(self.path, payload)

    def remove(self, job_id: str) -> None:
        """Remove a job entry (called when the job reaches a terminal state)."""
        job_id = str(job_id)
        with _LOCK:
            payload = self._load()
            jobs = payload.get("jobs", {})
            if job_id not in jobs:
                return
            jobs.pop(job_id, None)
            _write_json(self.path, payload)

    def pending_entries(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(job_id, entry)`` pairs for all non-terminal jobs, oldest first."""
        with _LOCK:
            payload = self._load()
            jobs = payload.get("jobs", {})
            entries = [(str(jid), dict(data)) for jid, data in jobs.items() if isinstance(data, dict)]
        entries.sort(key=lambda item: str(item[1].get("recorded_at") or ""))
        return entries

    def count(self) -> int:
        with _LOCK:
            return len(self._load().get("jobs", {}))
