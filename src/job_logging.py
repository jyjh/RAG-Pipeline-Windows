"""Per-job structured logging and run summaries.

When a 100GB ingestion fails it is currently hard to tell why: the root logger
defaults to WARNING on stderr, nothing is persisted to disk, and the in-memory
job-log tail is lost on a server crash. This module configures a per-job file
handler (one structured line per event) and writes a machine-readable run
summary that the job queue surfaces in the API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.atomic_io import write_json_atomic


# Formatter tag appended to every record so log scrapers can tell structured
# job events apart from library/dependency log lines.
JOB_LOGGER_NAME = "local_rag.job"
JOB_EVENT_MARKER = "JOB_EVENT"

_CONFIGURED_PATHS: set[str] = set()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_job_logging(log_path: str | Path) -> logging.Logger:
    """Attach a file handler at INFO to the job logger, writing to ``log_path``.

    Idempotent per ``log_path``: calling it repeatedly (e.g. across imports in
    the same subprocess) does not stack handlers. The parent directory is
    created. Returns the configured logger so callers can emit events directly.
    """
    resolved = str(Path(log_path).resolve())
    logger = logging.getLogger(JOB_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate lines via the root logger

    if resolved in _CONFIGURED_PATHS:
        return logger

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    _CONFIGURED_PATHS.add(resolved)
    return logger


def log_event(event: str, **fields: Any) -> None:
    """Emit one structured event on the job logger.

    ``event`` is a short stable identifier (e.g. ``"file_indexed"``); any extra
    ``fields`` are JSON-serialized so they stay on a single line and are easy
    to grep/parse.
    """
    logger = logging.getLogger(JOB_LOGGER_NAME)
    payload = {"event": event, "ts": utcnow_iso()}
    payload.update(fields)
    logger.info("%s %s", JOB_EVENT_MARKER, json.dumps(payload, sort_keys=True, default=str))


def write_run_summary(
    path: str | Path,
    *,
    phase: str,
    files_processed: int,
    files_failed: int,
    elapsed_s: float,
    disk_used_bytes: int | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a machine-readable run summary to ``path`` (atomically).

    The job queue reads this back to attach a structured tail to the in-memory
    job log returned by the API.
    """
    summary: dict[str, Any] = {
        "phase": phase,
        "files_processed": int(files_processed),
        "files_failed": int(files_failed),
        "elapsed_s": round(float(elapsed_s), 3),
        "completed_at": utcnow_iso(),
    }
    if disk_used_bytes is not None:
        summary["disk_used_bytes"] = int(disk_used_bytes)
    if errors:
        summary["errors"] = errors
    write_json_atomic(path, summary)
    return summary


class RunTimer:
    """Tiny monotonic stopwatch for phase timing."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._start
