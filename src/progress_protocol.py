"""Structured progress protocol for long-running pipeline subprocesses.

The ingestion and indexing phases of a 100GB-scale run take hours to days.
The web job runner (``src/web_app.py`` ``_run_job_subprocess``) spawns each
phase as a ``main.py`` subprocess whose combined stdout/stderr is streamed
into the job's log tail. To give the UI live progress (done/total/rate/ETA)
without coupling the subprocess to the server, the worker emits
machine-parseable progress lines prefixed with :data:`PROGRESS_PREFIX`.

The reader thread detects these lines, JSON-decodes the payload, and stores it
on the ``QueueJob`` (see ``queue_job.py``) so ``/api/jobs`` and
``/api/jobs/{id}`` can surface them. Emission is **always on** (not gated by
``progress_enabled``): it is cheap, single-line, and the whole point is that
the human-readable ``tqdm``/``_status`` output is suppressed via
``--no_progress`` in the web-driven path. Parsing failures degrade gracefully
-- an unparseable or partial line is treated as ordinary log output.
"""

from __future__ import annotations

import json
import sys
from typing import Any

# Sentinel prefix. The web reader splits on this to distinguish machine
# progress lines from ordinary log output. The trailing space is deliberate:
# it cleanly separates the prefix from the JSON payload.
PROGRESS_PREFIX = "__RAG_PROGRESS__ "


def emit_progress(
    *,
    phase: str,
    done: int,
    total: int | None = None,
    unit: str = "items",
    rate_per_min: float | None = None,
    eta_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
    stream=None,
) -> None:
    """Emit a structured progress line to ``stream`` (default stderr).

    Always emits regardless of ``progress_enabled`` so the web runner can
    surface progress even when the human-readable output is suppressed via
    ``--no_progress``. Safe to call from worker processes and threads; each
    call is a single ``print`` (one line, ``flush=True``) so the reader sees
    a whole line atomically.
    """
    payload: dict[str, Any] = {"phase": phase, "done": int(done), "unit": unit}
    if total is not None:
        payload["total"] = int(total)
    if rate_per_min is not None:
        payload["rate_per_min"] = round(float(rate_per_min), 2)
    if eta_seconds is not None:
        payload["eta_seconds"] = round(float(eta_seconds), 1)
    if extra:
        for key, value in extra.items():
            # Never let `extra` clobber the reserved fields; the canonical
            # values above win so the UI contract stays stable.
            if key not in payload:
                payload[key] = value
    try:
        line = PROGRESS_PREFIX + json.dumps(payload, separators=(",", ":"))
    except (TypeError, ValueError):
        # A non-serializable `extra` value must never crash the worker.
        return
    target = stream if stream is not None else sys.stderr
    print(line, file=target, flush=True)


def parse_progress_line(line: str) -> dict[str, Any] | None:
    """Parse a progress line.

    Returns the payload dict if ``line`` is a well-formed progress line,
    otherwise ``None`` (including for unparseable/partial lines, so callers
    can treat the line as ordinary log output).
    """
    text = (line or "").rstrip()
    if not text.startswith(PROGRESS_PREFIX):
        return None
    raw = text[len(PROGRESS_PREFIX):].strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def is_progress_line(line: str) -> bool:
    """Cheap membership check without JSON-decoding."""
    return (line or "").lstrip().startswith(PROGRESS_PREFIX)
