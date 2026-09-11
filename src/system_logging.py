"""System-wide logging: rotating file output with timestamps, plus a
structured per-request access log.

Before this module the web server's application log lines went nowhere: the
root logger had no handlers (uvicorn only shows its own startup/access lines
on the console), so ``logger.warning(...)`` calls across ``src/`` were lost
unless a developer ran with a console attached, and nothing survived a
crash. This module gives every process one shared, rotating file handler per
log file (so the root logger, the job logger, and the access logger can all
target ``logs/server.log`` without two handlers racing the same rotation)
and a structured access log keyed by client IP.

``setup_system_logging`` is re-entrant: calling it again (config reload,
tests) detaches the handlers a previous call attached and re-attaches to the
new targets. It never raises on logging problems it can avoid -- callers
wrap it in try/except OSError so a read-only workspace degrades to
console-only logging instead of blocking startup.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.defaults import (
    DEFAULT_ACCESS_LOG_FILE,
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_LEVEL,
)

# Logger that receives one structured JSON line per HTTP request. Kept
# separate from the root logger so operators can tail requests without
# application noise, and so it can be pointed at its own file.
ACCESS_LOGGER_NAME = "local_rag.access"

# Timestamps use local time (matching job logs and console output); the
# access-log JSON payloads carry an explicit UTC ISO timestamp for machine
# parsing across hosts.
SYSTEM_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# Rotate system/access logs so a long-lived server cannot grow them without
# bound (same ceiling as the job logs: 10 MiB x 5 backups = ~50 MiB/file).
SYSTEM_LOG_MAX_BYTES = 10 * 1024 * 1024
SYSTEM_LOG_BACKUP_COUNT = 5

# ``?token=`` is a documented credential transport for GET consumers that
# cannot set headers; it must never reach a persisted log line. The match is
# anchored to start-of-string or a ?/& delimiter so bare query strings
# ("token=...&x=1", as stored in access-log payloads) redact too.
TOKEN_QUERY_PATTERN = re.compile(r"(^|[?&])token=[^&\s]+")


def redact_token_query(value: str | None) -> str:
    """Replace ``token=<secret>`` query values with ``token=***``."""

    if not value:
        return value or ""
    return TOKEN_QUERY_PATTERN.sub(r"\1token=***", value)


class _TokenRedactingLogFilter(logging.Filter):
    """Redact ``token=<secret>`` query strings from log records.

    Applies to both the raw ``record.args`` (uvicorn access records format
    the request line lazily) and the already-rendered message.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.args:
                record.args = tuple(
                    redact_token_query(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            message = record.getMessage()
            if "token=" in message:
                record.msg = redact_token_query(message)
                record.args = None
        except Exception:  # noqa: BLE001 - redaction must never break logging
            pass
        return True


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# One RotatingFileHandler per resolved log path per process. Sharing a single
# handler between the root logger and (e.g.) the job logger is what makes two
# loggers safely target the same file: two independent RotatingFileHandlers
# on one path each keep their own rollover counters and corrupt the file at
# the first rotation.
#
# Reentrant: setup/reset hold it across shared_file_handler() calls.
_FILE_HANDLERS: dict[str, logging.Handler] = {}
_HANDLER_LOCK = threading.RLock()

# Handlers attached to the root logger by the most recent setup call, plus
# the access handler it attached -- tracked so a repeated setup detaches
# exactly what the previous one attached, even when the targets changed.
_ATTACHED_ROOT_HANDLERS: list[logging.Handler] = []
_ATTACHED_ACCESS_HANDLER: logging.Handler | None = None


def _resolve_log_path(raw_path: str | Path, workspace_root: Path | None) -> Path:
    path = Path(str(raw_path or "").strip())
    if not path.is_absolute():
        base = workspace_root or Path(__file__).resolve().parents[1]
        path = base / path
    return path


def shared_file_handler(log_path: str | Path) -> logging.Handler:
    """Return the process-wide rotating handler for ``log_path``.

    Creates the parent directory and the handler on first use; later callers
    (job logging, root logging, access logging) get the same instance.
    """
    resolved = _resolve_log_path(log_path, None)
    key = str(resolved)
    with _HANDLER_LOCK:
        handler = _FILE_HANDLERS.get(key)
        if handler is None:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            rotating = logging.handlers.RotatingFileHandler(
                resolved,
                maxBytes=SYSTEM_LOG_MAX_BYTES,
                backupCount=SYSTEM_LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            rotating.setFormatter(logging.Formatter(SYSTEM_LOG_FORMAT))
            rotating.addFilter(_TokenRedactingLogFilter())
            handler = rotating
            _FILE_HANDLERS[key] = handler
        return handler


def _access_logger() -> logging.Logger:
    logger = logging.getLogger(ACCESS_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # access lines go to the access file, not root
    return logger


def log_access_event(fields: dict[str, Any]) -> None:
    """Emit one structured access-log line (single-line JSON after a marker).

    The ``ACCESS_EVENT`` marker lets scrapers tell access records apart from
    library lines sharing the same file, mirroring ``JOB_EVENT``. The
    ``query`` field is token-redacted here (not only in the web middleware)
    so a future caller cannot leak a ``?token=`` secret into the log; the
    redaction is idempotent for already-redacted values.
    """
    payload = {"event": "http_request", "ts": utcnow_iso()}
    payload.update(fields)
    if payload.get("query"):
        payload["query"] = redact_token_query(payload["query"])
    _access_logger().info(
        "ACCESS_EVENT %s", json.dumps(payload, sort_keys=True, default=str)
    )


def setup_system_logging(
    *,
    level: str = DEFAULT_LOG_LEVEL,
    file: str | Path | None = DEFAULT_LOG_FILE,
    access_file: str | Path | None = DEFAULT_ACCESS_LOG_FILE,
    console: bool = True,
    workspace_root: str | Path | None = None,
) -> None:
    """Attach rotating file handlers to the root and access loggers.

    ``file`` receives every application/uvicorn/library record at ``level``
    or above, each line timestamped. ``access_file`` receives the structured
    per-request lines written via :func:`log_access_event`. An empty/None
    target disables that stream. ``console`` keeps (or adds) a stderr handler
    so the console view survives when this runs in place of
    ``logging.basicConfig``. Relative paths resolve against
    ``workspace_root`` (default: the repo root), never the process cwd.

    Re-entrant: a second call re-points logging at the new targets without
    stacking duplicate handlers. Root propagation of uvicorn's own loggers is
    arranged by the caller (``web_app.run_server`` passes a matching
    ``log_config`` to ``uvicorn.run``).
    """
    global _ATTACHED_ACCESS_HANDLER

    root = logging.getLogger()
    parsed_level = logging.getLevelName(str(level or DEFAULT_LOG_LEVEL).strip().upper())
    if not isinstance(parsed_level, int):
        parsed_level = logging.INFO
    root.setLevel(parsed_level)
    # The shared file handler never gets MORE restrictive than INFO: the job
    # logger (level INFO, noisily structured) and uvicorn's access logger
    # share these files, and raising the handler level with the app level
    # would silently drop their records. Records above the configured level
    # are already suppressed at the emitting logger via root's level.
    handler_level = parsed_level if parsed_level < logging.INFO else logging.INFO

    with _HANDLER_LOCK:
        for handler in _ATTACHED_ROOT_HANDLERS:
            root.removeHandler(handler)
        _ATTACHED_ROOT_HANDLERS.clear()
        if _ATTACHED_ACCESS_HANDLER is not None:
            _access_logger().removeHandler(_ATTACHED_ACCESS_HANDLER)
            _ATTACHED_ACCESS_HANDLER = None

        if file:
            handler = shared_file_handler(file)
            handler.setLevel(handler_level)
            root.addHandler(handler)
            _ATTACHED_ROOT_HANDLERS.append(handler)
        if console:
            stream = logging.StreamHandler(sys.stderr)
            stream.setFormatter(logging.Formatter(SYSTEM_LOG_FORMAT))
            stream.addFilter(_TokenRedactingLogFilter())
            stream.setLevel(parsed_level)
            root.addHandler(stream)
            _ATTACHED_ROOT_HANDLERS.append(stream)
        if access_file:
            access_handler = shared_file_handler(access_file)
            access_handler.setLevel(logging.INFO)
            _access_logger().addHandler(access_handler)
            _ATTACHED_ACCESS_HANDLER = access_handler


def reset_for_tests() -> None:
    """Detach everything setup_system_logging attached and drop the registry.

    Only for tests: restores the pristine "no handlers" state so a test that
    points logging at a temp file cannot leak it into later tests.
    """
    global _ATTACHED_ACCESS_HANDLER
    with _HANDLER_LOCK:
        root = logging.getLogger()
        for handler in _ATTACHED_ROOT_HANDLERS:
            root.removeHandler(handler)
        _ATTACHED_ROOT_HANDLERS.clear()
        if _ATTACHED_ACCESS_HANDLER is not None:
            _access_logger().removeHandler(_ATTACHED_ACCESS_HANDLER)
            _ATTACHED_ACCESS_HANDLER = None
        for handler in _FILE_HANDLERS.values():
            try:
                handler.close()
            except Exception:  # noqa: BLE001 - close is best-effort in tests
                pass
        _FILE_HANDLERS.clear()
