"""System logging ([logging] section): file output, redaction, access log.

Covers the persisted system log (timestamps/levels/logger names), the
per-request access log with client IP and token redaction, the shared
per-path rotating handler (one handler per file even when several loggers
target it), and the web app's outermost access-log middleware.
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response

import src.job_logging as job_logging
import src.system_logging as system_logging
from src.system_logging import (
    ACCESS_LOGGER_NAME,
    log_access_event,
    redact_token_query,
    reset_for_tests,
    setup_system_logging,
)


@pytest.fixture(autouse=True)
def _clean_logging_state(safe_tmp_path):
    """Isolate logging config per test and restore a pristine state after.

    Requests safe_tmp_path (even when a test does not) so its teardown runs
    AFTER this fixture's: closing the handlers must precede the directory
    deletion or Windows keeps the tree locked. Removes the handlers setup
    calls attach (including the one web_app's import attached to the repo
    logs dir) so tests never leak a temp-file handler -- or a closed one --
    into later tests.
    """
    try:
        yield
    finally:
        reset_for_tests()
        job_logger = logging.getLogger(job_logging.JOB_LOGGER_NAME)
        job_logger.handlers.clear()
        job_logging._CONFIGURED_PATHS.clear()


def _root_handlers_pointing_at(path) -> list[logging.Handler]:
    resolved = str(Path(path).resolve())
    return [
        h
        for h in logging.getLogger().handlers
        if getattr(h, "baseFilename", None) == resolved
    ]


def _read(path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_access_payload(record: logging.LogRecord) -> dict:
    message = record.getMessage()
    assert message.startswith("ACCESS_EVENT "), message
    return json.loads(message.split("ACCESS_EVENT ", 1)[1])


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_setup_writes_timestamped_lines_to_file(safe_tmp_path):
    log_file = safe_tmp_path / "sys.log"
    setup_system_logging(level="INFO", file=log_file, access_file=None, console=False)

    logging.getLogger("test.probe").warning("hello %s", "world")

    lines = _read(log_file).strip().splitlines()
    assert len(lines) == 1
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}[ T][\d:,]{8,} WARNING test\.probe hello world$",
        lines[0],
    ), lines[0]


def test_info_respects_configured_level(safe_tmp_path):
    log_file = safe_tmp_path / "sys.log"
    setup_system_logging(level="WARNING", file=log_file, access_file=None, console=False)

    logging.getLogger("test.probe").info("too chatty")
    logging.getLogger("test.probe").error("kept")

    lines = _read(log_file).strip().splitlines()
    assert len(lines) == 1
    assert "kept" in lines[0]


def test_setup_is_reentrant_and_does_not_stack_handlers(safe_tmp_path):
    first = safe_tmp_path / "first.log"
    second = safe_tmp_path / "second.log"
    setup_system_logging(level="INFO", file=first, access_file=None, console=False)
    setup_system_logging(level="INFO", file=second, access_file=None, console=False)

    # pytest adds its own root handlers; count only ours.
    assert not _root_handlers_pointing_at(first)
    assert len(_root_handlers_pointing_at(second)) == 1

    logging.getLogger("test.probe").info("final destination")
    assert "final destination" not in _read(first)
    assert "final destination" in _read(second)


def test_job_logger_and_system_logger_share_one_handler(safe_tmp_path):
    shared = safe_tmp_path / "shared.log"
    job_logging.setup_job_logging(shared)
    setup_system_logging(level="INFO", file=shared, access_file=None, console=False)

    job_handlers = logging.getLogger(job_logging.JOB_LOGGER_NAME).handlers
    # Compare RotatingFileHandlers only: in a full-suite run pytest's logging
    # plugin leaves its own LogCaptureHandlers on this logger, and pytest adds
    # a root FileHandler (\\.\nul) that shares the base class but not our
    # registry.
    job_rotating = [
        h for h in job_handlers if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    root_file_handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(job_rotating) == 1
    assert len(root_file_handlers) == 1
    assert job_rotating[0] is root_file_handlers[0]

    job_logging.log_event("unit_test_event", detail="ok")
    body = _read(shared)
    assert job_logging.JOB_EVENT_MARKER in body
    assert '"unit_test_event"' in body


def test_access_log_event_writes_structured_line(safe_tmp_path):
    access_file = safe_tmp_path / "access.log"
    setup_system_logging(level="INFO", file=None, access_file=access_file, console=False)

    log_access_event(
        {
            "ip": "192.0.2.7",
            "method": "GET",
            "path": "/api/health",
            "query": "",
            "status": 200,
            "duration_ms": 1.5,
            "key_id": None,
            "role": None,
        }
    )

    payload = json.loads(_read(access_file).strip().split("ACCESS_EVENT ", 1)[1])
    assert payload["event"] == "http_request"
    assert payload["ip"] == "192.0.2.7"
    assert payload["status"] == 200
    assert payload["duration_ms"] == 1.5
    assert payload["ts"]  # ISO timestamp present


def test_log_access_event_redacts_query_itself(safe_tmp_path):
    """Defense in depth: the emitter redacts even if the caller forgot."""
    access_file = safe_tmp_path / "access.log"
    setup_system_logging(level="INFO", file=None, access_file=access_file, console=False)

    log_access_event({"ip": "192.0.2.9", "query": "?token=hunter2&a=1", "status": 200})

    payload = json.loads(_read(access_file).strip().split("ACCESS_EVENT ", 1)[1])
    assert payload["query"] == "?token=***&a=1"


def test_token_redaction_util_and_file_records(safe_tmp_path):
    assert redact_token_query("a=1&token=abc&b=2") == "a=1&token=***&b=2"
    assert redact_token_query("?token=secret") == "?token=***"
    assert redact_token_query("") == ""
    assert redact_token_query(None) == ""

    log_file = safe_tmp_path / "sys.log"
    setup_system_logging(level="INFO", file=log_file, access_file=None, console=False)
    logger = logging.getLogger("test.uvicorn_style")
    # Uvicorn access records render the request line lazily from record.args.
    logger.info('%s - "%s %s HTTP/1.1" %d', "127.0.0.1", "GET", "/api/pdfs?token=sekrit", 200)

    body = _read(log_file)
    assert "token=sekrit" not in body
    assert "token=***" in body


def _make_request(
    *,
    method: str = "GET",
    path: str = "/api/health",
    query: bytes = b"",
    client=("192.0.2.44", 55555),
    headers=None,
) -> Request:
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query,
        "scheme": "http",
        "server": ("testserver", 80),
        "client": client,
        "headers": list(headers or []),
        "state": {},
    }
    return Request(scope)


def _capture_access_records(monkeypatch):
    """Attach a capture handler to the access logger; returns (records, cleanup)."""
    del monkeypatch
    capture = _CaptureHandler()
    access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
    access_logger.addHandler(capture)
    return capture.records, lambda: access_logger.removeHandler(capture)


def test_emit_access_record_payload_includes_ip_and_identity():
    import src.web_app as web_app

    records, cleanup = _capture_access_records(None)
    try:
        request = _make_request(
            method="POST",
            path="/api/chat/stream",
            query=b"token=sekrit&x=1",
            headers=[(b"user-agent", b"pytest-agent")],
        )
        request.state.api_identity = SimpleNamespace(key_id="rag_test", role="user")
        web_app._emit_access_record(request, 200, 0.1234)
    finally:
        cleanup()

    assert len(records) == 1
    payload = _parse_access_payload(records[0])
    assert payload["ip"] == "192.0.2.44"
    assert payload["method"] == "POST"
    assert payload["path"] == "/api/chat/stream"
    assert payload["query"] == "token=***&x=1"
    assert payload["status"] == 200
    assert payload["duration_ms"] == 123.4
    assert payload["user_agent"] == "pytest-agent"
    assert payload["key_id"] == "rag_test"
    assert payload["role"] == "user"


def test_emit_access_record_without_identity():
    import src.web_app as web_app

    records, cleanup = _capture_access_records(None)
    try:
        web_app._emit_access_record(_make_request(), 404, 0.001)
    finally:
        cleanup()

    payload = _parse_access_payload(records[0])
    assert payload["key_id"] is None
    assert payload["role"] is None
    assert payload["status"] == 404


def test_access_middleware_logs_and_passes_response_through():
    import src.web_app as web_app

    async def call_next(request):
        return Response("ok", status_code=201)

    records, cleanup = _capture_access_records(None)
    try:
        response = asyncio.run(
            web_app._access_log_request(_make_request(query=b"token=leak"), call_next)
        )
    finally:
        cleanup()

    assert response.status_code == 201
    payload = _parse_access_payload(records[0])
    assert payload["status"] == 201
    assert payload["query"] == "token=***"


def test_access_middleware_disabled_by_global_toggle(monkeypatch):
    import src.web_app as web_app

    monkeypatch.setattr(web_app, "ACCESS_LOG_ENABLED", False)

    async def call_next(request):
        return Response("ok")

    records, cleanup = _capture_access_records(None)
    try:
        response = asyncio.run(web_app._access_log_request(_make_request(), call_next))
    finally:
        cleanup()

    assert response.status_code == 200
    assert records == []


def test_access_middleware_logs_500_and_reraises_handler_errors():
    import src.web_app as web_app

    async def boom(request):
        raise RuntimeError("handler exploded")

    records, cleanup = _capture_access_records(None)
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(web_app._access_log_request(_make_request(), boom))
    finally:
        cleanup()

    assert _parse_access_payload(records[0])["status"] == 500


def test_logging_config_section_loads(safe_tmp_path):
    from src.config import load_config
    from src.web_app import _load_logging_config

    config_path = safe_tmp_path / "config.toml"
    config_path.write_text(
        "[logging]\n"
        'level = "debug"\n'
        'file = ""\n'
        'access_file = "logs/custom_access.log"\n',
        encoding="utf-8",
    )

    # The dataclass carries raw TOML values (upper-casing happens per
    # consumer, mirroring every other section).
    cfg = load_config(config_path)
    assert cfg.logging.level == "debug"
    assert cfg.logging.file == ""
    assert cfg.logging.access_file == "logs/custom_access.log"

    settings = _load_logging_config(config_path)
    assert settings["level"] == "DEBUG"
    assert settings["file"] == ""
    assert settings["access_file"] == "logs/custom_access.log"


def test_logging_config_invalid_level_falls_back_to_info(safe_tmp_path):
    from src.defaults import DEFAULT_LOG_LEVEL
    from src.web_app import _load_logging_config

    config_path = safe_tmp_path / "config.toml"
    config_path.write_text('[logging]\nlevel = "LOUD"\n', encoding="utf-8")
    settings = _load_logging_config(config_path)
    assert settings["level"] == DEFAULT_LOG_LEVEL
