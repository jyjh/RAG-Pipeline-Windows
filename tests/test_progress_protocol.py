"""Tests for the structured subprocess progress protocol.

Covers the round-trip used by the web job runner: a worker emits a
``__RAG_PROGRESS__`` line, the reader thread parses it, and the payload
lands on ``QueueJob.progress``. Parsing must degrade gracefully (return
``None``) for ordinary log lines, partial output, and malformed JSON so a
blip in the subprocess never breaks job-log streaming.
"""

from __future__ import annotations

import io

from src.progress_protocol import (
    PROGRESS_PREFIX,
    emit_progress,
    is_progress_line,
    parse_progress_line,
)


def test_emit_then_parse_round_trip():
    buf = io.StringIO()
    emit_progress(
        phase="indexing",
        done=42,
        total=1000,
        unit="files",
        rate_per_min=30.5,
        eta_seconds=1850.2,
        stream=buf,
    )
    line = buf.getvalue().strip()
    assert is_progress_line(line)
    payload = parse_progress_line(line)
    assert payload == {
        "phase": "indexing",
        "done": 42,
        "total": 1000,
        "unit": "files",
        "rate_per_min": 30.5,
        "eta_seconds": 1850.2,
    }


def test_emit_carries_extra_fields_without_clobbering_reserved():
    buf = io.StringIO()
    emit_progress(
        phase="indexing",
        done=5,
        extra={"records_written": 12345, "failed_files": 0},
        stream=buf,
    )
    payload = parse_progress_line(buf.getvalue().strip())
    assert payload is not None
    # Reserved field wins over an `extra` collision.
    assert payload["phase"] == "indexing"
    assert payload["records_written"] == 12345
    assert payload["failed_files"] == 0


def test_parse_non_progress_line_returns_none():
    assert parse_progress_line("ordinary log output") is None
    assert parse_progress_line("") is None
    assert parse_progress_line("Local index: found 3 Markdown file(s).") is None


def test_parse_malformed_payload_returns_none():
    # Correct prefix but broken JSON -- must not raise, must not match.
    assert parse_progress_line(PROGRESS_PREFIX + "{not json") is None
    assert parse_progress_line(PROGRESS_PREFIX) is None
    # A JSON array (not an object) is also rejected.
    assert parse_progress_line(PROGRESS_PREFIX + "[1, 2, 3]") is None


def test_emit_handles_non_serializable_extra_without_raising():
    buf = io.StringIO()
    # A non-JSON-serializable object in `extra` must not crash the worker.
    emit_progress(phase="indexing", done=1, extra={"bad": object()}, stream=buf)
    # Nothing was emitted (serialization failed silently).
    assert buf.getvalue() == ""


def test_is_progress_line_is_cheap_membership_check():
    assert is_progress_line(PROGRESS_PREFIX + '{"done": 1}') is True
    assert is_progress_line("  " + PROGRESS_PREFIX + '{"done": 1}') is True
    assert is_progress_line("not a progress line") is False


def test_optional_fields_omitted_when_none():
    buf = io.StringIO()
    emit_progress(phase="building_ann_index", done=0, stream=buf)
    payload = parse_progress_line(buf.getvalue().strip())
    assert payload == {"phase": "building_ann_index", "done": 0, "unit": "items"}
