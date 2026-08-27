"""Unit tests for the LLM source-group auto-tag plugin (src/auto_tag.py)."""

import json

import pytest

from src.auto_tag import (
    AutoTagDecision,
    AutoTagInput,
    build_user_prompt,
    classify_documents,
    parse_classification_response,
    pdf_excerpt,
)


def _reply(rows: list[dict]) -> str:
    return json.dumps(rows)


class TestParseClassificationResponse:
    def test_parses_plain_json_array(self):
        reply = _reply(
            [
                {"hash": "h1", "source_group": "official", "confidence": 0.9, "reason": "rules"},
                {"hash": "h2", "source_group": "student_research", "confidence": 0.8, "reason": "thesis"},
            ]
        )
        decisions = parse_classification_response(reply, ["h1", "h2"])
        assert set(decisions) == {"h1", "h2"}
        assert decisions["h1"].source_group == "official"
        assert decisions["h1"].confidence == 0.9
        assert decisions["h2"].source_group == "student_research"

    def test_parses_fenced_json_with_prose(self):
        reply = (
            "Here is the classification:\n```json\n"
            + _reply([{"hash": "h1", "source_group": "unofficial", "confidence": 0.75}])
            + "\n```\nDone."
        )
        decisions = parse_classification_response(reply, ["h1"])
        assert decisions["h1"].source_group == "unofficial"

    def test_drops_unknown_and_invalid_groups(self):
        reply = _reply(
            [
                {"hash": "unknown-hash", "source_group": "official", "confidence": 0.99},
                {"hash": "h1", "source_group": "ungrouped", "confidence": 0.99},
                {"hash": "h2", "source_group": "nonsense", "confidence": 0.99},
                {"hash": "h3", "source_group": "official", "confidence": 0.99},
            ]
        )
        decisions = parse_classification_response(reply, ["h1", "h2", "h3"])
        assert set(decisions) == {"h3"}

    def test_drops_below_confidence_floor(self):
        reply = _reply(
            [
                {"hash": "h1", "source_group": "official", "confidence": 0.4},
                {"hash": "h2", "source_group": "official", "confidence": 0.61},
            ]
        )
        decisions = parse_classification_response(reply, ["h1", "h2"], min_confidence=0.6)
        assert set(decisions) == {"h2"}

    def test_drops_malformed_rows_without_losing_batch(self):
        reply = _reply(
            [
                "not-a-dict",
                {"source_group": "official", "confidence": 0.9},
                {"hash": "h2", "source_group": "official"},
                {"hash": "h3", "source_group": "official", "confidence": "oops"},
                {"hash": "h4", "source_group": "official", "confidence": 0.9},
            ]
        )
        decisions = parse_classification_response(reply, ["h1", "h2", "h3", "h4"])
        assert set(decisions) == {"h4"}

    def test_first_decision_wins_on_duplicate_hash(self):
        reply = _reply(
            [
                {"hash": "h1", "source_group": "official", "confidence": 0.9},
                {"hash": "h1", "source_group": "unofficial", "confidence": 0.95},
            ]
        )
        decisions = parse_classification_response(reply, ["h1"])
        assert decisions["h1"].source_group == "official"

    def test_clamps_confidence_range(self):
        reply = _reply(
            [
                {"hash": "h1", "source_group": "official", "confidence": 7},
                {"hash": "h2", "source_group": "official", "confidence": -2},
            ]
        )
        # Floor 0.0 so the clamp itself is exercised (a clamped-to-0 value
        # would otherwise be dropped by the default 0.6 floor, by design).
        decisions = parse_classification_response(reply, ["h1", "h2"], min_confidence=0.0)
        assert decisions["h1"].confidence == 1.0
        assert decisions["h2"].confidence == 0.0

    def test_no_array_returns_empty(self):
        assert parse_classification_response("I cannot classify these.", ["h1"]) == {}
        assert parse_classification_response("", ["h1"]) == {}


class TestClassifyDocuments:
    def test_batches_items_one_call_per_batch(self):
        items = [AutoTagInput(f"h{i}", f"doc{i}.pdf") for i in range(5)]
        calls: list[list[AutoTagInput]] = []

        def fake_chat_fn(*, model, messages, timeout=None):
            batch_hashes = json.loads(messages[-1]["content"].split("\n\n", 1)[1])
            calls.append([row["hash"] for row in batch_hashes])
            return _reply(
                [
                    {"hash": row["hash"], "source_group": "official", "confidence": 0.9, "reason": "ok"}
                    for row in batch_hashes
                ]
            )

        decisions = classify_documents(
            items, model="gemma4:26b", chat_fn=fake_chat_fn, batch_size=2
        )

        assert calls == [["h0", "h1"], ["h2", "h3"], ["h4"]]
        assert set(decisions) == {f"h{i}" for i in range(5)}

    def test_failed_batch_is_skipped_not_fatal(self):
        items = [AutoTagInput(f"h{i}", f"doc{i}.pdf") for i in range(4)]

        def fake_chat_fn(*, model, messages, timeout=None):
            batch_hashes = json.loads(messages[-1]["content"].split("\n\n", 1)[1])
            if batch_hashes[0]["hash"] == "h0":
                raise RuntimeError("backend down")
            return _reply(
                [{"hash": row["hash"], "source_group": "unofficial", "confidence": 0.9} for row in batch_hashes]
            )

        decisions = classify_documents(items, chat_fn=fake_chat_fn, batch_size=2)

        assert set(decisions) == {"h2", "h3"}

    def test_prompt_carries_filename_and_excerpt(self):
        prompt = build_user_prompt([AutoTagInput("h1", "Rules 2026.pdf", "FSAE rules part T")])
        assert "h1" in prompt
        assert "Rules 2026.pdf" in prompt
        assert "FSAE rules part T" in prompt


class TestPdfExcerpt:
    def test_missing_file_returns_empty(self, tmp_path):
        assert pdf_excerpt(tmp_path / "nope.pdf") == ""

    def test_non_pdf_returns_empty(self, tmp_path):
        path = tmp_path / "fake.pdf"
        path.write_bytes(b"definitely not a pdf")
        assert pdf_excerpt(path) == ""

    def test_extracts_first_page_text(self, tmp_path):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        path = tmp_path / "doc.pdf"
        with path.open("wb") as handle:
            writer.write(handle)
        # Blank page -> no text; excerpt degrades to "" instead of raising.
        assert pdf_excerpt(path) == ""


def test_decision_dataclass_fields():
    decision = AutoTagDecision(source_group="official", confidence=0.9, reason="rules")
    assert decision.source_group == "official"
    assert decision.confidence == 0.9
    assert decision.reason == "rules"
