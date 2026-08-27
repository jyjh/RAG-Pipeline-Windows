"""Integration tests for LLM auto-tagging and the bulk trust-write paths."""

import json
import threading
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.web_app as web_app
from src.auto_tag import AutoTagDecision
from conftest import _rmtree_with_retry


@pytest.fixture
def workspace_tmp():
    path = Path.cwd() / f".tmp_test_web_app_auto_tag_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        _rmtree_with_retry(path)


@pytest.fixture(autouse=True)
def _inline_auto_tag_threads(monkeypatch):
    """Run the auto-tag worker synchronously instead of in a real thread.

    Deterministic assertions with no joins/sleeps, and immune to lingering
    auto-tag threads spawned by other tests in the same suite run.
    """

    class _InlineThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

    real_thread = threading.Thread
    monkeypatch.setattr(web_app.threading, "Thread", _InlineThread)
    yield
    monkeypatch.setattr(web_app.threading, "Thread", real_thread)


@pytest.fixture
def trust_env(monkeypatch, workspace_tmp):
    """Isolated registry/source-map/trust/db paths plus a quiet auto-tag state."""
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    trust_path = workspace_tmp / "trust.json"
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", trust_path)
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")
    with web_app._AUTO_TAG_LOCK:
        web_app._AUTO_TAG_STATE.update(
            {"running": False, "started_at": "", "finished_at": "", "queued": 0, "tagged": 0, "last_error": ""}
        )
    return trust_path


def _register_pdfs(trust_env, hashes, *, group=""):
    files = [{"filename": f"{h}.pdf", "hash": h, "staging_path": ""} for h in hashes]
    web_app.PdfRegistry(web_app.PDF_REGISTRY_PATH).register_queued(job_id="job", files=files)
    return files


def _stub_classifier(monkeypatch, decisions):
    calls: list[list[str]] = []

    def fake_classify(items, **kwargs):
        calls.append([item.source_hash for item in items])
        return decisions

    monkeypatch.setattr(web_app.auto_tag, "classify_documents", fake_classify)
    return calls


class TestTrustEntryAutoFlag:
    def test_normalize_preserves_auto_fields_and_coerces_confidence(self, trust_env):
        entry = web_app._normalize_trust_entry(
            "hash-a",
            {
                "source_group": "official",
                "auto_tagged": True,
                "auto_tag_model": "gemma4:26b",
                "auto_tagged_at": "2026-08-26T00:00:00+00:00",
                "auto_tag_reason": "rules document",
                "auto_tag_confidence": "1.7",
            },
        )
        assert entry["auto_tagged"] is True
        assert entry["auto_tag_model"] == "gemma4:26b"
        assert entry["auto_tag_reason"] == "rules document"
        assert entry["auto_tag_confidence"] == 1.0  # clamped, not string

    def test_default_entry_has_no_auto_flag(self, trust_env):
        entry = web_app._default_trust_entry("hash-a")
        assert entry["auto_tagged"] is False
        assert entry["auto_tag_model"] == ""
        assert entry["auto_tag_confidence"] is None

    def test_apply_auto_tag_decisions_flags_entries(self, trust_env):
        entries = web_app.apply_auto_tag_decisions(
            {"hash-a": AutoTagDecision("official", 0.9, "FSAE rules")},
            model="gemma4:26b",
        )
        entry = entries["hash-a"]
        assert entry["source_group"] == "official"
        assert entry["reliability_weight"] == 1.0
        assert entry["auto_tagged"] is True
        assert entry["auto_tag_model"] == "gemma4:26b"
        assert entry["auto_tag_confidence"] == 0.9
        assert entry["auto_tag_reason"] == "FSAE rules"
        assert entry["review_status"] == "unreviewed"  # auto tag is not approval

        persisted = json.loads(web_app.DOCUMENT_TRUST_PATH.read_text(encoding="utf-8"))
        assert persisted["documents"]["hash-a"]["auto_tagged"] is True

    def test_manual_source_group_update_clears_auto_flag(self, trust_env):
        web_app.apply_auto_tag_decisions(
            {"hash-a": AutoTagDecision("unofficial", 0.8, "forum guide")},
            model="gemma4:26b",
        )
        entry = web_app.update_document_trust("hash-a", {"source_group": "official"})
        assert entry["source_group"] == "official"
        assert entry["auto_tagged"] is False
        assert entry["auto_tag_model"] == ""
        assert entry["auto_tag_confidence"] is None
        assert entry["auto_tag_reason"] == ""

    def test_unrelated_manual_update_keeps_auto_flag(self, trust_env):
        web_app.apply_auto_tag_decisions(
            {"hash-a": AutoTagDecision("official", 0.9, "rules")},
            model="gemma4:26b",
        )
        entry = web_app.update_document_trust("hash-a", {"notes": "checked cover page"})
        assert entry["notes"] == "checked cover page"
        assert entry["auto_tagged"] is True


class TestBulkTrustWrites:
    def test_update_documents_trust_single_write_for_many_hashes(self, trust_env, monkeypatch):
        writes = {"count": 0}
        original = web_app._write_trust_registry

        def counting_write(payload, path=None):
            writes["count"] += 1
            return original(payload, path)

        monkeypatch.setattr(web_app, "_write_trust_registry", counting_write)

        entries = web_app.update_documents_trust(
            {f"hash-{i}": {"source_group": "official"} for i in range(50)}
        )

        assert len(entries) == 50
        assert writes["count"] == 1
        persisted = json.loads(web_app.DOCUMENT_TRUST_PATH.read_text(encoding="utf-8"))
        assert sum(1 for e in persisted["documents"].values() if e["source_group"] == "official") == 50

    def test_bulk_endpoint_builds_response_from_one_listing(self, trust_env, monkeypatch):
        _register_pdfs(trust_env, ["hash-a", "hash-b", "hash-c"])
        listings = {"count": 0}
        original = web_app.list_pdf_documents

        def counting_list(**kwargs):
            listings["count"] += 1
            return original(**kwargs)

        monkeypatch.setattr(web_app, "list_pdf_documents", counting_list)

        response = TestClient(web_app.app).post(
            "/api/pdfs/trust/bulk",
            json={"source_hashes": ["hash-a", "hash-b", "hash-c"], "source_group": "official"},
        )

        assert response.status_code == 200
        assert len(response.json()["updated"]) == 3
        assert listings["count"] == 1  # was 3+ full corpus listings before

    def test_bulk_endpoint_manual_tag_clears_auto_flag(self, trust_env, monkeypatch):
        _register_pdfs(trust_env, ["hash-a"])
        web_app.apply_auto_tag_decisions(
            {"hash-a": AutoTagDecision("unofficial", 0.7, "guess")}, model="gemma4:26b"
        )

        response = TestClient(web_app.app).post(
            "/api/pdfs/trust/bulk",
            json={"source_hashes": ["hash-a"], "source_group": "student_research"},
        )

        assert response.status_code == 200
        trust = response.json()["updated"][0]["trust"]
        assert trust["source_group"] == "student_research"
        assert trust["auto_tagged"] is False


class TestAutoTagEndpoints:
    def test_auto_tag_disabled_returns_409(self, trust_env, monkeypatch):
        monkeypatch.setattr(
            web_app,
            "_auto_tag_settings",
            lambda: {
                "enabled": False, "model": "gemma4:26b", "batch_size": 20,
                "min_confidence": 0.6, "excerpt_chars": 1200, "timeout": 120.0,
                "max_items_per_run": 200,
            },
        )
        response = TestClient(web_app.app).post("/api/pdfs/trust/auto-tag", json={})
        assert response.status_code == 409

    def test_auto_tag_endpoint_runs_and_flags_results(self, trust_env, monkeypatch):
        _register_pdfs(trust_env, ["hash-a", "hash-b"])
        # hash-b is already manually grouped -> must be left alone.
        web_app.update_document_trust("hash-b", {"source_group": "official"})
        _stub_classifier(
            monkeypatch,
            {
                "hash-a": AutoTagDecision("student_research", 0.85, "final-year thesis"),
                "hash-b": AutoTagDecision("unofficial", 0.9, "should not apply"),
            },
        )

        response = TestClient(web_app.app).post("/api/pdfs/trust/auto-tag", json={})

        assert response.status_code == 200
        assert response.json()["status"] == "running"
        assert response.json()["queued"] == ["hash-a"]

        persisted = json.loads(web_app.DOCUMENT_TRUST_PATH.read_text(encoding="utf-8"))
        assert persisted["documents"]["hash-a"]["source_group"] == "student_research"
        assert persisted["documents"]["hash-a"]["auto_tagged"] is True
        assert persisted["documents"]["hash-a"]["auto_tag_confidence"] == 0.85
        assert persisted["documents"]["hash-b"]["source_group"] == "official"
        assert persisted["documents"]["hash-b"]["auto_tagged"] is False

        status = TestClient(web_app.app).get("/api/pdfs/trust/auto-tag").json()
        assert status["running"] is False
        assert status["tagged"] == 1

    def test_auto_tag_endpoint_respects_requested_hashes(self, trust_env, monkeypatch):
        _register_pdfs(trust_env, ["hash-a", "hash-b"])
        calls = _stub_classifier(
            monkeypatch, {"hash-b": AutoTagDecision("unofficial", 0.8, "blog post")}
        )

        response = TestClient(web_app.app).post(
            "/api/pdfs/trust/auto-tag", json={"source_hashes": ["hash-b"]}
        )

        assert response.status_code == 200
        assert response.json()["queued"] == ["hash-b"]
        assert calls and calls[0] == ["hash-b"]

    def test_auto_tag_endpoint_without_ungrouped_is_idle(self, trust_env, monkeypatch):
        _register_pdfs(trust_env, ["hash-a"])
        web_app.update_document_trust("hash-a", {"source_group": "official"})

        response = TestClient(web_app.app).post("/api/pdfs/trust/auto-tag", json={})

        assert response.status_code == 200
        assert response.json()["status"] == "idle"
        assert response.json()["queued"] == []


class TestUploadAutoTagScheduling:
    def test_ungrouped_uploads_are_auto_tagged_in_background(self, trust_env, monkeypatch):
        uploads = [
            {"hash": "hash-u", "filename": "u.pdf", "staging_path": "", "source_group": "ungrouped"},
            {"hash": "hash-g", "filename": "g.pdf", "staging_path": "", "source_group": "official"},
        ]
        _stub_classifier(
            monkeypatch, {"hash-u": AutoTagDecision("official", 0.9, "manufacturer datasheet")}
        )

        web_app._schedule_upload_auto_tag(uploads)

        persisted = json.loads(web_app.DOCUMENT_TRUST_PATH.read_text(encoding="utf-8"))
        assert persisted["documents"]["hash-u"]["auto_tagged"] is True
        assert persisted["documents"]["hash-u"]["source_group"] == "official"
        # Manually grouped uploads bypass the auto-tagger entirely.
        assert "hash-g" not in persisted["documents"]

    def test_noop_when_disabled(self, trust_env, monkeypatch):
        monkeypatch.setattr(
            web_app,
            "_auto_tag_settings",
            lambda: {
                "enabled": False, "model": "gemma4:26b", "batch_size": 20,
                "min_confidence": 0.6, "excerpt_chars": 0, "timeout": 120.0,
                "max_items_per_run": 200,
            },
        )
        calls = _stub_classifier(monkeypatch, {})
        web_app._schedule_upload_auto_tag(
            [{"hash": "hash-u", "filename": "u.pdf", "staging_path": "", "source_group": "ungrouped"}]
        )
        assert calls == []
        assert not web_app.DOCUMENT_TRUST_PATH.exists()
