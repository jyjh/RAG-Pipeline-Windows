"""API surface for categories ("split databases").

Exercises the /api/categories CRUD, the /api/pdfs category facet, the bulk
move (transfer job) endpoint, and the chat category validation -- all against
a fully isolated workspace (patched DATA_DIR/DB_DIR/registry paths).
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.web_app as web_app
from src.pdf_registry import PdfRegistry, write_source_entry


class StubQueue:
    def __init__(self):
        self.transfer_calls = []

    def enqueue_transfer_sources(self, *, source_hashes, target_category, **kwargs):
        self.transfer_calls.append(
            {"source_hashes": list(source_hashes), "target_category": target_category}
        )
        return SimpleNamespace(to_dict=lambda: {"id": "job-1", "kind": "transfer_sources"})


@pytest.fixture
def workspace(safe_tmp_path, monkeypatch):
    """Isolated web_app workspace: patched data/db/registry/trust paths."""
    data_dir = safe_tmp_path / "data"
    db_dir = safe_tmp_path / "db"
    registry = data_dir / ".pdf_upload_registry.json"
    processed = safe_tmp_path / "processed_docs"
    data_dir.mkdir(parents=True)
    db_dir.mkdir()
    processed.mkdir()
    monkeypatch.setattr(web_app, "DATA_DIR", data_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", data_dir / ".document_trust.json")
    monkeypatch.setattr(web_app, "_index_mutation_blocker", lambda: "")
    stub = StubQueue()
    monkeypatch.setattr(web_app, "job_queue", stub)
    client = TestClient(web_app.app)
    return client, stub, registry, processed


def _register_source(registry_path, processed_dir, *, file_hash, filename):
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=processed_dir / f"{filename}.md",
        source_hash=file_hash,
        source_pdf_name=f"{filename}.pdf",
        source_pdf_path=str(processed_dir / f"{filename}.pdf"),
    )
    PdfRegistry(registry_path).register_queued(
        job_id="seed-job",
        files=[{"hash": file_hash, "filename": f"{filename}.pdf", "staging_path": ""}],
    )


def test_categories_crud_roundtrip(workspace):
    client, _, _, _ = workspace
    listed = client.get("/api/categories").json()
    assert [entry["key"] for entry in listed["categories"]] == ["general"]

    created = client.post("/api/categories", json={"key": "Team A", "label": "Team Design"})
    assert created.status_code == 200
    body = created.json()["category"]
    # Free-form names are slugged; the label keeps the user's text.
    assert body["key"] == "team-a"
    assert body["label"] == "Team Design"
    assert body["db_dir"] == str(web_app.DB_DIR / "categories" / "team-a")

    listed = client.get("/api/categories").json()
    by_key = {entry["key"]: entry for entry in listed["categories"]}
    assert set(by_key) == {"general", "team-a"}
    assert by_key["team-a"]["exists"] is False
    assert by_key["general"]["db_dir"] == str(web_app.DB_DIR)

    renamed = client.post("/api/categories/team-a/label", json={"label": "Team Docs"})
    assert renamed.status_code == 200
    assert renamed.json()["category"]["label"] == "Team Docs"

    deleted = client.delete("/api/categories/team-a")
    assert deleted.status_code == 200
    assert [entry["key"] for entry in client.get("/api/categories").json()["categories"]] == ["general"]


def test_category_validation_errors(workspace):
    client, _, _, _ = workspace
    # Free-form names are slugged, not rejected...
    assert client.post("/api/categories", json={"key": "bad key!"}).status_code == 200
    listed = client.get("/api/categories").json()
    assert "bad-key" in [entry["key"] for entry in listed["categories"]]
    # ...but input that slugifies to nothing is rejected, as are duplicates
    # and the reserved General key.
    assert client.post("/api/categories", json={"key": "///"}).status_code == 400
    assert client.post("/api/categories", json={"key": "general"}).status_code == 400
    assert client.post("/api/categories", json={"key": "ok", "label": "x"}).status_code == 200
    assert client.post("/api/categories", json={"key": "ok"}).status_code == 400  # duplicate
    assert client.post("/api/categories/ghost/label", json={"label": "x"}).status_code == 400
    assert client.post("/api/categories/general/label", json={"label": "x"}).status_code == 400
    assert client.delete("/api/categories/ghost").status_code == 404


def test_delete_category_with_members_conflicts(workspace):
    client, _, _, _ = workspace
    assert client.post("/api/categories", json={"key": "team"}).status_code == 200
    # Register a member directly through the store (as a transfer job would).
    web_app._category_store().set_memberships(["hash-1"], "team")
    conflict = client.delete("/api/categories/team")
    assert conflict.status_code == 409
    assert "document(s)" in conflict.json()["detail"]


def test_pdfs_category_facet_and_row_field(workspace):
    client, _, registry, processed = workspace
    _register_source(registry, processed, file_hash="a" * 64, filename="general-book")
    # No membership -> General on the row.
    rows = client.get("/api/pdfs", params={"limit": 10}).json()["pdfs"]
    assert rows[0]["category"] == "general"
    assert client.get("/api/pdfs", params={"category": "team"}).json()["total"] == 0
    assert client.get("/api/pdfs", params={"category": "general"}).json()["total"] == 1

    # Assign membership; the row moves and the facet follows.
    assert client.post("/api/categories", json={"key": "team"}).status_code == 200
    web_app._category_store().set_memberships(["a" * 64], "team")
    rows = client.get("/api/pdfs", params={"category": "team", "limit": 10}).json()["pdfs"]
    assert [row["hash"] for row in rows] == ["a" * 64]
    assert rows[0]["category"] == "team"
    assert client.get("/api/pdfs", params={"category": "general"}).json()["total"] == 0


def test_bulk_move_enqueues_transfer_job(workspace):
    client, stub, registry, processed = workspace
    _register_source(registry, processed, file_hash="b" * 64, filename="movable")
    assert client.post("/api/categories", json={"key": "team"}).status_code == 200

    response = client.post(
        "/api/pdfs/categories/bulk",
        json={"source_hashes": ["b" * 64, "f" * 64], "category": "team"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "transfer_sources"
    assert body["failed"] == [{"source_hash": "f" * 64, "error": "Unknown source hash."}]
    assert stub.transfer_calls == [
        {"source_hashes": ["b" * 64], "target_category": "team"}
    ]

    # Moving to an unknown category fails validation without enqueueing.
    assert client.post(
        "/api/pdfs/categories/bulk", json={"source_hashes": ["b" * 64], "category": "ghost"}
    ).status_code == 400
    assert len(stub.transfer_calls) == 1
    # Empty hash list after validation fails.
    assert client.post(
        "/api/pdfs/categories/bulk", json={"source_hashes": [""], "category": "team"}
    ).status_code == 400


def test_chat_unknown_category_is_clean_400(workspace):
    client, _, _, _ = workspace
    response = client.post(
        "/api/chat/stream", json={"question": "what is the fuel map?", "categories": ["ghost"]}
    )
    assert response.status_code == 400
    assert "Unknown category" in response.json()["detail"]


def test_index_endpoints_reject_unknown_category(workspace):
    client, _, _, _ = workspace
    assert client.get("/api/index", params={"category": "ghost"}).status_code == 400
    assert client.get("/api/index/summaries", params={"category": "ghost"}).status_code == 400
    assert client.get("/api/index/children", params={"parent_id": "x", "category": "ghost"}).status_code == 400
    assert client.get("/api/index/stream", params={"category": "ghost"}).status_code == 400
    assert client.post(
        "/api/index/vector-search", json={"query": "x", "category": "ghost"}
    ).status_code == 400
    assert client.post(
        "/api/index/delete", json={"record_ids": ["r1"], "category": "ghost"}
    ).status_code == 400
    assert client.post(
        "/api/index/update", json={"record_id": "r1", "content": "c", "category": "ghost"}
    ).status_code == 400
    # A known (but not-yet-built) category surfaces the standard 404.
    client.post("/api/categories", json={"key": "team"})
    assert client.get("/api/index", params={"category": "team"}).status_code == 404
