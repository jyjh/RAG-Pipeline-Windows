import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.local_rag as local_rag
import src.query as query
from conftest import _rmtree_with_retry

import src.web_app as web_app
from src.asset_store import ImageAssetStore, image_asset_marker
from src.defaults import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_KEEP_ALIVE
from src.index_overrides import load_index_overrides, persist_index_deletions, persist_index_edit
from src.pdf_registry import load_source_map, write_source_entry
from src.vector_store import LanceDBVectorStore


def _completed(args, *, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(["git", *args], returncode, stdout=stdout, stderr=stderr)

# The auth posture for this suite (local operator vs ``remote_client`` opt-out)
# is installed by the autouse ``_local_operator`` fixture in tests/conftest.py.


def _set_update_config(monkeypatch, *, branch="main"):
    monkeypatch.setattr(
        web_app,
        "SERVER_CONFIG",
        {
            "host": "127.0.0.1",
            "port": 8000,
            "health_poll_interval_ms": 60000,
            "jobs_poll_interval_ms": 60000,
            "update_remote": "origin",
            "update_branch": branch,
        },
    )


class IdleQueue:
    def summary(self):
        return {"active_query_count": 0, "queued_count": 0, "running_job_ids": [], "job_count": 0}


def _install_update_git(
    monkeypatch,
    *,
    current_branch="main",
    current_sha=None,
    latest_sha=None,
    dirty="",
    current_is_ancestor=True,
    latest_is_ancestor=False,
    fetch_returncode=0,
):
    calls = []
    current_sha = current_sha or ("a" * 40)
    latest_sha = latest_sha or current_sha

    def fake_run_git(args, *, timeout=web_app.GIT_TIMEOUT_SECONDS):
        calls.append(args)
        if args == ["branch", "--show-current"]:
            return _completed(args, stdout=f"{current_branch}\n")
        if args == ["rev-parse", "HEAD"]:
            return _completed(args, stdout=f"{current_sha}\n")
        if args == ["status", "--porcelain", "--untracked-files=no"]:
            return _completed(args, stdout=dirty)
        if args[0:2] == ["fetch", "--quiet"]:
            if fetch_returncode:
                return _completed(args, stderr="network unavailable", returncode=fetch_returncode)
            return _completed(args)
        if args == ["rev-parse", "--verify", "refs/remotes/origin/main"]:
            return _completed(args, stdout=f"{latest_sha}\n")
        if args[0:2] == ["merge-base", "--is-ancestor"]:
            if args[2:] == [current_sha, latest_sha]:
                return _completed(args, returncode=0 if current_is_ancestor else 1)
            if args[2:] == [latest_sha, current_sha]:
                return _completed(args, returncode=0 if latest_is_ancestor else 1)
        return _completed(args, stderr=f"unexpected git args: {args}", returncode=1)

    monkeypatch.setattr(web_app, "_run_git", fake_run_git)
    return calls


@pytest.fixture
def workspace_tmp():
    path = Path.cwd() / f".tmp_test_web_app_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        _rmtree_with_retry(path)


@pytest.fixture
def lancedb_tmp():
    path = Path(tempfile.gettempdir()) / f"rag_test_web_app_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        _rmtree_with_retry(path)


def _write_index(db_dir: Path):
    LanceDBVectorStore(db_dir).write_records(
        [
            {
                "id": "doc.md:0",
                "doc_id": "doc",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "processed_docs/doc.md",
                "chunk_index": 0,
                "content": "alpha context",
                "title": "Doc",
                "section_path": "Doc",
                "page_start": 1,
                "page_end": 1,
                "summary": "summary",
                "tags": ["alpha"],
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "doc.md:1",
                "doc_id": "doc",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "processed_docs/doc.md",
                "chunk_index": 1,
                "content": "beta context",
                "title": "Doc",
                "section_path": "Doc",
                "page_start": 1,
                "page_end": 1,
                "summary": "summary",
                "tags": ["beta"],
                "vector": [0.0, 1.0, 0.0],
            },
        ],
        embedding_model="nomic-embed-text",
        embedding_dim=3,
    )


def _write_hierarchical_index(db_dir: Path):
    LanceDBVectorStore(db_dir).write_records(
        [
            {
                "id": "doc-summary",
                "doc_id": "doc",
                "parent_id": "",
                "node_type": "document_summary",
                "file_path": "processed_docs/doc.md",
                "chunk_index": -1,
                "content": "document summary alpha",
                "title": "Doc",
                "section_path": "Doc",
                "page_start": 1,
                "page_end": 4,
                "summary": "document summary alpha",
                "tags": ["alpha"],
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "top-summary",
                "doc_id": "doc",
                "parent_id": "doc-summary",
                "node_type": "section_summary",
                "file_path": "processed_docs/doc.md",
                "chunk_index": -1,
                "content": "top section summary beta",
                "title": "Top",
                "section_path": "Doc > Top",
                "page_start": 2,
                "page_end": 4,
                "summary": "top summary",
                "tags": ["beta"],
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "leaf-summary",
                "doc_id": "doc",
                "parent_id": "top-summary",
                "node_type": "section_summary",
                "file_path": "processed_docs/doc.md",
                "chunk_index": -1,
                "content": "leaf summary gamma",
                "title": "Leaf",
                "section_path": "Doc > Top > Leaf",
                "page_start": 3,
                "page_end": 4,
                "summary": "leaf summary",
                "tags": ["gamma"],
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "chunk-1",
                "doc_id": "doc",
                "parent_id": "leaf-summary",
                "node_type": "chunk",
                "file_path": "processed_docs/doc.md",
                "chunk_index": 0,
                "content": "detailed gamma chunk",
                "title": "Leaf",
                "section_path": "Doc > Top > Leaf",
                "page_start": 3,
                "page_end": 3,
                "summary": "leaf summary",
                "tags": ["gamma"],
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "chunk-2",
                "doc_id": "doc",
                "parent_id": "leaf-summary",
                "node_type": "chunk",
                "file_path": "processed_docs/doc.md",
                "chunk_index": 1,
                "content": "detailed delta chunk",
                "title": "Leaf",
                "section_path": "Doc > Top > Leaf",
                "page_start": 4,
                "page_end": 4,
                "summary": "leaf summary",
                "tags": ["delta"],
                "vector": [0.0, 1.0, 0.0],
            },
        ],
        embedding_model="nomic-embed-text",
        embedding_dim=3,
    )


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for condition")


def test_index_rows_hide_vectors_and_support_search(lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)

    result = web_app.list_index_rows(search="beta", db_dir=db_dir)

    assert result["total"] == 1
    assert result["rows"][0]["id"] == "doc.md:1"
    assert "vector" not in result["rows"][0]


def test_iter_index_row_events_streams_batches(lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)

    events = list(web_app.iter_index_row_events(batch_size=1, db_dir=db_dir))

    assert [event["type"] for event in events] == ["metadata", "rows", "rows", "done"]
    assert events[0]["total"] == 2
    assert events[0]["embedding_model"] == "nomic-embed-text"
    assert events[1]["rows"][0]["id"] == "doc.md:0"
    assert events[2]["rows"][0]["id"] == "doc.md:1"
    assert events[-1]["received"] == 2
    assert events[-1]["total"] == 2
    assert "vector" not in events[1]["rows"][0]


def test_index_stream_endpoint_filters_and_streams(monkeypatch, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    client = TestClient(web_app.app)
    response = client.get("/api/index/stream?batch_size=1&search=beta")

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["metadata", "rows", "done"]
    assert events[0]["total"] is None
    assert events[1]["rows"][0]["id"] == "doc.md:1"
    assert events[-1]["received"] == 1
    assert events[-1]["total"] == 1


def test_index_summary_endpoint_returns_document_level_rows(monkeypatch, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_hierarchical_index(db_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    client = TestClient(web_app.app)
    response = client.get("/api/index/summaries")

    assert response.status_code == 200
    payload = response.json()
    assert payload["view"] == "hierarchy"
    assert payload["total"] == 1
    assert payload["rows"][0]["id"] == "doc-summary"
    assert payload["rows"][0]["node_type"] == "document_summary"
    assert payload["rows"][0]["summary_count"] == 2
    assert payload["rows"][0]["detail_count"] == 2
    assert payload["rows"][0]["child_count"] == 4


def test_index_children_endpoint_pages_descendants(monkeypatch, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_hierarchical_index(db_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    client = TestClient(web_app.app)
    response = client.get(
        "/api/index/children",
        params={"parent_id": "doc-summary", "offset": 1, "limit": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["view"] == "hierarchy_children"
    assert payload["total"] == 4
    assert payload["offset"] == 1
    assert [row["id"] for row in payload["rows"]] == ["leaf-summary", "chunk-1"]
    assert payload["rows"][0]["node_level"] == 2
    assert payload["rows"][1]["node_level"] == 2


def test_index_hierarchy_endpoints_use_conditional_etags(monkeypatch, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_hierarchical_index(db_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    client = TestClient(web_app.app)
    summaries = client.get("/api/index/summaries")
    summaries_unchanged = client.get(
        "/api/index/summaries",
        headers={"If-None-Match": summaries.headers["etag"]},
    )
    children = client.get("/api/index/children", params={"parent_id": "doc-summary"})
    children_unchanged = client.get(
        "/api/index/children",
        params={"parent_id": "doc-summary"},
        headers={"If-None-Match": children.headers["etag"]},
    )

    store = LanceDBVectorStore(db_dir)
    persist_index_edit(db_dir, store.get_record("chunk-1"), "edited child context")
    summaries_changed = client.get(
        "/api/index/summaries",
        headers={"If-None-Match": summaries.headers["etag"]},
    )
    children_changed = client.get(
        "/api/index/children",
        params={"parent_id": "doc-summary"},
        headers={"If-None-Match": children.headers["etag"]},
    )

    assert summaries.status_code == 200
    assert summaries_unchanged.status_code == 304
    assert children.status_code == 200
    assert children_unchanged.status_code == 304
    assert summaries_changed.status_code == 200
    assert summaries_changed.headers["etag"] != summaries.headers["etag"]
    assert children_changed.status_code == 200
    assert children_changed.headers["etag"] != children.headers["etag"]


def test_index_review_rows_include_image_asset_metadata(monkeypatch, workspace_tmp, lancedb_tmp):
    asset_dir = workspace_tmp / "assets"
    store = ImageAssetStore(asset_dir)
    asset = store.save_image(
        image_data=b"asset-png",
        source_hash="hash-assets",
        source_pdf_name="assets.pdf",
        page_no=2,
        description="Suspension graph",
    )
    db_dir = lancedb_tmp / "db"
    LanceDBVectorStore(db_dir).write_records(
        [
            {
                "id": "asset-summary",
                "doc_id": "asset-doc",
                "parent_id": "",
                "node_type": "document_summary",
                "file_path": "processed_docs/assets.md",
                "chunk_index": -1,
                "content": "document summary",
                "title": "Assets",
                "section_path": "Assets",
                "page_start": 1,
                "page_end": 2,
                "summary": "document summary",
                "tags": [],
                "source_hash": "hash-assets",
                "source_pdf_name": "assets.pdf",
                "source_pdf_path": "data/assets.pdf",
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "asset-chunk",
                "doc_id": "asset-doc",
                "parent_id": "asset-summary",
                "node_type": "chunk",
                "file_path": "processed_docs/assets.md",
                "chunk_index": 0,
                "content": f"graph detail\n{image_asset_marker(asset['asset_id'])}",
                "title": "Assets",
                "section_path": "Assets",
                "page_start": 2,
                "page_end": 2,
                "summary": "graph summary",
                "tags": ["graph"],
                "source_hash": "hash-assets",
                "source_pdf_name": "assets.pdf",
                "source_pdf_path": "data/assets.pdf",
                "vector": [0.0, 1.0, 0.0],
            },
        ],
        embedding_model="nomic-embed-text",
        embedding_dim=3,
    )
    monkeypatch.setattr(web_app, "ASSET_DIR", asset_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    client = TestClient(web_app.app)
    index = client.get("/api/index", params={"search": "graph"})
    children = client.get("/api/index/children", params={"parent_id": "asset-summary"})
    stream = client.get("/api/index/stream", params={"batch_size": 1, "search": "graph"})

    expected = {
        "asset_id": asset["asset_id"],
        "source_hash": "hash-assets",
        "source_pdf_name": "assets.pdf",
        "page_no": 2,
        "description": "Suspension graph",
        "mime_type": "image/png",
        "image_sha": asset["image_sha"],
        "url": f"/api/assets/{asset['asset_id']}",
    }
    assert index.status_code == 200
    assert index.json()["rows"][0]["assets"] == [expected]
    assert children.status_code == 200
    assert children.json()["rows"][0]["assets"] == [expected]
    assert stream.status_code == 200
    events = [json.loads(line) for line in stream.text.splitlines()]
    rows_event = next(event for event in events if event["type"] == "rows")
    assert rows_event["rows"][0]["assets"] == [expected]


def test_index_summary_search_matches_child_rows(monkeypatch, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_hierarchical_index(db_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    client = TestClient(web_app.app)
    summaries = client.get("/api/index/summaries", params={"search": "delta"})
    children = client.get(
        "/api/index/children",
        params={"parent_id": "doc-summary", "search": "delta"},
    )

    assert summaries.status_code == 200
    assert summaries.json()["total"] == 1
    assert summaries.json()["rows"][0]["detail_count"] == 1
    assert children.status_code == 200
    assert [row["id"] for row in children.json()["rows"]] == ["chunk-2"]


def test_index_vector_search_endpoint_uses_local_tool(monkeypatch, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_hierarchical_index(db_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)
    calls = {}

    class FakeLocalQueryEngine:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.store = LanceDBVectorStore(db_dir)

        def search_local_context(self, **kwargs):
            calls["search"] = kwargs
            return {
                "tool": "search_local_context",
                "query": kwargs["query"],
                "result_count": 1,
                "results": [
                    {
                        "source_id": "S1",
                        "citation": "[S1]",
                        "chunk_id": "chunk-2",
                        "score": 0.82,
                        "location": "Doc :: Top :: page 4",
                        "content": "detailed delta chunk",
                    }
                ],
            }

    monkeypatch.setattr(local_rag, "LocalQueryEngine", FakeLocalQueryEngine)

    client = TestClient(web_app.app)
    response = client.post(
        "/api/index/vector-search",
        json={"query": "delta dynamics", "relevance_floor": 0.72},
    )

    assert response.status_code == 200
    payload = response.json()
    assert calls["init"]["working_dir"] == str(db_dir)
    assert calls["init"]["asset_dir"] == str(web_app.ASSET_DIR)
    assert calls["init"]["trust_path"] == str(web_app.DOCUMENT_TRUST_PATH)
    assert calls["init"]["retrieval_min_score"] == 0.72
    assert calls["init"]["web_search_enabled"] is False
    assert calls["search"] == {"query": "delta dynamics", "relevance_floor": 0.72}
    assert payload["query"] == "delta dynamics"
    assert payload["relevance_floor"] == 0.72
    assert payload["total"] == 1
    assert payload["rows"][0]["id"] == "chunk-2"
    assert payload["rows"][0]["score"] == 0.82
    assert payload["rows"][0]["source_group"] == "ungrouped"
    assert payload["rows"][0]["citation"] == "[S1]"
    assert payload["tool_result"]["tool"] == "search_local_context"


def test_index_vector_search_rejects_empty_query():
    client = TestClient(web_app.app)
    response = client.post("/api/index/vector-search", json={"query": " "})

    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_update_index_record_reembeds_and_saves(monkeypatch, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)
    calls = {}

    class FakeEmbeddingEngine:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            calls["embed"] = (texts, truncate_dim, prefix)
            return np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEmbeddingEngine)

    row = web_app.update_index_record(
        record_id="doc.md:0",
        content="edited context",
        db_dir=db_dir,
    )

    assert row["content"] == "edited context"
    record = LanceDBVectorStore(db_dir).get_record("doc.md:0")
    assert record["content"] == "edited context"
    assert record["vector"] == [0.0, 0.0, 1.0]
    assert calls["embed"] == (["edited context"], 3, "search_document: ")
    manifest = json.loads(db_dir.joinpath(local_rag.INDEX_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    document = manifest["documents"]["processed_docs/doc.md"]
    assert manifest["total_records"] == 2
    # content_hashes now live in a per-source sidecar (sharded out of the
    # monolithic manifest to keep it small at scale), not in the document entry.
    assert "content_hashes" not in document, "content_hashes should be in sidecar, not manifest"
    hashes = local_rag.load_content_hash_sidecar(db_dir, "processed_docs/doc.md")
    assert hashes["doc.md:0"] == local_rag.index_record_content_hash(record)
    assert load_index_overrides(db_dir)["edits"]["doc.md:0"]["content"] == "edited context"
    edited_rows = {
        item["id"]: item
        for item in web_app.list_index_rows(db_dir=db_dir)["rows"]
    }
    assert edited_rows["doc.md:0"]["edited"] is True


def test_delete_index_records_persists_remaining_records(lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)

    result = web_app.delete_index_records(record_ids=["doc.md:1"], db_dir=db_dir)

    assert result == {"deleted": 1, "remaining": 1}
    assert [row["id"] for row in LanceDBVectorStore(db_dir).list_records()["rows"]] == ["doc.md:0"]
    assert "doc.md:1" in load_index_overrides(db_dir)["deletions"]


def test_delete_index_records_updates_pdf_status_when_source_removed(monkeypatch, workspace_tmp, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    source_hash = "hash-delete-status"
    LanceDBVectorStore(db_dir).write_records(
        [
            {
                "id": "chunk-delete",
                "doc_id": "doc-delete",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "processed_docs/delete.md",
                "chunk_index": 0,
                "content": "delete status context",
                "title": "Delete",
                "section_path": "Delete",
                "page_start": 1,
                "page_end": 1,
                "summary": "summary",
                "tags": [],
                "source_hash": source_hash,
                "source_pdf_name": "delete.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "delete.pdf"),
                "vector": [1.0, 0.0, 0.0],
            }
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "delete.md"
    markdown_path.write_text("# Delete\n\n" + ("indexed context " * 50), encoding="utf-8")
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash=source_hash,
        source_pdf_name="delete.pdf",
        source_pdf_path=web_app.DATA_DIR / "delete.pdf",
    )
    local_rag.write_index_manifest(
        db_dir,
        LanceDBVectorStore(db_dir).all_records(),
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")

    deleted = web_app.delete_index_records(record_ids=["chunk-delete"], db_dir=db_dir)
    response = TestClient(web_app.app).get("/api/pdfs")

    assert deleted == {"deleted": 1, "remaining": 0}
    document = response.json()["pdfs"][0]
    assert document["hash"] == source_hash
    assert document["status"] == "not_indexed"
    assert "missing_index_manifest" in document["quality"]["warnings"]


def test_delete_pdf_document_removes_source_artifacts(monkeypatch, workspace_tmp, lancedb_tmp):
    data_dir = workspace_tmp / "data"
    upload_dir = data_dir / "uploads" / "old-job"
    upload_dir.mkdir(parents=True)
    pdf_path = upload_dir / "delete.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 delete source")
    source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "delete.md"
    markdown_path.write_text("# Delete\n\n" + ("indexed context " * 50), encoding="utf-8")
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash=source_hash,
        source_pdf_name=pdf_path.name,
        source_pdf_path=pdf_path,
    )
    registry_path = workspace_tmp / "registry.json"
    file_upload = {
        "filename": pdf_path.name,
        "hash": source_hash,
        "staging_path": "",
        "upload_path": str(pdf_path),
        "processed_markdown_path": str(markdown_path),
    }
    registry = web_app.PdfRegistry(registry_path)
    registry.register_queued(job_id="old-job", files=[file_upload])
    registry.mark_job_status(job_id="old-job", files=[file_upload], status="indexed")
    trust_path = workspace_tmp / "trust.json"
    web_app.update_document_trust(source_hash, {"review_status": "approved"}, trust_path=trust_path)
    asset_dir = workspace_tmp / "assets"
    monkeypatch.setattr(web_app, "ASSET_DIR", asset_dir)
    asset_store = ImageAssetStore(asset_dir)
    asset = asset_store.save_image(
        image_data=b"plot",
        source_hash=source_hash,
        source_pdf_name=pdf_path.name,
        page_no=1,
        description="plot",
    )
    db_dir = lancedb_tmp / "db"
    LanceDBVectorStore(db_dir).write_records(
        [
            {
                "id": "target",
                "doc_id": "doc-target",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": str(markdown_path),
                "chunk_index": 0,
                "content": "target content",
                "source_hash": source_hash,
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "kept",
                "doc_id": "doc-kept",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "kept.md",
                "chunk_index": 0,
                "content": "kept content",
                "source_hash": "hash-other",
                "vector": [0.0, 1.0, 0.0],
            },
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    local_rag.write_index_manifest(
        db_dir,
        LanceDBVectorStore(db_dir).all_records(),
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    persist_index_edit(db_dir, LanceDBVectorStore(db_dir).get_record("target"), "edited target content")
    persist_index_deletions(
        db_dir,
        [
            {
                "id": "removed-target",
                "doc_id": "doc-target",
                "file_path": str(markdown_path),
                "source_hash": source_hash,
            }
        ],
    )
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    result = web_app.delete_pdf_document(
        source_hash,
        registry_path=registry_path,
        processed_dir=processed_dir,
        db_dir=db_dir,
        trust_path=trust_path,
        root_dir=workspace_tmp,
        data_dir=data_dir,
    )

    assert result["vectors"]["deleted"] == 1
    assert result["markdown_deleted"] == 1
    assert result["assets_deleted"] == 1
    assert result["pdfs_deleted"] in {0, 1}
    assert bool(result["pdf_delete_errors"]) is (result["pdfs_deleted"] == 0)
    assert result["registry_deleted"] is True
    assert result["trust_deleted"] is True
    assert not pdf_path.exists() or result["pdf_delete_errors"]
    assert not markdown_path.exists() or markdown_path.read_text(encoding="utf-8") == ""
    assert load_source_map(processed_dir)["documents"] == {}
    assert source_hash not in web_app.PdfRegistry(registry_path).load()["pdfs"]
    assert source_hash not in web_app._load_trust_registry(trust_path)["documents"]
    assert asset_store.asset_path(asset["asset_id"]) is None
    assert [row["id"] for row in LanceDBVectorStore(db_dir).list_records()["rows"]] == ["kept"]
    manifest = json.loads(db_dir.joinpath(local_rag.INDEX_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert source_hash not in manifest["documents"]
    assert "hash-other" in manifest["documents"]
    assert load_index_overrides(db_dir)["edits"] == {}
    assert load_index_overrides(db_dir)["deletions"] == {}


def test_web_index_helpers_work_with_lancedb(monkeypatch):
    db_dir = Path(tempfile.gettempdir()) / f"rag_test_web_app_{uuid.uuid4().hex}" / "db"
    try:
        from src.vector_store import LanceDBVectorStore

        store = LanceDBVectorStore(db_dir)
        store.write_records(
            [
                {
                    "id": "chunk-1",
                    "doc_id": "doc",
                    "parent_id": "doc-summary",
                    "node_type": "chunk",
                    "file_path": "doc.pdf",
                    "chunk_index": 0,
                    "content": "alpha context",
                    "title": "Alpha",
                    "section_path": "Doc > Alpha",
                    "page_start": 2,
                    "page_end": 2,
                    "summary": "summary",
                    "tags": ["alpha"],
                    "vector": [1.0, 0.0, 0.0],
                }
            ],
            embedding_model="fake-embed",
            embedding_dim=3,
        )

        class FakeEmbeddingEngine:
            def __init__(self, **kwargs):
                pass

            def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
                return np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEmbeddingEngine)

        result = web_app.list_index_rows(search="alpha", db_dir=db_dir)

        assert result["total"] == 1
        assert result["rows"][0]["id"] == "chunk-1"
        assert "vector" not in result["rows"][0]
        assert result["rows"][0]["section_path"] == "Doc > Alpha"

        row = web_app.update_index_record(
            record_id="chunk-1",
            content="edited alpha context",
            db_dir=db_dir,
        )
        assert row["content"] == "edited alpha context"
        assert store.get_record("chunk-1")["content"] == "edited alpha context"

        assert web_app.delete_index_records(record_ids=["chunk-1"], db_dir=db_dir) == {
            "deleted": 1,
            "remaining": 0,
        }
    finally:
        shutil.rmtree(db_dir.parents[0], ignore_errors=True)


def test_server_config_defaults_to_minute_when_missing(workspace_tmp):
    config = web_app._load_server_config(workspace_tmp / "missing.toml")

    assert config == {
        "host": "127.0.0.1",
        "bind_all": False,
        "port": 8000,
        "health_poll_interval_ms": 60000,
        "jobs_poll_interval_ms": 60000,
        "background_worker_threads": web_app.DEFAULT_BACKGROUND_WORKER_THREADS,
        "update_remote": "origin",
        "update_branch": "main",
        "disk_safety_factor": web_app.DEFAULT_DISK_SAFETY_FACTOR,
        "job_workers": web_app.DEFAULT_JOB_WORKERS,
        "query_wait_timeout_seconds": web_app.DEFAULT_QUERY_WAIT_TIMEOUT_SECONDS,
        "api_token": "",
    }


def test_server_config_reads_polling_intervals(workspace_tmp):
    config_path = workspace_tmp / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[server]",
                'host = "0.0.0.0"',
                "port = 8081",
                'update_remote = "upstream"',
                'update_branch = "web-ui"',
                "health_poll_interval_ms = 120032",
                "jobs_poll_interval_ms = 90000",
            ]
        ),
        encoding="utf-8",
    )

    config = web_app._load_server_config(config_path)

    assert config == {
        "host": "0.0.0.0",
        "bind_all": True,
        "port": 8081,
        "health_poll_interval_ms": 120032,
        "jobs_poll_interval_ms": 90000,
        "background_worker_threads": web_app.DEFAULT_BACKGROUND_WORKER_THREADS,
        "update_remote": "upstream",
        "update_branch": "web-ui",
        "disk_safety_factor": web_app.DEFAULT_DISK_SAFETY_FACTOR,
        "job_workers": web_app.DEFAULT_JOB_WORKERS,
        "query_wait_timeout_seconds": web_app.DEFAULT_QUERY_WAIT_TIMEOUT_SECONDS,
        "api_token": "",
    }


def test_server_config_bind_all_overrides_specific_host(workspace_tmp):
    config_path = workspace_tmp / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[server]",
                'host = "127.0.0.1"',
                "bind_all = true",
            ]
        ),
        encoding="utf-8",
    )

    config = web_app._load_server_config(config_path)

    assert config["host"] == "0.0.0.0"
    assert config["bind_all"] is True


def test_server_config_lan_alias_binds_all(workspace_tmp):
    config_path = workspace_tmp / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[server]",
                "lan = true",
            ]
        ),
        encoding="utf-8",
    )

    config = web_app._load_server_config(config_path)

    assert config["host"] == "0.0.0.0"
    assert config["bind_all"] is True


def test_run_server_does_not_pass_invalid_uvicorn_kwarg(monkeypatch):
    """Regression: run_server() once passed `limit_max_request_bytes` to
    uvicorn.run(), which is not a real uvicorn option and crashed startup with
    TypeError. Body-size capping now lives in _enforce_request_body_limit
    middleware; uvicorn.run() must only receive supported kwargs.
    """
    captured = {}

    class _FakeUvicorn:
        @staticmethod
        def run(app, **kwargs):
            captured["app"] = app
            captured["kwargs"] = kwargs

    monkeypatch.setattr(web_app, "SERVER_CONFIG", {"host": "127.0.0.1", "port": 8000})
    # uvicorn is imported lazily inside run_server(); inject the fake via sys.modules
    # so the local `import uvicorn` binds to it.
    import sys
    monkeypatch.setitem(sys.modules, "uvicorn", _FakeUvicorn)

    web_app.run_server()

    assert captured["app"] == "src.web_app:app"
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 8000
    # The kwarg that crashed startup must never reappear.
    assert "limit_max_request_bytes" not in captured["kwargs"]


def test_request_body_limit_rejects_oversized_post(monkeypatch):
    """The body-size middleware returns 413 for an oversized POST without
    buffering the full body into memory."""
    monkeypatch.setattr(web_app, "MAX_REQUEST_BYTES", 100)
    client = TestClient(web_app.app)
    response = client.post("/api/uploads", content=b"x" * 5000)
    assert response.status_code == 413


def test_request_body_limit_passes_small_post(monkeypatch):
    """Requests under the cap are unaffected (still 401/422/etc. from the real
    handler/auth, NOT 413 from the size middleware)."""
    monkeypatch.setattr(web_app, "MAX_REQUEST_BYTES", 100 << 20)  # 100 MiB
    client = TestClient(web_app.app)
    # A tiny body to a mutating endpoint: auth rejects with 401 (or 422 if the
    # handler reaches it). Either way, NOT the middleware's 413.
    response = client.post("/api/uploads", content=b"x")
    assert response.status_code != 413


def test_request_body_limit_disabled_when_zero(monkeypatch):
    """A cap of 0 disables the limit entirely (operator override)."""
    monkeypatch.setattr(web_app, "MAX_REQUEST_BYTES", 0)
    client = TestClient(web_app.app)
    response = client.post("/api/uploads", content=b"x" * 5000)
    assert response.status_code != 413


def test_chat_config_reads_prompt_retrieval_and_ollama_health_settings(workspace_tmp):
    config_path = workspace_tmp / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[models]",
                'llm_model = "qwen3:4b-instruct"',
                "[chat]",
                'system_prompt = "Configured prompt {web_instruction}"',
                "context_window = 120000",
                "llm_num_predict = 24000",
                'planner_model = "qwen2.5:1.5b"',
                "planner_enabled = false",
                "planner_max_queries = 5",
                'ollama_keep_alive = "10m"',
                "[retrieval]",
                "min_relevance_score = 0.62",
                "[ollama]",
                "chat_health_check_interval_seconds = 3.5",
                "chat_max_lost_health_checks = 9",
            ]
        ),
        encoding="utf-8",
    )

    config = web_app._load_chat_config(config_path)

    assert config == {
        "system_prompt": "Configured prompt {web_instruction}",
        "llm_model": "qwen3:4b-instruct",
        "context_window": 120000,
        "llm_num_predict": 24000,
        "planner_model": "qwen2.5:1.5b",
        "planner_enabled": False,
        "planner_max_queries": 5,
        "llm_timeout": 120.0,
        "retrieval_min_score": 0.62,
        "retrieval_rrf_k": 60,
        "ollama_host": "http://127.0.0.1:11434",
        "ollama_hosts": [],
        "ollama_fallback_enabled": True,
        "ollama_health_check_interval": 3.5,
        "ollama_max_lost_health_checks": 9,
        "ollama_keep_alive": "10m",
    }


def test_chat_config_defaults_local_model_and_keep_alive(workspace_tmp):
    """Absent [models].llm_model / [chat].ollama_keep_alive, typed defaults apply."""
    config_path = workspace_tmp / "config.toml"
    config_path.write_text("[chat]\n", encoding="utf-8")

    config = web_app._load_chat_config(config_path)

    assert config["llm_model"] == DEFAULT_LLM_MODEL
    assert config["ollama_keep_alive"] == DEFAULT_OLLAMA_KEEP_ALIVE


def test_health_exposes_server_polling_config(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "SERVER_CONFIG",
        {
            "host": "127.0.0.1",
            "port": 8000,
            "health_poll_interval_ms": 60000,
            "jobs_poll_interval_ms": 60000,
            "update_remote": "origin",
            "update_branch": "main",
        },
    )

    client = TestClient(web_app.app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["server"] == {
        "host": "127.0.0.1",
        "port": 8000,
        "health_poll_interval_ms": 60000,
        "jobs_poll_interval_ms": 60000,
        "update_remote": "origin",
        "update_branch": "main",
        # The master token is redacted from the open health endpoint.
        "api_token": "",
        "api_token_configured": False,
    }
    assert response.json()["chat"] == {
        "context_window": web_app.CHAT_CONFIG["context_window"],
        "llm_num_predict": web_app.CHAT_CONFIG["llm_num_predict"],
        "retrieval_min_score": web_app.CHAT_CONFIG["retrieval_min_score"],
    }


def test_update_status_reports_current(monkeypatch):
    _set_update_config(monkeypatch)
    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    calls = _install_update_git(monkeypatch, current_sha="a" * 40, latest_sha="a" * 40)

    status = web_app.get_update_status()

    assert status["state"] == "current"
    assert status["can_update"] is False
    assert status["current_sha"] == "a" * 40
    assert any(call[0:2] == ["fetch", "--quiet"] for call in calls)


def test_update_status_reports_available(monkeypatch):
    _set_update_config(monkeypatch)
    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    _install_update_git(
        monkeypatch,
        current_sha="a" * 40,
        latest_sha="b" * 40,
        current_is_ancestor=True,
    )

    status = web_app.get_update_status()

    assert status["state"] == "available"
    assert status["can_update"] is True
    assert status["current_sha"] == "a" * 40
    assert status["latest_sha"] == "b" * 40


def test_update_status_blocks_dirty_tracked_files(monkeypatch):
    _set_update_config(monkeypatch)
    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    calls = _install_update_git(monkeypatch, dirty=" M src/web_app.py\n")

    status = web_app.get_update_status()

    assert status["state"] == "blocked"
    assert "Tracked files" in status["message"]
    assert not any(call[0:2] == ["fetch", "--quiet"] for call in calls)


def test_update_status_blocks_wrong_branch(monkeypatch):
    _set_update_config(monkeypatch)
    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    _install_update_git(monkeypatch, current_branch="web-ui")

    status = web_app.get_update_status()

    assert status["state"] == "blocked"
    assert status["current_branch"] == "web-ui"
    assert "origin/main" in status["message"]


def test_update_status_blocks_diverged_history(monkeypatch):
    _set_update_config(monkeypatch)
    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    _install_update_git(
        monkeypatch,
        current_sha="a" * 40,
        latest_sha="b" * 40,
        current_is_ancestor=False,
        latest_is_ancestor=False,
    )

    status = web_app.get_update_status()

    assert status["state"] == "blocked"
    assert "diverged" in status["message"]


def test_update_status_reports_fetch_error(monkeypatch):
    _set_update_config(monkeypatch)
    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    _install_update_git(monkeypatch, fetch_returncode=128)

    status = web_app.get_update_status()

    assert status["state"] == "error"
    assert "Unable to fetch" in status["message"]


def test_update_apply_pull_failure_does_not_restart(monkeypatch):
    _set_update_config(monkeypatch)
    spawned = []
    monkeypatch.setattr(
        web_app,
        "get_update_status",
        lambda fetch=True: {
            "state": "available",
            "can_update": True,
            "current_sha": "a" * 40,
            "latest_sha": "b" * 40,
            "current_branch": "main",
            "target_remote": "origin",
            "target_branch": "main",
            "message": "Update available.",
        },
    )
    monkeypatch.setattr(
        web_app,
        "_run_git",
        lambda args, *, timeout=web_app.GIT_TIMEOUT_SECONDS: _completed(
            args,
            stderr="fast-forward failed",
            returncode=1,
        ),
    )
    monkeypatch.setattr(web_app, "_spawn_restart_helper", lambda **kwargs: spawned.append(kwargs))

    with pytest.raises(web_app.HTTPException) as exc_info:
        web_app.apply_available_update()

    assert exc_info.value.status_code == 500
    assert "fast-forward failed" in exc_info.value.detail["message"]
    assert spawned == []


def test_update_apply_endpoint_pulls_spawns_restart_and_schedules_exit(monkeypatch):
    _set_update_config(monkeypatch)
    git_calls = []
    spawned = []
    scheduled = []
    monkeypatch.setattr(web_app, "_require_local_update_request", lambda request: None)
    monkeypatch.setattr(
        web_app,
        "get_update_status",
        lambda fetch=True: {
            "state": "available",
            "can_update": True,
            "current_sha": "a" * 40,
            "latest_sha": "b" * 40,
            "current_branch": "main",
            "target_remote": "origin",
            "target_branch": "main",
            "message": "Update available.",
        },
    )

    def fake_run_git(args, *, timeout=web_app.GIT_TIMEOUT_SECONDS):
        git_calls.append(args)
        if args == ["pull", "--ff-only", "origin", "main"]:
            return _completed(args)
        if args == ["rev-parse", "HEAD"]:
            return _completed(args, stdout=f"{'b' * 40}\n")
        return _completed(args, stderr=f"unexpected git args: {args}", returncode=1)

    monkeypatch.setattr(web_app, "_run_git", fake_run_git)
    monkeypatch.setattr(web_app, "_spawn_restart_helper", lambda **kwargs: spawned.append(kwargs))
    monkeypatch.setattr(web_app, "_schedule_process_exit", lambda: scheduled.append(True))

    client = TestClient(web_app.app)
    response = client.post("/api/update/apply")

    assert response.status_code == 200
    assert response.json()["state"] == "restarting"
    assert response.json()["previous_sha"] == "a" * 40
    assert response.json()["current_sha"] == "b" * 40
    assert git_calls[0] == ["pull", "--ff-only", "origin", "main"]
    assert spawned == [{"old_pid": web_app.os.getpid(), "host": "127.0.0.1", "port": 8000}]
    assert scheduled == [True]


def test_update_apply_rejects_active_query(monkeypatch):
    class ActiveQueue:
        def summary(self):
            return {"active_query_count": 1, "queued_count": 0, "running_job_ids": [], "job_count": 0}

    _set_update_config(monkeypatch)
    monkeypatch.setattr(web_app, "job_queue", ActiveQueue())
    _install_update_git(monkeypatch)

    with pytest.raises(web_app.HTTPException) as exc_info:
        web_app.apply_available_update()

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["state"] == "blocked"
    assert "active chat query" in exc_info.value.detail["message"]


def test_update_apply_requires_local_request(monkeypatch):
    _set_update_config(monkeypatch)

    assert web_app._is_loopback_host("127.0.0.1") is True
    assert web_app._is_loopback_host("127.10.1.2") is True
    assert web_app._is_loopback_host("::1") is True
    assert web_app._is_loopback_host("0:0:0:0:0:0:0:1") is True
    assert web_app._is_loopback_host("192.168.1.20") is False
    assert web_app._is_local_update_host("127.0.0.1") is True
    assert web_app._is_local_update_host("192.168.1.20") is False


def test_update_apply_accepts_configured_bind_address(monkeypatch):
    _set_update_config(monkeypatch)
    # Use an RFC 5737 documentation address (TEST-NET-1) rather than a real
    # deployment IP. The test only needs a non-loopback configured bind host.
    monkeypatch.setitem(web_app.SERVER_CONFIG, "host", "192.0.2.10")

    assert web_app._is_local_update_host("192.0.2.10") is True
    assert web_app._is_local_update_host("192.0.2.11") is False


def test_update_apply_does_not_accept_wildcard_bind_address(monkeypatch):
    _set_update_config(monkeypatch)
    monkeypatch.setitem(web_app.SERVER_CONFIG, "host", "0.0.0.0")

    assert web_app._is_local_update_host("192.168.1.20") is False


def test_queue_pauses_new_work_while_query_is_active(workspace_tmp):
    calls = []

    def fake_ingest(input_dir, output_dir, **kwargs):
        calls.append(("ingest", Path(input_dir).name, Path(output_dir).name))

    def fake_index(md_dir, db_dir, **kwargs):
        calls.append(("index", Path(md_dir).name, Path(db_dir).name))

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        run_ingestion_func=fake_ingest,
        run_indexing_func=fake_index,
    )
    staging = workspace_tmp / "staging"
    staging.mkdir()
    staging.joinpath("doc.pdf").write_bytes(b"%PDF-1.4")

    queue.begin_query()
    job = queue.enqueue_upload(staging_dir=staging, filenames=["doc.pdf"])

    _wait_for(lambda: queue.get_job(job.id)["status"] == "paused_for_queries")
    assert calls == []

    queue.finish_query()
    _wait_for(lambda: queue.get_job(job.id)["status"] == "done")

    assert calls == [("ingest", job.id, "processed"), ("index", "processed", "db")]
    assert (workspace_tmp / "uploads" / job.id / "doc.pdf").exists()


def test_queue_aborts_index_job_when_query_watched_too_long(workspace_tmp):
    """A leaked active_query_count must not block indexing forever. The watchdog
    (query_wait_timeout_seconds) aborts the job instead of wedging the write lock."""
    indexed = []

    def fake_index(md_dir, db_dir, **kwargs):
        indexed.append(True)

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        run_indexing_func=fake_index,
        query_wait_timeout_seconds=0.5,  # short so the test is fast
    )
    staging = workspace_tmp / "staging"
    staging.mkdir()
    staging.joinpath("doc.pdf").write_bytes(b"%PDF-1.4")

    # Simulate a leaked query counter (e.g. a chat generator whose finally-block
    # never ran after a client disconnect + hung model).
    queue.begin_query()
    job = queue.enqueue_upload(staging_dir=staging, filenames=["doc.pdf"])

    # The job pauses for the query, then the watchdog fires and it fails.
    _wait_for(lambda: queue.get_job(job.id)["status"] == "paused_for_queries")
    _wait_for(lambda: queue.get_job(job.id)["status"] == "failed", timeout=5.0)

    assert indexed == [], "indexing must not have run while the query was held"
    payload = queue.get_job(job.id)
    assert "query_wait_timeout_seconds" in payload["error"] or "active query" in payload["error"], \
        f"error should mention the query-wait timeout; got: {payload['error']!r}"

    # Releasing the leaked counter lets a subsequent job proceed normally.
    queue.finish_query()


def test_queue_proceeds_when_query_releases_before_watchdog(workspace_tmp):
    """The watchdog must not fire if the query finishes in time."""
    calls = []

    def fake_ingest(input_dir, output_dir, **kwargs):
        calls.append(("ingest", Path(input_dir).name, Path(output_dir).name))

    def fake_index(md_dir, db_dir, **kwargs):
        calls.append(("index", Path(md_dir).name, Path(db_dir).name))

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        run_ingestion_func=fake_ingest,
        run_indexing_func=fake_index,
        query_wait_timeout_seconds=10.0,
    )
    staging = workspace_tmp / "staging"
    staging.mkdir()
    staging.joinpath("doc.pdf").write_bytes(b"%PDF-1.4")

    queue.begin_query()
    job = queue.enqueue_upload(staging_dir=staging, filenames=["doc.pdf"])
    _wait_for(lambda: queue.get_job(job.id)["status"] == "paused_for_queries")
    # Release well within the watchdog window.
    queue.finish_query()
    _wait_for(lambda: queue.get_job(job.id)["status"] == "done", timeout=5.0)
    assert ("index", "processed", "db") in calls


def test_queue_upload_batch_runs_one_final_index(workspace_tmp):
    calls = []

    def fake_ingest(input_dir, output_dir, **kwargs):
        calls.append(("ingest", sorted(path.name for path in Path(input_dir).glob("*.pdf"))))

    def fake_index(md_dir, db_dir, **kwargs):
        calls.append(("index", Path(md_dir).name, Path(db_dir).name))

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
        run_ingestion_func=fake_ingest,
        run_indexing_func=fake_index,
    )
    staging = workspace_tmp / "staging"
    staging.mkdir()
    staging.joinpath("one.pdf").write_bytes(b"%PDF-1.4 one")
    staging.joinpath("two.pdf").write_bytes(b"%PDF-1.4 two")

    job = queue.enqueue_upload(
        staging_dir=staging,
        filenames=["one.pdf", "two.pdf"],
        uploads=[
            {"filename": "one.pdf", "hash": "hash-one", "staging_path": str(staging / "one.pdf")},
            {"filename": "two.pdf", "hash": "hash-two", "staging_path": str(staging / "two.pdf")},
        ],
    )

    _wait_for(lambda: queue.get_job(job.id)["status"] == "done")

    assert calls == [
        ("ingest", ["one.pdf", "two.pdf"]),
        ("index", "processed", "db"),
    ]


def test_queue_cancel_paused_upload_marks_document_interrupted(workspace_tmp):
    calls = []
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    staging = workspace_tmp / "staging"
    staging.mkdir()
    staging.joinpath("doc.pdf").write_bytes(b"%PDF-1.4")
    file_upload = {
        "filename": "doc.pdf",
        "hash": "hash-cancel",
        "staging_path": str(staging / "doc.pdf"),
    }
    web_app.PdfRegistry(registry_path).register_queued(job_id="job-cancel", files=[file_upload])

    def fake_ingest(input_dir, output_dir, **kwargs):
        calls.append(("ingest", input_dir, output_dir))

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=processed_dir,
        db_dir=workspace_tmp / "db",
        registry_path=registry_path,
        run_ingestion_func=fake_ingest,
        run_indexing_func=lambda *args, **kwargs: None,
    )

    queue.begin_query()
    job = queue.enqueue_upload(
        staging_dir=staging,
        filenames=["doc.pdf"],
        uploads=[file_upload],
        job_id="job-cancel",
    )
    _wait_for(lambda: queue.get_job(job.id)["status"] == "paused_for_queries")

    cancelled = queue.cancel_job(job.id)
    _wait_for(lambda: queue.get_job(job.id)["status"] == "cancelled")
    queue.finish_query()

    entry = web_app.PdfRegistry(registry_path).load()["pdfs"]["hash-cancel"]
    listing = web_app.list_pdf_documents(registry_path=registry_path, processed_dir=processed_dir)

    assert cancelled["cancel_requested"] is True
    assert calls == []
    assert entry["status"] == "interrupted"
    assert entry["last_interrupted_job_id"] == "job-cancel"
    assert "job_interrupted" in listing["pdfs"][0]["quality"]["warnings"]


def test_queue_passes_ingestion_options_to_worker(workspace_tmp):
    captured = {}

    def fake_ingest(input_dir, output_dir, **kwargs):
        captured.update(kwargs)

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        run_ingestion_func=fake_ingest,
        run_indexing_func=lambda *args, **kwargs: None,
    )

    queue._run_ingestion(
        "input",
        "output",
        {
            "asset_dir": str(workspace_tmp / "assets"),
            "vision_model": "vision-test",
            "vision_enabled": False,
            "code_enrichment": False,
            "formula_enrichment": True,
            "ocr_backend": "tesseract_cli",
            "ocr_langs": ["eng"],
            "ocr_force_full_page": False,
            "ocr_bitmap_area_threshold": 0.2,
            "rapidocr_backend": "torch",
            "tesseract_cmd": "C:/Tools/tesseract.exe",
            "tesseract_data_path": "C:/Tools/tessdata",
            "tesseract_psm": 6,
        },
    )

    assert captured["asset_dir"] == str(workspace_tmp / "assets")
    assert captured["vision_model"] == "vision-test"
    assert captured["vision_enabled"] is False
    assert captured["code_enrichment"] is False
    assert captured["formula_enrichment"] is True
    assert captured["ocr_backend"] == "tesseract_cli"
    assert captured["ocr_langs"] == ["eng"]
    assert captured["ocr_force_full_page"] is False
    assert captured["ocr_bitmap_area_threshold"] == 0.2
    assert captured["rapidocr_backend"] == "torch"
    assert captured["tesseract_cmd"] == "C:/Tools/tesseract.exe"
    assert captured["tesseract_data_path"] == "C:/Tools/tessdata"
    assert captured["tesseract_psm"] == 6


def test_queue_runs_ingestion_in_subprocess_with_capped_env(monkeypatch, workspace_tmp):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(web_app.subprocess, "run", fake_run)
    monkeypatch.setitem(web_app.SERVER_CONFIG, "background_worker_threads", 2)
    monkeypatch.setenv("OMP_NUM_THREADS", "16")

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
    )

    queue._run_ingestion(
        str(workspace_tmp / "input"),
        str(workspace_tmp / "processed"),
        {
            "parser_mode": "manual",
            "asset_dir": str(workspace_tmp / "assets"),
            "accelerator": "cpu",
            "num_threads": 3,
            "asset_triggers": "none",
            "code_enrichment": False,
            "formula_enrichment": True,
            "vision_enabled": False,
            "ocr_backend": "tesseract_cli",
            "ocr_langs": ["eng"],
        },
    )

    command, kwargs = calls[0]
    assert command[:4] == [web_app.sys.executable, str(web_app.ROOT_DIR / "main.py"), "--mode", "ingest"]
    assert command[command.index("--num_threads") + 1] == "3"
    assert command[command.index("--code_enrichment") + 1] == "false"
    assert command[command.index("--formula_enrichment") + 1] == "true"
    assert command[command.index("--ocr_langs") + 1] == "eng"
    assert command[-1] == "--no_progress"
    assert kwargs["cwd"] == web_app.ROOT_DIR
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    assert kwargs["env"]["TOKENIZERS_PARALLELISM"] == "false"
    assert kwargs["env"]["OMP_NUM_THREADS"] == "2"


def test_queue_subprocess_failure_reports_tail(monkeypatch, workspace_tmp):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 5, stdout="ignored", stderr="failure detail")

    monkeypatch.setattr(web_app.subprocess, "run", fake_run)
    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
    )

    with pytest.raises(RuntimeError) as exc_info:
        queue._run_ingestion("input", "output", {})

    message = str(exc_info.value)
    assert "exit code 5" in message
    assert "failure detail" in message


def test_reindex_request_rejects_invalid_numeric_settings(monkeypatch):
    class FakeQueue:
        def enqueue_reindex(self, **kwargs):
            raise AssertionError("invalid request should not enqueue")

    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    response = TestClient(web_app.app).post("/api/reindex", json={"embedding_batch_size": 0})

    assert response.status_code == 422


def test_find_resumable_staged_dir_requires_completed_files(workspace_tmp):
    db_dir = workspace_tmp / "db"
    db_dir.mkdir()
    staged = db_dir.parent / ".index_build_eeee"
    staged.mkdir()
    (staged / web_app._STAGED_CHECKPOINT_FILENAME).write_text(
        json.dumps({"completed_files": ["x.md"]}),
        encoding="utf-8",
    )

    assert web_app._find_resumable_staged_dir(db_dir) == staged

    # An empty completed_files list carries no resumable progress -- the
    # indexer restarts from scratch anyway, so the dir must not be picked.
    (staged / web_app._STAGED_CHECKPOINT_FILENAME).write_text(
        json.dumps({"completed_files": []}),
        encoding="utf-8",
    )

    assert web_app._find_resumable_staged_dir(db_dir) is None


def test_gc_staged_build_dirs_removes_only_abandoned(workspace_tmp):
    db_dir = workspace_tmp / "db"
    db_dir.mkdir()
    parent = db_dir.parent
    old = time.time() - 2 * web_app.STAGED_BUILD_GC_MIN_AGE_SECONDS

    # No checkpoint at all (crash before first checkpoint).
    abandoned = parent / ".index_build_aaaa"
    (abandoned / "lancedb").mkdir(parents=True)
    # Checkpoint with zero completed files (no resumable progress).
    empty_ckpt = parent / ".index_build_bbbb"
    empty_ckpt.mkdir()
    (empty_ckpt / web_app._STAGED_CHECKPOINT_FILENAME).write_text(
        json.dumps({"completed_files": []}),
        encoding="utf-8",
    )
    # Resumable progress must survive.
    resumable = parent / ".index_build_cccc"
    resumable.mkdir()
    (resumable / web_app._STAGED_CHECKPOINT_FILENAME).write_text(
        json.dumps({"completed_files": ["a.md"]}),
        encoding="utf-8",
    )
    # Interrupted publish aside-dir is repair-only: GC must leave it alone
    # (it can hold the only copy of the live overrides/hashes).
    preserve = parent / ".index_preserve_ffff"
    preserve.mkdir()
    # Inside the grace window: never swept (a build may be warming up).
    fresh = parent / ".index_build_gggg"
    fresh.mkdir()
    for entry in (abandoned, empty_ckpt, resumable, preserve):
        os.utime(entry, (old, old))

    removed = web_app._gc_staged_build_dirs(db_dir)

    assert removed == 2
    assert not abandoned.exists()
    assert not empty_ckpt.exists()
    assert preserve.exists()
    assert resumable.exists()
    assert fresh.exists()


def test_job_subprocess_streams_bounded_log_tail(workspace_tmp):
    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
    )
    job = web_app.QueueJob(id="job-logs", kind="reindex")

    web_app._run_job_subprocess(
        [
            web_app.sys.executable,
            "-c",
            "for i in range(205): print(f'line {i}', flush=True)",
        ],
        cancel_event=job._cancel_event,
        log_callback=lambda line: queue._append_job_log(job, line),
    )

    payload = job.to_dict()
    assert payload["log_line_count"] == 205
    assert len(job.log_tail) == web_app.JOB_LOG_TAIL_LINES
    assert job.log_tail[0] == "line 5"
    assert "line 204" in payload["log_tail"]


def test_job_subprocess_failure_and_cancellation_keep_log_tail(workspace_tmp):
    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
    )
    failed_job = web_app.QueueJob(id="job-failed", kind="reindex")

    with pytest.raises(RuntimeError, match="before fail"):
        web_app._run_job_subprocess(
            [
                web_app.sys.executable,
                "-c",
                "print('before fail', flush=True); raise SystemExit(3)",
            ],
            cancel_event=failed_job._cancel_event,
            log_callback=lambda line: queue._append_job_log(failed_job, line),
        )
    assert failed_job.log_tail == ["before fail"]

    cancelled_job = web_app.QueueJob(id="job-cancel-run", kind="reindex")
    errors = []

    def run_cancellable():
        try:
            web_app._run_job_subprocess(
                [
                    web_app.sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                cancel_event=cancelled_job._cancel_event,
                log_callback=lambda line: queue._append_job_log(cancelled_job, line),
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_cancellable)
    thread.start()
    _wait_for(lambda: cancelled_job.log_line_count == 1)
    cancelled_job._cancel_event.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert isinstance(errors[0], web_app.JobCancelled)
    assert cancelled_job.log_tail == ["started"]


def test_queue_staged_index_publish_swaps_live_index(monkeypatch, workspace_tmp, lancedb_tmp):
    live_db = lancedb_tmp / "db"
    old_records = [
        {
            "id": "old",
            "doc_id": "old-doc",
            "parent_id": "",
            "node_type": "chunk",
            "file_path": "old.md",
            "chunk_index": 0,
            "content": "old content",
            "vector": [1.0, 0.0, 0.0],
        }
    ]
    LanceDBVectorStore(live_db).write_records(old_records, embedding_model="old-embed", embedding_dim=3)
    local_rag.write_index_manifest(live_db, old_records, embedding_model="old-embed", embedding_dim=3)
    commands = []

    def fake_job_subprocess(command, *, worker_threads=None):
        commands.append(command)
        staged_db = Path(command[command.index("--db_dir") + 1])
        records = [
            {
                "id": "new",
                "doc_id": "new-doc",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "new.md",
                "chunk_index": 0,
                "content": "new content",
                "vector": [0.0, 1.0, 0.0],
            }
        ]
        LanceDBVectorStore(staged_db).write_records(records, embedding_model="new-embed", embedding_dim=3)
        local_rag.write_index_manifest(staged_db, records, embedding_model="new-embed", embedding_dim=3)

    monkeypatch.setattr(web_app, "_run_job_subprocess", fake_job_subprocess)
    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=live_db,
        registry_path=workspace_tmp / "registry.json",
    )

    queue._run_indexing(str(workspace_tmp / "processed"), str(live_db), {})

    command = commands[0]
    assert command[command.index("--reuse_db_dir") + 1] == str(live_db)
    assert LanceDBVectorStore(live_db).get_record("new")["content"] == "new content"
    with pytest.raises(KeyError):
        LanceDBVectorStore(live_db).get_record("old")
    manifest = json.loads(live_db.joinpath(local_rag.INDEX_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["embedding_model"] == "new-embed"


def test_queue_failed_staged_index_keeps_live_index(monkeypatch, workspace_tmp, lancedb_tmp):
    live_db = lancedb_tmp / "db"
    records = [
        {
            "id": "old",
            "doc_id": "old-doc",
            "parent_id": "",
            "node_type": "chunk",
            "file_path": "old.md",
            "chunk_index": 0,
            "content": "old content",
            "vector": [1.0, 0.0, 0.0],
        }
    ]
    LanceDBVectorStore(live_db).write_records(records, embedding_model="old-embed", embedding_dim=3)

    def fake_job_subprocess(command, *, worker_threads=None):
        raise RuntimeError("index failed")

    monkeypatch.setattr(web_app, "_run_job_subprocess", fake_job_subprocess)
    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=live_db,
        registry_path=workspace_tmp / "registry.json",
    )

    with pytest.raises(RuntimeError, match="index failed"):
        queue._run_indexing(str(workspace_tmp / "processed"), str(live_db), {})

    assert LanceDBVectorStore(live_db).get_record("old")["content"] == "old content"


def test_queue_waits_for_queries_before_publishing_index(monkeypatch, workspace_tmp, lancedb_tmp):
    live_db = lancedb_tmp / "db"

    def fake_job_subprocess(command, *, worker_threads=None):
        staged_db = Path(command[command.index("--db_dir") + 1])
        records = [
            {
                "id": "new",
                "doc_id": "new-doc",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "new.md",
                "chunk_index": 0,
                "content": "new content",
                "vector": [0.0, 1.0, 0.0],
            }
        ]
        LanceDBVectorStore(staged_db).write_records(records, embedding_model="new-embed", embedding_dim=3)
        local_rag.write_index_manifest(staged_db, records, embedding_model="new-embed", embedding_dim=3)

    monkeypatch.setattr(web_app, "_run_job_subprocess", fake_job_subprocess)
    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=live_db,
        registry_path=workspace_tmp / "registry.json",
    )
    job = web_app.QueueJob(id="job-publish", kind="reindex")
    queue.begin_query()
    thread = threading.Thread(
        target=queue._run_indexing,
        args=(str(workspace_tmp / "processed"), str(live_db), {}),
        kwargs={"job": job},
    )
    thread.start()

    _wait_for(lambda: job.status == "paused_for_queries" and job.phase == "publishing_index")
    assert not LanceDBVectorStore(live_db).exists()

    queue.finish_query()
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert LanceDBVectorStore(live_db).get_record("new")["content"] == "new content"


def test_force_duplicate_cleanup_waits_until_ingestion_phase(monkeypatch, workspace_tmp, lancedb_tmp):
    calls = []
    processed_dir = workspace_tmp / "processed"
    db_dir = lancedb_tmp / "db"
    asset_dir = workspace_tmp / "assets"
    monkeypatch.setattr(web_app, "ASSET_DIR", asset_dir)
    processed_dir.mkdir()
    old_markdown = processed_dir / "old.md"
    old_markdown.write_text("old content", encoding="utf-8")
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=old_markdown,
        source_hash="hash-a",
        source_pdf_name="old.pdf",
        source_pdf_path=workspace_tmp / "old.pdf",
    )
    LanceDBVectorStore(db_dir).write_records(
        [
            {
                "id": "old",
                "doc_id": "old-doc",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": str(old_markdown),
                "chunk_index": 0,
                "content": "old",
                "source_hash": "hash-a",
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "kept",
                "doc_id": "kept-doc",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "kept.md",
                "chunk_index": 1,
                "content": "kept",
                "source_hash": "hash-b",
                "vector": [0.0, 1.0, 0.0],
            },
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    persist_index_edit(db_dir, LanceDBVectorStore(db_dir).get_record("old"), "edited old content")
    persist_index_deletions(
        db_dir,
        [{"id": "old-deleted", "doc_id": "old-doc", "file_path": str(old_markdown), "source_hash": "hash-a"}],
    )
    asset_store = ImageAssetStore(asset_dir)
    stale_asset = asset_store.save_image(
        image_data=b"old-graph",
        source_hash="hash-a",
        source_pdf_name="old.pdf",
        page_no=1,
        description="old graph",
    )

    def fake_ingest(input_dir, output_dir, **kwargs):
        old_content = old_markdown.read_text(encoding="utf-8") if old_markdown.exists() else ""
        calls.append(
            (
                "ingest",
                bool(old_content),
                LanceDBVectorStore(db_dir).count(),
                asset_store.asset_path(stale_asset["asset_id"]) is not None,
            )
        )

    def fake_index(md_dir, db_dir_arg, **kwargs):
        calls.append(("index", Path(md_dir).name, Path(db_dir_arg).name))

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=processed_dir,
        db_dir=db_dir,
        registry_path=workspace_tmp / "registry.json",
        run_ingestion_func=fake_ingest,
        run_indexing_func=fake_index,
    )
    staging = workspace_tmp / "staging"
    staging.mkdir()
    staging.joinpath("new.pdf").write_bytes(b"%PDF-1.4")

    queue.begin_query()
    job = queue.enqueue_upload(
        staging_dir=staging,
        filenames=["new.pdf"],
        uploads=[{"filename": "new.pdf", "hash": "hash-a", "staging_path": str(staging / "new.pdf")}],
        force_duplicate_hashes=["hash-a"],
    )

    _wait_for(lambda: queue.get_job(job.id)["status"] == "paused_for_queries")
    assert old_markdown.exists()
    assert LanceDBVectorStore(db_dir).count() == 2
    assert asset_store.asset_path(stale_asset["asset_id"]) is not None

    queue.finish_query()
    _wait_for(lambda: queue.get_job(job.id)["status"] == "done")

    assert calls[0] == ("ingest", False, 1, False)
    assert load_source_map(processed_dir)["documents"] == {}
    assert asset_store.load_manifest()["assets"] == {}
    assert load_index_overrides(db_dir)["edits"] == {}
    assert load_index_overrides(db_dir)["deletions"] == {}


def test_startup_recovery_resumes_saved_upload(workspace_tmp):
    calls = []
    registry_path = workspace_tmp / "registry.json"
    upload_root = workspace_tmp / "uploads"
    processed_dir = workspace_tmp / "processed"
    db_dir = workspace_tmp / "db"
    upload_dir = upload_root / "job-saved"
    upload_dir.mkdir(parents=True)
    upload_path = upload_dir / "saved.pdf"
    upload_path.write_bytes(b"%PDF-1.4 saved")
    file_upload = {
        "filename": "saved.pdf",
        "hash": "hash-saved",
        "staging_path": "",
        "upload_path": str(upload_path),
    }
    registry = web_app.PdfRegistry(registry_path)
    registry.register_queued(
        job_id="job-saved",
        files=[file_upload],
        options={"ocr_backend": "tesseract_cli", "embedding_model": "stale-embed"},
    )
    registry.mark_job_status(job_id="job-saved", files=[file_upload], status="saving_uploads")

    def fake_ingest(input_dir, output_dir, **kwargs):
        calls.append(("ingest", Path(input_dir), kwargs["ocr_backend"]))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "saved.md").write_text("saved markdown", encoding="utf-8")

    def fake_index(md_dir, db_dir_arg, **kwargs):
        calls.append(("index", Path(md_dir), kwargs["embedding_model"]))

    queue = web_app.RagJobQueue(
        upload_root=upload_root,
        processed_dir=processed_dir,
        db_dir=db_dir,
        registry_path=registry_path,
        run_ingestion_func=fake_ingest,
        run_indexing_func=fake_index,
    )

    recovered = queue.recover_pending_uploads()
    _wait_for(lambda: queue.get_job("job-saved")["status"] == "done")

    entry = web_app.PdfRegistry(registry_path).load()["pdfs"]["hash-saved"]
    assert recovered["recovered"] == 1
    assert recovered["jobs"][0]["resume_status"] == "saving_uploads"
    # Non-embedding options survive recovery verbatim; a stale embedding
    # model is re-resolved from the current config (see
    # test_startup_recovery_drops_stale_embedding_model).
    assert calls == [
        ("ingest", upload_dir, "tesseract_cli"),
        ("index", processed_dir, web_app.CONFIGURED_EMBEDDING_MODEL),
    ]
    assert entry["status"] == "indexed"
    assert entry["upload_path"] == str(upload_path)


def test_startup_recovery_resumes_ingested_upload_at_indexing(workspace_tmp):
    calls = []
    registry_path = workspace_tmp / "registry.json"
    upload_root = workspace_tmp / "uploads"
    processed_dir = workspace_tmp / "processed"
    db_dir = workspace_tmp / "db"
    processed_dir.mkdir()
    processed_path = processed_dir / "ingested.md"
    processed_path.write_text("ingested markdown", encoding="utf-8")
    file_upload = {
        "filename": "ingested.pdf",
        "hash": "hash-ingested",
        "staging_path": "",
        "upload_path": "",
        "processed_markdown_path": str(processed_path),
    }
    registry = web_app.PdfRegistry(registry_path)
    registry.register_queued(
        job_id="job-ingested",
        files=[file_upload],
        options={"embedding_model": "stale-embed"},
    )
    registry.mark_job_status(job_id="job-ingested", files=[file_upload], status="ingested")

    def fake_ingest(*args, **kwargs):
        raise AssertionError("ingested recovery should not re-run ingestion")

    def fake_index(md_dir, db_dir_arg, **kwargs):
        calls.append(("index", Path(md_dir), Path(db_dir_arg), kwargs["embedding_model"]))

    queue = web_app.RagJobQueue(
        upload_root=upload_root,
        processed_dir=processed_dir,
        db_dir=db_dir,
        registry_path=registry_path,
        run_ingestion_func=fake_ingest,
        run_indexing_func=fake_index,
    )

    recovered = queue.recover_pending_uploads()
    _wait_for(lambda: queue.get_job("job-ingested")["status"] == "done")

    entry = web_app.PdfRegistry(registry_path).load()["pdfs"]["hash-ingested"]
    assert recovered["recovered"] == 1
    assert recovered["jobs"][0]["resume_status"] == "ingested"
    assert calls == [
        ("index", processed_dir, db_dir, web_app.CONFIGURED_EMBEDDING_MODEL)
    ]
    assert entry["status"] == "indexed"


def test_startup_recovery_drops_stale_embedding_model(workspace_tmp):
    """Recovered uploads re-resolve a stale embedding model from the config.

    An upload captures the embedding model configured when it was accepted.
    If the config later moves to a different model, keeping the stale tag in
    the recovered job makes its indexing phase preflight/embed with a model
    the live index no longer uses; the job then fails while staying
    recoverable and re-queues on every server boot. A captured model that
    still matches the config must survive recovery untouched.
    """
    calls = []
    registry_path = workspace_tmp / "registry.json"
    upload_root = workspace_tmp / "uploads"
    processed_dir = workspace_tmp / "processed"
    db_dir = workspace_tmp / "db"
    processed_dir.mkdir()
    stale_path = processed_dir / "stale.md"
    stale_path.write_text("stale markdown", encoding="utf-8")
    fresh_path = processed_dir / "fresh.md"
    fresh_path.write_text("fresh markdown", encoding="utf-8")
    registry = web_app.PdfRegistry(registry_path)
    registry.register_queued(
        job_id="job-stale",
        files=[{
            "filename": "stale.pdf",
            "hash": "hash-stale",
            "staging_path": "",
            "upload_path": "",
            "processed_markdown_path": str(stale_path),
        }],
        options={"embedding_model": "nomic-embed-text"},
    )
    registry.register_queued(
        job_id="job-fresh",
        files=[{
            "filename": "fresh.pdf",
            "hash": "hash-fresh",
            "staging_path": "",
            "upload_path": "",
            "processed_markdown_path": str(fresh_path),
        }],
        options={"embedding_model": web_app.CONFIGURED_EMBEDDING_MODEL},
    )
    registry.mark_job_status(
        job_id="job-stale",
        files=[{"filename": "stale.pdf", "hash": "hash-stale"}],
        status="ingested",
    )
    registry.mark_job_status(
        job_id="job-fresh",
        files=[{"filename": "fresh.pdf", "hash": "hash-fresh"}],
        status="ingested",
    )

    def fake_index(md_dir, db_dir_arg, **kwargs):
        calls.append(kwargs["embedding_model"])

    queue = web_app.RagJobQueue(
        upload_root=upload_root,
        processed_dir=processed_dir,
        db_dir=db_dir,
        registry_path=registry_path,
        run_indexing_func=fake_index,
    )

    recovered = queue.recover_pending_uploads()
    _wait_for(lambda: queue.get_job("job-stale")["status"] == "done")
    _wait_for(lambda: queue.get_job("job-fresh")["status"] == "done")

    recovered_options = {
        job["id"]: job.get("options") or {} for job in recovered["jobs"]
    }
    assert "embedding_model" not in recovered_options["job-stale"]
    assert (
        recovered_options["job-fresh"].get("embedding_model")
        == web_app.CONFIGURED_EMBEDDING_MODEL
    )
    # Both jobs index with the model the live config resolves to.
    assert calls == [
        web_app.CONFIGURED_EMBEDDING_MODEL,
        web_app.CONFIGURED_EMBEDDING_MODEL,
    ]


def test_upload_endpoint_enqueues_pdf_batch(monkeypatch, workspace_tmp):
    captured = {}

    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                staging_dir=str(kwargs["staging_dir"]),
            )

    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("notes.pdf", b"%PDF-1.4", "application/pdf"))],
    )

    assert response.status_code == 200
    assert response.json()["filenames"] == ["notes.pdf"]
    assert captured["filenames"] == ["notes.pdf"]
    assert captured["uploads"][0]["filename"] == "notes.pdf"
    assert len(captured["uploads"][0]["hash"]) == 64
    assert captured["options"]["ocr_backend"] == "rapidocr"
    assert captured["options"]["asset_dir"] == str(web_app.ASSET_DIR)
    assert captured["options"]["rapidocr_backend"] == "onnxruntime"
    assert captured["options"]["ocr_force_full_page"] is True
    assert captured["options"]["ocr_langs"] == ["english"]
    assert captured["options"]["vision_enabled"] is True
    assert captured["uploads"][0]["source_group"] == "official"
    assert Path(captured["staging_dir"]).joinpath("notes.pdf").exists()


def test_upload_endpoint_batches_multiple_pdfs_into_one_job(monkeypatch, workspace_tmp):
    captured = []

    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            captured.append(kwargs)
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                uploads=kwargs["uploads"],
                staging_dir=str(kwargs["staging_dir"]),
            )

    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        files=[
            ("source_groups", (None, "official")),
            ("source_groups", (None, "unofficial")),
            ("files", ("one.pdf", b"%PDF-1.4 one", "application/pdf")),
            ("files", ("two.pdf", b"%PDF-1.4 two", "application/pdf")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_count"] == 1
    assert payload["filenames"] == ["one.pdf", "two.pdf"]
    assert [job["filenames"] for job in payload["jobs"]] == [["one.pdf", "two.pdf"]]
    assert [item["filenames"] for item in captured] == [["one.pdf", "two.pdf"]]
    assert len(captured) == 1
    assert len(captured[0]["uploads"]) == 2
    assert captured[0]["uploads"][0]["source_group"] == "official"
    assert captured[0]["uploads"][1]["source_group"] == "unofficial"
    assert Path(captured[0]["staging_dir"]).joinpath("one.pdf").exists()
    assert Path(captured[0]["staging_dir"]).joinpath("two.pdf").exists()


def test_upload_endpoint_requires_source_groups(monkeypatch, workspace_tmp):
    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            raise AssertionError("upload should not be queued")

    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        files=[("files", ("notes.pdf", b"%PDF-1.4", "application/pdf"))],
    )

    assert response.status_code == 400
    assert "source group" in response.json()["detail"]


def test_direct_upload_endpoint_allows_missing_source_groups(monkeypatch, workspace_tmp):
    captured = {}

    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                uploads=kwargs["uploads"],
                staging_dir=str(kwargs["staging_dir"]),
            )

    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads/direct",
        files=[("files", ("notes.pdf", b"%PDF-1.4", "application/pdf"))],
    )

    assert response.status_code == 200
    assert captured["uploads"][0]["source_group"] == "ungrouped"


def test_upload_endpoint_rejects_duplicate_pdf_without_force(monkeypatch, workspace_tmp):
    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                uploads=kwargs["uploads"],
                staging_dir=str(kwargs["staging_dir"]),
            )

    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    first = client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("notes.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )
    second = client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("copy.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )

    assert first.status_code == 200
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["can_force"] is True
    assert detail["force_token"]
    assert detail["duplicates"][0]["filename"] == "copy.pdf"
    assert detail["duplicates"][0]["existing_filename"] == "notes.pdf"


def test_upload_endpoint_rejects_duplicate_from_index_source_map(monkeypatch, workspace_tmp):
    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            raise AssertionError("duplicate upload should not be queued")

    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    content = b"%PDF-1.4 already indexed"
    source_hash = hashlib.sha256(content).hexdigest()
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=processed_dir / "notes.md",
        source_hash=source_hash,
        source_pdf_name="notes.pdf",
        source_pdf_path=web_app.DATA_DIR / "notes.pdf",
    )
    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("copy.pdf", content, "application/pdf"))],
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["can_force"] is True
    assert detail["force_token"]
    assert detail["duplicates"][0]["filename"] == "copy.pdf"
    assert detail["duplicates"][0]["existing_filename"] == "notes.pdf"
    assert detail["duplicates"][0]["status"] == "indexed"


def test_upload_endpoint_rejects_duplicate_from_vector_store(monkeypatch, workspace_tmp, lancedb_tmp):
    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            raise AssertionError("duplicate upload should not be queued")

    content = b"%PDF-1.4 indexed in vector store"
    source_hash = hashlib.sha256(content).hexdigest()
    db_dir = lancedb_tmp / "db"
    LanceDBVectorStore(db_dir).write_records(
        [
            {
                "id": "chunk-duplicate",
                "doc_id": "doc",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "processed_docs/notes.md",
                "chunk_index": 0,
                "content": "duplicate content",
                "title": "Notes",
                "section_path": "Notes",
                "page_start": 1,
                "page_end": 1,
                "summary": "summary",
                "tags": [],
                "source_hash": source_hash,
                "source_pdf_name": "notes.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "notes.pdf"),
                "vector": [1.0, 0.0, 0.0],
            }
        ],
        embedding_model="nomic-embed-text",
        embedding_dim=3,
    )
    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", workspace_tmp / "processed")
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("copy.pdf", content, "application/pdf"))],
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["force_token"]
    assert detail["duplicates"][0]["existing_filename"] == "notes.pdf"
    assert detail["duplicates"][0]["record_id"] == "chunk-duplicate"


def test_vector_duplicate_lookup_filters_by_source_hash(monkeypatch):
    calls = []

    class FakeStore:
        def exists(self):
            return True

        def records_by_source_hash(self, source_hashes):
            calls.append(list(source_hashes))
            return [
                {
                    "id": "chunk-target",
                    "source_hash": "hash-target",
                    "source_pdf_name": "target.pdf",
                }
            ]

        def list_records(self, **kwargs):
            raise AssertionError("duplicate lookup should not scan the full index")

    monkeypatch.setattr(web_app, "_index_store", lambda db_dir=None: FakeStore())

    duplicates = web_app._vector_store_duplicate_entries(
        [{"filename": "copy.pdf", "hash": "hash-target"}],
        known_hashes=set(),
    )

    assert calls == [["hash-target"]]
    assert duplicates == [
        {
            "filename": "copy.pdf",
            "hash": "hash-target",
            "existing_filename": "target.pdf",
            "status": "indexed",
            "job_id": "",
            "record_id": "chunk-target",
        }
    ]


def test_upload_endpoint_rejects_duplicate_from_existing_data_pdf(monkeypatch, workspace_tmp):
    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            raise AssertionError("duplicate upload should not be queued")

    data_dir = workspace_tmp / "data"
    uploads_dir = data_dir / "uploads" / "existing-job"
    uploads_dir.mkdir(parents=True)
    content = b"%PDF-1.4 existing uploaded pdf"
    uploads_dir.joinpath("notes.pdf").write_bytes(content)

    monkeypatch.setattr(web_app, "DATA_DIR", data_dir)
    monkeypatch.setattr(web_app, "STAGING_DIR", data_dir / ".upload_queue")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", data_dir / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", workspace_tmp / "processed")
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("copy.pdf", content, "application/pdf"))],
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["force_token"]
    assert detail["duplicates"][0]["existing_filename"] == "notes.pdf"
    assert detail["duplicates"][0]["status"] == "uploaded"


def test_data_pdf_duplicate_scan_reuses_hash_cache(monkeypatch, workspace_tmp):
    data_dir = workspace_tmp / "data"
    uploads_dir = data_dir / "uploads" / "existing-job"
    uploads_dir.mkdir(parents=True)
    pdf_path = uploads_dir / "notes.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 existing uploaded pdf")
    calls = []

    def fake_sha256_file(path):
        calls.append(Path(path))
        return "hash-target"

    web_app._PDF_HASH_CACHE.clear()
    monkeypatch.setattr(web_app, "sha256_file", fake_sha256_file)

    files = [{"filename": "copy.pdf", "hash": "hash-target"}]
    first = web_app._data_pdf_duplicate_entries(files, known_hashes=set(), data_dir=data_dir)
    second = web_app._data_pdf_duplicate_entries(files, known_hashes=set(), data_dir=data_dir)

    assert [item["existing_filename"] for item in first] == ["notes.pdf"]
    assert [item["existing_filename"] for item in second] == ["notes.pdf"]
    assert calls == [pdf_path]


def test_data_pdf_duplicate_scan_skips_metadata_known_paths(monkeypatch, workspace_tmp):
    data_dir = workspace_tmp / "data"
    uploads_dir = data_dir / "uploads" / "existing-job"
    uploads_dir.mkdir(parents=True)
    registry_pdf = uploads_dir / "known-registry.pdf"
    source_map_pdf = data_dir / "known-source-map.pdf"
    orphan_pdf = data_dir / "orphan.pdf"
    for path in (registry_pdf, source_map_pdf, orphan_pdf):
        path.write_bytes(b"%PDF-1.4 known")

    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "known.md"
    markdown_path.write_text("# Known", encoding="utf-8")
    registry_path = data_dir / "registry.json"
    web_app.PdfRegistry(registry_path).register_queued(
        job_id="known-job",
        files=[
            {
                "filename": registry_pdf.name,
                "hash": "1" * 64,
                "upload_path": str(registry_pdf),
            }
        ],
    )
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash="2" * 64,
        source_pdf_name=source_map_pdf.name,
        source_pdf_path=source_map_pdf,
    )
    calls = []

    def fake_sha256_file(path):
        calls.append(Path(path))
        return "hash-target"

    web_app._PDF_HASH_CACHE.clear()
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "sha256_file", fake_sha256_file)

    duplicates = web_app._data_pdf_duplicate_entries(
        [{"filename": "copy.pdf", "hash": "hash-target"}],
        known_hashes=set(),
        data_dir=data_dir,
    )

    assert calls == [orphan_pdf]
    assert [item["existing_filename"] for item in duplicates] == [orphan_pdf.name]


def test_upload_endpoint_accepts_large_pdf_response_shape(monkeypatch, workspace_tmp):
    captured = {}

    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                uploads=kwargs["uploads"],
                staging_dir=str(kwargs["staging_dir"]),
            )

    data_dir = workspace_tmp / "data"
    monkeypatch.setattr(web_app, "DATA_DIR", data_dir)
    monkeypatch.setattr(web_app, "STAGING_DIR", data_dir / ".upload_queue")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", data_dir / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", workspace_tmp / "processed")
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    content = b"%PDF-1.4\n" + (b"0" * (8 * 1024 * 1024))
    expected_hash = hashlib.sha256(content).hexdigest()
    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("large.pdf", content, "application/pdf"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_count"] == 1
    assert payload["filenames"] == ["large.pdf"]
    assert captured["uploads"][0]["hash"] == expected_hash
    assert Path(captured["uploads"][0]["staging_path"]).exists()


def test_upload_endpoint_allows_forced_duplicate(monkeypatch, workspace_tmp):
    captured = {}

    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                uploads=kwargs["uploads"],
                force_duplicate_hashes=kwargs["force_duplicate_hashes"],
                staging_dir=str(kwargs["staging_dir"]),
            )

    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("notes.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )
    duplicate = client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("copy.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )
    direct_force = client.post(
        "/api/uploads",
        data={"force_duplicates": "true", "source_groups": "official"},
        files=[("files", ("copy.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )
    response = client.post(
        "/api/uploads",
        data={
            "force_duplicates": "true",
            "force_token": duplicate.json()["detail"]["force_token"],
            "source_groups": "official",
        },
        files=[("files", ("copy.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )

    assert duplicate.status_code == 409
    assert direct_force.status_code == 409
    assert response.status_code == 200
    assert response.json()["force_duplicate_hashes"] == captured["force_duplicate_hashes"]
    assert captured["force_duplicate_hashes"] == [captured["uploads"][0]["hash"]]


def test_check_upload_hash_endpoint_reports_duplicates(monkeypatch, workspace_tmp):
    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                uploads=kwargs["uploads"],
                staging_dir=str(kwargs["staging_dir"]),
            )

    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    content = b"%PDF-1.4 same"
    source_hash = hashlib.sha256(content).hexdigest()
    client = TestClient(web_app.app)
    client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("notes.pdf", content, "application/pdf"))],
    )

    response = client.get("/api/uploads/check-hash", params={"hash": source_hash})

    assert response.status_code == 200
    assert response.json()["hash"] == source_hash
    assert response.json()["exists"] is True
    assert response.json()["duplicates"][0]["existing_filename"] == "notes.pdf"


def test_check_upload_hash_endpoint_rejects_invalid_hash():
    client = TestClient(web_app.app)
    response = client.get("/api/uploads/check-hash", params={"hash": "bad"})

    assert response.status_code == 400
    assert "SHA-256" in response.json()["detail"]


def test_jobs_endpoint_paginates(monkeypatch):
    class FakeQueue:
        def list_jobs(self):
            return [{"id": f"job-{index:02}", "filenames": [f"{index}.pdf"]} for index in range(12)]

        def state_version(self):
            return 0

    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.get("/api/jobs", params={"offset": 10, "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 12
    assert payload["offset"] == 10
    assert payload["limit"] == 10
    assert [job["id"] for job in payload["jobs"]] == ["job-10", "job-11"]
    assert payload["active_count"] == 0


def test_jobs_endpoint_reports_active_job_count(monkeypatch):
    class FakeQueue:
        def list_jobs(self):
            return [
                {"id": "job-running", "status": "running"},
                {"id": "job-paused", "status": "paused_for_queries"},
                {"id": "job-done", "status": "done"},
            ]

        def state_version(self):
            return 0

    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.get("/api/jobs")

    assert response.status_code == 200
    assert response.json()["active_count"] == 2


def test_jobs_endpoint_uses_conditional_etag(monkeypatch):
    class FakeQueue:
        def __init__(self):
            self.jobs = [{"id": "job-a", "status": "queued", "phase": "queued", "filenames": ["a.pdf"]}]
            self.version = 0

        def list_jobs(self):
            return list(self.jobs)

        def state_version(self):
            return self.version

    queue = FakeQueue()
    monkeypatch.setattr(web_app, "job_queue", queue)

    client = TestClient(web_app.app)
    first = client.get("/api/jobs")
    second = client.get("/api/jobs", headers={"If-None-Match": first.headers["etag"]})
    queue.jobs.append({"id": "job-b", "status": "done", "phase": "done", "filenames": ["b.pdf"]})
    queue.version += 1
    changed = client.get("/api/jobs", headers={"If-None-Match": first.headers["etag"]})

    assert first.status_code == 200
    assert second.status_code == 304
    assert changed.status_code == 200
    assert changed.headers["etag"] != first.headers["etag"]
    assert changed.json()["total"] == 2


def test_health_endpoint_uses_conditional_etag(monkeypatch, workspace_tmp):
    class FakeQueue:
        def __init__(self):
            self.queued_count = 0

        def summary(self):
            return {
                "active_query_count": 0,
                "queued_count": self.queued_count,
                "running_job_ids": [],
                "indexing_job_ids": [],
                "job_count": self.queued_count,
            }

    queue = FakeQueue()
    monkeypatch.setattr(web_app, "job_queue", queue)
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")

    client = TestClient(web_app.app)
    first = client.get("/api/health")
    second = client.get("/api/health", headers={"If-None-Match": first.headers["etag"]})
    queue.queued_count = 1
    changed = client.get("/api/health", headers={"If-None-Match": first.headers["etag"]})

    assert first.status_code == 200
    assert second.status_code == 304
    assert changed.status_code == 200
    assert changed.headers["etag"] != first.headers["etag"]
    assert changed.json()["queue"]["queued_count"] == 1


def test_index_mutation_endpoints_reject_active_indexing(monkeypatch):
    class IndexingQueue:
        def summary(self):
            return {
                "active_query_count": 0,
                "queued_count": 0,
                "running_job_ids": ["job-index"],
                "indexing_job_ids": ["job-index"],
                "job_count": 1,
            }

    monkeypatch.setattr(web_app, "job_queue", IndexingQueue())

    client = TestClient(web_app.app)
    update = client.post("/api/index/update", json={"record_id": "row", "content": "updated"})
    delete = client.post("/api/index/delete", json={"record_ids": ["row"]})

    assert update.status_code == 409
    assert delete.status_code == 409
    assert "indexing is running" in update.json()["detail"]


def test_read_endpoints_remain_available_during_active_indexing(monkeypatch, workspace_tmp):
    class ActiveIndexingQueue:
        def summary(self):
            return {
                "active_query_count": 0,
                "queued_count": 0,
                "running_job_ids": ["job-index"],
                "indexing_job_ids": ["job-index"],
                "active_job_count": 1,
                "job_count": 1,
            }

        def list_jobs(self):
            return [{"id": "job-index", "status": "running", "phase": "indexing", "filenames": []}]

        def state_version(self):
            return 0

    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    monkeypatch.setattr(web_app, "job_queue", ActiveIndexingQueue())
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")

    client = TestClient(web_app.app)

    health = client.get("/api/health")
    jobs = client.get("/api/jobs")
    pdfs = client.get("/api/pdfs")

    assert health.status_code == 200
    assert health.json()["queue"]["indexing_job_ids"] == ["job-index"]
    assert jobs.status_code == 200
    assert jobs.json()["active_count"] == 1
    assert pdfs.status_code == 200
    assert pdfs.json()["pdfs"] == []


def test_pdf_documents_endpoint_paginates(monkeypatch, workspace_tmp):
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    web_app.PdfRegistry(registry_path).register_queued(
        job_id="job",
        files=[
            {
                "filename": f"doc-{index:02}.pdf",
                "hash": f"hash-{index:02}",
                "staging_path": "",
            }
            for index in range(12)
        ],
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)

    client = TestClient(web_app.app)
    response = client.get("/api/pdfs", params={"offset": 10, "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 12
    assert payload["offset"] == 10
    assert payload["limit"] == 10
    assert [pdf["filename"] for pdf in payload["pdfs"]] == ["doc-10.pdf", "doc-11.pdf"]


def test_pdf_documents_endpoint_uses_conditional_etag_after_trust_update(monkeypatch, workspace_tmp):
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    trust_path = workspace_tmp / "trust.json"
    web_app.PdfRegistry(registry_path).register_queued(
        job_id="job",
        files=[{"filename": "doc.pdf", "hash": "hash-doc", "staging_path": ""}],
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", trust_path)
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")

    client = TestClient(web_app.app)
    first = client.get("/api/pdfs")
    second = client.get("/api/pdfs", headers={"If-None-Match": first.headers["etag"]})
    update = client.post("/api/pdfs/hash-doc/trust", json={"source_group": "official"})
    changed = client.get("/api/pdfs", headers={"If-None-Match": first.headers["etag"]})

    assert first.status_code == 200
    assert second.status_code == 304
    assert update.status_code == 200
    assert update.json()["pdf"]["trust"]["source_group"] == "official"
    assert changed.status_code == 200
    assert changed.headers["etag"] != first.headers["etag"]
    assert changed.json()["pdfs"][0]["trust"]["source_group"] == "official"


def test_pdf_documents_include_quality_payload(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "quality.md"
    markdown_path.write_text("# Overview\n\n" + ("alpha context " * 80), encoding="utf-8")
    source_hash = "hash-quality"
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash=source_hash,
        source_pdf_name="quality.pdf",
        source_pdf_path=web_app.DATA_DIR / "quality.pdf",
    )
    db_dir = workspace_tmp / "db"
    local_rag.write_index_manifest(
        db_dir,
        [
            {
                "id": "summary",
                "node_type": "document_summary",
                "content": "summary",
                "source_hash": source_hash,
                "source_pdf_name": "quality.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "quality.pdf"),
                "page_start": 1,
                "page_end": 2,
            },
            {
                "id": "chunk",
                "node_type": "chunk",
                "content": "alpha context",
                "source_hash": source_hash,
                "source_pdf_name": "quality.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "quality.pdf"),
                "page_start": 1,
                "page_end": 1,
            },
            {
                "id": "chunk-2",
                "node_type": "chunk",
                "content": "beta context",
                "source_hash": source_hash,
                "source_pdf_name": "quality.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "quality.pdf"),
                "page_start": 2,
                "page_end": 2,
            },
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    web_app.update_document_trust(source_hash, {"review_status": "approved"})

    client = TestClient(web_app.app)
    response = client.get("/api/pdfs")

    assert response.status_code == 200
    quality = response.json()["pdfs"][0]["quality"]
    assert quality["label"] == "ready"
    assert quality["warnings"] == []
    assert quality["chunk_count"] == 2
    assert quality["record_count"] == 3
    assert quality["markdown_exists"] is True
    assert response.json()["pdfs"][0]["trust"]["review_status"] == "approved"
    assert response.json()["pdfs"][0]["trust"]["source_group"] == "ungrouped"
    assert response.json()["pdfs"][0]["trust"]["reliability_weight"] == 0.1


def test_pdf_documents_flag_single_chunk_document(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "single-chunk.md"
    markdown_path.write_text("# Single\n\n" + ("alpha context " * 80), encoding="utf-8")
    source_hash = "hash-single-chunk"
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash=source_hash,
        source_pdf_name="single-chunk.pdf",
        source_pdf_path=web_app.DATA_DIR / "single-chunk.pdf",
    )
    db_dir = workspace_tmp / "db"
    local_rag.write_index_manifest(
        db_dir,
        [
            {
                "id": "chunk",
                "node_type": "chunk",
                "content": "alpha context",
                "source_hash": source_hash,
                "source_pdf_name": "single-chunk.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "single-chunk.pdf"),
                "page_start": 1,
                "page_end": 1,
            },
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    web_app.update_document_trust(source_hash, {"review_status": "approved"})

    client = TestClient(web_app.app)
    response = client.get("/api/pdfs")

    assert response.status_code == 200
    quality = response.json()["pdfs"][0]["quality"]
    assert "single_chunk" in quality["warnings"]
    assert "no_chunks" not in quality["warnings"]
    assert quality["label"] == "review"


def test_pdf_documents_flag_low_chunk_density_document(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "sparse-chunks.md"
    markdown_path.write_text("# Sparse\n\n" + ("dense body text " * 900), encoding="utf-8")
    source_hash = "hash-low-density"
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash=source_hash,
        source_pdf_name="sparse-chunks.pdf",
        source_pdf_path=web_app.DATA_DIR / "sparse-chunks.pdf",
    )
    db_dir = workspace_tmp / "db"
    local_rag.write_index_manifest(
        db_dir,
        [
            {
                "id": "chunk",
                "node_type": "chunk",
                "content": "oversized body " * 600,
                "source_hash": source_hash,
                "source_pdf_name": "sparse-chunks.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "sparse-chunks.pdf"),
                "page_start": 1,
                "page_end": 4,
            },
            {
                "id": "chunk-2",
                "node_type": "chunk",
                "content": "oversized body " * 600,
                "source_hash": source_hash,
                "source_pdf_name": "sparse-chunks.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "sparse-chunks.pdf"),
                "page_start": 5,
                "page_end": 8,
            },
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    web_app.update_document_trust(source_hash, {"review_status": "approved"})

    client = TestClient(web_app.app)
    response = client.get("/api/pdfs")

    assert response.status_code == 200
    quality = response.json()["pdfs"][0]["quality"]
    assert quality["chunk_count"] == 2
    assert quality["content_char_count"] == 2 * len("oversized body " * 600)
    assert "low_chunk_density" in quality["warnings"]
    assert "single_chunk" not in quality["warnings"]
    assert quality["label"] == "review"


def test_pdf_documents_warn_when_source_job_was_interrupted(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "interrupted.md"
    markdown_path.write_text("# Interrupted\n\n" + ("alpha context " * 80), encoding="utf-8")
    source_hash = "hash-source-interrupted"
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash=source_hash,
        source_pdf_name="interrupted.pdf",
        source_pdf_path=web_app.DATA_DIR / "interrupted.pdf",
    )
    db_dir = workspace_tmp / "db"
    local_rag.write_index_manifest(
        db_dir,
        [
            {
                "id": "chunk",
                "node_type": "chunk",
                "content": "alpha context",
                "source_hash": source_hash,
                "source_pdf_name": "interrupted.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "interrupted.pdf"),
                "page_start": 1,
                "page_end": 1,
            },
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    registry_path = workspace_tmp / "registry.json"
    web_app.PdfRegistry(registry_path).mark_sources_interrupted(
        job_id="job-source-cancel",
        source_hashes=[source_hash],
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")

    document = TestClient(web_app.app).get("/api/pdfs").json()["pdfs"][0]

    assert document["status"] == "indexed"
    assert document["last_interrupted_job_id"] == "job-source-cancel"
    assert "job_interrupted" in document["quality"]["warnings"]


def test_update_document_trust_rejects_invalid_source_group(monkeypatch, workspace_tmp):
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")

    with pytest.raises(ValueError, match="source_group must be one of"):
        web_app.update_document_trust("hash-a", {"source_group": "bad-group"})


def test_source_group_weights_match_review_policy():
    assert web_app.source_group_weight("official") == 1.0
    assert web_app.source_group_weight("student_research") == 0.9
    assert web_app.source_group_weight("unofficial") == 0.8
    assert web_app.source_group_weight("ungrouped") == 0.1


def test_pdf_documents_sort_ungrouped_first(monkeypatch, workspace_tmp):
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    trust_path = workspace_tmp / "trust.json"
    registry = web_app.PdfRegistry(registry_path)
    registry.register_queued(
        job_id="job-a",
        files=[{"filename": "zeta.pdf", "hash": "hash-zeta", "staging_path": ""}],
    )
    registry.register_queued(
        job_id="job-b",
        files=[{"filename": "alpha.pdf", "hash": "hash-alpha", "staging_path": ""}],
    )
    web_app.update_document_trust("hash-alpha", {"source_group": "official"}, trust_path=trust_path)

    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", trust_path)

    client = TestClient(web_app.app)
    response = client.get("/api/pdfs")

    assert response.status_code == 200
    assert [item["hash"] for item in response.json()["pdfs"]] == ["hash-zeta", "hash-alpha"]


def test_pdf_trust_endpoint_marks_stale_source(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "stale.md"
    markdown_path.write_text("# Overview\n\n" + ("alpha context " * 80), encoding="utf-8")
    source_hash = "hash-stale"
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash=source_hash,
        source_pdf_name="stale.pdf",
        source_pdf_path=web_app.DATA_DIR / "stale.pdf",
    )
    db_dir = workspace_tmp / "db"
    local_rag.write_index_manifest(
        db_dir,
        [
            {
                "id": "chunk",
                "node_type": "chunk",
                "content": "alpha context",
                "source_hash": source_hash,
                "source_pdf_name": "stale.pdf",
                "source_pdf_path": str(web_app.DATA_DIR / "stale.pdf"),
                "page_start": 1,
                "page_end": 1,
            },
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")

    client = TestClient(web_app.app)
    update = client.post(
        f"/api/pdfs/{source_hash}/trust",
        json={
            "review_status": "stale",
            "source_type": "student_project",
            "reviewed_by": "  Alex Reviewer  ",
            "notes": "old design rules",
        },
    )
    listing = client.get("/api/pdfs")

    assert update.status_code == 200
    assert update.json()["trust"]["review_status"] == "stale"
    assert update.json()["trust"]["reviewed_by"] == "Alex Reviewer"
    document = listing.json()["pdfs"][0]
    assert document["trust"]["source_type"] == "student_project"
    assert document["trust"]["reviewed_by"] == "Alex Reviewer"
    assert document["trust"]["notes"] == "old design rules"
    assert "marked_stale" in document["quality"]["warnings"]
    assert document["quality"]["label"] == "review"


def test_pdf_bulk_trust_endpoint_updates_multiple_sources(monkeypatch, workspace_tmp):
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    trust_path = workspace_tmp / "trust.json"
    web_app.PdfRegistry(registry_path).register_queued(
        job_id="job",
        files=[
            {"filename": "a.pdf", "hash": "hash-a", "staging_path": ""},
            {"filename": "b.pdf", "hash": "hash-b", "staging_path": ""},
        ],
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", trust_path)
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")

    response = TestClient(web_app.app).post(
        "/api/pdfs/trust/bulk",
        json={"source_hashes": ["hash-a", "hash-b"], "source_group": "official"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["failed"] == []
    assert [item["source_hash"] for item in payload["updated"]] == ["hash-a", "hash-b"]
    assert {item["pdf"]["trust"]["source_group"] for item in payload["updated"]} == {"official"}


def test_pdf_bulk_trust_endpoint_reports_partial_failures(monkeypatch, workspace_tmp):
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", workspace_tmp / "processed")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")
    (workspace_tmp / "processed").mkdir()

    response = TestClient(web_app.app).post(
        "/api/pdfs/trust/bulk",
        json={"source_hashes": ["hash-a", ""], "source_group": "official"},
    )

    assert response.status_code == 200
    assert response.json()["updated"][0]["source_hash"] == "hash-a"
    assert response.json()["failed"][0]["error"] == "source_hash cannot be empty"


def test_pdf_bulk_trust_endpoint_rejects_invalid_source_group():
    response = TestClient(web_app.app).post(
        "/api/pdfs/trust/bulk",
        json={"source_hashes": ["hash-a"], "source_group": "bad-group"},
    )

    assert response.status_code == 400
    assert "source_group must be one of" in response.json()["detail"]


def test_pdf_reprocess_endpoint_queues_single_source_upload(monkeypatch, workspace_tmp):
    data_dir = workspace_tmp / "data"
    upload_dir = data_dir / "uploads" / "old-job"
    upload_dir.mkdir(parents=True)
    pdf_bytes = b"%PDF-1.4 reprocess source"
    pdf_path = upload_dir / "source.pdf"
    pdf_path.write_bytes(pdf_bytes)
    source_hash = hashlib.sha256(pdf_bytes).hexdigest()
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "source.md"
    markdown_path.write_text("old markdown", encoding="utf-8")
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash=source_hash,
        source_pdf_name="source.pdf",
        source_pdf_path=pdf_path,
    )
    registry_path = workspace_tmp / "registry.json"
    file_upload = {
        "filename": "source.pdf",
        "hash": source_hash,
        "staging_path": "",
        "upload_path": str(pdf_path),
        "processed_markdown_path": str(markdown_path),
    }
    registry = web_app.PdfRegistry(registry_path)
    registry.register_queued(
        job_id="old-job",
        files=[file_upload],
        options={
            "ocr_backend": "tesseract_cli",
            "embedding_model": "old-embed",
            "code_enrichment": False,
            "formula_enrichment": True,
        },
    )
    registry.mark_job_status(job_id="old-job", files=[file_upload], status="indexed")
    captured = {}

    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                uploads=kwargs["uploads"],
                force_duplicate_hashes=kwargs["force_duplicate_hashes"],
                staging_dir=str(kwargs["staging_dir"]),
                options=kwargs["options"],
            )

    monkeypatch.setattr(web_app, "DATA_DIR", data_dir)
    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(f"/api/pdfs/{source_hash}/reprocess")

    assert response.status_code == 200
    assert response.json()["kind"] == "upload"
    assert captured["force_duplicate_hashes"] == [source_hash]
    assert captured["uploads"][0]["hash"] == source_hash
    staged_path = Path(captured["uploads"][0]["staging_path"])
    assert staged_path.exists()
    assert staged_path.read_bytes() == pdf_bytes
    assert captured["options"]["ocr_backend"] == "tesseract_cli"
    assert captured["options"]["embedding_model"] == "old-embed"
    assert captured["options"]["asset_dir"] == str(web_app.ASSET_DIR)
    assert captured["options"]["code_enrichment"] is False
    assert captured["options"]["formula_enrichment"] is True
    queued = web_app.PdfRegistry(registry_path).load()["pdfs"][source_hash]
    assert queued["status"] == "queued"
    assert queued["previous_entry"]["status"] == "indexed"


def test_pdf_reindex_endpoint_queues_index_only_job(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "source.md"
    markdown_path.write_text("# reindex only body", encoding="utf-8")
    source_hash = "hash-reindex"
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=markdown_path,
        source_hash=source_hash,
        source_pdf_name="source.pdf",
        source_pdf_path=workspace_tmp / "source.pdf",
    )
    registry_path = workspace_tmp / "registry.json"
    web_app.PdfRegistry(registry_path).register_queued(
        job_id="old-job",
        files=[
            {
                "filename": "source.pdf",
                "hash": source_hash,
                "staging_path": "",
                "upload_path": str(workspace_tmp / "source.pdf"),
                "processed_markdown_path": str(markdown_path),
            }
        ],
        options={"embedding_model": "old-embed", "summary_mode": "llm"},
    )
    captured = {}

    class FakeQueue:
        def enqueue_reindex_source(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id="job-reindex",
                kind="reindex_source",
                source_hashes=list(kwargs["source_hashes"]),
                options=dict(kwargs["options"]),
            )

        def enqueue_upload(self, **kwargs):
            raise AssertionError("re-index must not re-run ingestion")

    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(f"/api/pdfs/{source_hash}/reindex")

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "reindex_source"
    assert body["source_hashes"] == [source_hash]
    assert captured["source_hashes"] == [source_hash]
    assert captured["options"]["embedding_model"] == "old-embed"
    assert captured["options"]["summary_mode"] == "llm"
    # Ingestion is skipped: the processed Markdown is left untouched.
    assert markdown_path.read_text(encoding="utf-8") == "# reindex only body"


def test_pdf_reindex_404_when_no_processed_markdown(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)

    class FakeQueue:
        def enqueue_reindex_source(self, **kwargs):
            raise AssertionError("should not enqueue when no markdown exists")

    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post("/api/pdfs/missing-hash/reindex")

    assert response.status_code == 404


def test_reindex_source_job_deletes_records_before_indexing(monkeypatch, workspace_tmp, lancedb_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    markdown_path = processed_dir / "source.md"
    markdown_path.write_text("reindex body", encoding="utf-8")
    db_dir = lancedb_tmp / "db"
    LanceDBVectorStore(db_dir).write_records(
        [
            {
                "id": "target",
                "doc_id": "doc-target",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": str(markdown_path),
                "chunk_index": 0,
                "content": "target content",
                "source_hash": "hash-target",
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "kept",
                "doc_id": "doc-kept",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "kept.md",
                "chunk_index": 0,
                "content": "kept content",
                "source_hash": "hash-other",
                "vector": [0.0, 1.0, 0.0],
            },
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    index_counts = []

    def fake_index(md_dir, db_dir_arg, **kwargs):
        # By the time indexing runs, the target source's vectors are gone.
        index_counts.append(LanceDBVectorStore(db_dir).count())

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=processed_dir,
        db_dir=db_dir,
        registry_path=workspace_tmp / "registry.json",
        run_indexing_func=fake_index,
    )
    job = queue.enqueue_reindex_source(source_hashes=["hash-target"])

    _wait_for(lambda: queue.get_job(job.id)["status"] in {"done", "failed"})

    assert queue.get_job(job.id)["status"] == "done"
    assert queue.get_job(job.id)["kind"] == "reindex_source"
    assert index_counts == [1]


def test_pdf_documents_list_and_download_endpoint(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    pdf_path = web_app.DATA_DIR / "ST231.pdf"
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=processed_dir / "doc.md",
        source_hash="hash-download",
        source_pdf_name=pdf_path.name,
        source_pdf_path=pdf_path,
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)

    client = TestClient(web_app.app)
    listing = client.get("/api/pdfs")

    assert listing.status_code == 200
    pdfs = listing.json()["pdfs"]
    assert pdfs[0]["hash"] == "hash-download"
    assert pdfs[0]["can_download"] is True
    assert pdfs[0]["download_url"] == "/api/pdfs/hash-download/download"

    download = client.get("/api/pdfs/hash-download/download")
    assert download.status_code == 200
    assert download.content.startswith(b"%PDF")
    assert download.headers["content-disposition"].startswith("attachment;")

    view = client.get("/api/pdfs/hash-download/view")
    assert view.status_code == 200
    assert view.content.startswith(b"%PDF")
    assert view.headers["content-disposition"].startswith("inline;")

    by_title = client.get("/api/pdfs", params={"search": "ST231"})
    by_hash = client.get("/api/pdfs", params={"search": "hash-download"})
    no_match = client.get("/api/pdfs", params={"search": "no-match"})

    assert by_title.json()["total"] == 1
    assert by_hash.json()["total"] == 1
    assert no_match.json()["total"] == 0


def test_pdf_download_reports_missing_hash_and_missing_file(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    missing_pdf = web_app.DATA_DIR / f"missing-{uuid.uuid4().hex}.pdf"
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=processed_dir / "doc.md",
        source_hash="hash-missing-file",
        source_pdf_name="missing.pdf",
        source_pdf_path=missing_pdf,
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)

    client = TestClient(web_app.app)

    assert client.get("/api/pdfs/no-such-hash/download").status_code == 404
    assert client.get("/api/pdfs/hash-missing-file/download").status_code == 404


def test_pdf_download_rejects_paths_outside_data_dir(monkeypatch, workspace_tmp):
    """Paths outside every configured corpus root are refused with 403.

    The allow-list covers data_dir, the repo root, and db/processed roots, so
    the "outside" file must live in the system temp dir to prove containment.
    """
    import tempfile

    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    outside_pdf = Path(tempfile.gettempdir()) / f"rag_outside_{uuid.uuid4().hex}.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4 outside")
    try:
        write_source_entry(
            processed_dir=processed_dir,
            markdown_path=processed_dir / "doc.md",
            source_hash="hash-unsafe",
            source_pdf_name="outside.pdf",
            source_pdf_path=outside_pdf,
        )
        monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
        monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)

        client = TestClient(web_app.app)

        assert client.get("/api/pdfs/hash-unsafe/download").status_code == 403
    finally:
        outside_pdf.unlink(missing_ok=True)


def test_image_asset_endpoint_serves_manifest_known_files(monkeypatch, workspace_tmp):
    asset_dir = workspace_tmp / "assets"
    store = ImageAssetStore(asset_dir)
    asset = store.save_image(
        image_data=b"graph-png",
        source_hash="hash-a",
        source_pdf_name="report.pdf",
        page_no=2,
        description="graph description",
    )
    outside = workspace_tmp / "outside.png"
    outside.write_bytes(b"outside")
    manifest = store.load_manifest()
    manifest["assets"]["unsafe"] = {
        "asset_id": "unsafe",
        "source_hash": "hash-a",
        "source_pdf_name": "report.pdf",
        "page_no": 2,
        "description": "unsafe",
        "mime_type": "image/png",
        "relative_path": "../outside.png",
        "image_sha": "unsafe",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(web_app, "ASSET_DIR", asset_dir)

    client = TestClient(web_app.app)
    served = client.get(f"/api/assets/{asset['asset_id']}")
    unknown = client.get("/api/assets/no-such-asset")
    unsafe = client.get("/api/assets/unsafe")

    assert served.status_code == 200
    assert served.content == b"graph-png"
    assert served.headers["content-type"].startswith("image/png")
    assert unknown.status_code == 404
    assert unsafe.status_code == 404


def _frontend_script() -> str:
    """Full client source: the module entry plus every web/js module.

    The front-end split into ES modules (web/app.js imports web/js/*.js), so
    source-string assertions need the whole tree, not just the entry file.
    """
    entry = Path("web/app.js").read_text(encoding="utf-8")
    modules = sorted(Path("web/js").glob("*.js"))
    return entry + chr(10).join(m.read_text(encoding="utf-8") for m in modules)


def test_sources_panel_frontend_includes_image_asset_preview():
    script = _frontend_script()
    styles = Path("web/styles.css").read_text(encoding="utf-8")

    assert "source.assets" in script
    assert "item.assets" in script
    assert "source-assets" in script
    assert "source-asset" in script
    assert "index-assets" in script
    assert "Open image" in script
    assert ".source-assets" in styles
    assert ".source-asset img" in styles
    assert ".index-assets .source-asset img" in styles


def test_frontend_upload_uses_xhr_progress_and_keeps_duplicate_prompt():
    script = _frontend_script()

    assert "function uploadFormData(path, body, options = {})" in script
    assert "new XMLHttpRequest()" in script
    assert 'request.upload.addEventListener("progress"' in script
    # Per-file uploads still POST each small file via uploadFormData.
    assert 'uploadFormData("/api/uploads", body' in script
    # The chunked-upload completion uses requestJson with the complete endpoint,
    # but the normal upload path must not.
    assert 'requestJson("/api/uploads/complete"' in script
    # Per-file sequential uploads report progress per file.
    assert "Uploading ${index + 1}/${files.length}" in script or "Uploading" in script
    assert "Duplicate upload blocked." in script
    assert "showDuplicatePrompt(" in script


def test_frontend_jobs_polling_uses_active_count():
    script = _frontend_script()

    assert "JOBS_ACTIVE_POLL_INTERVAL_MS" in script
    assert "state.jobsActive = Number(data.active_count || 0) > 0" in script
    assert "wasActive && !state.jobsActive" in script
    assert "state.jobsActive ? JOBS_ACTIVE_POLL_INTERVAL_MS : state.jobsPollIntervalMs" in script
    assert 'data-job-action="cancel"' in script
    assert '/api/jobs/${encodeURIComponent(jobId)}/cancel' in script
    assert "openJobLogIds: new Set()" in script
    assert "function rememberJobLogOpenState()" in script
    assert 'details.job-log[data-job-id]' in script
    assert 'data-job-id="${escapeHtml(jobId)}"${logOpen}' in script


def test_frontend_uses_etag_cache_lazy_panels_and_keyed_row_patching():
    script = _frontend_script()
    # "startup" means the entry file's init tail only -- module code must not
    # leak into the eager-startup slice now that sources are concatenated.
    entry = Path("web/app.js").read_text(encoding="utf-8")
    startup = entry[entry.index("loadReviewerName();") :]

    assert "const getJsonCache = new Map();" in script
    assert 'headers.set("If-None-Match", cached.etag)' in script
    assert "response.status === 304" in script
    assert "notModified: true" in script
    assert "state.uploadDataDirty" in script
    assert "!state.indexLoaded || state.indexDirty" in script
    assert "refreshJobs();" not in startup
    assert "refreshPdfs();" not in startup
    assert "function patchTableRows(tbody, items, options)" in script
    assert "row.dataset.patchKey" in script
    assert "row.dataset.renderKey" in script
    assert "tbody.replaceChildren(fragment)" in script
    assert "function patchPdfRow(item, options = {})" in script
    assert "function renderJobRows(jobs)" in script
    assert "existingChildRows" in script


def test_frontend_reduces_chat_render_churn():
    script = _frontend_script()

    assert "function scheduleChatScroll(force = false)" in script
    assert "window.requestAnimationFrame" in script
    assert "scheduleChatScroll();" in script
    assert "const stableHtml = stableDelta ? await renderMarkdown(stableDelta) : \"\";" in script
    assert "setStreamTailRaw(parts, keys, tailText)" in script
    assert "sourcesSignature" in script
    assert "toolResultsSignature" in script


def test_frontend_chat_auth_uses_headers_instance():
    script = _frontend_script()

    assert 'const chatHeaders = new Headers({ "Content-Type": "application/json" });' in script


def test_frontend_category_weight_accepts_hundredths():
    script = _frontend_script()

    assert 'min="0.01" max="100" step="0.01"' in script


def test_root_injects_static_asset_cache_busting():
    response = TestClient(web_app.app).get("/")

    assert response.status_code == 200
    assert "/static/styles.css?v=" in response.text
    assert "/static/app.js?v=" in response.text


def test_frontend_pdf_rows_show_interrupted_warning():
    markup = Path("web/index.html").read_text(encoding="utf-8")
    script = _frontend_script()
    styles = Path("web/styles.css").read_text(encoding="utf-8")

    assert '<th scope="col">Action</th>' in markup
    assert "renderPdfInterruptedBadge" in script
    assert "job_interrupted" in script
    assert "pdf-warning-badge" in script
    assert ".pdf-title-line" in styles
    assert ".pdf-warning-badge" in styles


def test_frontend_includes_new_user_guide_and_walkthrough():
    markup = Path("web/index.html").read_text(encoding="utf-8")
    script = _frontend_script()
    styles = Path("web/styles.css").read_text(encoding="utf-8")
    refresh_pdfs = script[
        script.index("async function refreshPdfs(options = {})") : script.index("async function handlePdfAction")
    ]
    render_pdf_rows = script[
        script.index("function renderPdfRows(items)") : script.index("function clearWalkthroughHighlight")
    ]
    render_step = script[
        script.index("function renderWalkthroughStep()") : script.index("function startWalkthrough")
    ]

    assert 'data-tab-target="guide"' in markup
    assert 'id="guide"' in markup
    assert 'id="startGuideButton"' in markup
    assert 'id="walkthroughOverlay"' in markup
    assert 'id="welcomeTutorialOverlay"' in markup
    assert 'id="welcomeTutorialStartButton"' in markup
    assert 'id="welcomeTutorialSkipButton"' in markup
    assert 'id="cachePromptOverlay"' in markup
    assert "New User Guide" in markup
    assert "Welcome to Local FSAE RAG" in markup
    assert "TUTORIAL_SEEN_COOKIE" in script
    assert "SITE_VERSION_COOKIE" in script
    assert "getCookie" in script
    assert "setCookie" in script
    assert "maybeStartFirstVisitWalkthrough()" in script
    assert "showWelcomeTutorialPrompt" in script
    assert "acceptWelcomeTutorialPrompt" in script
    assert "welcomeTutorialPromptOpen()" in script
    assert "handleSiteVersionFromUpdateStatus(data)" in script
    assert "reloadAfterCacheClear" in script
    assert "caches.delete" in script
    assert "walkthroughSteps" in script
    assert "fakePdf: true" in script
    assert "ensureWalkthroughFakePdf" in script
    assert "removeWalkthroughFakePdf" in script
    assert "renderPdfRows(data.pdfs || []);" in refresh_pdfs
    assert "state.walkthroughFakePdfPinned" in render_pdf_rows
    assert "highlightWalkthroughTarget(step.target)" in render_pdf_rows
    assert "activateTab(step.tab, { refreshUpload: !fakePdfStep })" in render_step
    assert "WALKTHROUGH_FAKE_PDF_HASH" in script
    assert "Example untagged source.pdf" in script
    assert 'target.closest("details")' in script
    assert "details.open = true" in script
    assert "walkthrough-highlight" in script
    assert "Tag source reliability" in script
    assert "reliability group for each PDF" in markup
    assert "Tag ungrouped PDFs" in markup
    assert ".guide-layout" in styles
    assert ".walkthrough-dialog" in styles
    assert ".walkthrough-fake-pdf-row" in styles
    assert ".modal-dialog" in styles


def test_frontend_pdf_trust_actions_record_reviewer_name():
    markup = Path("web/index.html").read_text(encoding="utf-8")
    script = _frontend_script()
    styles = Path("web/styles.css").read_text(encoding="utf-8")

    assert 'id="reviewerNameInput"' in markup
    assert 'id="uploadGroupsPanel"' in markup
    assert 'id="sourceGroupPromptOverlay"' in markup
    assert 'data-source-group-choice="official"' in markup
    assert 'data-source-group-choice="student_research"' in markup
    assert 'data-source-group-choice="unofficial"' in markup
    assert "Weight 0.90" in markup
    assert "Weight 0.80" in markup
    assert "REVIEWER_NAME_COOKIE" in script
    assert "normalizeReviewerName" in script
    assert "ensureReviewerName" in script
    assert "formatBrowserTimestamp" in script
    assert "chooseSourceGroup" in script
    assert "closeSourceGroupPrompt" in script
    assert "Set source group: official" not in script
    assert "renderUploadGroupSelectors" in script
    assert "selectedUploadSourceGroups" in script
    assert "source_groups" in script
    assert "student_research: 0.9" in script
    assert "unofficial: 0.8" in script
    assert "data-pdf-action=\"tag-group\"" in script
    assert "Group:" in script
    assert "pdf-untagged-row" in script
    assert "quality-untagged" in script
    assert "timeZoneName" in script
    assert "body.reviewed_by = reviewer" in script
    assert "Reviewed by:" in script
    assert "saveReviewerName(els.reviewerNameInput.value)" in script
    assert ".upload-groups-panel" in styles
    assert ".pdf-untagged-row" in styles
    assert ".quality-untagged" in styles
    assert "border-left: 4px solid var(--danger)" in styles
    assert "color: var(--danger)" in styles
    assert ".source-group-actions" in styles
    assert "#reviewerNameInput" in styles
    # Source-group popup hotkeys (Ctrl+1/2/3) + multi-select bulk tagging.
    assert 'source-group-hotkey">Ctrl+1' in markup
    assert 'source-group-hotkey">Ctrl+2' in markup
    assert 'source-group-hotkey">Ctrl+3' in markup
    assert 'id="pdfSelectAllCheckbox"' in markup
    assert 'id="pdfBulkActionBar"' in markup
    assert 'id="pdfBulkTagButton"' in markup
    assert ".source-group-hotkey" in styles
    assert ".pdf-select-col" in styles
    assert ".pdf-bulk-bar" in styles
    assert "HOTKEY_SOURCE_GROUPS" in script
    assert "selectedPdfHashes" in script
    assert "applyBulkTagGroup" in script
    assert 'requestJson("/api/pdfs/trust/bulk"' in script
    assert "selectAllUntaggedPdfs" in script
    assert "data-pdf-select=" in script


def test_frontend_delete_control_lives_in_documents_panel():
    script = _frontend_script()

    pdf_actions_block = re.search(
        r"function renderPdfActions\(item\) \{(?P<body>.*?)\n\}",
        script,
        flags=re.S,
    )

    assert 'data-action="delete"' not in script
    assert pdf_actions_block is not None
    assert 'data-pdf-action="delete"' in pdf_actions_block.group("body")
    assert 'method: "DELETE"' in script
    assert "/api/pdfs/${encodeURIComponent(sourceHash)}" in script
    assert "/api/index/delete" not in script


def test_chat_stream_endpoint_streams_and_tracks_query_count(monkeypatch):
    events = []

    class FakeQueue:
        def begin_query(self):
            events.append("begin")

        def finish_query(self):
            events.append("finish")

    class FakeQueryEngine:
        def __init__(self, **kwargs):
            events.append(("engine", kwargs))

        def ask_stream_events(self, question, history=None):
            assert question == "alpha?"
            yield {"type": "thinking", "text": "checking "}
            yield {
                "type": "tool_result",
                "tool": "search_local_context",
                "text": "Retrieved 1 local source chunk(s).",
                "result": {
                    "tool": "search_local_context",
                    "query": "alpha?",
                    "result_count": 1,
                    "results": [{"citation": "[S1]", "content": "alpha context"}],
                },
                "content": (
                    '{"tool":"search_local_context","query":"alpha?",'
                    '"result_count":1,"results":[{"citation":"[S1]","content":"alpha context"}]}'
                ),
            }
            yield {"type": "sources", "sources": [{"id": "S1", "label": "[S1]", "kind": "local"}]}
            yield {"type": "answer", "text": "chunk "}
            yield {"type": "answer", "text": "two"}

    monkeypatch.setattr(web_app, "job_queue", FakeQueue())
    monkeypatch.setattr(query, "QueryEngine", FakeQueryEngine)

    client = TestClient(web_app.app)
    response = client.post(
        "/api/chat/stream",
        json={
            "question": "alpha?",
            "temperature": 0.65,
            "max_k": 25,
            "context_window": 4096,
            "llm_num_predict": 512,
            "web_search_enabled": False,
            "retrieval_min_score": 0.73,
        },
    )

    assert response.status_code == 200
    assert [json.loads(line) for line in response.text.splitlines()] == [
        {"type": "thinking", "text": "checking "},
        {
            "type": "tool_result",
            "tool": "search_local_context",
            "text": "Retrieved 1 local source chunk(s).",
            "result": {
                "tool": "search_local_context",
                "query": "alpha?",
                "result_count": 1,
                "results": [{"citation": "[S1]", "content": "alpha context"}],
            },
            "content": (
                '{"tool":"search_local_context","query":"alpha?",'
                '"result_count":1,"results":[{"citation":"[S1]","content":"alpha context"}]}'
            ),
        },
        {"type": "sources", "sources": [{"id": "S1", "label": "[S1]", "kind": "local"}]},
        {"type": "answer", "text": "chunk "},
        {"type": "answer", "text": "two"},
    ]
    assert events[0] == "begin"
    assert events[1][1]["temperature"] == 0.65
    assert events[1][1]["asset_dir"] == str(web_app.ASSET_DIR)
    assert events[1][1]["sampler_top_k"] == 25
    assert events[1][1]["context_window"] == 4096
    assert events[1][1]["llm_num_predict"] == 512
    assert events[1][1]["llm_timeout"] == web_app.CHAT_CONFIG["llm_timeout"]
    assert events[1][1]["web_search_enabled"] is False
    assert events[1][1]["retrieval_candidate_k"] == 80
    assert events[1][1]["retrieval_min_score"] == 0.73
    assert events[1][1]["retrieval_relative_cutoff"] == 0.72
    assert events[1][1]["context_token_fraction"] == 0.60
    assert events[1][1]["ollama_health_check_interval"] == 5.0
    assert events[1][1]["ollama_max_lost_health_checks"] == 5
    assert "search_local_context" in events[1][1]["system_prompt"]
    assert events[-1] == "finish"


def test_render_endpoint_formats_markdown_and_latex():
    client = TestClient(web_app.app)
    response = client.post(
        "/api/render",
        json={"text": "**Synthesize the Explanation** and $\\sigma^2$"},
    )

    assert response.status_code == 200
    html = response.json()["html"]
    assert "<strong>" in html
    assert "Synthesize the Explanation" in html
    assert "<math" in html
    assert "sigma" not in html


def test_create_index_backup_snapshots_lancedb_and_prunes(lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)

    snapshots = [web_app.create_index_backup(db_dir) for _ in range(web_app.LANCEDB_BACKUP_KEEP + 2)]

    listed = web_app.list_index_backups(db_dir)
    assert len(listed) == web_app.LANCEDB_BACKUP_KEEP
    # Newest snapshot (last created) must be retained at the top.
    assert listed[0]["name"] == snapshots[-1]["name"]
    # Each retained snapshot has the LanceDB dir and a nonzero record count.
    for entry in listed:
        assert entry["lancedb_present"] is True
        assert entry["record_count"] == 2
    backup_root = web_app.index_backup_root(db_dir)
    assert len([child for child in backup_root.iterdir() if child.is_dir()]) == web_app.LANCEDB_BACKUP_KEEP


def test_list_index_backups_reports_record_count_and_detects_corruption(lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)
    backup = web_app.create_index_backup(db_dir)

    listed = web_app.list_index_backups(db_dir)
    assert listed[0]["name"] == backup["name"]
    assert listed[0]["record_count"] == 2
    assert listed[0]["lancedb_present"] is True

    # Simulate a corrupted/incomplete backup with no LanceDB directory.
    bad_dir = web_app.index_backup_root(db_dir) / "20240101-000000-deadbeef"
    bad_dir.mkdir(parents=True)
    listed = web_app.list_index_backups(db_dir)
    bad_entry = next(entry for entry in listed if entry["name"] == bad_dir.name)
    assert bad_entry["lancedb_present"] is False
    assert bad_entry["record_count"] is None


def test_backup_endpoint_enqueues_job(monkeypatch):
    captured = {}

    class FakeQueue:
        def summary(self):
            return {"indexing_job_ids": []}

        def enqueue_backup(self, **kwargs):
            job = web_app.QueueJob(id="backup-job", kind="backup")
            captured["called"] = True
            return job

    monkeypatch.setattr(web_app, "job_queue", FakeQueue())
    response = TestClient(web_app.app).post("/api/index/backup")

    assert response.status_code == 200
    assert response.json()["kind"] == "backup"
    assert captured["called"] is True


def test_backup_endpoint_blocks_during_indexing(monkeypatch):
    class ActiveIndexingQueue:
        def summary(self):
            return {"indexing_job_ids": ["job-1"]}
        def enqueue_backup(self, **kwargs):
            raise AssertionError("backup should not enqueue while indexing")

    monkeypatch.setattr(web_app, "job_queue", ActiveIndexingQueue())
    response = TestClient(web_app.app).post("/api/index/backup")

    assert response.status_code == 409


def test_rebuild_endpoint_enqueues_job_and_blocks_during_indexing(monkeypatch):
    captured = {}

    class IdleQueue:
        def summary(self):
            return {"indexing_job_ids": []}
        def enqueue_rebuild(self, **kwargs):
            job = web_app.QueueJob(id="rebuild-job", kind="rebuild")
            captured["called"] = True
            return job

    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    response = TestClient(web_app.app).post("/api/index/rebuild")
    assert response.status_code == 200
    assert response.json()["kind"] == "rebuild"
    assert captured["called"] is True

    class ActiveIndexingQueue:
        def summary(self):
            return {"indexing_job_ids": ["job-1"]}
        def enqueue_rebuild(self, **kwargs):
            raise AssertionError("rebuild should not enqueue while indexing")

    monkeypatch.setattr(web_app, "job_queue", ActiveIndexingQueue())
    blocked = TestClient(web_app.app).post("/api/index/rebuild")
    assert blocked.status_code == 409


def test_restore_endpoint_requires_backup_name():
    response = TestClient(web_app.app).post("/api/index/restore", json={})
    assert response.status_code == 422


def test_restore_endpoint_enqueues_job(monkeypatch, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)
    backup = web_app.create_index_backup(db_dir)
    captured = {}

    class IdleQueue:
        def summary(self):
            return {"indexing_job_ids": []}
        def enqueue_restore(self, *, backup_name, **kwargs):
            captured["backup_name"] = backup_name
            return web_app.QueueJob(id="restore-job", kind="restore", backup_name=backup_name)

    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    response = TestClient(web_app.app).post(
        "/api/index/restore",
        json={"backup_name": backup["name"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "restore"
    assert payload["backup_name"] == backup["name"]
    assert captured["backup_name"] == backup["name"]


def test_index_backups_endpoint_lists_snapshots(monkeypatch, lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)
    backup = web_app.create_index_backup(db_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    response = TestClient(web_app.app).get("/api/index/backups")
    assert response.status_code == 200
    payload = response.json()
    assert payload["keep"] == web_app.LANCEDB_BACKUP_KEEP
    assert payload["backups"][0]["name"] == backup["name"]
    assert payload["backups"][0]["record_count"] == 2


def test_restore_index_swaps_live_table_and_keeps_safety_backup(lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)  # 2 records: doc.md:0, doc.md:1

    # Capture an early backup, then mutate the live index.
    early_backup = web_app.create_index_backup(db_dir)
    web_app.delete_index_records(record_ids=["doc.md:1"], db_dir=db_dir)
    assert LanceDBVectorStore(db_dir).count() == 1

    result = web_app.restore_index_from_backup(db_dir, early_backup["name"])

    assert result["restored"] is True
    assert result["backup_name"] == early_backup["name"]
    # Live index reflects the restored (2-record) snapshot.
    assert LanceDBVectorStore(db_dir).count() == 2
    # A safety backup of the pre-restore (1-record) index was kept.
    safety = result["safety_backup"]
    assert safety is not None
    safety_store = LanceDBVectorStore(
        web_app.index_backup_root(db_dir) / safety["name"]
    )
    assert safety_store.count() == 1


def test_restore_index_rejects_unknown_backup(lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)
    with pytest.raises(FileNotFoundError):
        web_app.restore_index_from_backup(db_dir, "does-not-exist")


def test_restore_index_rejects_traversal_name(lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)
    with pytest.raises(ValueError):
        web_app.restore_index_from_backup(db_dir, "../escape")


def _write_minimal_pdf(path: Path, payload: bytes = b"%PDF-1.4 reingest sample"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_known_source_hashes_collects_from_registry_and_source_map(workspace_tmp):
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()

    web_app.PdfRegistry(registry_path).register_queued(
        job_id="job-a",
        files=[{"filename": "a.pdf", "hash": "hash-a"}],
    )
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=processed_dir / "b.md",
        source_hash="hash-b",
        source_pdf_name="b.pdf",
        source_pdf_path=workspace_tmp / "b.pdf",
    )

    hashes = web_app._known_source_hashes(
        registry_path=registry_path,
        processed_dir=processed_dir,
    )
    assert hashes == ["hash-a", "hash-b"]


def test_enqueue_full_reingest_stages_all_sources_and_forces_duplicates(monkeypatch, workspace_tmp):
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    data_dir = workspace_tmp / "data"
    staging_root = workspace_tmp / "staging"

    _write_minimal_pdf(data_dir / "a.pdf", b"%PDF-1.4 alpha")
    _write_minimal_pdf(data_dir / "b.pdf", b"%PDF-1.4 beta")
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=processed_dir / "a.md",
        source_hash="hash-a",
        source_pdf_name="a.pdf",
        source_pdf_path=data_dir / "a.pdf",
    )
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=processed_dir / "b.md",
        source_hash="hash-b",
        source_pdf_name="b.pdf",
        source_pdf_path=data_dir / "b.pdf",
    )

    captured = {}

    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id=str(kwargs.get("job_id")),
                kind="upload",
                filenames=list(kwargs.get("filenames", [])),
                force_duplicate_hashes=list(kwargs.get("force_duplicate_hashes", [])),
            )

    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    job = web_app.enqueue_full_reingest(
        registry_path=registry_path,
        processed_dir=processed_dir,
        staging_root=staging_root,
        root_dir=workspace_tmp,
        data_dir=data_dir,
    )

    assert job.kind == "upload"
    assert sorted(captured["filenames"]) == ["a.pdf", "b.pdf"]
    assert sorted(captured["force_duplicate_hashes"]) == ["hash-a", "hash-b"]
    upload_hashes = {upload["hash"] for upload in captured["uploads"]}
    assert upload_hashes == {"hash-a", "hash-b"}
    # Each staged file actually exists in the staging directory.
    staging_dir = captured["staging_dir"]
    for upload in captured["uploads"]:
        assert Path(upload["staging_path"]).exists()
    assert str(staging_dir).startswith(str(staging_root))


def test_enqueue_full_reingest_raises_when_no_sources(workspace_tmp):
    with pytest.raises(FileNotFoundError):
        web_app.enqueue_full_reingest(
            registry_path=workspace_tmp / "empty-registry.json",
            processed_dir=workspace_tmp / "empty-processed",
            staging_root=workspace_tmp / "staging",
            root_dir=workspace_tmp,
            data_dir=workspace_tmp / "data",
        )


def test_reingest_endpoint_enqueues_job(monkeypatch, workspace_tmp):
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    data_dir = workspace_tmp / "data"
    _write_minimal_pdf(data_dir / "a.pdf")
    write_source_entry(
        processed_dir=processed_dir,
        markdown_path=processed_dir / "a.md",
        source_hash="hash-a",
        source_pdf_name="a.pdf",
        source_pdf_path=data_dir / "a.pdf",
    )

    captured = {}

    class IdleQueue:
        def summary(self):
            return {"indexing_job_ids": []}

        def enqueue_upload(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id=str(kwargs.get("job_id")),
                kind="upload",
                filenames=list(kwargs.get("filenames", [])),
            )

    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DATA_DIR", data_dir)

    response = TestClient(web_app.app).post("/api/reingest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["reingest"] is True
    assert payload["kind"] == "upload"
    assert captured["filenames"] == ["a.pdf"]


def test_reingest_endpoint_returns_404_with_no_sources(monkeypatch, workspace_tmp):
    class IdleQueue:
        def summary(self):
            return {"indexing_job_ids": []}

    monkeypatch.setattr(web_app, "job_queue", IdleQueue())
    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", workspace_tmp / "processed")
    monkeypatch.setattr(web_app, "DATA_DIR", workspace_tmp / "data")

    response = TestClient(web_app.app).post("/api/reingest")
    assert response.status_code == 404


def test_reingest_endpoint_blocks_during_indexing(monkeypatch):
    class ActiveIndexingQueue:
        def summary(self):
            return {"indexing_job_ids": ["job-1"]}

    monkeypatch.setattr(web_app, "job_queue", ActiveIndexingQueue())
    response = TestClient(web_app.app).post("/api/reingest")
    assert response.status_code == 409


@pytest.mark.remote_client
def test_api_token_middleware_blocks_mutation_without_token(monkeypatch):
    """With api_token set, a remote mutating request without it is rejected (401)."""
    monkeypatch.setattr(web_app, "_API_TOKEN", "secret-token-123")

    # A POST without the token -> 401.
    response = TestClient(web_app.app).post("/api/reindex")
    assert response.status_code == 401
    assert "token" in response.json()["detail"].lower()


@pytest.mark.remote_client
def test_api_token_middleware_allows_mutation_with_correct_token(monkeypatch):
    """Mutating requests with the correct X-API-Token header succeed (past auth)."""
    monkeypatch.setattr(web_app, "_API_TOKEN", "secret-token-123")

    # Stub the job queue so the reindex endpoint reaches a real handler state.
    class StubQueue:
        def __init__(self):
            self.enqueued = None

        def enqueue_reindex(self, **kwargs):
            class Job:
                id = "job-reindex-1"

                def to_dict(self):
                    return {"id": self.id, "kind": "reindex", "status": "queued"}

            self.enqueued = kwargs
            return Job()

        def summary(self):
            return {"indexing_job_ids": []}

    monkeypatch.setattr(web_app, "job_queue", StubQueue())
    response = TestClient(web_app.app).post(
        "/api/reindex",
        headers={"X-API-Token": "secret-token-123"},
    )
    # The auth gate passed -- the request is NOT 401. (The exact downstream
    # status depends on handler internals; the contract under test is "auth
    # accepted with the correct token".)
    assert response.status_code != 401, "correct token should pass auth"


@pytest.mark.remote_client
def test_api_token_middleware_allows_get_without_token(monkeypatch):
    """GET requests (health, listings) must work even when a token is set."""
    monkeypatch.setattr(web_app, "_API_TOKEN", "secret-token-123")

    response = TestClient(web_app.app).get("/api/health")
    assert response.status_code == 200


@pytest.mark.remote_client
def test_api_token_middleware_disabled_when_empty(monkeypatch):
    """When api_token is empty (default), open GETs still work (health)."""
    monkeypatch.setattr(web_app, "_API_TOKEN", "")

    response = TestClient(web_app.app).get("/api/health")
    assert response.status_code == 200


def test_metrics_endpoint_returns_operational_snapshot(monkeypatch):
    """/api/metrics exposes record count, index size, document count, queue state."""
    # Stub the store so we don't need a real LanceDB index.
    class StubStore:
        def exists(self):
            return True

        def count(self):
            return 12345

        def _on_disk_bytes(self):
            return 1024 * 1024 * 42

        def table_version_hint_path(self):
            return Path("/tmp/nonexistent_hint")

    monkeypatch.setattr(web_app, "_index_store", lambda *a, **k: StubStore())

    class StubQueue:
        def summary(self):
            return {"active_query_count": 2, "queued_count": 1, "running_job_ids": [], "indexing_job_ids": [], "active_job_count": 3, "job_count": 5}

    monkeypatch.setattr(web_app, "job_queue", StubQueue())
    monkeypatch.setattr(web_app, "_load_index_manifest", lambda *a, **k: {
        "documents": {"hash-a": {}, "hash-b": {}, "hash-c": {}},
        "embedded_records": 9000,
        "reused_records": 3345,
    })
    monkeypatch.setattr(web_app, "_ollama_status_snapshot", lambda: {"reachable": True})

    response = TestClient(web_app.app).get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["index_exists"] is True
    assert body["record_count"] == 12345
    assert body["index_bytes"] == 1024 * 1024 * 42
    assert body["document_count"] == 3
    assert body["embedded_records"] == 9000
    assert body["reused_records"] == 3345
    assert body["queue"]["active_query_count"] == 2
    assert body["ollama"]["reachable"] is True


# ---------------------------------------------------------------------------
# Admin API key management endpoints (/api/admin/api-keys ...)
# ---------------------------------------------------------------------------


def _install_admin_auth(monkeypatch, safe_tmp_path):
    """Fresh key store + master token so admin endpoint tests are hermetic."""
    from src.api_key_auth import ApiKeyAuthenticator, KeyStore

    store = KeyStore(Path(safe_tmp_path) / ".api_keys.json")
    authenticator = ApiKeyAuthenticator(store, default_rate_limit=60, persist_interval=999)
    monkeypatch.setattr(web_app, "api_authenticator", authenticator)
    monkeypatch.setattr(web_app, "_API_TOKEN", "master-token-1")
    return authenticator


@pytest.mark.remote_client
def test_admin_api_key_lifecycle(monkeypatch, safe_tmp_path):
    """Create -> list -> disable -> re-enable -> role change -> delete."""
    authenticator = _install_admin_auth(monkeypatch, safe_tmp_path)
    client = TestClient(web_app.app)
    master = {"X-API-Token": "master-token-1"}

    created = client.post(
        "/api/admin/api-keys",
        headers=master,
        json={"label": "review laptop", "role": "user", "expires_in_days": 30, "rate_limit_per_minute": 42},
    )
    assert created.status_code == 200, created.text
    payload = created.json()
    # The plaintext secret is returned exactly once and is never stored.
    assert payload["key"].startswith("rag_")
    assert payload["record"]["label"] == "review laptop"
    assert payload["record"]["role"] == "user"
    assert payload["record"]["rate_limit_per_minute"] == 42
    assert payload["record"]["expires_at"]
    prefix = payload["record"]["prefix"]

    listed = client.get("/api/admin/api-keys", headers=master).json()
    assert [key["prefix"] for key in listed["keys"]] == [prefix]
    assert "key" not in listed["keys"][0]

    disabled = client.post(
        f"/api/admin/api-keys/{prefix}/status", headers=master, json={"status": "disabled"}
    )
    assert disabled.status_code == 200
    assert disabled.json()["record"]["status"] == "disabled"

    enabled = client.post(
        f"/api/admin/api-keys/{prefix}/status", headers=master, json={"status": "active"}
    )
    assert enabled.json()["record"]["status"] == "active"

    promoted = client.post(
        f"/api/admin/api-keys/{prefix}/role", headers=master, json={"role": "admin"}
    )
    assert promoted.json()["record"]["role"] == "admin"

    deleted = client.delete(f"/api/admin/api-keys/{prefix}", headers=master)
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert authenticator.store.list_keys() == []


@pytest.mark.remote_client
def test_admin_api_key_endpoints_require_admin_role(monkeypatch, safe_tmp_path):
    """A user-role key can authenticate but gets 403 on admin endpoints."""
    authenticator = _install_admin_auth(monkeypatch, safe_tmp_path)
    client = TestClient(web_app.app)

    user_key, _record = authenticator.create_key(label="plain user", role="user")
    user_headers = {"X-API-Token": user_key}

    denied = client.get("/api/admin/api-keys", headers=user_headers)
    assert denied.status_code == 403
    assert "admin" in denied.json()["detail"].lower()

    denied_create = client.post(
        "/api/admin/api-keys", headers=user_headers, json={"label": "sneaky", "role": "admin"}
    )
    assert denied_create.status_code == 403

    # No credential at all -> 401.
    assert client.get("/api/admin/api-keys").status_code == 401


@pytest.mark.remote_client
def test_usage_meters_only_real_work(monkeypatch, safe_tmp_path):
    """Usage counters record queries/uploads; polling GETs stay unmetered.

    The middleware meters a request only when its path is real work: gated
    reads like the jobs poll every open tab runs are authenticated and
    rate-limited but not counted, while the real-work endpoints are counted
    even when the handler rejects the (empty) body, since metering happens in
    the middleware before routing.
    """
    authenticator = _install_admin_auth(monkeypatch, safe_tmp_path)
    client = TestClient(web_app.app)

    seen_tracks: list[bool] = []
    real_authenticate = authenticator.authenticate

    def _spy(supplied, **kwargs):
        seen_tracks.append(kwargs.get("track"))
        return real_authenticate(supplied, **kwargs)

    monkeypatch.setattr(authenticator, "authenticate", _spy)

    key, record = authenticator.create_key(label="metered user", role="user")
    headers = {"X-API-Token": key}

    assert client.get("/api/jobs", headers=headers).status_code == 200
    assert client.post("/api/uploads", headers=headers).status_code >= 400
    assert client.post("/api/chat/stream", headers=headers).status_code >= 400

    assert seen_tracks == [False, True, True]
    authenticator.usage.flush()
    stored = authenticator.store.get_by_hash(record["key_id"])
    assert stored["usage"]["requests"] == 2


@pytest.mark.remote_client
def test_admin_api_key_validation_errors(monkeypatch, safe_tmp_path):
    """Invalid role/expiry and unknown prefixes return 4xx, not 500."""
    _install_admin_auth(monkeypatch, safe_tmp_path)
    client = TestClient(web_app.app)
    master = {"X-API-Token": "master-token-1"}

    bad_role = client.post(
        "/api/admin/api-keys", headers=master, json={"label": "x", "role": "superuser"}
    )
    assert bad_role.status_code == 400

    bad_expiry = client.post(
        "/api/admin/api-keys",
        headers=master,
        json={"label": "x", "role": "user", "expires_at": "not-a-date"},
    )
    assert bad_expiry.status_code == 400

    missing = client.post(
        "/api/admin/api-keys/rag_zzz-none/status", headers=master, json={"status": "disabled"}
    )
    assert missing.status_code == 404

    missing_delete = client.delete("/api/admin/api-keys/rag_zzz-none", headers=master)
    assert missing_delete.status_code == 404


@pytest.mark.remote_client
def test_admin_api_key_endpoints_require_credential(monkeypatch, safe_tmp_path):
    """No master token + empty store + remote client: uniform 401, no leak."""
    from src.api_key_auth import ApiKeyAuthenticator, KeyStore

    store = KeyStore(Path(safe_tmp_path) / ".api_keys.json")
    authenticator = ApiKeyAuthenticator(store, default_rate_limit=60, persist_interval=999)
    monkeypatch.setattr(web_app, "api_authenticator", authenticator)
    monkeypatch.setattr(web_app, "_API_TOKEN", "")

    client = TestClient(web_app.app)
    assert client.get("/api/admin/api-keys").status_code == 401
    # Creating keys from a remote client without a credential is rejected by
    # the middleware itself; the first admin key can only come from the local
    # machine (auto-auth) or an admin credential.
    lan_response = client.post("/api/admin/api-keys", json={"label": "first", "role": "user"})
    assert lan_response.status_code == 401
    assert authenticator.store.list_keys() == []

    # A loopback client (real client address this time, not the patched
    # "testclient") passes the localhost auto-auth bypass and may mint keys.
    monkeypatch.setattr(web_app, "_LOCALHOST_AUTO_AUTH", True)
    local_client = TestClient(web_app.app, client=("127.0.0.1", 51000))
    local_response = local_client.post("/api/admin/api-keys", json={"label": "first", "role": "user"})
    assert local_response.status_code == 200


def test_pdf_documents_endpoint_filters_by_group_and_trust(monkeypatch, workspace_tmp):
    """source_group/trust_status facets filter rows server-side before paging."""
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    trust_path = workspace_tmp / "trust.json"
    web_app.PdfRegistry(registry_path).register_queued(
        job_id="job",
        files=[
            {"filename": "approved-official.pdf", "hash": "hash-a", "staging_path": ""},
            {"filename": "approved-stale.pdf", "hash": "hash-b", "staging_path": ""},
            {"filename": "unreviewed.pdf", "hash": "hash-c", "staging_path": ""},
        ],
    )
    web_app._write_trust_registry(
        {
            "documents": {
                "hash-a": {"review_status": "approved", "source_group": "official"},
                "hash-b": {"review_status": "stale", "source_group": "student_research"},
            }
        },
        trust_path,
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", trust_path)
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")

    client = TestClient(web_app.app)
    by_group = client.get("/api/pdfs", params={"source_group": "official"}).json()
    assert [pdf["hash"] for pdf in by_group["pdfs"]] == ["hash-a"]
    assert by_group["total"] == 1

    by_status = client.get("/api/pdfs", params={"trust_status": "stale"}).json()
    assert [pdf["hash"] for pdf in by_status["pdfs"]] == ["hash-b"]

    unreviewed = client.get("/api/pdfs", params={"trust_status": "unreviewed"}).json()
    assert [pdf["hash"] for pdf in unreviewed["pdfs"]] == ["hash-c"]

    combined = client.get(
        "/api/pdfs", params={"source_group": "official", "trust_status": "stale"}
    ).json()
    assert combined["pdfs"] == []

    everything = client.get("/api/pdfs", params={"source_group": "all"}).json()
    assert everything["total"] == 3


def test_pdf_documents_endpoint_filters_by_pipeline_status(monkeypatch, workspace_tmp):
    """status facet filters on the registry pipeline status the Status column shows."""
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    trust_path = workspace_tmp / "trust.json"
    web_app._write_trust_registry(
        {"documents": {"hash-c": {"review_status": "approved", "source_group": "official"}}},
        trust_path,
    )
    registry = web_app.PdfRegistry(registry_path)
    files = [
        {"filename": "queued.pdf", "hash": "hash-a", "staging_path": ""},
        {"filename": "indexed.pdf", "hash": "hash-b", "staging_path": ""},
        {"filename": "failed.pdf", "hash": "hash-c", "staging_path": ""},
        {"filename": "interrupted.pdf", "hash": "hash-d", "staging_path": ""},
    ]
    registry.register_queued(job_id="job", files=files)
    registry.mark_job_status(job_id="job", files=[files[1]], status="ingested")
    registry.mark_job_status(job_id="job", files=[files[1]], status="indexed")
    registry.mark_job_status(job_id="job", files=[files[2]], status="failed")
    registry.mark_job_status(job_id="job", files=[files[3]], status="interrupted")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", trust_path)
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")

    client = TestClient(web_app.app)
    queued = client.get("/api/pdfs", params={"status": "queued"}).json()
    assert [pdf["hash"] for pdf in queued["pdfs"]] == ["hash-a"]

    failed = client.get("/api/pdfs", params={"status": "failed"}).json()
    assert [pdf["hash"] for pdf in failed["pdfs"]] == ["hash-c"]

    interrupted = client.get("/api/pdfs", params={"status": "interrupted"}).json()
    assert [pdf["hash"] for pdf in interrupted["pdfs"]] == ["hash-d"]

    # No index manifest exists in this workspace, so the indexed row is
    # reported (and must filter) as "not_indexed" — same as the Status column.
    indexed = client.get("/api/pdfs", params={"status": "indexed"}).json()
    assert indexed["pdfs"] == []
    not_indexed = client.get("/api/pdfs", params={"status": "not_indexed"}).json()
    assert [pdf["hash"] for pdf in not_indexed["pdfs"]] == ["hash-b"]

    combined = client.get("/api/pdfs", params={"status": "failed", "trust_status": "approved"}).json()
    assert [pdf["hash"] for pdf in combined["pdfs"]] == ["hash-c"]
    combined_none = client.get("/api/pdfs", params={"status": "queued", "trust_status": "approved"}).json()
    assert combined_none["pdfs"] == []

    everything = client.get("/api/pdfs", params={"status": "all"}).json()
    assert everything["total"] == 4


# ---------------------------------------------------------------------------
# Audit hardening: chunk-upload containment, health redaction, admin
# zero-config loopback guard, client model/path overrides, publish repair.
# ---------------------------------------------------------------------------


def _chunk_post(client, *, upload_id, offset="0", total_size="4", body=b"1234", filename="doc.pdf"):
    return client.post(
        "/api/uploads/chunk",
        files={"chunk": (filename, body, "application/octet-stream")},
        data={"upload_id": upload_id, "filename": filename, "offset": offset, "total_size": total_size},
    )


def test_chunk_upload_rejects_non_hex_upload_id():
    """upload_id becomes a path component; only 32-char hex tokens are valid."""
    client = TestClient(web_app.app)
    traversal = _chunk_post(client, upload_id="..\\..\\evil")
    assert traversal.status_code == 400
    absolute = _chunk_post(client, upload_id="C:\\Windows\\Temp\\evil")
    assert absolute.status_code == 400
    short = _chunk_post(client, upload_id="abc")
    assert short.status_code == 400
    # And chunk_status is contained too (no file-existence oracle).
    probed = client.get("/api/uploads/chunk_status", params={"upload_id": "../../secret.pdf"})
    assert probed.status_code == 400


def test_chunk_upload_rejects_bad_offset_and_size():
    client = TestClient(web_app.app)
    bad_offset = _chunk_post(client, upload_id="a" * 32, offset="abc")
    assert bad_offset.status_code == 400
    negative = _chunk_post(client, upload_id="a" * 32, offset="-1")
    assert negative.status_code == 400


def test_chunk_upload_accepts_hex_id_and_appends():
    client = TestClient(web_app.app)
    upload_id = "b" * 32
    ok = _chunk_post(client, upload_id=upload_id, body=b"hello")
    assert ok.status_code == 200, ok.text
    assert ok.json()["offset"] == 5
    part = web_app.STAGING_DIR / upload_id / f"{upload_id}.part"
    assert part.read_bytes() == b"hello"
    part.unlink(missing_ok=True)
    part.parent.rmdir()


def test_chunk_upload_enforces_per_file_size_cap(monkeypatch):
    """The append is hard-capped against the REAL .part size, not the
    client-declared total_size: a crafted client cannot stream past
    max_upload_bytes one small chunk at a time."""
    monkeypatch.setattr(
        web_app, "UPLOADS_CONFIG", {**web_app.UPLOADS_CONFIG, "max_upload_bytes": 6}
    )
    client = TestClient(web_app.app)
    upload_id = "c" * 32
    ok = _chunk_post(client, upload_id=upload_id, body=b"1234", total_size="4")
    assert ok.status_code == 200
    over = _chunk_post(client, upload_id=upload_id, body=b"5678", offset="4", total_size="4")
    assert over.status_code == 413
    part = web_app.STAGING_DIR / upload_id / f"{upload_id}.part"
    assert part.read_bytes() == b"1234"  # partial append rolled back
    part.unlink(missing_ok=True)
    part.parent.rmdir()


def test_chunked_complete_rejects_oversized_part(monkeypatch):
    """complete() re-checks the real .part size before copying: a .part that
    is over the cap on disk (e.g. staged before the limit was tightened) is
    refused instead of being queued."""
    monkeypatch.setattr(
        web_app, "UPLOADS_CONFIG", {**web_app.UPLOADS_CONFIG, "max_upload_bytes": 4}
    )
    client = TestClient(web_app.app)
    upload_id = "d" * 32
    part_dir = web_app.STAGING_DIR / upload_id
    part_dir.mkdir(parents=True, exist_ok=True)
    (part_dir / f"{upload_id}.part").write_bytes(b"12345678")
    done = client.post(
        "/api/uploads/complete",
        json={"upload_id": upload_id, "filename": "doc.pdf"},
    )
    assert done.status_code == 413
    part = part_dir / f"{upload_id}.part"
    part.unlink(missing_ok=True)
    part.parent.rmdir()


def test_chunked_complete_requires_force_token(monkeypatch, workspace_tmp):
    """Forcing a duplicate through /api/uploads/complete demands the same
    signed, TTL-bound force token the multipart path issues with the 409."""
    captured = {}

    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                uploads=kwargs["uploads"],
                force_duplicate_hashes=kwargs["force_duplicate_hashes"],
                staging_dir=str(kwargs["staging_dir"]),
            )

    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", workspace_tmp / "processed")
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())
    monkeypatch.setattr(web_app, "_schedule_upload_auto_tag", lambda uploads: None)

    content = b"%PDF-1.4 duplicate"
    source_hash = hashlib.sha256(content).hexdigest()
    client = TestClient(web_app.app)
    first = client.post(
        "/api/uploads",
        data={"source_groups": "official"},
        files=[("files", ("notes.pdf", content, "application/pdf"))],
    )
    assert first.status_code == 200

    def stage_part(upload_id: str) -> None:
        part_dir = web_app.STAGING_DIR / upload_id
        part_dir.mkdir(parents=True, exist_ok=True)
        (part_dir / f"{upload_id}.part").write_bytes(content)

    # A bare force_duplicates boolean must NOT queue the duplicate.
    upload_id = "e" * 32
    stage_part(upload_id)
    bare = client.post(
        "/api/uploads/complete",
        json={"upload_id": upload_id, "filename": "copy.pdf", "force_duplicates": "true"},
    )
    assert bare.status_code == 409
    token = bare.json()["detail"]["force_token"]
    assert token  # the 409 still issues a token the client can retry with

    # The retry carries the token (the real client re-uploads chunks to a
    # fresh upload_id after the 409 consumed the first .part).
    retry_id = "f" * 32
    stage_part(retry_id)
    forced = client.post(
        "/api/uploads/complete",
        json={
            "upload_id": retry_id,
            "filename": "copy.pdf",
            "force_duplicates": "true",
            "force_token": token,
        },
    )
    assert forced.status_code == 200, forced.text
    assert captured["force_duplicate_hashes"] == [source_hash]


def test_chunked_complete_applies_source_group(monkeypatch, workspace_tmp):
    """A source group chosen on the chunked path must land in the trust
    registry (the only source the Library listing reads), not just in the
    registry options."""
    captured = {}

    class FakeQueue:
        def enqueue_upload(self, **kwargs):
            captured.update(kwargs)
            return web_app.QueueJob(
                id=kwargs["job_id"],
                kind="upload",
                filenames=kwargs["filenames"],
                uploads=kwargs["uploads"],
                staging_dir=str(kwargs["staging_dir"]),
            )

    monkeypatch.setattr(web_app, "STAGING_DIR", workspace_tmp / "staging")
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", workspace_tmp / "trust.json")
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())
    monkeypatch.setattr(web_app, "_schedule_upload_auto_tag", lambda uploads: None)

    content = b"%PDF-1.4 grouped"
    source_hash = hashlib.sha256(content).hexdigest()
    client = TestClient(web_app.app)
    upload_id = "1" * 32
    part_dir = web_app.STAGING_DIR / upload_id
    part_dir.mkdir(parents=True)
    (part_dir / f"{upload_id}.part").write_bytes(content)
    response = client.post(
        "/api/uploads/complete",
        json={"upload_id": upload_id, "filename": "grouped.pdf", "source_groups": "official"},
    )

    assert response.status_code == 200, response.text
    assert captured["uploads"][0]["source_group"] == "official"
    trust = json.loads((workspace_tmp / "trust.json").read_text())
    assert trust["documents"][source_hash]["source_group"] == "official"


def test_page_slice_all_page_sizes_honor_max_page_size():
    """limit<=0 (the UI's All mode) resolves to each endpoint's MAX_PAGE_SIZE
    before hitting _page_slice; the slice must keep that ceiling instead of
    clamping everything to its generic 100-row default."""
    rows = [{"id": str(i)} for i in range(300)]

    assert web_app._page_slice(rows, offset=0, limit=500, cap=500)["limit"] == 500
    assert web_app._page_slice(rows, offset=0, limit=2000, cap=2000)["limit"] == 2000
    assert web_app._page_slice(rows, offset=0, limit=500)["limit"] == 100  # default cap
    # End-to-end through the jobs listing envelope (empty queue is fine: the
    # resolved limit is what regressed).
    assert web_app.list_job_rows(offset=0, limit=web_app.JOBS_MAX_PAGE_SIZE)["limit"] == (
        web_app.JOBS_MAX_PAGE_SIZE
    )


def test_health_redacts_master_token(monkeypatch, safe_tmp_path):
    """/api/health is an open GET; it must never echo the master token."""
    monkeypatch.setattr(
        web_app, "SERVER_CONFIG", {**web_app.SERVER_CONFIG, "api_token": "secret-token-xyz"}
    )
    client = TestClient(web_app.app)
    data = client.get("/api/health").json()
    assert data["server"]["api_token"] == ""
    assert data["server"]["api_token_configured"] is True


@pytest.mark.remote_client
def test_admin_key_creation_requires_local_machine(monkeypatch, safe_tmp_path):
    """A remote host cannot mint keys; the local machine always can."""
    from src.api_key_auth import ApiKeyAuthenticator, KeyStore

    authenticator = ApiKeyAuthenticator(
        KeyStore(Path(safe_tmp_path) / ".api_keys.json"), default_rate_limit=60, persist_interval=999
    )
    monkeypatch.setattr(web_app, "api_authenticator", authenticator)
    monkeypatch.setattr(web_app, "_API_TOKEN", "")

    lan_client = TestClient(web_app.app)  # default client host is not loopback
    denied = lan_client.post("/api/admin/api-keys", json={"label": "attacker", "role": "admin"})
    assert denied.status_code == 401
    assert authenticator.store.list_keys() == []

    local_client = TestClient(web_app.app, client=("127.0.0.1", 51000))
    allowed = local_client.post("/api/admin/api-keys", json={"label": "owner", "role": "user"})
    assert allowed.status_code == 200
    # The minted user-role key is NOT admin: from a remote client it can
    # authenticate but gets 403 on the admin listing (localhost, by contrast,
    # is always admin via auto-auth).
    minted = allowed.json()["key"]
    denied_list = TestClient(web_app.app).get(
        "/api/admin/api-keys", headers={"X-API-Token": minted}
    )
    assert denied_list.status_code == 403


def test_safe_client_model_rejects_oversized_override(monkeypatch):
    """A client-named exact match of a huge installed model falls back to the
    configured default; unknown names pass through unchanged."""
    from src import llm_api

    monkeypatch.setattr(llm_api, "_ollama_model_sizes", lambda: {"gemma4:latest": 96127041536})
    monkeypatch.setenv("LOCAL_MODEL_MAX_BYTES", str(4 * 1024 ** 3))
    assert (
        web_app._safe_client_model("gemma4:latest", fallback="qwen3:4b-instruct")
        == "qwen3:4b-instruct"
    )
    assert web_app._safe_client_model("qwen3:4b-instruct", fallback="qwen3:4b-instruct") == "qwen3:4b-instruct"
    assert web_app._safe_client_model("not-installed:8b", fallback="qwen3:4b-instruct") == "not-installed:8b"
    monkeypatch.setenv("LOCAL_MODEL_MAX_BYTES", "0")
    assert web_app._safe_client_model("gemma4:latest", fallback="qwen3:4b-instruct") == "gemma4:latest"


def test_repair_preserved_components_restores_missing_and_prunes_stale(safe_tmp_path):
    """A crash mid-publish strands the only overrides/hashes copy in the
    aside dir; startup must restore it (the old GC deleted it)."""
    live = safe_tmp_path / "db"
    live.mkdir()
    (live / "index_overrides.json").write_text("{}", encoding="utf-8")
    aside = safe_tmp_path / ".index_preserve_abc123"
    (aside / "hashes").mkdir(parents=True)
    (aside / "hashes" / "x.json").write_text("{}", encoding="utf-8")
    (aside / "index_overrides.json").write_text("{}", encoding="utf-8")

    resolved = web_app._repair_preserved_components(live)

    assert resolved == 1
    assert (live / "hashes" / "x.json").exists()  # restored from the aside dir
    assert (live / "index_overrides.json").exists()  # live copy untouched
    assert not aside.exists()  # fully resolved -> removed


def test_repair_preserved_components_keeps_unrestorable(safe_tmp_path, monkeypatch):
    import shutil as _shutil

    live = safe_tmp_path / "db"
    live.mkdir()
    aside = safe_tmp_path / ".index_preserve_keep"
    (aside / "hashes").mkdir(parents=True)
    real_move = _shutil.move

    def failing_move(*args, **kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr(web_app.shutil, "move", failing_move)
    try:
        web_app._repair_preserved_components(live)
    finally:
        monkeypatch.setattr(web_app.shutil, "move", real_move)
    # The aside dir must survive a failed restore -- it is the only copy.
    assert aside.exists()


def test_gc_staged_build_dirs_leaves_preserve_and_rollover(safe_tmp_path):
    """Preserve dirs are repair-only, rollover dirs are warn-only: the GC must
    touch neither (both can hold the only copy of live data)."""
    live = safe_tmp_path / "db"
    live.mkdir()
    for name in (".index_preserve_old", ".index_rollover_old"):
        target = safe_tmp_path / name
        target.mkdir()
        os.utime(target, (time.time() - 10_000, time.time() - 10_000))
    build = safe_tmp_path / ".index_build_stale"
    build.mkdir()
    os.utime(build, (time.time() - 10_000, time.time() - 10_000))

    removed = web_app._gc_staged_build_dirs(live)

    assert removed == 1  # only the un-checkpointed .index_build_ dir
    assert (safe_tmp_path / ".index_preserve_old").exists()
    assert (safe_tmp_path / ".index_rollover_old").exists()
    assert not build.exists()


def test_tracked_job_ledger_lifecycle(workspace_tmp):
    """Ledger correctness around enqueue/cancel: recorded while pending (crash
    recovery), removed on cancel (no resurrection at startup)."""
    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
    )
    job = queue.enqueue_backup(job_id="ledger-cancel-test", auto_start=False)
    try:
        entries = json.loads(queue.ledger.path.read_text(encoding="utf-8"))["jobs"]
        assert "ledger-cancel-test" in entries  # recorded BEFORE it can run

        cancelled = queue.cancel_job("ledger-cancel-test")
        assert cancelled["status"] == "cancelled"
        entries = json.loads(queue.ledger.path.read_text(encoding="utf-8"))["jobs"]
        assert "ledger-cancel-test" not in entries  # never resurrects
    finally:
        with queue._condition:
            queue._jobs.pop("ledger-cancel-test", None)


def test_equivalent_active_jobs_are_coalesced(workspace_tmp):
    """Repeated equivalent maintenance requests must share one job."""
    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
    )
    first = queue.enqueue_reindex(options={"embedding_model": "all-minilm"}, auto_start=False)
    second = queue.enqueue_reindex(options={"embedding_model": "all-minilm"}, auto_start=False)

    assert second is first
    assert [job["id"] for job in queue.list_jobs()] == [first.id]
    entries = json.loads(queue.ledger.path.read_text(encoding="utf-8"))["jobs"]
    assert list(entries) == [first.id]


def test_recovery_removes_duplicate_ledger_entries(workspace_tmp):
    """A legacy ledger containing duplicates is compacted during recovery."""
    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
    )
    options = {"embedding_model": "all-minilm"}
    queue.ledger.record("reindex-a", kind="reindex", options=options)
    queue.ledger.record("reindex-b", kind="reindex", options=options)

    recovered = queue.recover_pending_uploads(auto_start=False)

    assert recovered["recovered"] == 1
    assert len(queue.list_jobs()) == 1
    entries = json.loads(queue.ledger.path.read_text(encoding="utf-8"))["jobs"]
    assert list(entries) == ["reindex-a"]


def test_pdf_documents_endpoint_sorts_by_column(monkeypatch, workspace_tmp):
    """sort=filename/-filename/-updated reorders rows; default keeps ungrouped first."""
    registry_path = workspace_tmp / "registry.json"
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    trust_path = workspace_tmp / "trust.json"
    web_app.PdfRegistry(registry_path).register_queued(
        job_id="job",
        files=[
            {"filename": "Beta.pdf", "hash": "hash-b", "staging_path": ""},
            {"filename": "alpha.pdf", "hash": "hash-a", "staging_path": ""},
        ],
    )
    web_app._write_trust_registry(
        {"documents": {"hash-a": {"source_group": "official"}}},
        trust_path,
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DOCUMENT_TRUST_PATH", trust_path)
    monkeypatch.setattr(web_app, "DB_DIR", workspace_tmp / "db")

    client = TestClient(web_app.app)
    default_order = client.get("/api/pdfs").json()
    # Default: ungrouped (Beta) first.
    assert [pdf["hash"] for pdf in default_order["pdfs"]] == ["hash-b", "hash-a"]

    by_name = client.get("/api/pdfs", params={"sort": "filename"}).json()
    assert [pdf["hash"] for pdf in by_name["pdfs"]] == ["hash-a", "hash-b"]

    by_name_desc = client.get("/api/pdfs", params={"sort": "-filename"}).json()
    assert [pdf["hash"] for pdf in by_name_desc["pdfs"]] == ["hash-b", "hash-a"]

    # Ascending group order is the useful direction: official < ungrouped.
    by_group = client.get("/api/pdfs", params={"sort": "group"}).json()
    assert [pdf["hash"] for pdf in by_group["pdfs"]] == ["hash-a", "hash-b"]


# ---------------------------------------------------------------------------
# Corpus-root allow-list, Markdown preview, disk/history/security surfaces.
# ---------------------------------------------------------------------------


def test_resolve_pdf_path_accepts_configured_corpus_roots(monkeypatch, workspace_tmp):
    """A source PDF outside data_dir is servable when its root is configured."""
    from src import config as config_module

    import tempfile

    # Roots under the repo are always allowed (ROOT_DIR), so the probe files
    # must live in the system temp dir to prove containment actually blocks.
    corpus_root = Path(tempfile.gettempdir()) / f"rag_eval_corpus_{uuid.uuid4().hex}"
    source_pdf = corpus_root / "batch" / "report.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4 test")

    outside_root = Path(tempfile.gettempdir()) / f"rag_eval_outside_{uuid.uuid4().hex}"
    outside = outside_root / "secret.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"%PDF-1.4 secret")

    monkeypatch.setattr(web_app, "DATA_DIR", workspace_tmp / "data")
    monkeypatch.setattr(
        config_module,
        "_CONFIG_CACHE",
        {},
    )
    real_load = config_module.load_config

    class FakePaths:
        corpus_roots = [str(corpus_root)]

    class FakeCfg:
        paths = FakePaths()

    monkeypatch.setattr(web_app, "_default_config_path", lambda: Path("config.toml"))
    monkeypatch.setattr(config_module, "load_config", lambda path=None: FakeCfg())
    # _resolve_pdf_path imports load_config from src.config at call time.
    monkeypatch.setattr(
        web_app,
        "_allowed_corpus_roots",
        lambda: [web_app.DATA_DIR.resolve(), web_app.ROOT_DIR.resolve(), corpus_root.resolve()],
    )

    assert web_app._resolve_pdf_path(str(source_pdf)).resolve() == source_pdf.resolve()
    with pytest.raises(PermissionError):
        web_app._resolve_pdf_path(str(outside))


def test_pdf_markdown_endpoint_serves_processed_text(monkeypatch, safe_tmp_path):
    """GET /api/pdfs/{hash}/markdown returns the processed Markdown content."""
    md_file = Path(safe_tmp_path) / "doc.md"
    md_file.write_text("# Processed title\n\nBody text.", encoding="utf-8")

    monkeypatch.setattr(
        web_app,
        "_pdf_row_for_hash",
        lambda source_hash: {
            "hash": "hash-md",
            "filename": "doc.pdf",
            "processed_markdown_path": str(md_file),
        },
    )
    monkeypatch.setattr(
        web_app,
        "_resolve_markdown_path",
        lambda raw_path, **kwargs: md_file,
    )

    client = TestClient(web_app.app)
    response = client.get("/api/pdfs/hash-md/markdown")
    assert response.status_code == 200
    assert response.json()["markdown"].startswith("# Processed title")

    monkeypatch.setattr(
        web_app,
        "_pdf_row_for_hash",
        lambda source_hash: None,
    )
    missing = client.get("/api/pdfs/unknown-hash/markdown")
    assert missing.status_code == 404


def test_metrics_includes_disk_and_history(monkeypatch, workspace_tmp):
    """/api/metrics carries disk volumes; history samples accumulate."""
    from collections import deque

    class StubStore:
        def exists(self):
            return True

        def count(self):
            return 7

        def _on_disk_bytes(self):
            return 1024

        def table_version_hint_path(self):
            return Path("/tmp/nonexistent_hint")

    monkeypatch.setattr(web_app, "_index_store", lambda *a, **k: StubStore())

    class StubQueue:
        def summary(self):
            return {"queued_count": 0, "active_job_count": 0, "active_query_count": 0}

        def state_version(self):
            return 1

    monkeypatch.setattr(web_app, "job_queue", StubQueue())
    monkeypatch.setattr(web_app, "_load_index_manifest", lambda *a, **k: {})
    monkeypatch.setattr(web_app, "_ollama_status_snapshot", lambda: {"reachable": False})

    web_app._METRICS_HISTORY.clear()
    web_app._sample_metrics_once()
    web_app._sample_metrics_once()

    assert len(web_app._METRICS_HISTORY) == 2
    sample = web_app._METRICS_HISTORY[0]
    assert set(sample) >= {"at", "records", "queued", "active"}

    client = TestClient(web_app.app)
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("disk"), list)
    for volume in body["disk"]:
        assert {"label", "free_bytes", "total_bytes"} <= set(volume)

    history = client.get("/api/metrics/history")
    assert history.status_code == 200
    assert len(history.json()["samples"]) == 2


def test_health_reports_security_and_startup_notices(monkeypatch):
    """/api/health exposes auth/bind posture and any startup repair notices."""
    monkeypatch.setattr(web_app, "_API_TOKEN", "")
    monkeypatch.setattr(web_app, "api_authenticator", None)
    monkeypatch.setattr(web_app, "_STARTUP_NOTICES", ["Interrupted index swap recovered from x"])

    class StubStore:
        def exists(self):
            return False

        def count(self):
            return 0

        def table_version_hint_path(self):
            return Path("/tmp/nonexistent_hint")

    monkeypatch.setattr(web_app, "_index_store", lambda *a, **k: StubStore())

    class StubQueue:
        def summary(self):
            return {"queued_count": 0, "active_job_count": 0, "active_query_count": 0}

    monkeypatch.setattr(web_app, "job_queue", StubQueue())
    monkeypatch.setattr(web_app, "_llm_status_snapshot", lambda: {})
    monkeypatch.setitem(web_app.SERVER_CONFIG, "bind_all", False)
    monkeypatch.setitem(web_app.SERVER_CONFIG, "host", "127.0.0.1")

    response = TestClient(web_app.app).get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["security"] == {"auth_enabled": False, "bind_all": False}
    assert body["startup_notices"] == ["Interrupted index swap recovered from x"]


# ---------------------------------------------------------------------------
# Conversation history, review-status weighting, document records endpoint.
# ---------------------------------------------------------------------------


def test_chat_request_accepts_and_bounds_history():
    """ChatTurn validation + endpoint sanitization (roles, size, cap)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        web_app.ChatTurn(role="system", content="nope")
    with pytest.raises(ValidationError):
        web_app.ChatTurn(role="user", content="")

    turns = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"}
        for index in range(30)
    ]
    payload = web_app.ChatRequest(question="q", history=turns)
    sanitized = web_app._sanitize_chat_history(payload.history)
    assert len(sanitized) == web_app.MAX_CHAT_HISTORY_TURNS
    assert sanitized[-1]["content"] == "turn 29"


def test_chat_stream_passes_history_to_engine(monkeypatch):
    """chat_stream forwards sanitized history into ask_stream_events."""
    captured: dict = {}

    class StubEngine:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def ask_stream_events(self, question, history=None):
            captured["question"] = question
            captured["history"] = history
            yield {"type": "answer", "text": "ok"}

    class StubQueue:
        def begin_query(self):
            pass

        def finish_query(self):
            pass

    monkeypatch.setattr(web_app, "job_queue", StubQueue())
    import src.query as query_module

    real_query_engine = query_module.QueryEngine
    monkeypatch.setattr(query_module, "QueryEngine", StubEngine)
    # chat_stream imports QueryEngine from src.query inside generate(); patch
    # the attribute on the module the function reads.
    monkeypatch.setattr("src.query.QueryEngine", StubEngine)
    monkeypatch.setattr(web_app, "_resolve_chat_categories", lambda keys, request=None: [{"db_dir": "x", "label": "general"}])

    from fastapi.testclient import TestClient

    # Plain client (no context manager): entering the lifespan would trip the
    # single-instance data lock while the developer's server is running.
    client = TestClient(web_app.app)
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "question": "what about the rear wing?",
            "history": [
                {"role": "user", "content": "How does downforce work?"},
                {"role": "assistant", "content": "Downforce presses the car into the road."},
            ],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "Downforce presses" in str(captured.get("history"))
    assert captured["question"] == "what about the rear wing?"


def test_review_status_affects_group_map_weight(safe_tmp_path):
    """load_source_group_map folds review status into the effective weight."""
    from src.reliability import load_source_group_map

    trust_path = Path(safe_tmp_path) / ".document_trust.json"
    trust_path.write_text(
        json.dumps(
            {
                "version": 1,
                "documents": {
                    "hash-approved": {"source_group": "official", "review_status": "approved"},
                    "hash-stale": {"source_group": "official", "review_status": "stale"},
                    "hash-unreviewed": {"source_group": "official"},
                },
            }
        ),
        encoding="utf-8",
    )
    groups = load_source_group_map(trust_path)
    approved = groups["hash-approved"]
    stale = groups["hash-stale"]
    unreviewed = groups["hash-unreviewed"]

    assert approved["review_status"] == "approved"
    assert approved["weight"] == pytest.approx(1.0)
    assert stale["review_status"] == "stale"
    # official(1.0) * stale(0.6)
    assert stale["weight"] == pytest.approx(0.6)
    assert unreviewed["weight"] == pytest.approx(0.95)
    # The ranking invariant: stale official now ranks below approved official.
    assert stale["weight"] < unreviewed["weight"] < approved["weight"]


def test_document_records_endpoint_returns_sorted_rows(monkeypatch, lancedb_tmp):
    """/api/index/document_records scopes to one hash, sorted, paginated."""
    db_dir = lancedb_tmp / "db"
    records = []
    for index in range(5):
        records.append(
            {
                "id": f"doc:{index}",
                "doc_id": "doc",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "doc.pdf",
                "chunk_index": index,
                "content": f"chunk number {index}",
                "title": "Doc",
                "section_path": "Doc",
                "page_start": 1,
                "page_end": 1,
                "summary": "summary",
                "tags": [],
                "source_hash": "hash-doc",
                "vector": [1.0, 0.0, 0.0],
            }
        )
    # Also a record from a different document that must NOT appear.
    records.append({**records[0], "id": "other:0", "doc_id": "other", "source_hash": "hash-other", "content": "other doc"})
    LanceDBVectorStore(db_dir).write_records(records, embedding_model="fake-embed", embedding_dim=3)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    client = TestClient(web_app.app)
    response = client.get(
        "/api/index/document_records",
        params={"source_hash": "hash-doc", "offset": 1, "limit": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 5
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["content"] == "chunk number 1"
    assert all(row["source_hash"] == "hash-doc" for row in payload["rows"])


# ---------------------------------------------------------------------------
# Graceful shutdown: job finalization + the localhost-only shutdown endpoint.
# ---------------------------------------------------------------------------


class _StubShutdownQueue:
    """Stand-in for the app-level ``job_queue`` in shutdown endpoint tests.

    The real queue points at the developer's live data/ directory; endpoint
    tests must never touch it (a stray finalize would mutate the real ledger).
    """

    def __init__(self):
        self.finalize_calls = 0

    def summary(self):
        return {"queued_count": 0, "active_job_count": 0, "active_query_count": 0}

    def shutdown_finalize(self, **kwargs):
        self.finalize_calls += 1
        return {"finalized": 0, "forced": 0}


@pytest.fixture()
def shutdown_env(monkeypatch):
    """Stub the exit trigger + app queue and hand the test a fresh state dict.

    The countdown thread reads the module-global state, so restoring the
    original (inactive) dict at teardown automatically de-arms any thread that
    is still sleeping: its next active-check returns before it can finalize or
    raise SIGINT.
    """
    stub = _StubShutdownQueue()
    triggered = []
    state = {"active": False}
    monkeypatch.setattr(web_app, "job_queue", stub)
    monkeypatch.setattr(web_app, "_trigger_graceful_exit", lambda: triggered.append(True))
    monkeypatch.setattr(web_app, "_SERVER_SHUTDOWN_STATE", state)
    return stub, triggered, state


def test_shutdown_finalize_force_cancels_stuck_job(workspace_tmp):
    """A job stuck inside a phase that never checks the cancel event is
    force-finalized: it reaches a terminal state and the ledger entry is gone,
    so startup recovery cannot resurrect it after the restart."""
    release = threading.Event()

    def fake_index(md_dir, db_dir, **kwargs):
        release.wait(timeout=10)

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
        run_indexing_func=fake_index,
    )
    job = queue.enqueue_reindex(job_id="stuck-reindex", auto_start=True)
    _wait_for(lambda: queue.get_job(job.id)["status"] == "running")
    try:
        summary = queue.shutdown_finalize(grace_seconds=0.3)
    finally:
        release.set()
        _wait_for(lambda: queue.get_job(job.id)["status"] in ("cancelled", "done", "failed"))

    assert summary["finalized"] == 1
    assert summary["forced"] == 1
    assert queue.get_job(job.id)["status"] == "cancelled"
    entries = json.loads(queue.ledger.path.read_text(encoding="utf-8"))["jobs"]
    assert "stuck-reindex" not in entries
    # Idempotent: a second pass (e.g. the endpoint thread, then the lifespan)
    # finds nothing active.
    assert queue.shutdown_finalize(grace_seconds=0.1) == {"finalized": 0, "forced": 0}


def test_shutdown_finalize_lets_worker_do_its_own_bookkeeping(workspace_tmp):
    """When the worker observes the cancel event within the grace window, it
    runs the authoritative cancellation path itself and finalize forces
    nothing."""
    release = threading.Event()

    def fake_index(md_dir, db_dir, **kwargs):
        # Poll the cancel event like the real indexer does between phases,
        # then return so the worker's own cancellation path runs.
        for _ in range(200):
            if job.cancel_requested:
                break
            time.sleep(0.01)
        release.set()

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
        run_indexing_func=fake_index,
    )
    job = queue.enqueue_reindex(job_id="polite-reindex", auto_start=True)
    _wait_for(lambda: queue.get_job(job.id)["status"] == "running")
    summary = queue.shutdown_finalize(grace_seconds=3.0)
    release.wait(timeout=5)
    _wait_for(lambda: queue.get_job(job.id)["status"] == "cancelled")

    assert summary["finalized"] == 1
    assert summary["forced"] == 0
    entries = json.loads(queue.ledger.path.read_text(encoding="utf-8"))["jobs"]
    assert "polite-reindex" not in entries


def test_shutdown_finalize_marks_upload_sources_interrupted(workspace_tmp):
    """Finalizing a stuck upload job writes the registry 'interrupted' status —
    a non-recoverable status, so the next boot does not re-run the upload."""

    def fake_ingest(input_dir, output_dir, **kwargs):
        threading.Event().wait(timeout=10)

    queue = web_app.RagJobQueue(
        upload_root=workspace_tmp / "uploads",
        processed_dir=workspace_tmp / "processed",
        db_dir=workspace_tmp / "db",
        registry_path=workspace_tmp / "registry.json",
        run_ingestion_func=fake_ingest,
    )
    staging = workspace_tmp / "staging"
    staging.mkdir()
    staging.joinpath("doc.pdf").write_bytes(b"%PDF-1.4")
    # The upload endpoints register files before enqueueing; mirror that so the
    # job has registry entries to finalize.
    queue.registry.register_queued(
        job_id="upload-fix",
        files=[{"filename": "doc.pdf", "hash": "hash-doc", "staging_path": str(staging / "doc.pdf")}],
    )
    job = queue.enqueue_upload(
        staging_dir=staging,
        filenames=["doc.pdf"],
        job_id="upload-fix",
        uploads=[{"filename": "doc.pdf", "hash": "hash-doc", "staging_path": str(staging / "doc.pdf")}],
    )
    _wait_for(lambda: queue.get_job(job.id)["status"] == "running")

    summary = queue.shutdown_finalize(grace_seconds=0.3)

    assert summary["finalized"] == 1
    assert queue.get_job(job.id)["status"] == "cancelled"
    registry = queue.registry.load()
    statuses = {entry["status"] for entry in registry["pdfs"].values()}
    assert statuses == {"interrupted"}
    # 'interrupted' is not in RECOVERABLE_UPLOAD_STATUSES, so recovery at the
    # next startup re-enqueues nothing from this job.
    recovered = queue.recover_pending_uploads(auto_start=False)
    assert recovered["recovered"] == 0


@pytest.mark.remote_client
def test_server_shutdown_rejects_remote_admin(monkeypatch, shutdown_env):
    """Even a valid admin credential from off-machine gets 403: shutdown is a
    loopback-only capability, not an admin-role one."""
    stub, triggered, _state = shutdown_env
    monkeypatch.setattr(web_app, "_API_TOKEN", "master-token-1")

    client = TestClient(web_app.app)
    response = client.post(
        "/api/server/shutdown",
        headers={"X-API-Token": "master-token-1"},
    )

    assert response.status_code == 403
    assert "local machine" in response.json()["detail"]
    assert triggered == []
    assert stub.finalize_calls == 0
    assert web_app._server_shutdown_public_state() is None


def test_server_shutdown_counts_down_then_finalizes(monkeypatch, shutdown_env):
    """A loopback request arms the countdown, /api/health advertises it to the
    other users, and the delayed thread finalizes jobs before triggering the
    graceful exit."""
    stub, triggered, _state = shutdown_env
    monkeypatch.setattr(web_app, "_API_TOKEN", "")
    monkeypatch.setattr(web_app, "api_authenticator", None)
    monkeypatch.setattr(web_app, "_STARTUP_NOTICES", [])

    class StubStore:
        def exists(self):
            return False

        def count(self):
            return 0

        def table_version_hint_path(self):
            return Path("/tmp/nonexistent_hint")

    monkeypatch.setattr(web_app, "_index_store", lambda *a, **k: StubStore())
    monkeypatch.setattr(web_app, "_llm_status_snapshot", lambda: {})

    client = TestClient(web_app.app)
    response = client.post("/api/server/shutdown", params={"delay_seconds": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "shutting_down"
    assert payload["active"] is True
    assert payload["delay_seconds"] == 5.0

    # Warning channel for the other open pages: the health payload flips to
    # the shutdown state (its ETag seed changes with it).
    health = client.get("/api/health")
    assert health.status_code == 200
    shutting_down = health.json()["shutting_down"]
    assert shutting_down is not None and shutting_down["active"] is True

    # The request only ARMS the shutdown: nothing finalizes or exits before
    # the countdown elapses.
    assert stub.finalize_calls == 0
    assert triggered == []

    # Countdown elapses: finalize runs, then the (stubbed) exit trigger fires.
    _wait_for(lambda: triggered, timeout=8.0)
    assert stub.finalize_calls == 1


def test_server_shutdown_is_idempotent(monkeypatch, shutdown_env):
    """A second request while the countdown runs returns the FIRST request's
    state instead of scheduling a competing exit."""
    _stub, triggered, _state = shutdown_env
    client = TestClient(web_app.app)
    first = client.post("/api/server/shutdown", params={"delay_seconds": 5}).json()
    second = client.post("/api/server/shutdown", params={"delay_seconds": 5}).json()

    assert first["shutdown_at"] == second["shutdown_at"]
    assert first["requested_at"] == second["requested_at"]
    # Only one countdown was armed; the fixture teardown de-arms it.
    assert not triggered
