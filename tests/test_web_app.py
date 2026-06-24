import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.query as query
import src.web_app as web_app
from src.pdf_registry import load_source_map, write_source_entry
from src.vector_store import JsonVectorStore


@pytest.fixture
def workspace_tmp():
    path = Path.cwd() / f".tmp_test_web_app_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_index(db_dir: Path):
    db_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "local_vector",
        "embedding_model": "nomic-embed-text",
        "embedding_dim": 3,
        "records": [
            {
                "id": "doc.md:0",
                "file_path": "processed_docs/doc.md",
                "chunk_index": 0,
                "content": "alpha context",
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "doc.md:1",
                "file_path": "processed_docs/doc.md",
                "chunk_index": 1,
                "content": "beta context",
                "vector": [0.0, 1.0, 0.0],
            },
        ],
    }
    web_app.vector_index_path(db_dir).write_text(json.dumps(payload), encoding="utf-8")


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for condition")


def test_index_rows_hide_vectors_and_support_search(workspace_tmp):
    db_dir = workspace_tmp / "db"
    _write_index(db_dir)

    result = web_app.list_index_rows(search="beta", db_dir=db_dir)

    assert result["total"] == 1
    assert result["rows"][0]["id"] == "doc.md:1"
    assert "vector" not in result["rows"][0]


def test_update_index_record_reembeds_and_saves(monkeypatch, workspace_tmp):
    db_dir = workspace_tmp / "db"
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

    payload = json.loads(web_app.vector_index_path(db_dir).read_text(encoding="utf-8"))
    assert row["content"] == "edited context"
    assert payload["records"][0]["content"] == "edited context"
    assert payload["records"][0]["vector"] == [0.0, 0.0, 1.0]
    assert calls["embed"] == (["edited context"], 3, "search_document: ")


def test_delete_index_records_persists_remaining_records(workspace_tmp):
    db_dir = workspace_tmp / "db"
    _write_index(db_dir)

    result = web_app.delete_index_records(record_ids=["doc.md:1"], db_dir=db_dir)

    payload = json.loads(web_app.vector_index_path(db_dir).read_text(encoding="utf-8"))
    assert result == {"deleted": 1, "remaining": 1}
    assert [record["id"] for record in payload["records"]] == ["doc.md:0"]


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
        "health_poll_interval_ms": 60000,
        "jobs_poll_interval_ms": 60000,
    }


def test_server_config_reads_polling_intervals(workspace_tmp):
    config_path = workspace_tmp / "config.toml"
    config_path.write_text(
        "[server]\nhealth_poll_interval_ms = 120000\njobs_poll_interval_ms = 90000\n",
        encoding="utf-8",
    )

    config = web_app._load_server_config(config_path)

    assert config == {
        "health_poll_interval_ms": 120000,
        "jobs_poll_interval_ms": 90000,
    }


def test_health_exposes_server_polling_config(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "SERVER_CONFIG",
        {"health_poll_interval_ms": 60000, "jobs_poll_interval_ms": 60000},
    )

    client = TestClient(web_app.app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["server"] == {
        "health_poll_interval_ms": 60000,
        "jobs_poll_interval_ms": 60000,
    }


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


def test_force_duplicate_cleanup_waits_until_ingestion_phase(workspace_tmp):
    calls = []
    processed_dir = workspace_tmp / "processed"
    db_dir = workspace_tmp / "db"
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
    JsonVectorStore(db_dir).write_records(
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

    def fake_ingest(input_dir, output_dir, **kwargs):
        old_content = old_markdown.read_text(encoding="utf-8") if old_markdown.exists() else ""
        calls.append(("ingest", bool(old_content), JsonVectorStore(db_dir).count()))

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
    assert JsonVectorStore(db_dir).count() == 2

    queue.finish_query()
    _wait_for(lambda: queue.get_job(job.id)["status"] == "done")

    assert calls[0] == ("ingest", False, 1)
    assert load_source_map(processed_dir)["documents"] == {}


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
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        files=[("files", ("notes.pdf", b"%PDF-1.4", "application/pdf"))],
    )

    assert response.status_code == 200
    assert response.json()["filenames"] == ["notes.pdf"]
    assert captured["filenames"] == ["notes.pdf"]
    assert captured["uploads"][0]["filename"] == "notes.pdf"
    assert len(captured["uploads"][0]["hash"]) == 64
    assert Path(captured["staging_dir"]).joinpath("notes.pdf").exists()


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
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    first = client.post(
        "/api/uploads",
        files=[("files", ("notes.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )
    second = client.post(
        "/api/uploads",
        files=[("files", ("copy.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )

    assert first.status_code == 200
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["can_force"] is True
    assert detail["duplicates"][0]["filename"] == "copy.pdf"
    assert detail["duplicates"][0]["existing_filename"] == "notes.pdf"


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
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    client.post(
        "/api/uploads",
        files=[("files", ("notes.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )
    response = client.post(
        "/api/uploads",
        data={"force_duplicates": "true"},
        files=[("files", ("copy.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )

    assert response.status_code == 200
    assert response.json()["force_duplicate_hashes"] == captured["force_duplicate_hashes"]
    assert captured["force_duplicate_hashes"] == [captured["uploads"][0]["hash"]]


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

        def ask_stream_events(self, question):
            assert question == "alpha?"
            yield {"type": "thinking", "text": "checking "}
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
        },
    )

    assert response.status_code == 200
    assert [json.loads(line) for line in response.text.splitlines()] == [
        {"type": "thinking", "text": "checking "},
        {"type": "answer", "text": "chunk "},
        {"type": "answer", "text": "two"},
    ]
    assert events[0] == "begin"
    assert events[1][1]["temperature"] == 0.65
    assert events[1][1]["sampler_top_k"] == 25
    assert events[1][1]["context_window"] == 4096
    assert events[1][1]["llm_num_predict"] == 512
    assert events[1][1]["llm_timeout"] == 120.0
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
