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
from src.vector_store import LanceDBVectorStore


@pytest.fixture
def workspace_tmp():
    path = Path.cwd() / f".tmp_test_web_app_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def lancedb_tmp():
    path = Path(tempfile.gettempdir()) / f"rag_test_web_app_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


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


def test_delete_index_records_persists_remaining_records(lancedb_tmp):
    db_dir = lancedb_tmp / "db"
    _write_index(db_dir)

    result = web_app.delete_index_records(record_ids=["doc.md:1"], db_dir=db_dir)

    assert result == {"deleted": 1, "remaining": 1}
    assert [row["id"] for row in LanceDBVectorStore(db_dir).list_records()["rows"]] == ["doc.md:0"]


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
        "[server]\nhealth_poll_interval_ms = 120032\njobs_poll_interval_ms = 90000\n",
        encoding="utf-8",
    )

    config = web_app._load_server_config(config_path)

    assert config == {
        "health_poll_interval_ms": 120032,
        "jobs_poll_interval_ms": 90000,
    }


def test_chat_config_reads_prompt_retrieval_and_ollama_health_settings(workspace_tmp):
    config_path = workspace_tmp / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[chat]",
                'system_prompt = "Configured prompt {web_instruction}"',
                "context_window = 120000",
                "llm_num_predict = 24000",
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
        "context_window": 120000,
        "llm_num_predict": 24000,
        "retrieval_min_score": 0.62,
        "ollama_health_check_interval": 3.5,
        "ollama_max_lost_health_checks": 9,
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
    assert response.json()["chat"] == {
        "context_window": web_app.CHAT_CONFIG["context_window"],
        "llm_num_predict": web_app.CHAT_CONFIG["llm_num_predict"],
        "retrieval_min_score": web_app.CHAT_CONFIG["retrieval_min_score"],
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


def test_force_duplicate_cleanup_waits_until_ingestion_phase(workspace_tmp, lancedb_tmp):
    calls = []
    processed_dir = workspace_tmp / "processed"
    db_dir = lancedb_tmp / "db"
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

    def fake_ingest(input_dir, output_dir, **kwargs):
        old_content = old_markdown.read_text(encoding="utf-8") if old_markdown.exists() else ""
        calls.append(("ingest", bool(old_content), LanceDBVectorStore(db_dir).count()))

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
    processed_dir = workspace_tmp / "processed"
    processed_dir.mkdir()
    outside_pdf = workspace_tmp / "outside.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4 outside")
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
        {"type": "sources", "sources": [{"id": "S1", "label": "[S1]", "kind": "local"}]},
        {"type": "answer", "text": "chunk "},
        {"type": "answer", "text": "two"},
    ]
    assert events[0] == "begin"
    assert events[1][1]["temperature"] == 0.65
    assert events[1][1]["sampler_top_k"] == 25
    assert events[1][1]["context_window"] == 4096
    assert events[1][1]["llm_num_predict"] == 512
    assert events[1][1]["llm_timeout"] == 120.0
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
