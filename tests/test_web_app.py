import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.query as query
import src.web_app as web_app


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
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        files=[("files", ("notes.pdf", b"%PDF-1.4", "application/pdf"))],
    )

    assert response.status_code == 200
    assert response.json()["filenames"] == ["notes.pdf"]
    assert captured["filenames"] == ["notes.pdf"]
    assert Path(captured["staging_dir"]).joinpath("notes.pdf").exists()


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
