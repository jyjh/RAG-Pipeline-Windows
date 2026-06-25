import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.local_rag as local_rag
import src.query as query
import src.web_app as web_app
from src.pdf_registry import load_source_map, write_source_entry
from src.vector_store import LanceDBVectorStore


def _completed(args, *, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(["git", *args], returncode, stdout=stdout, stderr=stderr)


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
    assert calls["init"]["retrieval_min_score"] == 0.72
    assert calls["init"]["web_search_enabled"] is False
    assert calls["search"] == {"query": "delta dynamics", "relevance_floor": 0.72}
    assert payload["query"] == "delta dynamics"
    assert payload["relevance_floor"] == 0.72
    assert payload["total"] == 1
    assert payload["rows"][0]["id"] == "chunk-2"
    assert payload["rows"][0]["score"] == 0.82
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
        "host": "127.0.0.1",
        "port": 8000,
        "health_poll_interval_ms": 60000,
        "jobs_poll_interval_ms": 60000,
        "update_remote": "origin",
        "update_branch": "main",
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
        "port": 8081,
        "health_poll_interval_ms": 120032,
        "jobs_poll_interval_ms": 90000,
        "update_remote": "upstream",
        "update_branch": "web-ui",
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
    monkeypatch.setitem(web_app.SERVER_CONFIG, "host", "100.87.142.5")

    assert web_app._is_local_update_host("100.87.142.5") is True
    assert web_app._is_local_update_host("100.87.142.6") is False


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
            "vision_model": "vision-test",
            "vision_enabled": False,
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

    assert captured["vision_model"] == "vision-test"
    assert captured["vision_enabled"] is False
    assert captured["ocr_backend"] == "tesseract_cli"
    assert captured["ocr_langs"] == ["eng"]
    assert captured["ocr_force_full_page"] is False
    assert captured["ocr_bitmap_area_threshold"] == 0.2
    assert captured["rapidocr_backend"] == "torch"
    assert captured["tesseract_cmd"] == "C:/Tools/tesseract.exe"
    assert captured["tesseract_data_path"] == "C:/Tools/tessdata"
    assert captured["tesseract_psm"] == 6


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
        options={"ocr_backend": "tesseract_cli", "embedding_model": "persisted-embed"},
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
    assert calls == [
        ("ingest", upload_dir, "tesseract_cli"),
        ("index", processed_dir, "persisted-embed"),
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
        options={"embedding_model": "persisted-embed"},
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
    assert calls == [("index", processed_dir, db_dir, "persisted-embed")]
    assert entry["status"] == "indexed"


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
    assert captured["options"]["ocr_backend"] == "rapidocr"
    assert captured["options"]["rapidocr_backend"] == "onnxruntime"
    assert captured["options"]["ocr_force_full_page"] is True
    assert captured["options"]["ocr_langs"] == ["english"]
    assert captured["options"]["vision_enabled"] is True
    assert Path(captured["staging_dir"]).joinpath("notes.pdf").exists()


def test_upload_endpoint_splits_multiple_pdfs_into_jobs(monkeypatch, workspace_tmp):
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
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        files=[
            ("files", ("one.pdf", b"%PDF-1.4 one", "application/pdf")),
            ("files", ("two.pdf", b"%PDF-1.4 two", "application/pdf")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_count"] == 2
    assert payload["filenames"] == ["one.pdf", "two.pdf"]
    assert [job["filenames"] for job in payload["jobs"]] == [["one.pdf"], ["two.pdf"]]
    assert [item["filenames"] for item in captured] == [["one.pdf"], ["two.pdf"]]
    assert captured[0]["job_id"] != captured[1]["job_id"]
    assert captured[0]["staging_dir"] != captured[1]["staging_dir"]
    assert Path(captured[0]["staging_dir"]).joinpath("one.pdf").exists()
    assert Path(captured[1]["staging_dir"]).joinpath("two.pdf").exists()


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
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
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
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        files=[("files", ("copy.pdf", content, "application/pdf"))],
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["force_token"]
    assert detail["duplicates"][0]["existing_filename"] == "notes.pdf"
    assert detail["duplicates"][0]["record_id"] == "chunk-duplicate"


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
    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/uploads",
        files=[("files", ("copy.pdf", content, "application/pdf"))],
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["force_token"]
    assert detail["duplicates"][0]["existing_filename"] == "notes.pdf"
    assert detail["duplicates"][0]["status"] == "uploaded"


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
    duplicate = client.post(
        "/api/uploads",
        files=[("files", ("copy.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )
    direct_force = client.post(
        "/api/uploads",
        data={"force_duplicates": "true"},
        files=[("files", ("copy.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )
    response = client.post(
        "/api/uploads",
        data={
            "force_duplicates": "true",
            "force_token": duplicate.json()["detail"]["force_token"],
        },
        files=[("files", ("copy.pdf", b"%PDF-1.4 same", "application/pdf"))],
    )

    assert duplicate.status_code == 409
    assert direct_force.status_code == 409
    assert response.status_code == 200
    assert response.json()["force_duplicate_hashes"] == captured["force_duplicate_hashes"]
    assert captured["force_duplicate_hashes"] == [captured["uploads"][0]["hash"]]


def test_jobs_endpoint_paginates(monkeypatch):
    class FakeQueue:
        def list_jobs(self):
            return [{"id": f"job-{index:02}", "filenames": [f"{index}.pdf"]} for index in range(12)]

    monkeypatch.setattr(web_app, "job_queue", FakeQueue())

    client = TestClient(web_app.app)
    response = client.get("/api/jobs", params={"offset": 10, "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 12
    assert payload["offset"] == 10
    assert payload["limit"] == 10
    assert [job["id"] for job in payload["jobs"]] == ["job-10", "job-11"]


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
        ],
        embedding_model="fake-embed",
        embedding_dim=3,
    )
    monkeypatch.setattr(web_app, "PDF_REGISTRY_PATH", workspace_tmp / "registry.json")
    monkeypatch.setattr(web_app, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(web_app, "DB_DIR", db_dir)

    client = TestClient(web_app.app)
    response = client.get("/api/pdfs")

    assert response.status_code == 200
    quality = response.json()["pdfs"][0]["quality"]
    assert quality["label"] == "ready"
    assert quality["chunk_count"] == 1
    assert quality["record_count"] == 2
    assert quality["markdown_exists"] is True


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
