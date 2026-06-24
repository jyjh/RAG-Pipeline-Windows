import json
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

import src.local_rag as local_rag
from src.vector_store import LanceDBVectorStore


def _write_lancedb_records(db_dir: Path, records: list[dict]):
    enriched = []
    for index, record in enumerate(records):
        item = {
            "id": record.get("id", f"record-{index}"),
            "doc_id": record.get("doc_id", f"doc-{index}"),
            "parent_id": record.get("parent_id", ""),
            "node_type": record.get("node_type", "chunk"),
            "file_path": record.get("file_path", "doc.md"),
            "chunk_index": record.get("chunk_index", index),
            "content": record.get("content", ""),
            "title": record.get("title", "Doc"),
            "section_path": record.get("section_path", "Doc"),
            "page_start": record.get("page_start", 1),
            "page_end": record.get("page_end", 1),
            "summary": record.get("summary", "summary"),
            "tags": record.get("tags", []),
            "vector": record.get("vector", [1.0, 0.0, 0.0]),
        }
        item.update(record)
        enriched.append(item)
    LanceDBVectorStore(db_dir).write_records(
        enriched,
        embedding_model="fake-embed",
        embedding_dim=3,
    )


def test_chunk_markdown_splits_long_text_without_empty_chunks():
    text = "alpha\n\n" + ("beta " * 1000)

    chunks = local_rag.chunk_markdown(text, max_chars=100, overlap=10)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert chunks[0] == "alpha"
    assert len(chunks) > 2


def test_ollama_chat_posts_directly_to_api_chat(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, *, body=b"", lines=None):
            self.body = body
            self.lines = lines or []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return self.body

        def __iter__(self):
            return iter(self.lines)

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": payload,
                "content_type": request.get_header("Content-type"),
            }
        )
        if payload["stream"]:
            return FakeResponse(lines=[b'{"message":{"content":"streamed"},"done":true}\n'])
        return FakeResponse(body=b'{"message":{"content":"answer"}}')

    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    monkeypatch.setattr(local_rag.urllib.request, "urlopen", fake_urlopen)

    response = local_rag._ollama_chat(
        model="gemma4",
        messages=[{"role": "user", "content": "hi"}],
        options={"num_predict": 1},
        stream=False,
        timeout=7.5,
    )
    events = list(
        local_rag._ollama_chat(
            model="gemma4",
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1},
            stream=True,
            timeout=8.5,
        )
    )

    assert response["message"]["content"] == "answer"
    assert events == [{"message": {"content": "streamed"}, "done": True}]
    assert [call["url"] for call in calls] == [
        "http://127.0.0.1:11434/api/chat",
        "http://127.0.0.1:11434/api/chat",
    ]
    assert calls[0]["timeout"] == 7.5
    assert calls[1]["timeout"] == 8.5
    assert calls[0]["payload"]["model"] == "gemma4"
    assert calls[1]["payload"]["stream"] is True


def test_local_query_engine_uses_local_index_and_ollama(monkeypatch):
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        _write_lancedb_records(
            tmp_path,
            [
                {
                    "id": "a:0",
                    "doc_id": "a",
                    "file_path": "a.md",
                    "chunk_index": 0,
                    "content": "alpha context",
                    "vector": [1.0, 0.0, 0.0],
                },
                {
                    "id": "b:0",
                    "doc_id": "b",
                    "file_path": "b.md",
                    "chunk_index": 0,
                    "content": "beta context",
                    "vector": [0.0, 1.0, 0.0],
                },
            ],
        )

        class FakeEngine:
            def __init__(self, **kwargs):
                pass

            def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
                return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

        calls = {}

        def fake_chat(**kwargs):
            calls["kwargs"] = kwargs
            return {"message": {"content": "answer"}}

        monkeypatch.setattr(local_rag, "_ollama_chat", fake_chat)
        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
        )
        answer = engine.ask("alpha?")

        assert answer == "answer"
        assert calls["kwargs"]["model"] == "gemma4"
        assert calls["kwargs"]["options"]["temperature"] == 0.9
        assert calls["kwargs"]["options"]["top_k"] == 40
        assert calls["kwargs"]["options"]["num_ctx"] == 8192
        assert calls["kwargs"]["options"]["num_predict"] == 4096
        user_prompt = calls["kwargs"]["messages"][1]["content"]
        assert "alpha context" in user_prompt
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_query_engine_streams_ollama_chunks(monkeypatch):
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        _write_lancedb_records(
            tmp_path,
            [
                {
                    "id": "a:0",
                    "doc_id": "a",
                    "file_path": "a.md",
                    "chunk_index": 0,
                    "content": "alpha context",
                    "vector": [1.0, 0.0, 0.0],
                }
            ],
        )

        class FakeEngine:
            def __init__(self, **kwargs):
                pass

            def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
                return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

        calls = {}

        def fake_chat(**kwargs):
            calls["kwargs"] = kwargs
            return iter(
                [
                    {"message": {"thinking": "considering "}},
                    {"message": {"content": "chunk "}},
                    {"message": {"content": "two"}},
                ]
            )

        monkeypatch.setattr(local_rag, "_ollama_chat", fake_chat)
        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
        )
        chunks = list(engine.ask_stream("alpha?"))

        assert chunks == ["chunk ", "two"]
        assert calls["kwargs"]["stream"] is True
        assert calls["kwargs"]["options"]["temperature"] == 0.9
        assert calls["kwargs"]["options"]["top_k"] == 40
        assert calls["kwargs"]["options"]["num_ctx"] == 8192
        assert calls["kwargs"]["options"]["num_predict"] == 4096

        events = list(engine.ask_stream_events("alpha?"))
        assert events == [
            {"type": "notice", "text": "Embedding query and retrieving context..."},
            {"type": "notice", "text": "Retrieved 1 context chunk(s). Requesting answer from gemma4..."},
            {"type": "thinking", "text": "considering "},
            {"type": "answer", "text": "chunk "},
            {"type": "answer", "text": "two"},
        ]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_query_engine_reports_thinking_only_cutoff(monkeypatch):
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        _write_lancedb_records(
            tmp_path,
            [
                {
                    "id": "a:0",
                    "doc_id": "a",
                    "file_path": "a.md",
                    "chunk_index": 0,
                    "content": "alpha context",
                    "vector": [1.0, 0.0, 0.0],
                }
            ],
        )

        class FakeEngine:
            def __init__(self, **kwargs):
                pass

            def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
                return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

        def fake_chat(**kwargs):
            return iter(
                [
                    {"message": {"content": "<think>still thinking"}},
                    {"done": True, "done_reason": "length", "message": {"content": ""}},
                ]
            )

        monkeypatch.setattr(local_rag, "_ollama_chat", fake_chat)
        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
        )
        events = list(engine.ask_stream_events("alpha?"))

        assert events[0] == {"type": "notice", "text": "Embedding query and retrieving context..."}
        assert events[1] == {"type": "notice", "text": "Retrieved 1 context chunk(s). Requesting answer from gemma4..."}
        assert events[2] == {"type": "thinking", "text": "still thinking"}
        assert events[-1]["type"] == "notice"
        assert "token limit" in events[-1]["text"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_query_engine_expands_lancedb_summary_hits(monkeypatch):
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        from src.vector_store import LanceDBVectorStore

        LanceDBVectorStore(tmp_path).write_records(
            [
                {
                    "id": "doc-summary",
                    "doc_id": "doc",
                    "parent_id": "",
                    "node_type": "document_summary",
                    "file_path": "doc.pdf",
                    "chunk_index": -1,
                    "content": "document summary only",
                    "title": "Doc",
                    "section_path": "Doc",
                    "page_start": 1,
                    "page_end": 2,
                    "summary": "summary",
                    "tags": ["alpha"],
                    "vector": [1.0, 0.0, 0.0],
                },
                {
                    "id": "chunk-1",
                    "doc_id": "doc",
                    "parent_id": "doc-summary",
                    "node_type": "chunk",
                    "file_path": "doc.pdf",
                    "chunk_index": 0,
                    "content": "alpha leaf context",
                    "title": "Alpha",
                    "section_path": "Doc > Alpha",
                    "page_start": 2,
                    "page_end": 2,
                    "summary": "summary",
                    "tags": ["alpha"],
                    "vector": [0.0, 1.0, 0.0],
                },
            ],
            embedding_model="fake-embed",
            embedding_dim=3,
        )

        class FakeEngine:
            def __init__(self, **kwargs):
                pass

            def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
                return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

        calls = {}

        def fake_chat(**kwargs):
            calls["kwargs"] = kwargs
            return {"message": {"content": "answer"}}

        monkeypatch.setattr(local_rag, "_ollama_chat", fake_chat)
        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
            top_k=1,
        )

        assert engine.ask("alpha?") == "answer"
        user_prompt = calls["kwargs"]["messages"][1]["content"]
        assert "alpha leaf context" in user_prompt
        assert "document summary only" not in user_prompt
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_indexer_batches_chunk_embeddings(monkeypatch):
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    md_dir = tmp_path / "md"
    db_dir = tmp_path / "db"
    md_dir.mkdir(parents=True)
    try:
        md_dir.joinpath("doc.md").write_text(
            "\n\n".join(f"paragraph {i}" for i in range(5)),
            encoding="utf-8",
        )
        calls = []

        class FakeEngine:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))

            def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
                calls.append(("embed", list(texts)))
                return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        def fake_build_section_records(*args, **kwargs):
            return [
                {
                    "id": f"record-{i}",
                    "doc_id": "doc",
                    "parent_id": "doc" if i else "",
                    "node_type": "chunk" if i else "document_summary",
                    "file_path": "doc.md",
                    "chunk_index": i,
                    "content": f"chunk {i}",
                    "title": f"Section {i}",
                    "section_path": f"Doc > Section {i}",
                    "page_start": i + 1,
                    "page_end": i + 1,
                    "summary": f"summary {i}",
                    "tags": ["chunk"],
                }
                for i in range(5)
            ]

        monkeypatch.setattr(local_rag, "build_section_records", fake_build_section_records)

        indexer = local_rag.LocalVectorIndexer(
            working_dir=str(db_dir),
            embedding_batch_size=2,
            progress_enabled=False,
        )
        indexer.index_markdown(str(md_dir))

        embed_calls = [
            payload
            for kind, payload in calls
            if kind == "embed" and payload != ["embedding health check"]
        ]
        assert [len(payload) for payload in embed_calls] == [2, 2, 1]
        from src.vector_store import LanceDBVectorStore

        assert LanceDBVectorStore(db_dir).count() == 5
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_indexer_fails_fast_when_ollama_preflight_fails(monkeypatch):
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    md_dir = tmp_path / "md"
    db_dir = tmp_path / "db"
    md_dir.mkdir(parents=True)
    try:
        md_dir.joinpath("doc.md").write_text("content", encoding="utf-8")

        class FakeEngine:
            def __init__(self, **kwargs):
                pass

            def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
                raise RuntimeError("timeout")

        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        indexer = local_rag.LocalVectorIndexer(
            working_dir=str(db_dir),
            progress_enabled=False,
        )
        try:
            indexer.index_markdown(str(md_dir))
        except RuntimeError as exc:
            assert "Ollama embedding preflight failed" in str(exc)
        else:
            raise AssertionError("Expected preflight failure")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
