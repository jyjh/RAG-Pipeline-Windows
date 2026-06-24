import json
import shutil
import sys
import types
import uuid
from pathlib import Path

import numpy as np

import src.local_rag as local_rag


def test_chunk_markdown_splits_long_text_without_empty_chunks():
    text = "alpha\n\n" + ("beta " * 1000)

    chunks = local_rag.chunk_markdown(text, max_chars=100, overlap=10)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert chunks[0] == "alpha"
    assert len(chunks) > 2


def test_local_query_engine_uses_local_index_and_ollama(monkeypatch):
    tmp_path = Path.cwd() / f".tmp_test_local_rag_{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        index = {
            "backend": "local_vector",
            "embedding_dim": 3,
            "records": [
                {
                    "id": "a:0",
                    "file_path": "a.md",
                    "chunk_index": 0,
                    "content": "alpha context",
                    "vector": [1.0, 0.0, 0.0],
                },
                {
                    "id": "b:0",
                    "file_path": "b.md",
                    "chunk_index": 0,
                    "content": "beta context",
                    "vector": [0.0, 1.0, 0.0],
                },
            ],
        }
        tmp_path.joinpath(local_rag.INDEX_FILENAME).write_text(
            json.dumps(index),
            encoding="utf-8",
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

        monkeypatch.setitem(
            sys.modules,
            "ollama",
            types.SimpleNamespace(chat=fake_chat),
        )
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
    tmp_path = Path.cwd() / f".tmp_test_local_rag_{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        index = {
            "backend": "local_vector",
            "embedding_dim": 3,
            "records": [
                {
                    "id": "a:0",
                    "file_path": "a.md",
                    "chunk_index": 0,
                    "content": "alpha context",
                    "vector": [1.0, 0.0, 0.0],
                }
            ],
        }
        tmp_path.joinpath(local_rag.INDEX_FILENAME).write_text(
            json.dumps(index),
            encoding="utf-8",
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

        monkeypatch.setitem(
            sys.modules,
            "ollama",
            types.SimpleNamespace(chat=fake_chat),
        )
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
            {"type": "thinking", "text": "considering "},
            {"type": "answer", "text": "chunk "},
            {"type": "answer", "text": "two"},
        ]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_query_engine_reports_thinking_only_cutoff(monkeypatch):
    tmp_path = Path.cwd() / f".tmp_test_local_rag_{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        index = {
            "backend": "local_vector",
            "embedding_dim": 3,
            "records": [
                {
                    "id": "a:0",
                    "file_path": "a.md",
                    "chunk_index": 0,
                    "content": "alpha context",
                    "vector": [1.0, 0.0, 0.0],
                }
            ],
        }
        tmp_path.joinpath(local_rag.INDEX_FILENAME).write_text(
            json.dumps(index),
            encoding="utf-8",
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

        monkeypatch.setitem(
            sys.modules,
            "ollama",
            types.SimpleNamespace(chat=fake_chat),
        )
        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
        )
        events = list(engine.ask_stream_events("alpha?"))

        assert events[0] == {"type": "thinking", "text": "still thinking"}
        assert events[-1]["type"] == "notice"
        assert "token limit" in events[-1]["text"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_indexer_batches_chunk_embeddings(monkeypatch):
    tmp_path = Path.cwd() / f".tmp_test_local_rag_{uuid.uuid4().hex}"
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
        monkeypatch.setattr(local_rag, "chunk_markdown", lambda content: [f"chunk {i}" for i in range(5)])

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
        index = json.loads(db_dir.joinpath(local_rag.INDEX_FILENAME).read_text(encoding="utf-8"))
        assert len(index["records"]) == 5
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_indexer_fails_fast_when_ollama_preflight_fails(monkeypatch):
    tmp_path = Path.cwd() / f".tmp_test_local_rag_{uuid.uuid4().hex}"
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
