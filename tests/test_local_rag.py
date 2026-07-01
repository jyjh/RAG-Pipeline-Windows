import json
import shutil
import tempfile
import urllib.error
import uuid
from pathlib import Path

import numpy as np

import src.local_rag as local_rag
from src.asset_store import ImageAssetStore, image_asset_marker
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


def _tool_call(name="search_local_context", arguments=None):
    return {"function": {"name": name, "arguments": arguments or {"query": "alpha?"}}}


def _fake_local_tool_chat(calls, *, final_events=None):
    final_events = final_events or [{"message": {"content": "answer [S1]"}}]

    def fake_chat(**kwargs):
        calls.append(kwargs)
        if kwargs["stream"]:
            return iter(final_events)
        has_tool_result = any(message.get("role") == "tool" for message in kwargs["messages"])
        if not has_tool_result:
            return {"message": {"tool_calls": [_tool_call()]}}
        return {"message": {"content": ""}}

    return fake_chat


def test_chunk_markdown_splits_long_text_without_empty_chunks():
    text = "alpha\n\n" + ("beta " * 1000)

    chunks = local_rag.chunk_markdown(text, max_chars=100, overlap=10)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert chunks[0] == "alpha"
    assert len(chunks) > 2


def test_write_index_manifest_summarizes_source_quality_counts():
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_manifest_{uuid.uuid4().hex}"
    try:
        records = [
            {
                "id": "doc",
                "node_type": "document_summary",
                "content": "summary",
                "source_hash": "hash-a",
                "source_pdf_name": "doc.pdf",
                "source_pdf_path": "data/doc.pdf",
                "page_start": 1,
                "page_end": 3,
            },
            {
                "id": "chunk",
                "node_type": "chunk",
                "content": "alpha context",
                "source_hash": "hash-a",
                "source_pdf_name": "doc.pdf",
                "source_pdf_path": "data/doc.pdf",
                "page_start": 2,
                "page_end": 2,
            },
        ]

        manifest = local_rag.write_index_manifest(
            tmp_path,
            records,
            embedding_model="fake-embed",
            embedding_dim=3,
        )

        assert tmp_path.joinpath(local_rag.INDEX_MANIFEST_FILENAME).exists()
        assert manifest["total_records"] == 2
        assert manifest["documents"]["hash-a"]["record_count"] == 2
        assert manifest["documents"]["hash-a"]["chunk_count"] == 1
        assert manifest["documents"]["hash-a"]["summary_count"] == 1
        assert manifest["documents"]["hash-a"]["page_start"] == 1
        assert manifest["documents"]["hash-a"]["page_end"] == 3
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_citation_support_warnings_flag_weak_cited_claims():
    messages = [
        {
            "role": "tool",
            "content": (
                '{"results":[{"citation":"[S1]",'
                '"content":"The suspension report discusses spring stiffness and damping ratios."}]}'
            ),
        }
    ]

    supported = local_rag.citation_support_warnings(
        "The report discusses spring stiffness and damping ratios [S1].",
        messages,
    )
    weak = local_rag.citation_support_warnings(
        "Aerodynamic vortices define accumulator voltage limits [S1].",
        messages,
    )

    assert supported == []
    assert weak
    assert "[S1]" in weak[0]


def test_local_citation_includes_pdf_page_link():
    citations = local_rag.CitationRegistry()

    source = citations.add_local(
        {
            "id": "chunk-a",
            "source_hash": "hash/a",
            "source_pdf_name": "report.pdf",
            "page_start": 7,
            "page_end": 9,
            "content": "alpha context",
        }
    )

    assert source["open_url"] == "/api/pdfs/hash%2Fa/view#page=7"
    assert source["download_url"] == "/api/pdfs/hash%2Fa/download"
    assert source["page_label"] == "pages 7-9"


def test_local_tool_resolves_image_asset_markers_without_url_in_tool_payload(safe_tmp_path):
    store = ImageAssetStore(safe_tmp_path / "assets")
    asset = store.save_image(
        image_data=b"graph-png",
        source_hash="hash-a",
        source_pdf_name="report.pdf",
        page_no=4,
        description="A line graph of roll gradient.",
    )
    record = {
        "id": "chunk-a",
        "source_hash": "hash-a",
        "source_pdf_name": "report.pdf",
        "page_start": 4,
        "page_end": 4,
        "content": f"> [Vision Analysis]: A line graph of roll gradient.\n{image_asset_marker(asset['asset_id'])}",
        "score": 0.91,
    }
    engine = local_rag.LocalQueryEngine.__new__(local_rag.LocalQueryEngine)
    engine.asset_store = store
    engine._context_token_budget = lambda: 4000
    engine._retrieve = lambda *args, **kwargs: [record]
    citations = local_rag.CitationRegistry(asset_store=store)

    result = local_rag.LocalQueryEngine._local_tool_result(
        engine,
        query="roll gradient graph",
        exclude_ids=set(),
        citations=citations,
        token_budget=4000,
    )

    assert result["result_count"] == 1
    assert "/api/assets/" not in json.dumps(result)
    sources = citations.all_sources()
    assert sources[0]["assets"][0]["asset_id"] == asset["asset_id"]
    assert sources[0]["assets"][0]["url"] == f"/api/assets/{asset['asset_id']}"
    assert sources[0]["assets"][0]["description"] == "A line graph of roll gradient."


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
    assert calls[0]["timeout"] is None
    assert calls[1]["timeout"] is None
    assert calls[0]["payload"]["model"] == "gemma4"
    assert calls[1]["payload"]["stream"] is True


def test_ollama_chat_cancels_after_lost_health_check_cycles(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append({"url": request.full_url, "timeout": timeout})
        raise urllib.error.URLError("server down")

    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    monkeypatch.setattr(local_rag.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(local_rag.time, "sleep", lambda seconds: None)

    try:
        local_rag._ollama_chat(
            model="gemma4",
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1},
            stream=False,
            timeout=7.5,
            health_check_interval=0.1,
            max_lost_health_checks=2,
        )
    except RuntimeError as exc:
        assert "did not recover within 2 health check cycle" in str(exc)
    else:
        raise AssertionError("Expected connection loss failure")

    assert calls[0]["url"] == "http://127.0.0.1:11434/api/chat"
    assert calls[0]["timeout"] is None
    assert [call["url"] for call in calls[1:]] == [
        "http://127.0.0.1:11434/api/version",
        "http://127.0.0.1:11434/api/version",
    ]


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

        calls = []

        monkeypatch.setattr(local_rag, "_ollama_chat", _fake_local_tool_chat(calls))
        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
            planner_enabled=False,
        )
        answer = engine.ask("alpha?")

        assert answer == "answer [S1]"
        assert calls[0]["model"] == "gemma4"
        assert calls[0]["tools"][0]["function"]["name"] == "search_local_context"
        assert calls[-1]["stream"] is True
        assert calls[-1]["options"]["temperature"] == 0.3
        assert calls[-1]["options"]["top_k"] == 40
        assert calls[-1]["options"]["num_ctx"] == 8192
        assert calls[-1]["options"]["num_predict"] == 4096
        tool_messages = [message for message in calls[-1]["messages"] if message.get("role") == "tool"]
        assert tool_messages
        assert "alpha context" in tool_messages[0]["content"]
        assert '"citation": "[S1]"' in tool_messages[0]["content"]
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

        calls = []

        monkeypatch.setattr(
            local_rag,
            "_ollama_chat",
            _fake_local_tool_chat(
                calls,
                final_events=[
                    {"message": {"thinking": "considering "}},
                    {"message": {"content": "chunk "}},
                    {"message": {"content": "two [S1]"}},
                ],
            ),
        )
        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
            planner_enabled=False,
        )
        chunks = list(engine.ask_stream("alpha?"))

        assert chunks == ["chunk ", "two [S1]"]
        assert calls[-1]["stream"] is True
        assert calls[-1]["options"]["temperature"] == 0.3
        assert calls[-1]["options"]["top_k"] == 40
        assert calls[-1]["options"]["num_ctx"] == 8192
        assert calls[-1]["options"]["num_predict"] == 4096

        events = list(engine.ask_stream_events("alpha?"))
        assert events[0] == {"type": "notice", "text": "Planning retrieval tool calls..."}
        assert {"type": "tool_call", "tool": "search_local_context", "text": "Running search_local_context..."} in events
        tool_results = [event for event in events if event.get("type") == "tool_result"]
        assert tool_results
        assert tool_results[0]["tool"] == "search_local_context"
        assert tool_results[0]["result"]["tool"] == "search_local_context"
        assert tool_results[0]["result"]["results"][0]["citation"] == "[S1]"
        assert json.loads(tool_results[0]["content"]) == tool_results[0]["result"]
        assert any(event.get("type") == "sources" and event["sources"][0]["label"] == "[S1]" for event in events)
        assert {"type": "thinking", "text": "considering "} in events
        assert {"type": "answer", "text": "chunk "} in events
        assert {"type": "answer", "text": "two [S1]"} in events
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_retrieval_uses_relevance_cutoff_without_fixed_chunk_limit(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    class FakeStore:
        def exists(self):
            return True

        def count(self):
            return 3

        def search(self, vector, *, top_k):
            assert top_k == 80
            return [
                {"id": "a", "node_type": "chunk", "content": "alpha", "score": 1.0},
                {"id": "b", "node_type": "chunk", "content": "beta", "score": 0.75},
                {"id": "c", "node_type": "chunk", "content": "gamma", "score": 0.71},
            ]

        def child_chunks(self, parent, *, limit):
            return []

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(working_dir=str(tmp_path), progress_enabled=False)
        engine.store = FakeStore()
        engine.record_count = 3

        matches = engine._retrieve("alpha?")

        assert [match["id"] for match in matches] == ["a", "b"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_retrieval_reranks_vector_candidates_with_query_terms(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    class FakeStore:
        def exists(self):
            return True

        def count(self):
            return 2

        def search(self, vector, *, top_k):
            return [
                {
                    "id": "vector-best",
                    "node_type": "chunk",
                    "title": "General notes",
                    "content": "generic optimization background",
                    "score": 0.95,
                    "source_group": "official",
                },
                {
                    "id": "lexical-best",
                    "node_type": "chunk",
                    "title": "Ridge regularization",
                    "content": "ridge regularization uses a lambda penalty",
                    "score": 0.90,
                    "source_group": "official",
                },
            ]

        def child_chunks(self, parent, *, limit):
            return []

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(working_dir=str(tmp_path), progress_enabled=False)
        engine.store = FakeStore()
        engine.record_count = 2

        matches = engine._retrieve("ridge regularization lambda")

        assert [match["id"] for match in matches] == ["lexical-best", "vector-best"]
        assert matches[0]["vector_score"] == 0.9
        assert matches[0]["lexical_score"] > matches[1]["lexical_score"]
        assert matches[0]["score"] == matches[0]["hybrid_score"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_retrieval_applies_source_group_reliability_weight(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    class FakeStore:
        def exists(self):
            return True

        def count(self):
            return 2

        def search(self, vector, *, top_k):
            return [
                {
                    "id": "unofficial-best",
                    "node_type": "chunk",
                    "content": "ridge regularization uses lambda",
                    "score": 0.92,
                    "source_group": "unofficial",
                },
                {
                    "id": "official-second",
                    "node_type": "chunk",
                    "content": "ridge regularization uses lambda",
                    "score": 0.88,
                    "source_group": "official",
                },
            ]

        def child_chunks(self, parent, *, limit):
            return []

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(working_dir=str(tmp_path), progress_enabled=False)
        engine.store = FakeStore()
        engine.record_count = 2

        matches = engine._retrieve("ridge regularization lambda")

        assert [match["id"] for match in matches] == ["official-second", "unofficial-best"]
        assert matches[0]["source_group"] == "official"
        assert matches[0]["reliability_modifier"] == 1.0
        assert matches[1]["source_group"] == "unofficial"
        assert matches[1]["reliability_modifier"] == 0.8
        assert matches[1]["vector_score"] > matches[0]["vector_score"]
        assert matches[1]["hybrid_score"] > matches[0]["hybrid_score"]
        assert matches[0]["score"] > matches[1]["score"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_retrieval_lexical_weight_zero_keeps_vector_order(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    class FakeStore:
        def exists(self):
            return True

        def count(self):
            return 2

        def search(self, vector, *, top_k):
            return [
                {
                    "id": "vector-best",
                    "node_type": "chunk",
                    "content": "generic",
                    "score": 0.95,
                    "source_group": "official",
                },
                {
                    "id": "lexical-best",
                    "node_type": "chunk",
                    "content": "ridge regularization lambda",
                    "score": 0.90,
                    "source_group": "official",
                },
            ]

        def child_chunks(self, parent, *, limit):
            return []

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            retrieval_lexical_weight=0.0,
            progress_enabled=False,
        )
        engine.store = FakeStore()
        engine.record_count = 2

        matches = engine._retrieve("ridge regularization lambda")

        assert [match["id"] for match in matches] == ["vector-best", "lexical-best"]
        assert [match["score"] for match in matches] == [0.95, 0.9]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_retrieval_stops_at_context_token_budget_after_first_match(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    class FakeStore:
        def exists(self):
            return True

        def count(self):
            return 2

        def search(self, vector, *, top_k):
            return [
                {"id": "a", "node_type": "chunk", "content": "alpha " * 220, "score": 1.0},
                {"id": "b", "node_type": "chunk", "content": "beta " * 220, "score": 0.9},
            ]

        def child_chunks(self, parent, *, limit):
            return []

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            context_window=1024,
            context_token_fraction=0.25,
            progress_enabled=False,
        )
        engine.store = FakeStore()
        engine.record_count = 2

        matches = engine._retrieve("alpha?")

        assert [match["id"] for match in matches] == ["a"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_retrieval_truncates_oversized_first_match_to_context_budget(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    class FakeStore:
        def exists(self):
            return True

        def count(self):
            return 1

        def search(self, vector, *, top_k):
            return [
                {"id": "a", "node_type": "chunk", "content": "alpha " * 2000, "score": 1.0},
            ]

        def child_chunks(self, parent, *, limit):
            return []

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            context_window=1024,
            context_token_fraction=0.25,
            progress_enabled=False,
        )
        engine.store = FakeStore()
        engine.record_count = 1

        matches = engine._retrieve("alpha?")

        assert len(matches) == 1
        assert matches[0]["estimated_tokens"] <= engine._context_token_budget()
        assert matches[0]["content_truncated"] is True
        assert matches[0]["content"].endswith(local_rag.CONTEXT_TRUNCATION_NOTICE)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_tool_result_exposes_reliability_fields():
    record = {
        "id": "chunk-a",
        "source_hash": "hash-a",
        "source_pdf_name": "report.pdf",
        "page_start": 4,
        "page_end": 4,
        "content": "alpha context",
        "score": 0.72,
        "vector_score": 0.9,
        "lexical_score": 0.8,
        "hybrid_score": 0.9,
        "reliability_modifier": 0.8,
        "source_group": "student_research",
    }
    engine = local_rag.LocalQueryEngine.__new__(local_rag.LocalQueryEngine)
    engine.asset_store = None
    engine._context_token_budget = lambda: 4000
    engine._retrieve = lambda *args, **kwargs: [record]
    citations = local_rag.CitationRegistry()

    result = local_rag.LocalQueryEngine._local_tool_result(
        engine,
        query="alpha context",
        exclude_ids=set(),
        citations=citations,
        token_budget=4000,
    )

    item = result["results"][0]
    assert item["vector_score"] == 0.9
    assert item["lexical_score"] == 0.8
    assert item["hybrid_score"] == 0.9
    assert item["reliability_modifier"] == 0.8
    assert item["source_group"] == "student_research"


def test_forced_local_tool_call_keeps_final_prompt_under_input_context_budget(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    class FakeStore:
        def exists(self):
            return True

        def count(self):
            return 1

        def search(self, vector, *, top_k):
            return [
                {"id": "a", "node_type": "chunk", "content": "alpha " * 4000, "score": 1.0},
            ]

        def child_chunks(self, parent, *, limit):
            return []

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            context_window=4096,
            context_token_fraction=0.95,
            progress_enabled=False,
        )
        engine.store = FakeStore()
        engine.record_count = 1
        messages = engine._tool_messages("alpha?")
        citations = local_rag.CitationRegistry()

        result, _ = engine._forced_local_tool_call(messages, question="alpha?", citations=citations)
        messages.append({"role": "user", "content": engine._final_answer_instruction()})

        assert result["result_count"] == 1
        assert result["context_truncated"] is True
        assert engine._context_token_budget() == int(4096 * 0.60)
        assert local_rag.estimate_prompt_tokens(messages) <= engine._context_token_budget()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_web_search_skips_when_prompt_already_exceeds_input_context_budget(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    def fail_web_search(*args, **kwargs):
        raise AssertionError("web search should not be called")

    monkeypatch.setattr(local_rag, "web_search_duckduckgo_lite", fail_web_search)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            context_window=1024,
            context_token_fraction=0.60,
            progress_enabled=False,
        )
        messages = engine._tool_messages("alpha?")
        messages.append({"role": "tool", "tool_name": "search_local_context", "content": "x" * 4000})

        tools = engine._tool_definitions(include_web_search=engine._web_search_allowed(messages))
        _, result, text = engine._execute_tool_call(
            _tool_call(name="web_search", arguments={"query": "current alpha"}),
            question="alpha?",
            citations=local_rag.CitationRegistry(),
            messages=messages,
        )

        assert all(tool["function"]["name"] != "web_search" for tool in tools)
        assert result["result_count"] == 0
        assert "current prompt" in result["error"]
        assert text == result["error"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_repeated_local_tool_call_excludes_already_returned_chunks(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    class FakeStore:
        def exists(self):
            return True

        def count(self):
            return 2

        def search(self, vector, *, top_k):
            return [
                {"id": "a", "node_type": "chunk", "content": "alpha", "score": 1.0},
                {"id": "b", "node_type": "chunk", "content": "beta", "score": 0.9},
            ]

        def child_chunks(self, parent, *, limit):
            return []

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(working_dir=str(tmp_path), progress_enabled=False)
        engine.store = FakeStore()
        engine.record_count = 2
        citations = local_rag.CitationRegistry()

        first = engine._execute_tool_call(
            _tool_call(arguments={"query": "alpha?"}),
            question="alpha?",
            citations=citations,
        )[1]
        second = engine._execute_tool_call(
            _tool_call(arguments={"query": "alpha?"}),
            question="alpha?",
            citations=citations,
        )[1]

        assert [result["chunk_id"] for result in first["results"]] == ["a", "b"]
        assert second["results"] == []
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_local_query_engine_uses_configurable_system_prompt(monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    captured = {}

    def fake_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return {"message": {"content": ""}}

    monkeypatch.setattr(local_rag, "_ollama_chat", fake_chat)
    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    try:
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            web_search_enabled=False,
            system_prompt="Custom system. {web_instruction}",
        )
        list(engine._run_tool_rounds("alpha?"))

        assert captured["messages"][0]["content"] == "Custom system. Do not use web_search; it is disabled for this request."
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_web_search_duckduckgo_lite_parses_results(monkeypatch):
    import requests

    class FakeResponse:
        text = """
        <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpaper">Paper</a>
        <td class="result-snippet">Useful snippet</td>
        """

        def raise_for_status(self):
            pass

    calls = {}

    def fake_get(url, timeout, headers):
        calls["url"] = url
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    result = local_rag.web_search_duckduckgo_lite("query", max_results=3, timeout=4.0)

    assert calls["timeout"] == 4.0
    assert "q=query" in calls["url"]
    assert result["results"] == [
        {
            "title": "Paper",
            "url": "https://example.com/paper",
            "snippet": "Useful snippet",
            "provider": "duckduckgo_lite",
        }
    ]


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

        calls = []

        monkeypatch.setattr(
            local_rag,
            "_ollama_chat",
            _fake_local_tool_chat(
                calls,
                final_events=[
                    {"message": {"content": "<think>still thinking"}},
                    {"done": True, "done_reason": "length", "message": {"content": ""}},
                ],
            ),
        )
        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
            planner_enabled=False,
        )
        events = list(engine.ask_stream_events("alpha?"))

        assert events[0] == {"type": "notice", "text": "Planning retrieval tool calls..."}
        assert {"type": "thinking", "text": "still thinking"} in events
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

        calls = []

        monkeypatch.setattr(local_rag, "_ollama_chat", _fake_local_tool_chat(calls))
        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
            planner_enabled=False,
        )

        assert engine.ask("alpha?") == "answer [S1]"
        tool_messages = [message for message in calls[-1]["messages"] if message.get("role") == "tool"]
        assert "alpha leaf context" in tool_messages[0]["content"]
        assert "document summary only" not in tool_messages[0]["content"]
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


def test_local_indexer_reuses_unchanged_vectors(monkeypatch):
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    md_dir = tmp_path / "md"
    db_dir = tmp_path / "db"
    md_dir.mkdir(parents=True)
    try:
        md_dir.joinpath("doc.md").write_text("content", encoding="utf-8")
        reused_vector = [1.0] + [0.0] * 767
        changed_vector = [0.0, 1.0] + [0.0] * 766
        new_vector = [0.0, 0.0, 1.0] + [0.0] * 765

        LanceDBVectorStore(db_dir).write_records(
            [
                {
                    "id": "same",
                    "doc_id": "doc",
                    "parent_id": "",
                    "node_type": "chunk",
                    "file_path": "doc.md",
                    "chunk_index": 0,
                    "content": "same content",
                    "title": "Same",
                    "section_path": "Doc > Same",
                    "page_start": 1,
                    "page_end": 1,
                    "summary": "summary",
                    "tags": [],
                    "vector": reused_vector,
                },
                {
                    "id": "changed",
                    "doc_id": "doc",
                    "parent_id": "",
                    "node_type": "chunk",
                    "file_path": "doc.md",
                    "chunk_index": 1,
                    "content": "old content",
                    "title": "Changed",
                    "section_path": "Doc > Changed",
                    "page_start": 2,
                    "page_end": 2,
                    "summary": "summary",
                    "tags": [],
                    "vector": [0.5] + [0.0] * 767,
                },
            ],
            embedding_model="nomic-embed-text",
            embedding_dim=768,
        )

        calls = []

        class FakeEngine:
            def __init__(self, **kwargs):
                pass

            def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
                calls.append(list(texts))
                if list(texts) == ["embedding health check"]:
                    return np.asarray([[1.0] + [0.0] * 7], dtype=np.float32)
                vectors = []
                for text in texts:
                    if text == "changed content":
                        vectors.append(changed_vector)
                    elif text == "new content":
                        vectors.append(new_vector)
                    else:
                        raise AssertionError(f"Unexpected embedding request: {text}")
                return np.asarray(vectors, dtype=np.float32)

        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)

        def fake_build_section_records(*args, **kwargs):
            base = {
                "doc_id": "doc",
                "parent_id": "",
                "node_type": "chunk",
                "file_path": "doc.md",
                "summary": "summary",
                "tags": [],
            }
            return [
                {
                    **base,
                    "id": "same",
                    "chunk_index": 0,
                    "content": "same content",
                    "title": "Same",
                    "section_path": "Doc > Same",
                    "page_start": 1,
                    "page_end": 1,
                },
                {
                    **base,
                    "id": "changed",
                    "chunk_index": 1,
                    "content": "changed content",
                    "title": "Changed",
                    "section_path": "Doc > Changed",
                    "page_start": 2,
                    "page_end": 2,
                },
                {
                    **base,
                    "id": "new",
                    "chunk_index": 2,
                    "content": "new content",
                    "title": "New",
                    "section_path": "Doc > New",
                    "page_start": 3,
                    "page_end": 3,
                },
            ]

        monkeypatch.setattr(local_rag, "build_section_records", fake_build_section_records)

        indexer = local_rag.LocalVectorIndexer(working_dir=str(db_dir), embedding_batch_size=8, progress_enabled=False)
        indexer.index_markdown(str(md_dir))

        assert calls == [["embedding health check"], ["changed content", "new content"]]
        store = LanceDBVectorStore(db_dir)
        assert store.count() == 3
        assert store.get_record("same")["vector"] == reused_vector
        assert store.get_record("changed")["vector"] == changed_vector
        manifest = json.loads(db_dir.joinpath(local_rag.INDEX_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["reused_records"] == 1
        assert manifest["embedded_records"] == 2
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


def test_parse_query_list_handles_json_array():
    assert local_rag._parse_query_list('["alpha", "beta"]', max_queries=5) == ["alpha", "beta"]


def test_parse_query_list_strips_json_fences():
    assert local_rag._parse_query_list("```json\n[\"alpha\", \"beta\"]\n```", max_queries=5) == [
        "alpha",
        "beta",
    ]


def test_parse_query_list_falls_back_to_lines():
    assert local_rag._parse_query_list("alpha\nbeta\n", max_queries=5) == ["alpha", "beta"]


def test_parse_query_list_dedupes_and_caps():
    parsed = local_rag._parse_query_list('["alpha", "alpha", "beta", "gamma"]', max_queries=2)
    assert parsed == ["alpha", "beta"]


def test_parse_query_list_empty_returns_empty():
    assert local_rag._parse_query_list("", max_queries=5) == []


def test_generate_search_queries_falls_back_to_question_on_error(monkeypatch):
    def raise_chat(**kwargs):
        raise RuntimeError("planner model unavailable")

    monkeypatch.setattr(local_rag, "_ollama_chat", raise_chat)

    queries = local_rag.generate_search_queries(
        "What is ridge regression?",
        model="qwen2.5:1.5b",
        max_queries=3,
        timeout=5.0,
    )

    assert queries == ["What is ridge regression?"]


def test_generate_search_queries_prepends_original_question(monkeypatch):
    def fake_chat(**kwargs):
        return {"message": {"content": '["ridge regularization", "L2 penalty"]'}}

    monkeypatch.setattr(local_rag, "_ollama_chat", fake_chat)

    queries = local_rag.generate_search_queries(
        "What is ridge regression?",
        model="qwen2.5:1.5b",
        max_queries=3,
        timeout=5.0,
    )

    assert queries[0] == "What is ridge regression?"
    assert "ridge regularization" in queries
    assert "L2 penalty" in queries


def test_generate_search_queries_returns_only_question_when_max_zero(monkeypatch):
    queries = local_rag.generate_search_queries(
        "What is ridge regression?",
        model="qwen2.5:1.5b",
        max_queries=0,
        timeout=5.0,
    )
    assert queries == ["What is ridge regression?"]


def test_eager_local_tool_call_merges_multi_query_results(monkeypatch):
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
                # Both query embeddings point at the alpha chunk so retrieval
                # is deterministic; the merge must still dedupe it.
                return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)
        # Force the planner to emit two queries; both resolve to the same chunk
        # via the FakeEngine embedding, so we can assert dedup happens.
        monkeypatch.setattr(
            local_rag,
            "generate_search_queries",
            lambda *args, **kwargs: ["alpha?", "alpha variant"],
        )

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
            planner_model="qwen2.5:1.5b",
            planner_max_queries=2,
        )
        citations = local_rag.CitationRegistry()
        messages = engine._tool_messages("alpha?")

        outcome = engine._eager_local_tool_call(
            question="alpha?",
            citations=citations,
            messages=messages,
        )

        assert outcome is not None
        result, text = outcome
        assert result["tool"] == "search_local_context"
        assert result["planner_queries"] == ["alpha?", "alpha variant"]
        # The alpha chunk must appear at most once despite two queries.
        chunk_ids = [block["chunk_id"] for block in result["results"]]
        assert chunk_ids.count("a:0") == 1
        assert result["result_count"] >= 1
        # An assistant tool_calls message and a tool result are appended so the
        # loop sees the search as already completed.
        roles = [message.get("role") for message in messages]
        assert "assistant" in roles
        assert "tool" in roles
        assert "Retrieved" in text
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_eager_local_tool_call_skipped_when_disabled():
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
            ],
        )

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            planner_enabled=False,
        )
        citations = local_rag.CitationRegistry()
        messages = engine._tool_messages("alpha?")

        assert engine._eager_local_tool_call(
            question="alpha?",
            citations=citations,
            messages=messages,
        ) is None
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_eager_retrieval_runs_in_tool_rounds_and_prefetches_context(monkeypatch):
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
            ],
        )

        class FakeEngine:
            def __init__(self, **kwargs):
                pass

            def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
                return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)
        monkeypatch.setattr(
            local_rag,
            "generate_search_queries",
            lambda *args, **kwargs: ["alpha?"],
        )
        # The model never calls tools itself; the eager pre-fetch must still
        # supply context so the loop reaches the final-answer instruction.
        monkeypatch.setattr(
            local_rag,
            "_ollama_chat",
            _fake_local_tool_chat([], final_events=[{"message": {"content": "answer [S1]"}}]),
        )

        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            model="gemma4",
            planner_enabled=True,
            planner_max_queries=1,
        )

        events = list(engine.ask_stream_events("alpha?"))
        # With the planner on and a populated index, the opening notice changes.
        assert events[0] == {"type": "notice", "text": "Searching local context..."}
        tool_call_events = [e for e in events if e.get("type") == "tool_call"]
        assert tool_call_events
        assert tool_call_events[0]["tool"] == "search_local_context"
        tool_results = [e for e in events if e.get("type") == "tool_result"]
        assert tool_results
        assert tool_results[0]["result"]["planner_queries"] == ["alpha?"]
        assert any(e.get("type") == "answer" and "answer [S1]" in e["text"] for e in events)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_eager_prompt_used_when_planner_active_and_index_populated(monkeypatch):
    """With the planner on and a populated index, the eager system prompt is
    selected automatically (no custom prompt configured)."""
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
            ],
        )
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            planner_enabled=True,
        )
        assert engine._eager_retrieval_active() is True

        system_content = engine._tool_messages("alpha?")[0]["content"]

        assert "has already been retrieved" in system_content
        assert "you do not need to call search_local_context before answering" in system_content
        # The non-eager mandate must not appear.
        assert "must call" not in system_content
        # The {web_instruction} placeholder must have been filled.
        assert "{web_instruction}" not in system_content
        assert "web_search" in system_content
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_non_eager_prompt_used_when_planner_disabled(monkeypatch):
    """With the planner off, the original mandate-to-search prompt is used."""
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
            ],
        )
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            planner_enabled=False,
        )
        assert engine._eager_retrieval_active() is False

        system_content = engine._tool_messages("alpha?")[0]["content"]

        assert "must call" in system_content
        assert "at least once" in system_content
        assert "has already been retrieved" not in system_content
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_non_eager_prompt_used_when_index_empty():
    """Eager retrieval can't run on an empty index, so the non-eager prompt
    (mandating the model search itself) is selected even with the planner on."""
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_local_rag_{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            planner_enabled=True,
        )
        assert engine._eager_retrieval_active() is False

        system_content = engine._tool_messages("alpha?")[0]["content"]

        assert "must call" in system_content
        assert "has already been retrieved" not in system_content
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_custom_prompt_gets_eager_suffix_when_planner_active():
    """A user-supplied system_prompt is preserved verbatim, but in eager mode
    the steering suffix is appended so the model still treats pre-fetched
    context as already provided."""
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
            ],
        )
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            planner_enabled=True,
            system_prompt="You are a notes helper. {web_instruction}",
        )

        system_content = engine._tool_messages("alpha?")[0]["content"]

        # The custom prompt text must be preserved at the start.
        assert system_content.startswith("You are a notes helper.")
        # The {web_instruction} placeholder must have been filled.
        assert "{web_instruction}" not in system_content
        # The eager steering suffix must be appended.
        assert local_rag.EAGER_CONTEXT_SUFFIX.strip() in system_content
        assert "has already been retrieved" in system_content
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_custom_prompt_used_verbatim_when_planner_disabled():
    """With the planner off, a custom system_prompt is used verbatim with no
    eager suffix appended."""
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
            ],
        )
        engine = local_rag.LocalQueryEngine(
            working_dir=str(tmp_path),
            progress_enabled=False,
            planner_enabled=False,
            system_prompt="You are a notes helper. {web_instruction}",
        )

        system_content = engine._tool_messages("alpha?")[0]["content"]

        assert system_content.startswith("You are a notes helper.")
        assert local_rag.EAGER_CONTEXT_SUFFIX.strip() not in system_content
        # web_instruction placeholder still filled.
        assert "{web_instruction}" not in system_content
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
