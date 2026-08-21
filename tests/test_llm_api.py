"""Tests for the SoCLAaS API client + backend selector (src/llm_api.py) and the
soclaas/ollama dispatcher in src/local_rag.py / src/embeddings.py / src/ingestion.py.

The dormant local-Ollama transport has its own coverage in test_embeddings.py /
test_local_rag.py (which force LLM_BACKEND=ollama). These tests pin the new
primary SoCLAaS path: backend selection, key resolution, prefix resolution,
option mapping, HTTP request/response parsing (chat/embed/vision), retry,
health probe, the chat dispatcher, and the embeddings + vision backend branch.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src import llm_api


@pytest.fixture(autouse=True)
def _soclaas_env(monkeypatch):
    # config.example.toml sets [llm_api].backend = "soclaas"; ensure env never
    # overrides it, and provide a key so require_api_key() succeeds for HTTP tests.
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    # Embeddings have their own selector ([embeddings].backend, default
    # "ollama"); isolate it from the developer's environment here.
    monkeypatch.delenv("EMBEDDINGS_BACKEND", raising=False)
    monkeypatch.setenv("SOCLAAS_API_KEY", "sk-test-key")


# --------------------------------------------------------------------------- #
# Backend selection + key resolution
# --------------------------------------------------------------------------- #


def test_active_backend_env_overrides_config(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    assert llm_api.active_backend() == "ollama"
    assert llm_api.is_soclaas() is False


def test_active_backend_defaults_to_soclaas():
    assert llm_api.active_backend() == "soclaas"
    assert llm_api.is_soclaas() is True


def test_resolve_api_key_prefers_named_env(monkeypatch):
    monkeypatch.setenv("SOCLAAS_API_KEY", "sk-from-soclaas-env")
    monkeypatch.setenv("LLM_API_KEY", "sk-from-llm-env")
    assert llm_api.resolve_api_key() == "sk-from-soclaas-env"


def test_resolve_api_key_falls_back_to_llm_api_key(monkeypatch):
    monkeypatch.delenv("SOCLAAS_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-from-llm-env")
    assert llm_api.resolve_api_key() == "sk-from-llm-env"


def test_require_api_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SOCLAAS_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    # config.example.toml has api_key = ""
    with pytest.raises(RuntimeError, match="SoCLAaS API key not configured"):
        llm_api.require_api_key()


# --------------------------------------------------------------------------- #
# Option mapping + prefix resolution
# --------------------------------------------------------------------------- #


def test_map_chat_options_translates_num_predict_and_drops_num_ctx():
    mapped = llm_api.map_chat_options(
        {"temperature": 0.4, "top_k": 40, "num_predict": 1024, "num_ctx": 8192, "top_p": 0.9}
    )
    assert mapped == {"temperature": 0.4, "top_k": 40, "max_tokens": 1024, "top_p": 0.9}
    assert "num_ctx" not in mapped  # OpenAI servers reject it


def test_resolve_embedding_prefix_instruction_free_for_bge():
    assert llm_api.resolve_embedding_prefix("bge-m3", "doc") == ""
    assert llm_api.resolve_embedding_prefix("bge-m3", "query") == ""


def test_resolve_embedding_prefix_applies_for_nomic_and_e5():
    assert llm_api.resolve_embedding_prefix("nomic-embed-text", "doc") == "search_document: "
    assert llm_api.resolve_embedding_prefix("nomic-embed-text", "query") == "search_query: "
    assert llm_api.resolve_embedding_prefix("intfloat/multilingual-e5-large", "query") == "search_query: "


# --------------------------------------------------------------------------- #
# HTTP request/response parsing (chat / embed / vision) with mocked transport
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """Stand-in for the urllib response object returned by llm_api._request."""

    def __init__(self, body_bytes: bytes, *, lines: list[bytes] | None = None):
        self._body = body_bytes
        self._lines = lines or []
        self.status = 200

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_request(monkeypatch, response_factory):
    """Replace llm_api._request with a callable returning response_factory(url, body)."""
    captured = {}

    def fake_request(url, *, headers, body=None, method="POST", timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        captured["method"] = method
        return response_factory(url, body)

    monkeypatch.setattr(llm_api, "_request", fake_request)
    return captured


def test_soclaas_chat_once_parses_openai_message(monkeypatch):
    payload = {"choices": [{"message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}]}

    cap = _patch_request(
        monkeypatch,
        lambda url, body: _FakeResponse(json.dumps(payload).encode()),
    )
    resp = llm_api.soclaas_chat_once(
        model="gemma4:26b",
        messages=[{"role": "user", "content": "hi"}],
        options={"temperature": 0.3, "num_predict": 50, "num_ctx": 8192},
        tools=[{"type": "function", "function": {"name": "x"}}],
    )
    assert resp == payload
    # num_predict -> max_tokens, num_ctx dropped, tools + stream forwarded.
    assert cap["body"] == {
        "model": "gemma4:26b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 50,
        "tools": [{"type": "function", "function": {"name": "x"}}],
    }
    assert cap["headers"]["Authorization"] == "Bearer sk-test-key"


def test_soclaas_chat_stream_parses_sse(monkeypatch):
    # Two content chunks then [DONE]; plus a non-data line that must be skipped.
    lines = [
        b": ping\n",
        b'data: {"choices": [{"delta": {"content": "hel"}, "finish_reason": null}]}\n',
        b'data: {"choices": [{"delta": {"content": "lo"}, "finish_reason": null}]}\n',
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n',
        b"data: [DONE]\n",
        b'data: {"choices": [{"delta": {"content": "after-done-should-not-emit"}}]}\n',
    ]
    _patch_request(monkeypatch, lambda url, body: _FakeResponse(b"", lines=lines))
    chunks = list(
        llm_api.soclaas_chat_stream(model="gemma4:26b", messages=[{"role": "user", "content": "hi"}])
    )
    contents = [c["choices"][0]["delta"].get("content") for c in chunks]
    assert contents == ["hel", "lo", None]
    assert len(chunks) == 3  # [DONE] terminates and is not yielded


def test_soclaas_embed_preserves_input_order(monkeypatch):
    # OpenAI returns embedding objects unordered by index.
    data = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
            {"index": 2, "embedding": [0.0, 0.0]},
        ]
    }
    _patch_request(monkeypatch, lambda url, body: _FakeResponse(json.dumps(data).encode()))
    vectors = llm_api.soclaas_embed(model="bge-m3", input_texts=["a", "b", "c"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]


def test_soclaas_embed_empty_input_skips_request(monkeypatch):
    called = {"n": 0}

    def factory(url, body):
        called["n"] += 1
        return _FakeResponse(b"{}")

    _patch_request(monkeypatch, factory)
    assert llm_api.soclaas_embed(model="bge-m3", input_texts=[]) == []
    assert called["n"] == 0


def test_soclaas_embed_raises_on_count_mismatch(monkeypatch):
    data = {"data": [{"index": 0, "embedding": [1.0]}]}  # 1 for 2 inputs
    _patch_request(monkeypatch, lambda url, body: _FakeResponse(json.dumps(data).encode()))
    with pytest.raises(llm_api.SoclaasError, match="2 input"):
        llm_api.soclaas_embed(model="bge-m3", input_texts=["a", "b"])


def test_soclaas_vision_builds_image_url_parts(monkeypatch):
    captured = {}

    def factory(url, body):
        captured["body"] = body
        return _FakeResponse(
            json.dumps({"choices": [{"message": {"content": "a torque chart"}}]}).encode()
        )

    _patch_request(monkeypatch, factory)
    text = llm_api.soclaas_vision(
        model="qwen3-vl:32b", prompt="describe", images_b64=["QUJD"], media_type="image/jpeg"
    )
    assert text == "a torque chart"
    content = captured["body"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
    }


# --------------------------------------------------------------------------- #
# Retry + probe
# --------------------------------------------------------------------------- #


def test_with_retry_retries_transient_then_raises(monkeypatch):
    monkeypatch.setenv("RAG_PIPELINE_CONFIG", "config.example.toml")
    attempts = {"n": 0}

    def always_transient():
        attempts["n"] += 1
        raise llm_api.SoclaasError("SoCLAaS returned HTTP 503 at x: busy")

    monkeypatch.setattr(llm_api.time, "sleep", lambda s: None)
    with pytest.raises(llm_api.SoclaasError, match="HTTP 503"):
        llm_api._with_retry(always_transient, description="t")
    assert attempts["n"] == 3  # retries config (3) bounds the attempts


def test_with_retry_does_not_retry_4xx(monkeypatch):
    attempts = {"n": 0}

    def auth_error():
        attempts["n"] += 1
        raise llm_api.SoclaasError("SoCLAaS returned HTTP 401 at x: bad key")

    monkeypatch.setattr(llm_api.time, "sleep", lambda s: None)
    with pytest.raises(llm_api.SoclaasError, match="HTTP 401"):
        llm_api._with_retry(auth_error, description="t")
    assert attempts["n"] == 1


def test_probe_soclaas_reports_unconfigured_without_key(monkeypatch):
    monkeypatch.delenv("SOCLAAS_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    ok, detail = llm_api.probe_soclaas(timeout=1)
    assert ok is False
    assert "api key" in detail


# --------------------------------------------------------------------------- #
# Dispatcher integration: chat (local_rag), embeddings, vision
# --------------------------------------------------------------------------- #


def test_llm_chat_dispatcher_routes_to_soclaas(monkeypatch):
    import src.local_rag as local_rag

    seen = {}

    def fake_once(*, model, messages, options=None, tools=None, timeout=None):
        seen["once"] = {"model": model, "options": options}
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(llm_api, "soclaas_chat_once", fake_once)
    resp = local_rag._llm_chat(
        model="gemma4:26b",
        messages=[{"role": "user", "content": "hi"}],
        options={"temperature": 0.3, "num_predict": 10},
        stream=False,
    )
    assert seen["once"]["model"] == "gemma4:26b"
    assert local_rag._ollama_response_content(resp) == "ok"


def test_embedding_engine_uses_soclaas_path(monkeypatch):
    from src.embeddings import EmbeddingEngine

    # The default embeddings backend is local Ollama; opt this transport in
    # explicitly (the split-selection matrix is covered in test_embeddings.py).
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "soclaas")

    seen = {"model": None, "texts": []}

    def fake_embed(*, model, input_texts, timeout=None):
        seen["model"] = model
        seen["texts"].extend(list(input_texts))
        return [[0.1] * 1024 for _ in input_texts]

    monkeypatch.setattr(llm_api, "soclaas_embed", fake_embed)
    eng = EmbeddingEngine(model_name="bge-m3", ollama_batch_size=2, ollama_timeout=10)
    vecs = eng.get_mrl_embeddings(["a", "b", "c"], truncate_dim=1024, prefix="")
    assert vecs.shape == (3, 1024)
    assert seen["model"] == "bge-m3"
    assert seen["texts"] == ["a", "b", "c"]  # batching must not lose/reorder inputs
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_ingestion_generate_routes_vision_to_soclaas(monkeypatch):
    import base64

    from src.ingestion import _ollama_generate

    seen = {}

    def fake_vision(*, model, prompt, images_b64, options=None, timeout=None, **_kw):
        seen["model"] = model
        seen["prompt"] = prompt
        seen["images"] = images_b64
        return "figure description"

    monkeypatch.setattr(llm_api, "soclaas_vision", fake_vision)
    # Warm-up call is a no-op on the soclaas path.
    assert _ollama_generate(model="qwen3-vl:32b", prompt="Hello", keep_alive="30m").response == ""
    # Real vision call routes to soclaas_vision.
    resp = _ollama_generate(
        model="qwen3-vl:32b",
        prompt="describe",
        images=[base64.b64encode(b"img").decode()],
        options={"num_ctx": 8192},
    )
    assert resp.response == "figure description"
    assert seen["model"] == "qwen3-vl:32b"
    assert len(seen["images"]) == 1


# --------------------------------------------------------------------------- #
# Embedding-dimension mismatch guard (the mandatory re-index path)
# --------------------------------------------------------------------------- #


def test_query_engine_raises_on_dim_mismatch(monkeypatch, tmp_path):
    import src.local_rag as local_rag
    from src.vector_store import LanceDBVectorStore

    # Build a 3-d index; the configured dim (config.example.toml) is 1024.
    store = LanceDBVectorStore(str(tmp_path))
    store.write_records(
        [
            {
                "id": "x:0",
                "doc_id": "x",
                "node_type": "chunk",
                "file_path": "x.md",
                "chunk_index": 0,
                "content": "alpha",
                "vector": [1.0, 0.0, 0.0],
            }
        ],
        embedding_model="nomic-embed-text",
        embedding_dim=3,
    )

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def get_mrl_embeddings(self, texts, truncate_dim=768, prefix=""):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr("src.embeddings.EmbeddingEngine", FakeEngine)
    engine = local_rag.LocalQueryEngine(
        working_dir=str(tmp_path), progress_enabled=False, planner_enabled=False
    )
    with pytest.raises(RuntimeError, match="embedding dimension"):
        engine._ensure_compatible_dim()
