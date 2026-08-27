import numpy as np
import pytest

from src.embeddings import EmbeddingEngine


# These tests exercise the dormant local-Ollama embedding transport (they patch
# the `_ollama_api` seam). Force that backend so the seam is on the active path;
# the SoCLAaS API path has its own coverage in tests/test_llm_api.py.
@pytest.fixture(autouse=True)
def _force_ollama_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")


def test_ollama_embeddings_use_local_ollama_api(monkeypatch):
    engine = EmbeddingEngine(model_name="nomic-embed-text")
    monkeypatch.setattr(
        engine,
        "_ollama_api",
        lambda path, payload, host=None: {
            "embeddings": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        },
    )

    vectors = engine.get_mrl_embeddings(["alpha", "beta"], truncate_dim=4)

    assert vectors.shape == (2, 4)
    assert np.allclose(vectors[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(vectors[1], [0.0, 1.0, 0.0, 0.0])


def test_ollama_embeddings_are_batched(monkeypatch):
    calls = []
    engine = EmbeddingEngine(
        model_name="nomic-embed-text",
        ollama_batch_size=2,
    )

    def fake_ollama_api(path, payload, host=None):
        input_value = payload["input"]
        input_list = [input_value] if isinstance(input_value, str) else list(input_value)
        calls.append(input_list)
        embeddings = []
        for index, _ in enumerate(input_list):
            embeddings.append([float(len(calls)), float(index + 1)])
        return {"embeddings": embeddings}

    monkeypatch.setattr(engine, "_ollama_api", fake_ollama_api)

    vectors = engine.get_mrl_embeddings(["a", "b", "c", "d", "e"], truncate_dim=2)

    assert [len(call) for call in calls] == [2, 2, 1]
    assert vectors.shape == (5, 2)


def test_ollama_single_embedding_uses_string_input(monkeypatch):
    calls = []
    engine = EmbeddingEngine(
        model_name="nomic-embed-text",
        ollama_batch_size=1,
    )

    def fake_ollama_api(path, payload, host=None):
        calls.append(payload)
        assert isinstance(payload["input"], str)
        return {"embeddings": [[1.0, 0.0]]}

    monkeypatch.setattr(engine, "_ollama_api", fake_ollama_api)

    vectors = engine.get_mrl_embeddings(["query text"], truncate_dim=2)

    assert calls[0]["input"] == "search_document: query text"
    assert vectors.shape == (1, 2)


def test_multi_repolla_hosts_round_robin_batches(monkeypatch):
    """With OLLAMA_EMBED_HOSTS set, batches are round-robined across replicas."""
    monkeypatch.setenv("OLLAMA_EMBED_HOSTS", "http://gpu-a:11434, http://gpu-b:11434")
    monkeypatch.setenv("OLLAMA_EMBED_CONCURRENCY", "1")
    try:
        engine = EmbeddingEngine(
            model_name="nomic-embed-text",
            ollama_batch_size=1,
        )
        # Record which host each input text was sent to. Completion order is
        # nondeterministic under parallelism, so we check the input->host map,
        # not the call order.
        input_to_host: dict[str, str] = {}

        def fake_ollama_api(path, payload, host=None):
            input_value = payload["input"]
            input_to_host[input_value] = host
            return {"embeddings": [[1.0, 0.0]]}

        monkeypatch.setattr(engine, "_ollama_api", fake_ollama_api)

        engine.get_mrl_embeddings(["a", "b", "c", "d"], truncate_dim=2)

        # Batch assignment is by batch index: batch 1->gpu-a, 2->gpu-b, 3->gpu-a,
        # 4->gpu-b. The input the fake sees carries the prefix, so key on the
        # trailing char.
        expected = {
            "a": "http://gpu-a:11434",
            "b": "http://gpu-b:11434",
            "c": "http://gpu-a:11434",
            "d": "http://gpu-b:11434",
        }
        actual = {key[-1]: host for key, host in input_to_host.items()}
        assert actual == expected, f"round-robin assignment mismatch: {actual}"
    finally:
        monkeypatch.delenv("OLLAMA_EMBED_HOSTS", raising=False)
        monkeypatch.delenv("OLLAMA_EMBED_CONCURRENCY", raising=False)


def test_parallel_batches_preserve_order(monkeypatch):
    """Parallel dispatch (concurrency>1) must return vectors in input order."""
    monkeypatch.setenv("OLLAMA_EMBED_CONCURRENCY", "4")
    try:
        engine = EmbeddingEngine(
            model_name="nomic-embed-text",
            ollama_batch_size=1,
        )
        # Each call returns a DISTINCT unit vector so that after L2-normalization
        # the rows are still distinguishable and order can be checked. We vary the
        # per-call latency so the thread pool interleaves -- if ordering were
        # broken by concurrency, this test would catch it.
        import time

        def fake_ollama_api(path, payload, host=None):
            text = payload["input"]
            # Use the last char of the prefixed input as the identity axis.
            axis = ord(text[-1]) % 8
            vec = [0.0] * 8
            vec[axis] = 1.0
            time.sleep(0.01 * ((ord(text[-1]) % 5) + 1))
            return {"embeddings": [vec]}

        monkeypatch.setattr(engine, "_ollama_api", fake_ollama_api)

        texts = ["a", "b", "c", "d", "e", "f"]
        vectors = engine.get_mrl_embeddings(texts, truncate_dim=8)

        assert vectors.shape == (6, 8)
        # Each row should be a unit vector on the axis derived from its own
        # last char (after the prefix). Confirms ordering survived the pool.
        for i, text in enumerate(texts):
            expected_axis = ord(text[-1]) % 8
            # The prefix "search_document: " ends in a space, so the input the
            # fake sees is "search_document: a" -- its last char is the text's.
            row = vectors[i]
            assert np.isclose(np.sum(row), 1.0), f"row {i} not unit: {row}"
            assert np.isclose(np.argmax(row), expected_axis), (
                f"row {i} axis {np.argmax(row)} != {expected_axis} (text={text!r})"
            )
    finally:
        monkeypatch.delenv("OLLAMA_EMBED_CONCURRENCY", raising=False)


def test_single_host_default_is_serial(monkeypatch):
    """With default env (one host, concurrency=1) batches run serially."""
    from src.embeddings import _resolve_ollama_hosts, _resolve_embed_concurrency

    # Default env has no OLLAMA_EMBED_HOSTS -> single host list.
    monkeypatch.delenv("OLLAMA_EMBED_HOSTS", raising=False)
    monkeypatch.delenv("OLLAMA_EMBED_CONCURRENCY", raising=False)
    assert _resolve_ollama_hosts() == ["http://127.0.0.1:11434"]
    assert _resolve_embed_concurrency() == 1


def test_multi_replica_forces_concurrency(monkeypatch):
    """Even with concurrency=1, multiple hosts raise effective parallelism."""
    from src.embeddings import _resolve_ollama_hosts, _resolve_embed_concurrency

    monkeypatch.setenv("OLLAMA_EMBED_HOSTS", "http://a:1,http://b:1")
    monkeypatch.delenv("OLLAMA_EMBED_CONCURRENCY", raising=False)
    try:
        hosts = _resolve_ollama_hosts()
        concurrency = _resolve_embed_concurrency()
        effective = max(concurrency, len(hosts))
        assert len(hosts) == 2
        assert effective == 2
    finally:
        monkeypatch.delenv("OLLAMA_EMBED_HOSTS", raising=False)


def test_dedup_hosts_preserves_order(monkeypatch):
    """Duplicate hosts in OLLAMA_EMBED_HOSTS are de-duplicated in order."""
    from src.embeddings import _resolve_ollama_hosts

    monkeypatch.setenv("OLLAMA_EMBED_HOSTS", "http://a:1, http://a:1, http://b:1")
    try:
        assert _resolve_ollama_hosts() == ["http://a:1", "http://b:1"]
    finally:
        monkeypatch.delenv("OLLAMA_EMBED_HOSTS", raising=False)


# --- per-modality backend selection ([embeddings].backend) --------------------
#
# The embeddings transport is selected independently of the chat/vision backend
# so the default split deployment works: SoCLAaS chat/vision + locally hosted
# nomic-embed-text embeddings.


def test_embeddings_backend_default_is_local_ollama(monkeypatch):
    from src.embeddings import embeddings_use_soclaas, resolve_embeddings_backend

    monkeypatch.delenv("EMBEDDINGS_BACKEND", raising=False)
    monkeypatch.delenv("RAG_PIPELINE_CONFIG", raising=False)
    # Defaults: [embeddings].backend = "ollama" even with chat on soclaas.
    monkeypatch.setenv("LLM_BACKEND", "soclaas")
    assert resolve_embeddings_backend() == "ollama"
    assert embeddings_use_soclaas() is False


def test_embeddings_backend_empty_follows_chat_backend(monkeypatch, tmp_path):
    from src.embeddings import embeddings_use_soclaas, resolve_embeddings_backend

    monkeypatch.setenv("LLM_BACKEND", "soclaas")
    # Without a SoCLAaS key the chat backend itself falls back to ollama, and
    # embeddings inherit that fallback (cloud unavailable -> everything local).
    monkeypatch.delenv("SOCLAAS_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = tmp_path / "split.toml"
    cfg.write_text('[embeddings]\nbackend = ""\n', encoding="utf-8")
    monkeypatch.setenv("RAG_PIPELINE_CONFIG", str(cfg))
    assert resolve_embeddings_backend() == ""
    assert embeddings_use_soclaas() is False

    # With a key the soclaas chat selection is real and embeddings inherit it.
    monkeypatch.setenv("SOCLAAS_API_KEY", "sk-test")
    assert embeddings_use_soclaas() is True


def test_chat_backend_falls_back_to_ollama_without_key(monkeypatch, tmp_path):
    """Cloud unavailable (no key) -> effective backend is local Ollama."""
    from src import llm_api

    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_STRICT_BACKEND", raising=False)
    monkeypatch.delenv("SOCLAAS_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = tmp_path / "fb.toml"
    cfg.write_text(
        '[llm_api]\nbackend = "soclaas"\napi_key = ""\n', encoding="utf-8"
    )
    monkeypatch.setenv("RAG_PIPELINE_CONFIG", str(cfg))
    assert llm_api.active_backend() == "ollama"
    assert llm_api.is_soclaas() is False

    # LLM_STRICT_BACKEND opts out of the fallback (loud failure instead).
    monkeypatch.setenv("LLM_STRICT_BACKEND", "1")
    assert llm_api.active_backend() == "soclaas"

    # A configured key keeps soclaas active.
    monkeypatch.delenv("LLM_STRICT_BACKEND", raising=False)
    monkeypatch.setenv("SOCLAAS_API_KEY", "sk-test")
    assert llm_api.active_backend() == "soclaas"


def test_embeddings_backend_env_overrides_config(monkeypatch, tmp_path):
    from src.embeddings import resolve_embeddings_backend

    cfg = tmp_path / "split.toml"
    cfg.write_text('[embeddings]\nbackend = "soclaas"\n', encoding="utf-8")
    monkeypatch.setenv("RAG_PIPELINE_CONFIG", str(cfg))
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "ollama")
    assert resolve_embeddings_backend() == "ollama"


def test_embeddings_backend_invalid_env_raises(monkeypatch):
    from src.embeddings import resolve_embeddings_backend

    monkeypatch.setenv("EMBEDDINGS_BACKEND", "gpu")
    with pytest.raises(ValueError, match="EMBEDDINGS_BACKEND"):
        resolve_embeddings_backend()


def test_engine_routes_to_ollama_with_chat_on_soclaas(monkeypatch):
    """The default split: chat/vision on SoCLAaS, embeddings on local Ollama."""
    monkeypatch.delenv("EMBEDDINGS_BACKEND", raising=False)
    monkeypatch.delenv("RAG_PIPELINE_CONFIG", raising=False)
    monkeypatch.setenv("LLM_BACKEND", "soclaas")

    engine = EmbeddingEngine(model_name="nomic-embed-text")
    assert engine._backend == "ollama"
    monkeypatch.setattr(
        engine,
        "_ollama_api",
        lambda path, payload, host=None: {"embeddings": [[1.0, 0.0]]},
    )
    vectors = engine.get_mrl_embeddings(["alpha"], truncate_dim=2)
    assert vectors.shape == (1, 2)


def test_engine_routes_to_soclaas_with_chat_on_ollama(monkeypatch):
    """The mirror split: offline chat via Ollama, embeddings via SoCLAaS."""
    from src import llm_api

    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "soclaas")

    monkeypatch.setattr(
        llm_api,
        "soclaas_embed",
        lambda *, model, input_texts, timeout=None: [[1.0, 0.0] for _ in input_texts],
    )
    engine = EmbeddingEngine(model_name="bge-m3")
    assert engine._backend == "soclaas"
    vectors = engine.get_mrl_embeddings(["alpha"], truncate_dim=2, prefix="")
    assert vectors.shape == (1, 2)
