import numpy as np
from src.embeddings import EmbeddingEngine


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
