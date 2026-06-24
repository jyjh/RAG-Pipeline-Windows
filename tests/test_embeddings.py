import numpy as np
from src.embeddings import EmbeddingEngine


def test_hash_embeddings_are_deterministic_and_normalized():
    engine = EmbeddingEngine(backend="hash")

    first = engine.get_mrl_embeddings(["alpha beta"], truncate_dim=16)
    second = engine.get_mrl_embeddings(["alpha beta"], truncate_dim=16)

    assert first.shape == (1, 16)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0)


def test_hash_embeddings_separate_texts():
    engine = EmbeddingEngine(backend="hash")

    vectors = engine.get_mrl_embeddings(["alpha beta", "gamma delta"], truncate_dim=16)

    assert vectors.shape == (2, 16)
    assert not np.allclose(vectors[0], vectors[1])


def test_ollama_embeddings_use_local_ollama_backend(monkeypatch):
    engine = EmbeddingEngine(backend="ollama", model_name="nomic-embed-text")
    monkeypatch.setattr(
        engine,
        "_ollama_api",
        lambda path, payload: {
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
        backend="ollama",
        model_name="nomic-embed-text",
        ollama_batch_size=2,
    )

    def fake_ollama_api(path, payload):
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
        backend="ollama",
        model_name="nomic-embed-text",
        ollama_batch_size=1,
    )

    def fake_ollama_api(path, payload):
        calls.append(payload)
        assert isinstance(payload["input"], str)
        return {"embeddings": [[1.0, 0.0]]}

    monkeypatch.setattr(engine, "_ollama_api", fake_ollama_api)

    vectors = engine.get_mrl_embeddings(["query text"], truncate_dim=2)

    assert calls[0]["input"] == "search_document: query text"
    assert vectors.shape == (1, 2)
