import src.query as query


def test_query_engine_delegates_to_local_query_engine(monkeypatch):
    calls = {}

    class FakeLocalQueryEngine:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def ask(self, question):
            calls["question"] = question
            return "answer"

    monkeypatch.setattr("src.local_rag.LocalQueryEngine", FakeLocalQueryEngine)

    engine = query.QueryEngine(
        working_dir="db",
        model="gemma4",
        embedding_model="nomic-embed-text",
        embedding_batch_size=4,
        embedding_timeout=12.0,
        llm_num_predict=256,
        progress_enabled=False,
    )

    assert engine.ask("What is aero balance?") == "answer"
    assert calls == {
        "init": {
            "working_dir": "db",
            "model": "gemma4",
            "embedding_model": "nomic-embed-text",
            "embedding_batch_size": 4,
            "embedding_timeout": 12.0,
            "num_predict": 256,
            "progress_enabled": False,
        },
        "question": "What is aero balance?",
    }
