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
        llm_timeout=45.0,
        temperature=0.7,
        sampler_top_k=24,
        context_window=4096,
        retrieval_candidate_k=55,
        retrieval_min_score=0.41,
        retrieval_relative_cutoff=0.66,
        context_token_fraction=0.5,
        web_search_enabled=False,
        web_search_timeout=6.0,
        web_search_max_results=3,
        ollama_health_check_interval=2.5,
        ollama_max_lost_health_checks=7,
        system_prompt="Custom prompt {web_instruction}",
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
            "llm_timeout": 45.0,
            "temperature": 0.7,
            "sampler_top_k": 24,
            "context_window": 4096,
            "retrieval_candidate_k": 55,
            "retrieval_min_score": 0.41,
            "retrieval_relative_cutoff": 0.66,
            "context_token_fraction": 0.5,
            "web_search_enabled": False,
            "web_search_timeout": 6.0,
            "web_search_max_results": 3,
            "ollama_health_check_interval": 2.5,
            "ollama_max_lost_health_checks": 7,
            "system_prompt": "Custom prompt {web_instruction}",
            "progress_enabled": False,
        },
        "question": "What is aero balance?",
    }
