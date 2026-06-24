import main


def test_main_ingest_dispatches_to_current_ingestion(monkeypatch):
    calls = {}

    def fake_run_ingestion(
        data_dir,
        md_dir,
        *,
        parser_mode,
        accelerator,
        asset_triggers,
        progress_enabled,
    ):
        calls["data_dir"] = data_dir
        calls["md_dir"] = md_dir
        calls["parser_mode"] = parser_mode
        calls["accelerator"] = accelerator
        calls["asset_triggers"] = asset_triggers
        calls["progress_enabled"] = progress_enabled

    monkeypatch.setattr(main, "run_ingestion", fake_run_ingestion)

    result = main.main(
        [
            "--mode",
            "ingest",
            "--data_dir",
            "data_in",
            "--md_dir",
            "md_out",
            "--parser_mode",
            "docling",
            "--accelerator",
            "cpu",
            "--asset_triggers",
            "none",
            "--no_progress",
        ]
    )

    assert result == 0
    assert calls == {
        "data_dir": "data_in",
        "md_dir": "md_out",
        "parser_mode": "docling",
        "accelerator": "cpu",
        "asset_triggers": "none",
        "progress_enabled": False,
    }


def test_main_index_dispatches_to_current_indexing(monkeypatch):
    calls = {}

    def fake_run_indexing(
        md_dir,
        db_dir,
        *,
        progress_enabled,
        embedding_model,
        embedding_batch_size,
        embedding_timeout,
        index_backend,
        summary_mode,
        chunk_target_tokens,
        chunk_overlap_tokens,
    ):
        calls["md_dir"] = md_dir
        calls["db_dir"] = db_dir
        calls["progress_enabled"] = progress_enabled
        calls["embedding_model"] = embedding_model
        calls["embedding_batch_size"] = embedding_batch_size
        calls["embedding_timeout"] = embedding_timeout
        calls["index_backend"] = index_backend
        calls["summary_mode"] = summary_mode
        calls["chunk_target_tokens"] = chunk_target_tokens
        calls["chunk_overlap_tokens"] = chunk_overlap_tokens

    monkeypatch.setattr(main, "run_indexing", fake_run_indexing)

    result = main.main(["--mode", "index", "--md_dir", "md_in", "--db_dir", "db_out"])

    assert result == 0
    assert calls == {
        "md_dir": "md_in",
        "db_dir": "db_out",
        "progress_enabled": True,
        "embedding_model": "nomic-embed-text",
        "embedding_batch_size": 8,
        "embedding_timeout": 30.0,
        "index_backend": "lancedb",
        "summary_mode": "hybrid",
        "chunk_target_tokens": 900,
        "chunk_overlap_tokens": 120,
    }


def test_main_query_dispatches_to_current_query_engine(monkeypatch, capsys):
    calls = {}

    class FakeQueryEngine:
        def __init__(
            self,
            working_dir,
            model,
            embedding_model,
            embedding_batch_size,
            embedding_timeout,
            llm_num_predict,
            llm_timeout,
            temperature,
            sampler_top_k,
            context_window,
            progress_enabled,
        ):
            calls["working_dir"] = working_dir
            calls["model"] = model
            calls["embedding_model"] = embedding_model
            calls["embedding_batch_size"] = embedding_batch_size
            calls["embedding_timeout"] = embedding_timeout
            calls["llm_num_predict"] = llm_num_predict
            calls["llm_timeout"] = llm_timeout
            calls["temperature"] = temperature
            calls["sampler_top_k"] = sampler_top_k
            calls["context_window"] = context_window
            calls["progress_enabled"] = progress_enabled

        def ask(self, question):
            calls["question"] = question
            return "answer text"

    monkeypatch.setattr(main, "QueryEngine", FakeQueryEngine)

    result = main.main(
        [
            "--mode",
            "query",
            "--db_dir",
            "db_in",
            "--question",
            "What is regularization?",
            "--llm_model",
            "custom-model",
        ]
    )

    assert result == 0
    assert calls == {
        "working_dir": "db_in",
        "model": "custom-model",
        "embedding_model": "nomic-embed-text",
        "embedding_batch_size": 8,
        "embedding_timeout": 30.0,
        "llm_num_predict": 4096,
        "llm_timeout": 120.0,
        "temperature": 0.9,
        "sampler_top_k": 40,
        "context_window": 8192,
        "progress_enabled": True,
        "question": "What is regularization?",
    }
    assert capsys.readouterr().out.strip() == "answer text"
