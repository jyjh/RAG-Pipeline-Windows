import main
from src.lightrag_compat import LightRAGDependencyError


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
        model,
        progress_enabled,
        embedding_backend,
        embedding_model,
        embedding_local_files_only,
        embedding_batch_size,
        embedding_timeout,
        tokenizer_backend,
        rag_backend,
        lightrag_import_timeout,
    ):
        calls["md_dir"] = md_dir
        calls["db_dir"] = db_dir
        calls["model"] = model
        calls["progress_enabled"] = progress_enabled
        calls["embedding_backend"] = embedding_backend
        calls["embedding_model"] = embedding_model
        calls["embedding_local_files_only"] = embedding_local_files_only
        calls["embedding_batch_size"] = embedding_batch_size
        calls["embedding_timeout"] = embedding_timeout
        calls["tokenizer_backend"] = tokenizer_backend
        calls["rag_backend"] = rag_backend
        calls["lightrag_import_timeout"] = lightrag_import_timeout

    monkeypatch.setattr(main, "run_indexing", fake_run_indexing)

    result = main.main(["--mode", "index", "--md_dir", "md_in", "--db_dir", "db_out"])

    assert result == 0
    assert calls == {
        "md_dir": "md_in",
        "db_dir": "db_out",
        "model": "gemma4",
        "progress_enabled": True,
        "embedding_backend": "ollama",
        "embedding_model": "nomic-embed-text",
        "embedding_local_files_only": True,
        "embedding_batch_size": 8,
        "embedding_timeout": 30.0,
        "tokenizer_backend": "byte",
        "rag_backend": "auto",
        "lightrag_import_timeout": 10,
    }


def test_main_query_dispatches_to_current_query_engine(monkeypatch, capsys):
    calls = {}

    class FakeQueryEngine:
        def __init__(
            self,
            working_dir,
            model,
            embedding_backend,
            embedding_model,
            embedding_local_files_only,
            embedding_batch_size,
            embedding_timeout,
            tokenizer_backend,
            rag_backend,
            lightrag_import_timeout,
            progress_enabled,
        ):
            calls["working_dir"] = working_dir
            calls["model"] = model
            calls["embedding_backend"] = embedding_backend
            calls["embedding_model"] = embedding_model
            calls["embedding_local_files_only"] = embedding_local_files_only
            calls["embedding_batch_size"] = embedding_batch_size
            calls["embedding_timeout"] = embedding_timeout
            calls["tokenizer_backend"] = tokenizer_backend
            calls["rag_backend"] = rag_backend
            calls["lightrag_import_timeout"] = lightrag_import_timeout
            calls["progress_enabled"] = progress_enabled

        def ask(self, question, mode="hybrid"):
            calls["question"] = question
            calls["mode"] = mode
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
            "--query_mode",
            "local",
            "--llm_model",
            "custom-model",
        ]
    )

    assert result == 0
    assert calls == {
        "working_dir": "db_in",
        "model": "custom-model",
        "embedding_backend": "ollama",
        "embedding_model": "nomic-embed-text",
        "embedding_local_files_only": True,
        "embedding_batch_size": 8,
        "embedding_timeout": 30.0,
        "tokenizer_backend": "byte",
        "rag_backend": "auto",
        "lightrag_import_timeout": 10,
        "progress_enabled": True,
        "question": "What is regularization?",
        "mode": "local",
    }
    assert capsys.readouterr().out.strip() == "answer text"


def test_main_reports_lightrag_dependency_error(monkeypatch, capsys):
    class BrokenQueryEngine:
        def __init__(self, **kwargs):
            raise LightRAGDependencyError("install lightrag-hku")

    monkeypatch.setattr(main, "QueryEngine", BrokenQueryEngine)

    result = main.main(["--mode", "query", "--question", "test"])

    assert result == 2
    assert "install lightrag-hku" in capsys.readouterr().err
