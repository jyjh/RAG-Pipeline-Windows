import main


def test_main_ingest_dispatches_to_current_ingestion(monkeypatch):
    calls = {}

    def fake_run_ingestion(
        data_dir,
        md_dir,
        *,
        parser_mode,
        asset_dir,
        accelerator,
        num_threads,
        asset_triggers,
        code_enrichment,
        formula_enrichment,
        vision_model,
        vision_enabled,
        ocr_backend,
        ocr_langs,
        ocr_force_full_page,
        ocr_bitmap_area_threshold,
        rapidocr_backend,
        tesseract_cmd,
        tesseract_data_path,
        tesseract_psm,
        progress_enabled,
    ):
        calls["data_dir"] = data_dir
        calls["md_dir"] = md_dir
        calls["parser_mode"] = parser_mode
        calls["asset_dir"] = asset_dir
        calls["accelerator"] = accelerator
        calls["num_threads"] = num_threads
        calls["asset_triggers"] = asset_triggers
        calls["code_enrichment"] = code_enrichment
        calls["formula_enrichment"] = formula_enrichment
        calls["vision_model"] = vision_model
        calls["vision_enabled"] = vision_enabled
        calls["ocr_backend"] = ocr_backend
        calls["ocr_langs"] = ocr_langs
        calls["ocr_force_full_page"] = ocr_force_full_page
        calls["ocr_bitmap_area_threshold"] = ocr_bitmap_area_threshold
        calls["rapidocr_backend"] = rapidocr_backend
        calls["tesseract_cmd"] = tesseract_cmd
        calls["tesseract_data_path"] = tesseract_data_path
        calls["tesseract_psm"] = tesseract_psm
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
            "--asset_dir",
            "asset_out",
            "--accelerator",
            "cpu",
            "--num_threads",
            "3",
            "--asset_triggers",
            "none",
            "--code_enrichment",
            "false",
            "--formula_enrichment",
            "true",
            "--vision_model",
            "vision-test",
            "--vision_enabled",
            "false",
            "--ocr_backend",
            "tesseract_cli",
            "--ocr_langs",
            "eng",
            "--ocr_force_full_page",
            "false",
            "--ocr_bitmap_area_threshold",
            "0.2",
            "--rapidocr_backend",
            "torch",
            "--tesseract_cmd",
            "C:/Tools/tesseract.exe",
            "--tesseract_data_path",
            "C:/Tools/tessdata",
            "--tesseract_psm",
            "6",
            "--no_progress",
        ]
    )

    assert result == 0
    assert calls == {
        "data_dir": "data_in",
        "md_dir": "md_out",
        "parser_mode": "docling",
        "asset_dir": "asset_out",
        "accelerator": "cpu",
        "num_threads": 3,
        "asset_triggers": "none",
        "code_enrichment": False,
        "formula_enrichment": True,
        "vision_model": "vision-test",
        "vision_enabled": False,
        "ocr_backend": "tesseract_cli",
        "ocr_langs": ["eng"],
        "ocr_force_full_page": False,
        "ocr_bitmap_area_threshold": 0.2,
        "rapidocr_backend": "torch",
        "tesseract_cmd": "C:/Tools/tesseract.exe",
        "tesseract_data_path": "C:/Tools/tessdata",
        "tesseract_psm": 6,
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
        reuse_db_dir,
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
        calls["reuse_db_dir"] = reuse_db_dir
        calls["summary_mode"] = summary_mode
        calls["chunk_target_tokens"] = chunk_target_tokens
        calls["chunk_overlap_tokens"] = chunk_overlap_tokens

    monkeypatch.setattr(main, "run_indexing", fake_run_indexing)

    result = main.main(
        ["--mode", "index", "--md_dir", "md_in", "--db_dir", "db_out", "--reuse_db_dir", "db_live"]
    )

    assert result == 0
    assert calls == {
        "md_dir": "md_in",
        "db_dir": "db_out",
        "progress_enabled": True,
        "embedding_model": "nomic-embed-text",
        "embedding_batch_size": 8,
        "embedding_timeout": 30.0,
        "index_backend": "lancedb",
        "reuse_db_dir": "db_live",
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
            asset_dir,
            model,
            embedding_model,
            embedding_batch_size,
            embedding_timeout,
            llm_num_predict,
            llm_timeout,
            temperature,
            sampler_top_k,
            context_window,
            retrieval_candidate_k,
            retrieval_min_score,
            retrieval_relative_cutoff,
            context_token_fraction,
            web_search_enabled,
            web_search_timeout,
            web_search_max_results,
            ollama_health_check_interval,
            ollama_max_lost_health_checks,
            system_prompt,
            planner_model,
            planner_enabled,
            planner_max_queries,
            progress_enabled,
        ):
            calls["working_dir"] = working_dir
            calls["asset_dir"] = asset_dir
            calls["model"] = model
            calls["embedding_model"] = embedding_model
            calls["embedding_batch_size"] = embedding_batch_size
            calls["embedding_timeout"] = embedding_timeout
            calls["llm_num_predict"] = llm_num_predict
            calls["llm_timeout"] = llm_timeout
            calls["temperature"] = temperature
            calls["sampler_top_k"] = sampler_top_k
            calls["context_window"] = context_window
            calls["retrieval_candidate_k"] = retrieval_candidate_k
            calls["retrieval_min_score"] = retrieval_min_score
            calls["retrieval_relative_cutoff"] = retrieval_relative_cutoff
            calls["context_token_fraction"] = context_token_fraction
            calls["web_search_enabled"] = web_search_enabled
            calls["web_search_timeout"] = web_search_timeout
            calls["web_search_max_results"] = web_search_max_results
            calls["ollama_health_check_interval"] = ollama_health_check_interval
            calls["ollama_max_lost_health_checks"] = ollama_max_lost_health_checks
            calls["system_prompt"] = system_prompt
            calls["planner_model"] = planner_model
            calls["planner_enabled"] = planner_enabled
            calls["planner_max_queries"] = planner_max_queries
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
        "asset_dir": "db/assets",
        "model": "custom-model",
        "embedding_model": "nomic-embed-text",
        "embedding_batch_size": 8,
        "embedding_timeout": 30.0,
        "llm_num_predict": 4096,
        "llm_timeout": 120.0,
        "temperature": 0.3,
        "sampler_top_k": 40,
        "context_window": 8192,
        "retrieval_candidate_k": 80,
        "retrieval_min_score": 0.5,
        "retrieval_relative_cutoff": 0.72,
        "context_token_fraction": 0.6,
        "web_search_enabled": True,
        "web_search_timeout": 8.0,
        "web_search_max_results": 5,
        "ollama_health_check_interval": 5.0,
        "ollama_max_lost_health_checks": 5,
        "system_prompt": None,
        "planner_model": "qwen2.5:1.5b",
        "planner_enabled": True,
        "planner_max_queries": 3,
        "progress_enabled": True,
        "question": "What is regularization?",
    }
    assert capsys.readouterr().out.strip() == "answer text"
