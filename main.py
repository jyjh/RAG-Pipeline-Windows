from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from src.coerce import (
    as_bool as _as_bool,
    as_float as _as_float,
    as_optional_int as _as_optional_int,
    as_positive_int as _as_positive_int,
    as_string_list as _as_langs,
)
from src.system_logging import setup_system_logging
from src.defaults import (
    DEFAULT_ASSET_DIR,
    DEFAULT_ASSET_TRIGGERS,
    DEFAULT_CODE_ENRICHMENT,
    DEFAULT_CONTEXT_TOKEN_FRACTION,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_DOCLING_ACCELERATOR,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FORMULA_ENRICHMENT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_NUM_PREDICT,
    DEFAULT_OCR_BACKEND,
    DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
    DEFAULT_OCR_FORCE_FULL_PAGE,
    DEFAULT_OCR_LANGS,
    DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL,
    DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS,
    DEFAULT_PDF_PARSER_MODE,
    DEFAULT_PLANNER_MAX_QUERIES,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_RAPIDOCR_BACKEND,
    DEFAULT_RETRIEVAL_CANDIDATE_K,
    DEFAULT_RETRIEVAL_MIN_SCORE,
    DEFAULT_RETRIEVAL_RRF_K,
    DEFAULT_RETRIEVAL_RELATIVE_CUTOFF,
    DEFAULT_SAMPLER_TOP_K,
    DEFAULT_SCANNED_OCR_ENGINE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TESSERACT_CMD,
    DEFAULT_TESSERACT_DATA_PATH,
    DEFAULT_TESSERACT_PSM,
    DEFAULT_UNLIMITED_OCR_DPI,
    DEFAULT_UNLIMITED_OCR_MODEL,
    DEFAULT_VISION_OCR_DPI,
    DEFAULT_VISION_OCR_MODEL,
    DEFAULT_VISION_ENABLED,
    DEFAULT_VISION_MODEL,
    DEFAULT_WEB_SEARCH_ENABLED,
    DEFAULT_WEB_SEARCH_MAX_RESULTS,
    DEFAULT_WEB_SEARCH_TIMEOUT,
    SUPPORTED_OCR_BACKENDS,
    SUPPORTED_RAPIDOCR_BACKENDS,
)


# Timestamped console output plus a rotating CLI log (logs/cli.log) so
# library warnings/errors from CLI runs persist like the web server's system
# log. Per-run job detail still goes to logs/job_<mode>_<pid>.log; the
# per-request access log is web-only and stays disabled here. Relative log
# paths resolve against the repo root, not the process cwd.
try:
    setup_system_logging(file="logs/cli.log", access_file=None)
except OSError:
    pass


def run_ingestion(*args, **kwargs):
    print("Importing ingestion module...", file=sys.stderr, flush=True)
    from src.ingestion import run_ingestion as _run_ingestion

    print("Ingestion module imported.", file=sys.stderr, flush=True)
    return _run_ingestion(*args, **kwargs)


def run_indexing(*args, **kwargs):
    print("Importing indexing module...", file=sys.stderr, flush=True)
    from src.indexing import run_indexing as _run_indexing

    print("Indexing module imported.", file=sys.stderr, flush=True)
    return _run_indexing(*args, **kwargs)


class QueryEngine:
    """Lazy-loading stand-in so query-mode imports don't slow ingest/index runs."""

    def __new__(cls, *args, **kwargs):
        from src.query import QueryEngine as _QueryEngine

        return _QueryEngine(*args, **kwargs)


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _pipeline_config(config_path: Path | None = None):
    """Typed config via the shared cached loader (src.config.load_config).

    Without an explicit path or ``RAG_PIPELINE_CONFIG``, resolves the repo
    ``config.toml`` (working-directory independent) instead of falling back
    to pure dataclass defaults.
    """
    from src.config import default_config_path, load_config

    return load_config(config_path if config_path is not None else default_config_path())


def _load_ingestion_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg = _pipeline_config(config_path)
    ingestion = cfg.ingestion
    return {
        "asset_dir": str(cfg.paths.asset_dir or DEFAULT_ASSET_DIR),
        "parser_mode": str(ingestion.parser_mode or DEFAULT_PDF_PARSER_MODE),
        "accelerator": str(ingestion.accelerator or DEFAULT_DOCLING_ACCELERATOR),
        "num_threads": _as_positive_int(ingestion.num_threads, 8),
        "asset_triggers": str(ingestion.asset_triggers or DEFAULT_ASSET_TRIGGERS),
        "code_enrichment": _as_bool(ingestion.code_enrichment, DEFAULT_CODE_ENRICHMENT),
        "formula_enrichment": _as_bool(ingestion.formula_enrichment, DEFAULT_FORMULA_ENRICHMENT),
        "vision_model": str(cfg.models.vision_model or DEFAULT_VISION_MODEL),
        "vision_enabled": _as_bool(ingestion.vision_enabled, DEFAULT_VISION_ENABLED),
        "ocr_backend": str(ingestion.ocr_backend or DEFAULT_OCR_BACKEND),
        "ocr_langs": _as_langs(ingestion.ocr_langs, DEFAULT_OCR_LANGS),
        "ocr_force_full_page": _as_bool(ingestion.ocr_force_full_page, DEFAULT_OCR_FORCE_FULL_PAGE),
        "ocr_bitmap_area_threshold": _as_float(
            ingestion.ocr_bitmap_area_threshold,
            DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
        ),
        "rapidocr_backend": str(ingestion.rapidocr_backend or DEFAULT_RAPIDOCR_BACKEND),
        "tesseract_cmd": str(ingestion.tesseract_cmd or DEFAULT_TESSERACT_CMD),
        "tesseract_data_path": str(ingestion.tesseract_data_path or DEFAULT_TESSERACT_DATA_PATH),
        "tesseract_psm": _as_optional_int(ingestion.tesseract_psm, DEFAULT_TESSERACT_PSM),
        "scanned_ocr_engine": str(ingestion.scanned_ocr_engine or DEFAULT_SCANNED_OCR_ENGINE),
        "unlimited_ocr_model": str(ingestion.unlimited_ocr_model or DEFAULT_UNLIMITED_OCR_MODEL),
        "unlimited_ocr_dpi": _as_positive_int(ingestion.unlimited_ocr_dpi, DEFAULT_UNLIMITED_OCR_DPI),
        "vision_ocr_model": str(ingestion.vision_ocr_model or DEFAULT_VISION_OCR_MODEL),
        "vision_ocr_dpi": _as_positive_int(ingestion.vision_ocr_dpi, DEFAULT_VISION_OCR_DPI),
        "ingestion_workers": _as_positive_int(ingestion.ingestion_workers, 1),
        "max_pages_whole_doc": max(0, _as_optional_int(ingestion.max_pages_whole_doc, 0) or 0),
    }


def _load_indexing_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the raw ``[indexing]`` section for ANN tuning.

    Values are applied to the module-level ANN constants in
    :mod:`src.vector_store` via :func:`apply_indexing_config` before indexing
    starts, so ``create_vector_index()`` and query-time search honor config.toml.
    """
    return dict(_pipeline_config(config_path).indexing)


def _load_query_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg = _pipeline_config(config_path)
    chat = cfg.chat
    retrieval = cfg.retrieval
    web_search = cfg.web_search
    ollama = cfg.ollama
    return {
        "llm_model": str(cfg.models.llm_model or DEFAULT_LLM_MODEL),
        "embedding_model": str(cfg.models.embedding_model or DEFAULT_EMBEDDING_MODEL),
        "embedding_dim": _as_positive_int(cfg.models.embedding_dim, DEFAULT_EMBEDDING_DIM),
        "llm_num_predict": _as_positive_int(chat.llm_num_predict, DEFAULT_NUM_PREDICT),
        "llm_timeout": _as_float(chat.llm_timeout, DEFAULT_LLM_TIMEOUT),
        "temperature": _as_float(chat.temperature, DEFAULT_TEMPERATURE),
        "max_k": _as_positive_int(chat.max_k, DEFAULT_SAMPLER_TOP_K),
        "context_window": _as_positive_int(chat.context_window, DEFAULT_CONTEXT_WINDOW),
        "retrieval_candidate_k": _as_positive_int(retrieval.candidate_top_k, DEFAULT_RETRIEVAL_CANDIDATE_K),
        "retrieval_min_score": _as_float(retrieval.min_relevance_score, DEFAULT_RETRIEVAL_MIN_SCORE),
        "retrieval_relative_cutoff": _as_float(retrieval.relative_relevance_cutoff, DEFAULT_RETRIEVAL_RELATIVE_CUTOFF),
        "retrieval_rrf_k": _as_positive_int(retrieval.rrf_k, DEFAULT_RETRIEVAL_RRF_K),
        "context_token_fraction": _as_float(retrieval.context_token_fraction, DEFAULT_CONTEXT_TOKEN_FRACTION),
        "web_search_enabled": _as_bool(web_search.enabled, DEFAULT_WEB_SEARCH_ENABLED),
        "web_search_timeout": _as_float(web_search.timeout_seconds, DEFAULT_WEB_SEARCH_TIMEOUT),
        "web_search_max_results": _as_positive_int(web_search.max_results, DEFAULT_WEB_SEARCH_MAX_RESULTS),
        "ollama_health_check_interval": _as_float(ollama.chat_health_check_interval_seconds, DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL),
        "ollama_max_lost_health_checks": _as_positive_int(ollama.chat_max_lost_health_checks, DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS),
        "system_prompt": str(chat.system_prompt or "") or None,
        "planner_model": str(chat.planner_model or DEFAULT_PLANNER_MODEL),
        "planner_enabled": _as_bool(chat.planner_enabled, True),
        "planner_max_queries": _as_positive_int(chat.planner_max_queries, DEFAULT_PLANNER_MAX_QUERIES),
    }


def _ingestion_args(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_ingestion_config()
    return {
        "parser_mode": args.parser_mode or config["parser_mode"],
        "asset_dir": args.asset_dir or config["asset_dir"],
        "accelerator": args.accelerator or config["accelerator"],
        "num_threads": args.num_threads if args.num_threads is not None else config["num_threads"],
        "asset_triggers": args.asset_triggers or config["asset_triggers"],
        "code_enrichment": _as_bool(args.code_enrichment, config["code_enrichment"]),
        "formula_enrichment": _as_bool(args.formula_enrichment, config["formula_enrichment"]),
        "vision_model": args.vision_model or config["vision_model"],
        "vision_enabled": _as_bool(args.vision_enabled, config["vision_enabled"]),
        "ocr_backend": args.ocr_backend or config["ocr_backend"],
        "ocr_langs": _as_langs(args.ocr_langs, tuple(config["ocr_langs"])),
        "ocr_force_full_page": _as_bool(args.ocr_force_full_page, config["ocr_force_full_page"]),
        "ocr_bitmap_area_threshold": (
            args.ocr_bitmap_area_threshold
            if args.ocr_bitmap_area_threshold is not None
            else config["ocr_bitmap_area_threshold"]
        ),
        "rapidocr_backend": args.rapidocr_backend or config["rapidocr_backend"],
        "tesseract_cmd": args.tesseract_cmd or config["tesseract_cmd"],
        "tesseract_data_path": args.tesseract_data_path or config["tesseract_data_path"],
        "tesseract_psm": args.tesseract_psm if args.tesseract_psm is not None else config["tesseract_psm"],
        "scanned_ocr_engine": config["scanned_ocr_engine"],
        "unlimited_ocr_model": config["unlimited_ocr_model"],
        "unlimited_ocr_dpi": config["unlimited_ocr_dpi"],
        "vision_ocr_model": config["vision_ocr_model"],
        "vision_ocr_dpi": config["vision_ocr_dpi"],
        "ingestion_workers": (
            args.ingestion_workers
            if getattr(args, "ingestion_workers", None) is not None
            else config.get("ingestion_workers", 1)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local FSAE RAG Pipeline")
    parser.add_argument("--mode", choices=["ingest", "index", "query"], required=True)
    parser.add_argument("--data_dir", default="data", help="Input PDF directory or PDF file")
    parser.add_argument("--md_dir", default="processed_docs", help="Generated Markdown directory")
    parser.add_argument("--db_dir", default="db", help="Local vector index directory")
    parser.add_argument("--asset_dir", default=None, help="Directory for stored image assets.")
    parser.add_argument("--llm_model", default=None, help="Ollama LLM model")
    parser.add_argument(
        "--embedding_model",
        default=None,
        help="Ollama embedding model name",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Embedding vector dimension (must match the selected model).",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=None,
        help="Texts/chunks per Ollama embedding request.",
    )
    parser.add_argument(
        "--embedding_timeout",
        type=float,
        default=None,
        help="Seconds before one Ollama embedding HTTP request fails.",
    )
    parser.add_argument(
        "--index_backend",
        choices=["lancedb"],
        default=None,
        help="Vector index storage backend for index mode (default: [chunking].index_backend, lancedb). "
        "LanceDB is the only supported backend.",
    )
    parser.add_argument(
        "--reuse_db_dir",
        default=None,
        help="Existing DB directory to inspect for reusable vectors while writing a new index.",
    )
    parser.add_argument(
        "--summary_mode",
        choices=["hybrid", "deterministic", "llm"],
        default=None,
        help="How to derive document and section summary records during indexing "
        "(default: [chunking].summary_mode).",
    )
    parser.add_argument(
        "--chunk_target_tokens",
        type=int,
        default=None,
        help="Target chunk size used within one detected section (default: [chunking].max_tokens).",
    )
    parser.add_argument(
        "--chunk_overlap_tokens",
        type=int,
        default=None,
        help="Overlap used only when splitting an oversized detected section "
        "(default: [chunking].overlap_tokens).",
    )
    parser.add_argument(
        "--source_hashes",
        default="",
        help="Comma-separated source hashes for incremental indexing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Index mode only: resume a previously-interrupted full build from "
            "its checkpoint (skips files already written). No effect on "
            "incremental (--source_hashes) runs."
        ),
    )
    parser.add_argument("--question", help="Question to ask in query mode")
    parser.add_argument(
        "--llm_num_predict",
        type=int,
        default=None,
        help="Maximum answer tokens to request from Ollama in query mode.",
    )
    parser.add_argument(
        "--llm_timeout",
        type=float,
        default=None,
        help="Deprecated compatibility option; Ollama chat generation no longer uses a request timeout.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Ollama sampling temperature in query mode.",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=None,
        help="Ollama top_k sampler setting in query mode.",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=None,
        help="Ollama num_ctx context window in query mode.",
    )
    parser.add_argument(
        "--retrieval_candidate_k",
        type=int,
        default=None,
        help="Candidate pool size for local vector retrieval before relevance cutoffs.",
    )
    parser.add_argument(
        "--retrieval_min_score",
        type=float,
        default=None,
        help="Minimum normalized relevance score for local context chunks.",
    )
    parser.add_argument(
        "--retrieval_relative_cutoff",
        type=float,
        default=None,
        help="Keep chunks with score at least this fraction of the best hit.",
    )
    parser.add_argument(
        "--retrieval_rrf_k",
        type=int,
        default=None,
        help="RRF damping constant for fusing vector + BM25 rankings (default: [retrieval].rrf_k).",
    )
    parser.add_argument(
        "--context_token_fraction",
        type=float,
        default=None,
        help="Fraction of the model context window allowed for input prompt context, capped at 0.60.",
    )
    parser.add_argument(
        "--no_web_search",
        action="store_true",
        help="Disable the web_search tool in query mode.",
    )
    parser.add_argument(
        "--web_search_timeout",
        type=float,
        default=None,
        help="Seconds before a keyless web-search request fails.",
    )
    parser.add_argument(
        "--web_search_max_results",
        type=int,
        default=None,
        help="Maximum web-search results returned to the model per call.",
    )
    parser.add_argument(
        "--planner_model",
        default=None,
        help="Small Ollama model used to expand the question into retrieval queries before the main model runs.",
    )
    parser.add_argument(
        "--no_planner",
        action="store_true",
        help="Disable eager multi-query retrieval; let the main model decide when to search.",
    )
    parser.add_argument(
        "--planner_max_queries",
        type=int,
        default=None,
        help="Number of diverse search queries the planner model should generate.",
    )
    parser.add_argument(
        "--ollama_health_check_interval",
        type=float,
        default=None,
        help="Seconds between Ollama health checks after chat connection loss.",
    )
    parser.add_argument(
        "--ollama_max_lost_health_checks",
        type=int,
        default=None,
        help="Cancel chat only after this many failed Ollama health checks after connection loss.",
    )
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Override the default RAG chat system prompt. Use {web_instruction} to place the web-search instruction.",
    )
    parser.add_argument(
        "--parser_mode",
        choices=["hybrid", "manual", "docling"],
        default=None,
        help="PDF parser selection for ingest mode",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        default=None,
        help="Docling accelerator selection",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Docling/native worker thread count for ingest mode.",
    )
    parser.add_argument(
        "--asset_triggers",
        choices=["none", "images", "auto", "all"],
        default=None,
        help=(
            "When hybrid/manual ingest should run Docling asset enrichment after pypdf text succeeds. "
            "Use 'auto' for pictures/code/formulas or 'all' to include table heuristics."
        ),
    )
    parser.add_argument(
        "--code_enrichment",
        choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
        default=None,
        help="Enable Docling code enrichment for detected code pages.",
    )
    parser.add_argument(
        "--formula_enrichment",
        choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
        default=None,
        help="Enable Docling formula enrichment for detected formula pages.",
    )
    parser.add_argument("--vision_model", default=None, help="Ollama vision model for figure/page analysis.")
    parser.add_argument(
        "--vision_enabled",
        choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
        default=None,
        help="Enable local vision analysis for figures and scanned-page fallback.",
    )
    parser.add_argument(
        "--ocr_backend",
        choices=list(SUPPORTED_OCR_BACKENDS),
        default=None,
        help="Docling OCR backend for scanned/image-only PDFs.",
    )
    parser.add_argument("--ocr_langs", default=None, help="Comma-separated OCR languages.")
    parser.add_argument(
        "--ocr_force_full_page",
        choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
        default=None,
        help="Force full-page OCR for scanned PDFs.",
    )
    parser.add_argument(
        "--ocr_bitmap_area_threshold",
        type=float,
        default=None,
        help="Docling OCR bitmap area threshold.",
    )
    parser.add_argument(
        "--rapidocr_backend",
        choices=list(SUPPORTED_RAPIDOCR_BACKENDS),
        default=None,
        help="RapidOCR inference backend.",
    )
    parser.add_argument("--tesseract_cmd", default=None, help="Tesseract executable path or command.")
    parser.add_argument("--tesseract_data_path", default=None, help="Tesseract language data directory.")
    parser.add_argument("--tesseract_psm", type=int, default=None, help="Tesseract page segmentation mode.")
    parser.add_argument(
        "--ingestion_workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for PDF ingestion (1 = serial). "
        "Each worker loads its own parser models, so cap for GPU memory.",
    )
    parser.add_argument(
        "--max_pages_whole_doc",
        type=int,
        default=None,
        help="Force page-at-a-time Docling parsing above this page count (0 disables).",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable ingestion progress bars",
    )
    parser.add_argument(
        "--job_log_dir",
        default=None,
        help="Directory for per-job structured log files (ingest/index runs). "
        "Defaults to <workspace>/logs.",
    )
    return parser


def _setup_job_logger(mode: str, job_log_dir: str | None) -> logging.Logger | None:
    """Attach a per-run structured file handler for ingest/index modes.

    Returns the configured logger (or None if setup failed). The log file is
    ``logs/job_<mode>_<pid>.log`` under ``job_log_dir`` (defaulting to a
    ``logs`` dir next to the workspace root). Structured events written via
    ``log_event`` land here and survive a subprocess crash, unlike the
    in-memory job-log tail.
    """
    from src.job_logging import setup_job_logging

    base = Path(job_log_dir) if job_log_dir else Path("logs")
    try:
        log_path = base / f"job_{mode}_{os.getpid()}.log"
        return setup_job_logging(log_path)
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    print("Starting RAG pipeline CLI...", file=sys.stderr, flush=True)
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode == "ingest":
            print("Starting ingestion mode...", file=sys.stderr, flush=True)
            _setup_job_logger("ingest", args.job_log_dir)
            ingestion_options = _ingestion_args(args)
            configured_max_pages = config_max_pages = _load_ingestion_config().get("max_pages_whole_doc", 50)
            if getattr(args, "max_pages_whole_doc", None) is not None or config_max_pages != 50:
                ingestion_options["max_pages_whole_doc"] = (
                    args.max_pages_whole_doc
                    if getattr(args, "max_pages_whole_doc", None) is not None
                    else configured_max_pages
                )
            run_ingestion(
                args.data_dir,
                args.md_dir,
                **ingestion_options,
                progress_enabled=not args.no_progress,
            )
            return 0

        if args.mode == "index":
            print("Starting index mode...", file=sys.stderr, flush=True)
            # Acquire the cross-process index lock on the target dir. For a
            # direct CLI build into the live db/ this prevents racing the
            # server's publish/backup/restore; for a staged build (db_dir is a
            # fresh unique dir) the lock is trivially acquired and harmless.
            from src.file_lock import acquire_index_lock

            with acquire_index_lock(args.db_dir):
                # Apply [indexing] ANN tuning to the module constants before the
                # indexer runs (covers nprobes/refine_factor/min_rows/retrain
                # threshold). Safe no-op when the section is absent.
                from src.vector_store import apply_indexing_config

                apply_indexing_config(_load_indexing_config())
                pipeline_cfg = _pipeline_config()
                embedding_cfg = pipeline_cfg.embeddings
                chunking_cfg = pipeline_cfg.chunking
                index_kwargs = dict(
                    md_dir=args.md_dir,
                    db_dir=args.db_dir,
                    progress_enabled=not args.no_progress,
                    embedding_model=(
                        args.embedding_model
                        or pipeline_cfg.models.embedding_model
                        or DEFAULT_EMBEDDING_MODEL
                    ),
                    embedding_dim=(
                        args.embedding_dim
                        or _as_positive_int(pipeline_cfg.models.embedding_dim, DEFAULT_EMBEDDING_DIM)
                    ),
                    embedding_batch_size=(
                        args.embedding_batch_size
                        or _as_positive_int(embedding_cfg.batch_size, DEFAULT_EMBEDDING_BATCH_SIZE)
                    ),
                    embedding_timeout=(
                        args.embedding_timeout
                        or _as_float(embedding_cfg.timeout_seconds, DEFAULT_EMBEDDING_TIMEOUT)
                    ),
                    index_backend=(
                        args.index_backend
                        or str(chunking_cfg.index_backend or "").strip().lower()
                        or "lancedb"
                    ),
                    reuse_db_dir=args.reuse_db_dir,
                    # "Or-style" flags: CLI value wins, else [chunking], else
                    # the run_indexing defaults (hybrid / 900 / 120).
                    summary_mode=(
                        args.summary_mode
                        or str(chunking_cfg.summary_mode or "").strip().lower()
                        or "hybrid"
                    ),
                    chunk_target_tokens=(
                        args.chunk_target_tokens
                        or _as_positive_int(chunking_cfg.max_tokens, 900)
                    ),
                    chunk_overlap_tokens=(
                        args.chunk_overlap_tokens
                        if args.chunk_overlap_tokens is not None
                        else max(0, int(chunking_cfg.overlap_tokens or 0))
                    ),
                    resume=bool(args.resume),
                )
                source_hashes = [value.strip() for value in args.source_hashes.split(",") if value.strip()]
                if source_hashes:
                    index_kwargs["source_hashes"] = source_hashes
                run_indexing(**index_kwargs)
            return 0

        if args.mode == "query":
            if not args.question:
                parser.error("--question is required when --mode query")
            query_config = _load_query_config()
            # CLI flags override config values. "Or-style" flags fall back to
            # config on 0/empty (matching the historical argparse defaults);
            # numeric/str flags fall back only when unset (None) so explicit
            # zeros (e.g. retrieval_min_score=0) are honored.
            cli_overrides = {
                key: value
                for key, value in {
                    "model": args.llm_model,
                    "embedding_model": args.embedding_model,
                    "embedding_batch_size": args.embedding_batch_size,
                    "embedding_timeout": args.embedding_timeout,
                    "llm_num_predict": args.llm_num_predict,
                    "sampler_top_k": args.max_k,
                    "context_window": args.context_window,
                    "retrieval_candidate_k": args.retrieval_candidate_k,
                    "web_search_max_results": args.web_search_max_results,
                    "planner_model": args.planner_model,
                    "planner_max_queries": args.planner_max_queries,
                }.items()
                if value
            }
            cli_overrides.update(
                {
                    key: value
                    for key, value in {
                        "llm_timeout": args.llm_timeout,
                        "temperature": args.temperature,
                        "retrieval_min_score": args.retrieval_min_score,
                        "retrieval_relative_cutoff": args.retrieval_relative_cutoff,
                        "retrieval_rrf_k": args.retrieval_rrf_k,
                        "context_token_fraction": args.context_token_fraction,
                        "web_search_timeout": args.web_search_timeout,
                        "ollama_health_check_interval": args.ollama_health_check_interval,
                        "ollama_max_lost_health_checks": args.ollama_max_lost_health_checks,
                        "system_prompt": args.system_prompt,
                    }.items()
                    if value is not None
                }
            )
            model = query_config.pop("llm_model")
            sampler_top_k = query_config.pop("max_k")
            engine_kwargs = {
                **query_config,
                "model": model,
                "sampler_top_k": sampler_top_k,
                **cli_overrides,
                "working_dir": args.db_dir,
                "asset_dir": args.asset_dir or _load_ingestion_config()["asset_dir"],
                # Always passed (not only when the CLI set them) so fake
                # engines in tests and the real QueryEngine signature stay stable.
                "embedding_batch_size": cli_overrides.pop("embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE),
                "embedding_timeout": cli_overrides.pop("embedding_timeout", DEFAULT_EMBEDDING_TIMEOUT),
                "web_search_enabled": query_config["web_search_enabled"] and not args.no_web_search,
                "planner_enabled": query_config["planner_enabled"] and not args.no_planner,
                "progress_enabled": not args.no_progress,
            }
            answer = QueryEngine(**engine_kwargs).ask(args.question)
            print(answer)
            return 0
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        raise

    raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
