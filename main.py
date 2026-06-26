from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.defaults import (
    DEFAULT_ASSET_TRIGGERS,
    DEFAULT_CODE_ENRICHMENT,
    DEFAULT_DOCLING_ACCELERATOR,
    DEFAULT_FORMULA_ENRICHMENT,
    DEFAULT_OCR_BACKEND,
    DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
    DEFAULT_OCR_FORCE_FULL_PAGE,
    DEFAULT_OCR_LANGS,
    DEFAULT_PDF_PARSER_MODE,
    DEFAULT_RAPIDOCR_BACKEND,
    DEFAULT_TESSERACT_CMD,
    DEFAULT_TESSERACT_DATA_PATH,
    DEFAULT_TESSERACT_PSM,
    DEFAULT_VISION_ENABLED,
    DEFAULT_VISION_MODEL,
    SUPPORTED_OCR_BACKENDS,
    SUPPORTED_RAPIDOCR_BACKENDS,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def default_llm_model() -> str:
    try:
        from src.defaults import DEFAULT_LLM_MODEL
    except Exception:
        return "gemma4"
    return DEFAULT_LLM_MODEL


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


def _load_toml_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or Path("config.toml")
    if not config_path.exists():
        return {}
    try:
        import tomllib

        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_langs(value: Any, default: tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or list(default)
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return parts or list(default)
    return list(default)


def _load_ingestion_config(config_path: Path | None = None) -> dict[str, Any]:
    payload = _load_toml_config(config_path)
    ingestion = payload.get("ingestion", {}) if isinstance(payload.get("ingestion"), dict) else {}
    models = payload.get("models", {}) if isinstance(payload.get("models"), dict) else {}
    return {
        "parser_mode": str(ingestion.get("parser_mode") or DEFAULT_PDF_PARSER_MODE),
        "accelerator": str(ingestion.get("accelerator") or DEFAULT_DOCLING_ACCELERATOR),
        "asset_triggers": str(ingestion.get("asset_triggers") or DEFAULT_ASSET_TRIGGERS),
        "code_enrichment": _as_bool(ingestion.get("code_enrichment"), DEFAULT_CODE_ENRICHMENT),
        "formula_enrichment": _as_bool(ingestion.get("formula_enrichment"), DEFAULT_FORMULA_ENRICHMENT),
        "vision_model": str(models.get("vision_model") or DEFAULT_VISION_MODEL),
        "vision_enabled": _as_bool(ingestion.get("vision_enabled"), DEFAULT_VISION_ENABLED),
        "ocr_backend": str(ingestion.get("ocr_backend") or DEFAULT_OCR_BACKEND),
        "ocr_langs": _as_langs(ingestion.get("ocr_langs"), DEFAULT_OCR_LANGS),
        "ocr_force_full_page": _as_bool(ingestion.get("ocr_force_full_page"), DEFAULT_OCR_FORCE_FULL_PAGE),
        "ocr_bitmap_area_threshold": _as_float(
            ingestion.get("ocr_bitmap_area_threshold"),
            DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
        ),
        "rapidocr_backend": str(ingestion.get("rapidocr_backend") or DEFAULT_RAPIDOCR_BACKEND),
        "tesseract_cmd": str(ingestion.get("tesseract_cmd") or DEFAULT_TESSERACT_CMD),
        "tesseract_data_path": str(ingestion.get("tesseract_data_path") or DEFAULT_TESSERACT_DATA_PATH),
        "tesseract_psm": _as_optional_int(ingestion.get("tesseract_psm"), DEFAULT_TESSERACT_PSM),
    }


def _ingestion_args(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_ingestion_config()
    return {
        "parser_mode": args.parser_mode or config["parser_mode"],
        "accelerator": args.accelerator or config["accelerator"],
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
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local FSAE RAG Pipeline")
    parser.add_argument("--mode", choices=["ingest", "index", "query"], required=True)
    parser.add_argument("--data_dir", default="data", help="Input PDF directory or PDF file")
    parser.add_argument("--md_dir", default="processed_docs", help="Generated Markdown directory")
    parser.add_argument("--db_dir", default="db", help="Local vector index directory")
    parser.add_argument("--llm_model", default=default_llm_model(), help="Ollama LLM model")
    parser.add_argument(
        "--embedding_model",
        default="nomic-embed-text",
        help="Ollama embedding model name",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=8,
        help="Texts/chunks per Ollama embedding request.",
    )
    parser.add_argument(
        "--embedding_timeout",
        type=float,
        default=30.0,
        help="Seconds before one Ollama embedding HTTP request fails.",
    )
    parser.add_argument(
        "--index_backend",
        choices=["lancedb"],
        default="lancedb",
        help="Vector index storage backend for index mode. LanceDB is the only supported backend.",
    )
    parser.add_argument(
        "--summary_mode",
        choices=["hybrid", "deterministic", "llm"],
        default="hybrid",
        help="How to derive document and section summary records during indexing.",
    )
    parser.add_argument(
        "--chunk_target_tokens",
        type=int,
        default=900,
        help="Target chunk size used within one detected section.",
    )
    parser.add_argument(
        "--chunk_overlap_tokens",
        type=int,
        default=120,
        help="Overlap used only when splitting an oversized detected section.",
    )
    parser.add_argument("--question", help="Question to ask in query mode")
    parser.add_argument(
        "--llm_num_predict",
        type=int,
        default=4096,
        help="Maximum answer tokens to request from Ollama in query mode.",
    )
    parser.add_argument(
        "--llm_timeout",
        type=float,
        default=120.0,
        help="Deprecated compatibility option; Ollama chat generation no longer uses a request timeout.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Ollama sampling temperature in query mode.",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=40,
        help="Ollama top_k sampler setting in query mode.",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=8192,
        help="Ollama num_ctx context window in query mode.",
    )
    parser.add_argument(
        "--retrieval_candidate_k",
        type=int,
        default=80,
        help="Candidate pool size for local vector retrieval before relevance cutoffs.",
    )
    parser.add_argument(
        "--retrieval_min_score",
        type=float,
        default=0.50,
        help="Minimum normalized relevance score for local context chunks.",
    )
    parser.add_argument(
        "--retrieval_relative_cutoff",
        type=float,
        default=0.72,
        help="Keep chunks with score at least this fraction of the best hit.",
    )
    parser.add_argument(
        "--context_token_fraction",
        type=float,
        default=0.60,
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
        default=8.0,
        help="Seconds before a keyless web-search request fails.",
    )
    parser.add_argument(
        "--web_search_max_results",
        type=int,
        default=5,
        help="Maximum web-search results returned to the model per call.",
    )
    parser.add_argument(
        "--ollama_health_check_interval",
        type=float,
        default=5.0,
        help="Seconds between Ollama health checks after chat connection loss.",
    )
    parser.add_argument(
        "--ollama_max_lost_health_checks",
        type=int,
        default=5,
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
        "--no_progress",
        action="store_true",
        help="Disable ingestion progress bars",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    print("Starting RAG pipeline CLI...", file=sys.stderr, flush=True)
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode == "ingest":
            print("Starting ingestion mode...", file=sys.stderr, flush=True)
            ingestion_options = _ingestion_args(args)
            run_ingestion(
                args.data_dir,
                args.md_dir,
                **ingestion_options,
                progress_enabled=not args.no_progress,
            )
            return 0

        if args.mode == "index":
            print("Starting index mode...", file=sys.stderr, flush=True)
            run_indexing(
                args.md_dir,
                args.db_dir,
                progress_enabled=not args.no_progress,
                embedding_model=args.embedding_model,
                embedding_batch_size=args.embedding_batch_size,
                embedding_timeout=args.embedding_timeout,
                index_backend=args.index_backend,
                summary_mode=args.summary_mode,
                chunk_target_tokens=args.chunk_target_tokens,
                chunk_overlap_tokens=args.chunk_overlap_tokens,
            )
            return 0

        if args.mode == "query":
            if not args.question:
                parser.error("--question is required when --mode query")
            answer = QueryEngine(
                working_dir=args.db_dir,
                model=args.llm_model,
                embedding_model=args.embedding_model,
                embedding_batch_size=args.embedding_batch_size,
                embedding_timeout=args.embedding_timeout,
                llm_num_predict=args.llm_num_predict,
                llm_timeout=args.llm_timeout,
                temperature=args.temperature,
                sampler_top_k=args.max_k,
                context_window=args.context_window,
                retrieval_candidate_k=args.retrieval_candidate_k,
                retrieval_min_score=args.retrieval_min_score,
                retrieval_relative_cutoff=args.retrieval_relative_cutoff,
                context_token_fraction=args.context_token_fraction,
                web_search_enabled=not args.no_web_search,
                web_search_timeout=args.web_search_timeout,
                web_search_max_results=args.web_search_max_results,
                ollama_health_check_interval=args.ollama_health_check_interval,
                ollama_max_lost_health_checks=args.ollama_max_lost_health_checks,
                system_prompt=args.system_prompt,
                progress_enabled=not args.no_progress,
            ).ask(args.question)
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
