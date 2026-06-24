from __future__ import annotations

import argparse
import logging
import sys


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
        help="Seconds before one Ollama chat request fails in query mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
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
        "--parser_mode",
        choices=["hybrid", "manual", "docling"],
        default="hybrid",
        help="PDF parser selection for ingest mode",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        default="auto",
        help="Docling accelerator selection",
    )
    parser.add_argument(
        "--asset_triggers",
        choices=["none", "images", "all"],
        default="none",
        help=(
            "When hybrid/manual ingest should run Docling asset enrichment after pypdf text succeeds. "
            "Use 'all' to include table/equation heuristics."
        ),
    )
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
            run_ingestion(
                args.data_dir,
                args.md_dir,
                parser_mode=args.parser_mode,
                accelerator=args.accelerator,
                asset_triggers=args.asset_triggers,
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
