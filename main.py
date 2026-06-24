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


def is_lightrag_dependency_error(exc: BaseException) -> bool:
    try:
        from src.lightrag_compat import LightRAGDependencyError
    except Exception:
        return False
    return isinstance(exc, LightRAGDependencyError)


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
    parser.add_argument("--db_dir", default="db", help="LightRAG working directory")
    parser.add_argument("--llm_model", default=default_llm_model(), help="Ollama LLM model")
    parser.add_argument(
        "--embedding_backend",
        choices=["hash", "ollama", "sentence-transformers"],
        default="ollama",
        help="Embedding backend for index/query setup",
    )
    parser.add_argument(
        "--embedding_model",
        default="nomic-embed-text",
        help="Embedding model name for the selected backend",
    )
    parser.add_argument(
        "--allow_embedding_download",
        action="store_true",
        help="Allow SentenceTransformers to download the embedding model if it is not cached",
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
        "--tokenizer_backend",
        choices=["byte", "deepseek"],
        default="byte",
        help="Tokenizer backend for LightRAG chunking",
    )
    parser.add_argument(
        "--rag_backend",
        choices=["auto", "lightrag", "local"],
        default="auto",
        help="RAG backend for index/query. 'auto' falls back to local vector RAG if LightRAG import stalls.",
    )
    parser.add_argument(
        "--lightrag_import_timeout",
        type=int,
        default=10,
        help="Seconds to wait for LightRAG core import in auto backend mode.",
    )
    parser.add_argument("--question", help="Question to ask in query mode")
    parser.add_argument(
        "--query_mode",
        choices=["local", "global", "hybrid"],
        default="hybrid",
        help="LightRAG query mode",
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
                model=args.llm_model,
                progress_enabled=not args.no_progress,
                embedding_backend=args.embedding_backend,
                embedding_model=args.embedding_model,
                embedding_local_files_only=not args.allow_embedding_download,
                embedding_batch_size=args.embedding_batch_size,
                embedding_timeout=args.embedding_timeout,
                tokenizer_backend=args.tokenizer_backend,
                rag_backend=args.rag_backend,
                lightrag_import_timeout=args.lightrag_import_timeout,
            )
            return 0

        if args.mode == "query":
            if not args.question:
                parser.error("--question is required when --mode query")
            answer = QueryEngine(
                working_dir=args.db_dir,
                model=args.llm_model,
                embedding_backend=args.embedding_backend,
                embedding_model=args.embedding_model,
                embedding_local_files_only=not args.allow_embedding_download,
                embedding_batch_size=args.embedding_batch_size,
                embedding_timeout=args.embedding_timeout,
                tokenizer_backend=args.tokenizer_backend,
                rag_backend=args.rag_backend,
                lightrag_import_timeout=args.lightrag_import_timeout,
                progress_enabled=not args.no_progress,
            ).ask(
                args.question,
                mode=args.query_mode,
            )
            print(answer)
            return 0
    except Exception as exc:
        if is_lightrag_dependency_error(exc):
            print(str(exc), file=sys.stderr)
            return 2
        if isinstance(exc, RuntimeError):
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        raise

    raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
