from __future__ import annotations

import argparse
import json
import logging
import sys

from src.config import load_config
from src.generation import QueryService
from src.indexing_v1 import StructuredIndexer
from src.ingestion_v1 import StructuredIngestor
from src.service import run_app
from src.store import SQLiteBlockStore
from src.validation import validate_runtime


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local FSAE RAG Pipeline")
    parser.add_argument("--config", help="Path to JSON or TOML config file")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Parse PDFs/markdown into citation-safe blocks")
    ingest.add_argument("--input", default=None, help="Input file or directory")
    ingest.add_argument("--reset", action="store_true", help="Clear the structured store before ingestion")

    index = subcommands.add_parser("index", help="Build chunks, embeddings, and vector index")
    index.add_argument("--rebuild", action="store_true", default=True, help="Rebuild the index")

    query = subcommands.add_parser("query", help="Ask a question")
    query.add_argument("question", help="Question to ask")
    query.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    query.add_argument("--top-k", type=int, default=None)
    query.add_argument("--json", action="store_true", help="Print full JSON response")

    serve = subcommands.add_parser("serve", help="Run the local FastAPI service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--lan", action="store_true", help="Bind for LAN use; requires --api-token or config token")
    serve.add_argument("--api-token", default=None, help="Required token for LAN mode")

    subcommands.add_parser("validate", help="Check runtime dependencies and indexed state")
    subcommands.add_parser("ui", help="Print Streamlit launch command")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    store = SQLiteBlockStore(cfg.paths.db_dir)

    if args.command == "ingest":
        if args.reset:
            store.reset()
        source = args.input or cfg.paths.data_dir
        doc_ids = StructuredIngestor(cfg, store).ingest_path(source)
        print(json.dumps({"document_ids": doc_ids}, indent=2))
        return 0

    if args.command == "index":
        result = StructuredIndexer(cfg, store).rebuild()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "query":
        response = QueryService(cfg, store).ask(args.question, mode=args.mode, top_k=args.top_k)
        payload = response.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(payload["answer"])
            if payload["citations"]:
                print("\nCitations:")
                for idx, citation in enumerate(payload["citations"], start=1):
                    print(
                        f"[C{idx}] {citation['document_title']} p.{citation['page']} "
                        f"{citation['modality']} {citation['block_id']}"
                    )
        return 0

    if args.command == "serve":
        if args.lan:
            cfg.server.lan = True
            cfg.server.host = args.host or "0.0.0.0"
            if args.api_token:
                cfg.server.api_token = args.api_token
            if not cfg.server.api_token:
                raise SystemExit("LAN mode requires --api-token or server.api_token in config.")
        elif args.host:
            cfg.server.host = args.host
        if args.port:
            cfg.server.port = args.port
        run_app(host=cfg.server.host, port=cfg.server.port, config=cfg)
        return 0

    if args.command == "validate":
        print(json.dumps(validate_runtime(cfg), indent=2))
        return 0

    if args.command == "ui":
        print("Run: streamlit run src/ui.py")
        print("Optional env: RAG_API_URL=http://127.0.0.1:8000 RAG_API_TOKEN=<token>")
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
