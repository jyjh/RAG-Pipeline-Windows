import argparse
import logging
import os
from src.ingestion import run_ingestion
from src.indexing import run_indexing
from src.query import QueryEngine
from src.utils import manage_vram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Local STEM RAG Pipeline Orchestrator")
    parser.add_argument("--mode", choices=["ingest", "index", "query", "all"], required=True, help="Mode to run")
    parser.add_argument("--data_dir", default="data", help="Directory containing input PDFs")
    parser.add_argument("--md_dir", default="processed_docs", help="Directory for intermediate markdown")
    parser.add_argument("--db_dir", default="db", help="Directory for the database")
    parser.add_argument("--question", help="The question to ask in query mode")
    parser.add_argument("--query_mode", choices=["local", "global", "hybrid"], default="hybrid", help="LightRAG query mode")

    args = parser.parse_args()

    # Ensure working directories exist regardless of how main() is invoked
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.md_dir, exist_ok=True)
    os.makedirs(args.db_dir, exist_ok=True)

    LLM_MODEL = "deepseek-r1:32b"
    VISION_MODEL = "qwen2.5vl:7b"

    if args.mode in ("ingest", "all"):
        logger.info("Starting Ingestion Phase...")
        # Vision model is loaded lazily on first figure detection — no preload needed
        run_ingestion(args.data_dir, args.md_dir)
        logger.info("Ingestion Phase Complete.")

    if args.mode in ("index", "all"):
        logger.info("Starting Indexing Phase...")
        manage_vram(LLM_MODEL, model_to_unload=VISION_MODEL)
        run_indexing(args.md_dir, args.db_dir)
        logger.info("Indexing Phase Complete.")

    if args.mode == "query":
        if not args.question:
            logger.error("--question is required for query mode.")
            return

        logger.info("Starting Query Phase...")
        manage_vram(LLM_MODEL)
        engine = QueryEngine(working_dir=args.db_dir, model=LLM_MODEL)
        response = engine.ask(args.question, mode=args.query_mode)
        print("\n--- RESPONSE ---\n")
        print(response)
        print("\n----------------\n")


if __name__ == "__main__":
    main()
