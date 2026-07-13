import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.ingestion import run_ingestion
from src.indexing import run_indexing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Bulk Ingestion and Indexing CLI")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to directory containing PDFs")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion and only run indexing")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing and only run ingestion")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_dir}")
        sys.exit(1)
        
    config = load_config()
    processed_dir = config.paths.processed_dir
    db_dir = config.paths.db_dir
    asset_dir = config.paths.asset_dir
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)
    
    logger.info("="*50)
    logger.info(f"Starting bulk pipeline")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Processed: {processed_dir}")
    logger.info(f"DB: {db_dir}")
    logger.info("="*50)
    
    if not args.skip_ingest:
        logger.info("--- PHASE 1: INGESTION ---")
        try:
            run_ingestion(
                input_dir=str(input_dir),
                output_dir=processed_dir,
                parser_mode=config.ingestion.ocr_strategy,
                accelerator=config.ingestion.accelerator,
                num_threads=config.ingestion.num_threads,
                asset_dir=asset_dir,
                code_enrichment=False,
                formula_enrichment=config.ingestion.formula_enrichment,
                vision_model=config.models.vision_model,
                vision_enabled=config.ingestion.vision_enabled,
                ocr_backend=config.ingestion.ocr_backend,
                progress_enabled=True,
            )
            logger.info("Ingestion completed successfully.")
        except Exception as e:
            logger.exception("Ingestion failed")
            sys.exit(1)
            
    if not args.skip_index:
        logger.info("--- PHASE 2: INDEXING ---")
        try:
            run_indexing(
                md_dir=processed_dir,
                db_dir=db_dir,
                progress_enabled=True,
                embedding_model=config.models.embedding_model,
                index_backend="lancedb",
                summary_mode="hybrid",
                chunk_target_tokens=config.chunking.max_tokens,
                chunk_overlap_tokens=config.chunking.overlap_tokens,
            )
            logger.info("Indexing completed successfully.")
        except Exception as e:
            logger.exception("Indexing failed")
            sys.exit(1)
            
    logger.info("Bulk processing finished.")

if __name__ == "__main__":
    main()
