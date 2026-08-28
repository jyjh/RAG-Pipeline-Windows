"""Cluster-side bulk ingestion/indexing entry point (runs inside the SIF).

Invoked by the PBS job script. Forwards the FULL ``[ingestion]`` config to
``run_ingestion`` -- not just a hand-picked subset -- so a cluster parse
behaves identically to a local one (worker count, OCR settings, asset
triggers, ...). ``--input-dir``/``--processed-dir``/``--db-dir``/``--asset-dir``
default to the staged ``config.toml`` paths and can be overridden for corpus
directories that live outside the repo (see ``HpcBackend.push_corpus_dir``).
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import default_config_path, load_config
from src.ingestion import run_ingestion
from src.indexing import run_indexing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _ingestion_kwargs(config) -> dict:
    """Map ``[ingestion]`` + vision model onto ``run_ingestion`` parameters.

    Listed explicitly (rather than splatting the dataclass) so an unknown
    IngestionConfig field can never crash the call with a TypeError.
    """
    ingestion = config.ingestion
    return {
        "parser_mode": ingestion.parser_mode,
        "accelerator": ingestion.accelerator,
        "num_threads": ingestion.num_threads,
        "asset_triggers": ingestion.asset_triggers,
        "code_enrichment": ingestion.code_enrichment,
        "formula_enrichment": ingestion.formula_enrichment,
        "vision_model": config.models.vision_model,
        "vision_enabled": ingestion.vision_enabled,
        "ocr_backend": ingestion.ocr_backend,
        "ocr_langs": ingestion.ocr_langs,
        "ocr_force_full_page": ingestion.ocr_force_full_page,
        "ocr_bitmap_area_threshold": ingestion.ocr_bitmap_area_threshold,
        "rapidocr_backend": ingestion.rapidocr_backend,
        "tesseract_cmd": ingestion.tesseract_cmd,
        "tesseract_data_path": ingestion.tesseract_data_path,
        "tesseract_psm": ingestion.tesseract_psm,
        "ingestion_workers": ingestion.ingestion_workers,
        "max_pages_whole_doc": ingestion.max_pages_whole_doc,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bulk Ingestion and Indexing CLI")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to directory containing PDFs")
    parser.add_argument("--processed-dir", type=str, default=None, help="Output directory for processed Markdown (default: config [paths].processed_dir)")
    parser.add_argument("--db-dir", type=str, default=None, help="LanceDB output directory (default: config [paths].db_dir)")
    parser.add_argument("--asset-dir", type=str, default=None, help="Image-asset output directory (default: config [paths].asset_dir)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion and only run indexing")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing and only run ingestion")

    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_dir}")
        sys.exit(1)

    # Discovered config (repo config.toml / RAG_PIPELINE_CONFIG); a bare
    # load_config() would read pure defaults and ignore config.toml.
    config = load_config(default_config_path())
    processed_dir = Path(args.processed_dir) if args.processed_dir else Path(config.paths.processed_dir)
    db_dir = Path(args.db_dir) if args.db_dir else Path(config.paths.db_dir)
    asset_dir = Path(args.asset_dir) if args.asset_dir else Path(config.paths.asset_dir)

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
                output_dir=str(processed_dir),
                asset_dir=asset_dir,
                progress_enabled=True,
                **_ingestion_kwargs(config),
            )
            logger.info("Ingestion completed successfully.")
        except Exception as e:
            logger.exception("Ingestion failed")
            sys.exit(1)

    if not args.skip_index:
        logger.info("--- PHASE 2: INDEXING ---")
        try:
            run_indexing(
                md_dir=str(processed_dir),
                db_dir=str(db_dir),
                progress_enabled=True,
                embedding_model=config.models.embedding_model,
                index_backend="lancedb",
                summary_mode="hybrid",
            )
            logger.info("Indexing completed successfully.")
        except Exception as e:
            logger.exception("Indexing failed")
            sys.exit(1)

    logger.info("Bulk processing finished.")

if __name__ == "__main__":
    main()
