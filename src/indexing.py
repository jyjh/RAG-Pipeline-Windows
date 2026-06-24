import logging
import os
import sys

logger = logging.getLogger(__name__)


def _progress_status(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def run_indexing(
    md_dir: str,
    db_dir: str,
    *,
    progress_enabled: bool = True,
    embedding_model: str = "nomic-embed-text",
    embedding_batch_size: int | None = None,
    embedding_timeout: float | None = None,
    index_backend: str = "lancedb",
    summary_mode: str = "hybrid",
    chunk_target_tokens: int = 900,
    chunk_overlap_tokens: int = 120,
):
    _progress_status("Starting Ollama local indexing pipeline...", enabled=progress_enabled)
    _progress_status(f"Checking Markdown directory: {md_dir}", enabled=progress_enabled)
    if not os.path.isdir(md_dir):
        logger.warning("Markdown directory does not exist: %s", md_dir)
        _progress_status(f"Markdown directory does not exist: {md_dir}", enabled=progress_enabled)
        return

    md_files = sorted(f for f in os.listdir(md_dir) if f.endswith(".md"))
    _progress_status(f"Found {len(md_files)} Markdown file(s).", enabled=progress_enabled)
    if not md_files:
        logger.warning("No markdown files found in %s", md_dir)
        return

    from src.local_rag import LocalVectorIndexer

    indexer = LocalVectorIndexer(
        working_dir=db_dir,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        embedding_timeout=embedding_timeout,
        index_backend=index_backend,
        summary_mode=summary_mode,
        chunk_target_tokens=chunk_target_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        progress_enabled=progress_enabled,
    )
    indexer.index_markdown(md_dir)


if __name__ == "__main__":
    run_indexing("processed_docs", "db")
