from pathlib import Path

from src.config import PipelineConfig
from src.generation import QueryService
from src.indexing_v1 import StructuredIndexer
from src.ingestion_v1 import StructuredIngestor
from src.store import SQLiteBlockStore


def _config(tmp_path: Path) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.paths.data_dir = str(tmp_path / "data")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_dir = str(tmp_path / "db")
    cfg.paths.asset_dir = str(tmp_path / "db" / "assets")
    cfg.models.embedding_model = "hash"
    cfg.models.embedding_dim = 64
    cfg.models.allow_hash_embeddings = True
    cfg.ingestion.vision_enabled = False
    cfg.chunking.max_tokens = 60
    cfg.chunking.overlap_tokens = 10
    cfg.retrieval.top_k = 3
    cfg.ensure_dirs()
    return cfg


def test_markdown_ingest_index_query_returns_block_citations(tmp_path):
    cfg = _config(tmp_path)
    source = Path(cfg.paths.data_dir) / "accumulator.md"
    source.write_text(
        """
# Accumulator Cooling

The accumulator pack temperature is managed by forced airflow across the cell tabs.
Temperature sensors should be placed near high resistance busbar joints.

# Aero Loads

The front wing increases normal load at the tire contact patch.
        """.strip(),
        encoding="utf-8",
    )
    store = SQLiteBlockStore(cfg.paths.db_dir)
    doc_ids = StructuredIngestor(cfg, store).ingest_path(source)
    result = StructuredIndexer(cfg, store).rebuild()
    response = QueryService(cfg, store).ask("Where should accumulator temperature sensors be placed?")
    payload = response.to_dict()
    assert len(doc_ids) == 1
    assert result["chunks"] >= 1
    assert payload["citations"]
    assert payload["citations"][0]["page"] == 1
    assert payload["retrieved_blocks"]
