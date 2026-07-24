from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class PathsConfig:
    data_dir: str = "data"
    processed_dir: str = "processed_docs"
    db_dir: str = "db"
    asset_dir: str = "db/assets"


@dataclass
class ModelConfig:
    llm_model: str = "deepseek-r1:32b"
    vision_model: str = "qwen2.5vl:7b"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    reranker_model: str = ""
    embedding_dim: int = 768
    allow_hash_embeddings: bool = True
    native_embeddings: bool = False


@dataclass
class IngestionConfig:
    ocr_backend: str = "tesseract_cli"
    ocr_strategy: str = "auto"
    vision_enabled: bool = True
    figure_crop_enabled: bool = True
    formula_enrichment: bool = False
    table_structure: bool = True
    accelerator: str = "cuda"
    num_threads: int = 8


@dataclass
class ChunkingConfig:
    max_tokens: int = 900
    overlap_tokens: int = 120
    adjacent_block_window: int = 1


@dataclass
class RetrievalConfig:
    top_k: int = 8
    vector_top_k: int = 24
    bm25_top_k: int = 24
    rrf_k: int = 60
    rerank_top_k: int = 12


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    lan: bool = False
    api_token: str = ""


@dataclass
class EmbeddingsConfig:
    """Embedding-engine tuning. These are the dominant cost at corpus scale
    (a 100GB cold index is weeks of embedding work on a single host), so they
    are surfaced in config rather than env-only.

    Precedence at construction time (see ``EmbeddingEngine.__init__``) is:
    explicit constructor arg > env var (``OLLAMA_EMBED_*``) > these config
    values > hardcoded defaults. The env vars remain the escape hatch for
    ad-hoc overrides without editing the config file.
    """

    batch_size: int = 128
    timeout_seconds: float = 30.0
    retries: int = 3
    cache_max_entries: int = 50_000
    # Scale the per-request timeout with batch size so a large batch on a slow
    # GPU does not silently trip the retry loop. The effective timeout is
    # ``timeout_seconds * max(1.0, batch_size / timeout_batch_baseline)``,
    # capped at ``timeout_max_seconds``. Set ``timeout_batch_baseline`` equal to
    # the batch size at which ``timeout_seconds`` was measured (default 128).
    timeout_batch_baseline: int = 128
    timeout_max_seconds: float = 600.0
    # List of Ollama replica URLs for multi-host embedding (the primary
    # scale-out lever). Empty = single host via ``OLLAMA_HOST``.
    hosts: list[str] = field(default_factory=list)
    # In-flight embedding batches per host. Set >1 when the Ollama server runs
    # with ``OLLAMA_NUM_PARALLEL>1``. 1 = one batch per host at a time.
    concurrency: int = 1


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)

    def ensure_dirs(self) -> None:
        for value in (self.paths.data_dir, self.paths.processed_dir, self.paths.db_dir, self.paths.asset_dir):
            Path(value).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".toml", ".tml"}:
        import tomllib

        with path.open("rb") as fh:
            return tomllib.load(fh)
    raise ValueError(f"Unsupported config format: {path}")


def _merge_dataclass(target: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(target, key, value)
    return target


def load_config(path: str | os.PathLike[str] | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    chosen = path or os.environ.get("RAG_PIPELINE_CONFIG")
    if chosen:
        _merge_dataclass(cfg, _load_mapping(Path(chosen)))
    cfg.ensure_dirs()
    return cfg

