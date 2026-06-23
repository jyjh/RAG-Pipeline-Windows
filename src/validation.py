from __future__ import annotations

import importlib.util
from pathlib import Path

from src.config import PipelineConfig
from src.store import SQLiteBlockStore


RUNTIME_PACKAGES = {
    "docling": "PDF layout, OCR, tables, formulas",
    "lancedb": "vector storage",
    "sentence_transformers": "Nomic embeddings and optional reranking",
    "ollama": "local LLM/VLM generation",
    "fastapi": "API service",
    "uvicorn": "API server",
    "streamlit": "browser UI",
    "pypdf": "PDF page classification",
}


def validate_runtime(config: PipelineConfig) -> dict:
    packages = {
        name: {
            "installed": importlib.util.find_spec(name) is not None,
            "purpose": purpose,
        }
        for name, purpose in RUNTIME_PACKAGES.items()
    }
    store = SQLiteBlockStore(config.paths.db_dir)
    documents = store.list_documents()
    chunks = store.list_chunks()
    store.close()
    return {
        "paths": {
            "data_dir": _path_state(config.paths.data_dir),
            "db_dir": _path_state(config.paths.db_dir),
            "asset_dir": _path_state(config.paths.asset_dir),
        },
        "packages": packages,
        "store": {
            "documents": len(documents),
            "chunks": len(chunks),
            "sqlite": str(Path(config.paths.db_dir) / "rag.sqlite"),
        },
        "models": {
            "llm_model": config.models.llm_model,
            "vision_model": config.models.vision_model,
            "embedding_model": config.models.embedding_model,
        },
    }


def _path_state(path: str) -> dict:
    p = Path(path)
    return {"path": str(p), "exists": p.exists(), "is_dir": p.is_dir()}

