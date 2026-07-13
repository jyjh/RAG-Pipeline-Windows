"""DEPRECATED: legacy SQLite-backed v1 pipeline module.

Not imported by the active web app or the LanceDB-based ingestion/indexing
pipeline. Retained for legacy test coverage. See src/store.py for the full
deprecation note. New work belongs in the LanceDB-backed modules.
"""
from __future__ import annotations

import logging
import time

from src.chunking import build_chunks_for_blocks
from src.config import PipelineConfig
from src.embedding_backends import EmbeddingProvider
from src.store import SQLiteBlockStore
from src.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class StructuredIndexer:
    def __init__(self, config: PipelineConfig, store: SQLiteBlockStore):
        self.config = config
        self.store = store
        self.embedder = EmbeddingProvider(config.models)
        self.vector_index = VectorIndex(config.paths.db_dir, store)

    def rebuild(self) -> dict:
        start = time.perf_counter()
        self.store.clear_index()
        chunks = []
        groups: dict[str, list[str]] = {}
        docs = self.store.list_documents()
        for doc in docs:
            result = build_chunks_for_blocks(doc.doc_id, self.store.list_blocks(doc.doc_id), self.config.chunking)
            for chunk in result.chunks:
                self.store.upsert_chunk(chunk)
                chunks.append(chunk)
            for gid, ids in result.duplicate_groups.items():
                groups.setdefault(gid, []).extend(ids)
        vectors = {}
        backend = self.vector_index.backend
        if chunks:
            embedded = self.embedder.embed_documents([c.text for c in chunks])
            rows = []
            for chunk, vector in zip(chunks, embedded.vectors):
                vectors[chunk.chunk_id] = vector
                rows.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "text": chunk.text,
                             "duplicate_group_id": chunk.duplicate_group_id, "vector": vector})
            self.vector_index.upsert(rows)
            backend = f"{embedded.backend}+{self.vector_index.backend}"
        elapsed = time.perf_counter() - start
        logger.info("Indexed %s documents, %s chunks using %s in %.2fs", len(docs), len(chunks), backend, elapsed)
        return {"documents": len(docs), "chunks": len(chunks), "vectors": len(vectors),
                "duplicate_groups": sum(1 for ids in groups.values() if len(ids) > 1),
                "backend": backend, "elapsed_seconds": elapsed}

