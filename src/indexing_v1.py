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
        documents = self.store.list_documents()
        chunks = []
        duplicate_groups: dict[str, list[str]] = {}

        for doc in documents:
            blocks = self.store.list_blocks(doc.doc_id)
            result = build_chunks_for_blocks(doc.doc_id, blocks, self.config.chunking)
            for chunk in result.chunks:
                self.store.upsert_chunk(chunk)
                chunks.append(chunk)
            for group_id, chunk_ids in result.duplicate_groups.items():
                duplicate_groups.setdefault(group_id, []).extend(chunk_ids)

        vectors = {}
        if chunks:
            texts = [chunk.text for chunk in chunks]
            embedding_result = self.embedder.embed_documents(texts)
            rows = []
            for chunk, vector in zip(chunks, embedding_result.vectors):
                vectors[chunk.chunk_id] = vector
                rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "duplicate_group_id": chunk.duplicate_group_id,
                        "vector": vector,
                    }
                )
            self.vector_index.upsert(rows)
            backend = f"{embedding_result.backend}+{self.vector_index.backend}"
        else:
            backend = self.vector_index.backend

        grouped_duplicates = {
            group_id: ids for group_id, ids in duplicate_groups.items() if len(ids) > 1
        }
        elapsed = time.perf_counter() - start
        logger.info(
            "Indexed %s documents, %s chunks using %s in %.2fs",
            len(documents),
            len(chunks),
            backend,
            elapsed,
        )
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "vectors": len(vectors),
            "duplicate_groups": len(grouped_duplicates),
            "backend": backend,
            "elapsed_seconds": elapsed,
        }

