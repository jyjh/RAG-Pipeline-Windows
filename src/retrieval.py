"""DEPRECATED: legacy SQLite-backed v1 pipeline module.

Not imported by the active web app or the LanceDB-based ingestion/indexing
pipeline. Retained for legacy test coverage. See src/store.py for the full
deprecation note. New work belongs in the LanceDB-backed modules.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from src.config import PipelineConfig
from src.embedding_backends import EmbeddingProvider
from src.schema import Citation, RetrievedBlock
from src.store import SQLiteBlockStore
from src.vector_index import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    citations: list[Citation]
    retrieved_blocks: list[RetrievedBlock]
    context: str
    timings: dict[str, float]


class Retriever:
    def __init__(self, config: PipelineConfig, store: SQLiteBlockStore):
        self.config = config
        self.store = store
        self.embedder = EmbeddingProvider(config.models)
        self.vector_index = VectorIndex(config.paths.db_dir, store)
        self.reranker = LocalReranker(config.models.reranker_model)

    def retrieve(self, question: str, mode: str = "hybrid", top_k: int | None = None,
                 doc_filters: list[str] | None = None) -> RetrievalResult:
        top_k = top_k or self.config.retrieval.top_k
        mode = (mode or "hybrid").lower()
        timings: dict[str, float] = {}
        start = time.perf_counter()
        vector_hits = []
        if mode in {"hybrid", "vector", "local"}:
            vector_hits = self.vector_index.search(self.embedder.embed_query(question).vectors[0], self.config.retrieval.vector_top_k)
            vector_hits = _filter(vector_hits, doc_filters)
        timings["vector_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        bm25_hits = []
        if mode in {"hybrid", "bm25", "text", "global"}:
            bm25_hits = _filter(self.store.search_chunks_fts(question, self.config.retrieval.bm25_top_k), doc_filters)
        timings["bm25_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], self.config.retrieval.rrf_k)
        chunks = self.store.get_chunks_by_ids([x["chunk_id"] for x in fused])
        candidates = [{**x, "text": chunks[x["chunk_id"]].text, "doc_id": chunks[x["chunk_id"]].doc_id}
                      for x in fused if x["chunk_id"] in chunks]
        reranked = self.reranker.rerank(question, candidates)[: self.config.retrieval.rerank_top_k]
        timings["fusion_rerank_seconds"] = time.perf_counter() - start
        blocks, citations = self._expand(reranked, top_k)
        context = "\n\n".join(f"[C{i}] {b.document_title}, page {b.page}, block {b.block_id}, {b.modality}\n{b.text}"
                              for i, b in enumerate(blocks, start=1))
        return RetrievalResult(citations, blocks, context, timings)

    def _expand(self, hits: list[dict], top_k: int) -> tuple[list[RetrievedBlock], list[Citation]]:
        retrieved: list[RetrievedBlock] = []
        citations: list[Citation] = []
        seen: set[str] = set()
        max_citations = max(top_k, top_k * 4)
        for hit in hits[:top_k]:
            for block in self.store.get_blocks_for_chunk(hit["chunk_id"]):
                for expanded in self.store.get_adjacent_blocks(block, self.config.chunking.adjacent_block_window):
                    if expanded.block_id in seen:
                        continue
                    seen.add(expanded.block_id)
                    doc = self.store.get_document(expanded.doc_id)
                    title = doc.title if doc else expanded.doc_id
                    asset_url = f"/assets/{expanded.asset_id}" if expanded.asset_id else ""
                    text = expanded.content_for_index()
                    rb = RetrievedBlock(expanded.block_id, hit["chunk_id"], expanded.doc_id, title, expanded.page_num,
                                        expanded.modality, float(hit.get("score", 0.0)), text, asset_url)
                    retrieved.append(rb)
                    if text and len(citations) < max_citations:
                        citations.append(Citation(expanded.doc_id, title, expanded.page_num, expanded.block_id,
                                                  expanded.modality, rb.score, expanded.snippet(), asset_url))
        return retrieved, citations


class LocalReranker:
    def __init__(self, model_name: str = ""):
        self._model = None
        if model_name:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(model_name)
            except Exception as exc:
                logger.warning("Reranker unavailable, using lexical rerank: %s", exc)

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        if self._model is not None:
            scores = self._model.predict([(question, c["text"]) for c in candidates])
            for c, score in zip(candidates, scores):
                c["score"] = float(score)
        else:
            q = set(_terms(question))
            for c in candidates:
                c["score"] = float(c.get("score", 0.0)) + len(q & set(_terms(c.get("text", "")))) / max(1, len(q))
        return sorted(candidates, key=lambda x: x["score"], reverse=True)


def reciprocal_rank_fusion(result_sets: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for result in result_sets:
        for rank, row in enumerate(result, start=1):
            cid = row["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            rows.setdefault(cid, row)
    fused = [{**rows[cid], "score": score} for cid, score in scores.items()]
    return sorted(fused, key=lambda x: x["score"], reverse=True)


def _filter(rows: list[dict], docs: list[str] | None) -> list[dict]:
    return rows if not docs else [r for r in rows if r.get("doc_id") in set(docs)]


def _terms(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]{2,}", text or "")]

