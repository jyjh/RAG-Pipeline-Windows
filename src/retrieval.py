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

    def retrieve(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        doc_filters: list[str] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.config.retrieval.top_k
        mode = (mode or "hybrid").lower()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        vector_hits = []
        if mode in {"hybrid", "vector", "local"}:
            query_vector = self.embedder.embed_query(question).vectors[0]
            vector_hits = self.vector_index.search(query_vector, self.config.retrieval.vector_top_k)
            vector_hits = _filter_docs(vector_hits, doc_filters)
        timings["vector_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        bm25_hits = []
        if mode in {"hybrid", "bm25", "text", "global"}:
            bm25_hits = self.store.search_chunks_fts(question, self.config.retrieval.bm25_top_k)
            bm25_hits = _filter_docs(bm25_hits, doc_filters)
        timings["bm25_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [vector_hits, bm25_hits],
            rrf_k=self.config.retrieval.rrf_k,
        )
        chunk_map = self.store.get_chunks_by_ids([item["chunk_id"] for item in fused])
        candidates = [
            {
                **item,
                "text": chunk_map[item["chunk_id"]].text,
                "doc_id": chunk_map[item["chunk_id"]].doc_id,
            }
            for item in fused
            if item["chunk_id"] in chunk_map
        ]
        reranked = self.reranker.rerank(question, candidates)[: self.config.retrieval.rerank_top_k]
        timings["fusion_rerank_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        retrieved_blocks, citations = self._expand_blocks(reranked, top_k)
        context = _context_from_blocks(retrieved_blocks)
        timings["context_seconds"] = time.perf_counter() - started
        return RetrievalResult(
            citations=citations,
            retrieved_blocks=retrieved_blocks,
            context=context,
            timings=timings,
        )

    def _expand_blocks(self, hits: list[dict], top_k: int) -> tuple[list[RetrievedBlock], list[Citation]]:
        retrieved: list[RetrievedBlock] = []
        citations: list[Citation] = []
        seen_blocks: set[str] = set()
        seen_citations: set[str] = set()
        max_citations = max(top_k, top_k * 4)

        for hit in hits[:top_k]:
            chunk_id = hit["chunk_id"]
            score = float(hit.get("score", 0.0))
            blocks = self.store.get_blocks_for_chunk(chunk_id)
            expanded = []
            for block in blocks:
                expanded.extend(
                    self.store.get_adjacent_blocks(
                        block,
                        window=self.config.chunking.adjacent_block_window,
                    )
                )
            for block in expanded or blocks:
                if block.block_id in seen_blocks:
                    continue
                seen_blocks.add(block.block_id)
                document = self.store.get_document(block.doc_id)
                title = document.title if document else block.doc_id
                asset_url = f"/assets/{block.asset_id}" if block.asset_id else ""
                text = block.content_for_index()
                retrieved.append(
                    RetrievedBlock(
                        block_id=block.block_id,
                        chunk_id=chunk_id,
                        doc_id=block.doc_id,
                        document_title=title,
                        page=block.page_num,
                        modality=block.modality,
                        score=score,
                        text=text,
                        asset_url=asset_url,
                    )
                )
                if block.block_id not in seen_citations and text and len(citations) < max_citations:
                    seen_citations.add(block.block_id)
                    citations.append(
                        Citation(
                            doc_id=block.doc_id,
                            document_title=title,
                            page=block.page_num,
                            block_id=block.block_id,
                            modality=block.modality,
                            score=score,
                            snippet=block.snippet(),
                            asset_url=asset_url,
                        )
                    )
        return retrieved, citations


class LocalReranker:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self._model = None
        if model_name:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name)
            except Exception as exc:
                logger.warning("Reranker unavailable, using lexical rerank: %s", exc)

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        if self._model is not None:
            pairs = [(question, c["text"]) for c in candidates]
            scores = self._model.predict(pairs)
            for candidate, score in zip(candidates, scores):
                candidate["score"] = float(score)
            return sorted(candidates, key=lambda item: item["score"], reverse=True)

        q_terms = set(_terms(question))
        for candidate in candidates:
            c_terms = set(_terms(candidate.get("text", "")))
            lexical = len(q_terms & c_terms) / max(1, len(q_terms))
            candidate["score"] = float(candidate.get("score", 0.0)) + lexical
        return sorted(candidates, key=lambda item: item["score"], reverse=True)


def reciprocal_rank_fusion(result_sets: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    source_rows: dict[str, dict] = {}
    for rows in result_sets:
        for rank, row in enumerate(rows, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
            source_rows.setdefault(chunk_id, row)
    fused = []
    for chunk_id, score in scores.items():
        row = dict(source_rows[chunk_id])
        row["score"] = score
        fused.append(row)
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused


def _filter_docs(rows: list[dict], doc_filters: list[str] | None) -> list[dict]:
    if not doc_filters:
        return rows
    allowed = set(doc_filters)
    return [row for row in rows if row.get("doc_id") in allowed]


def _context_from_blocks(blocks: list[RetrievedBlock]) -> str:
    parts = []
    for index, block in enumerate(blocks, start=1):
        parts.append(
            f"[C{index}] {block.document_title}, page {block.page}, "
            f"block {block.block_id}, {block.modality}\n{block.text}"
        )
    return "\n\n".join(parts)


def _terms(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]{2,}", text or "")]
