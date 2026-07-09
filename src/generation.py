from __future__ import annotations

import logging
import time

from src.config import PipelineConfig
from src.retrieval import Retriever
from src.schema import QueryResponse
from src.store import SQLiteBlockStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are the local NUS FSAE knowledge-transfer assistant. Answer only from supplied context and cite claims with [C1]-style citations."


class QueryService:
    def __init__(self, config: PipelineConfig, store: SQLiteBlockStore):
        self.config = config
        self.store = store
        self.retriever = Retriever(config, store)

    def ask(self, question: str, mode: str = "hybrid", top_k: int | None = None,
            doc_filters: list[str] | None = None) -> QueryResponse:
        start = time.perf_counter()
        retrieval = self.retriever.retrieve(question, mode, top_k, doc_filters)
        answer, gen_time = self._generate(question, retrieval.context)
        timings = dict(retrieval.timings)
        timings["generation_seconds"] = gen_time
        timings["total_seconds"] = time.perf_counter() - start
        return QueryResponse(answer, retrieval.citations, retrieval.retrieved_blocks, timings)

    def _generate(self, question: str, context: str) -> tuple[str, float]:
        start = time.perf_counter()
        if not context.strip():
            return "I could not find relevant local source blocks for that question.", time.perf_counter() - start
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}\n\nContext:\n{context}\n\nAnswer concisely with citations."
        try:
            import ollama
            response = ollama.generate(model=self.config.models.llm_model, prompt=prompt, keep_alive="30m", stream=False)
            text = getattr(response, "response", "") or response.get("response", "")
            if text.strip():
                return text.strip(), time.perf_counter() - start
        except Exception as exc:
            logger.warning("Ollama generation unavailable, using extractive fallback: %s", exc)
        return _fallback(context), time.perf_counter() - start


def _fallback(context: str) -> str:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    headers = [line for line in lines if line.startswith("[C")]
    snippets = []
    for header in headers[:4]:
        idx = lines.index(header)
        snippets.append(f"{header}: {lines[idx + 1][:300] if idx + 1 < len(lines) else ''}")
    return "Local generator unavailable. Most relevant retrieved evidence:\n" + "\n".join(snippets)

