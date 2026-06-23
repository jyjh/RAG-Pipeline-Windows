from __future__ import annotations

import logging
import time

from src.config import PipelineConfig
from src.retrieval import Retriever
from src.schema import QueryResponse
from src.store import SQLiteBlockStore

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the local NUS FSAE knowledge-transfer assistant.
Answer only from the supplied context. Cite every factual claim with citation keys
like [C1] or [C2]. If the context is insufficient, say what is missing."""


class QueryService:
    def __init__(self, config: PipelineConfig, store: SQLiteBlockStore):
        self.config = config
        self.store = store
        self.retriever = Retriever(config, store)

    def ask(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        doc_filters: list[str] | None = None,
    ) -> QueryResponse:
        started = time.perf_counter()
        retrieval = self.retriever.retrieve(question, mode=mode, top_k=top_k, doc_filters=doc_filters)
        answer, generation_seconds = self._generate_answer(question, retrieval.context)
        timings = dict(retrieval.timings)
        timings["generation_seconds"] = generation_seconds
        timings["total_seconds"] = time.perf_counter() - started
        return QueryResponse(
            answer=answer,
            citations=retrieval.citations,
            retrieved_blocks=retrieval.retrieved_blocks,
            timings=timings,
        )

    def _generate_answer(self, question: str, context: str) -> tuple[str, float]:
        started = time.perf_counter()
        if not context.strip():
            return "I could not find relevant local source blocks for that question.", time.perf_counter() - started

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer with concise technical reasoning and citations."
        )
        try:
            import ollama

            response = ollama.generate(
                model=self.config.models.llm_model,
                prompt=prompt,
                keep_alive="30m",
                stream=False,
            )
            text = getattr(response, "response", "") or response.get("response", "")
            if text.strip():
                return text.strip(), time.perf_counter() - started
        except Exception as exc:
            logger.warning("Ollama generation unavailable, using extractive fallback: %s", exc)

        return _extractive_fallback(context), time.perf_counter() - started


def _extractive_fallback(context: str) -> str:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    cited = [line for line in lines if line.startswith("[C")]
    snippets = []
    for i, header in enumerate(cited[:4], start=1):
        try:
            start = lines.index(header)
        except ValueError:
            start = 0
        body = lines[start + 1] if start + 1 < len(lines) else ""
        snippets.append(f"{header}: {body[:300]}")
    if not snippets:
        return "Relevant source blocks were retrieved, but no concise extractive answer could be formed."
    return "Local generator unavailable. Most relevant retrieved evidence:\n" + "\n".join(snippets)

