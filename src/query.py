import logging
from src.defaults import DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(
        self,
        working_dir="./db",
        asset_dir: str | None = None,
        trust_path: str | None = None,
        model=DEFAULT_LLM_MODEL,
        embedding_model: str = "nomic-embed-text",
        embedding_batch_size: int | None = None,
        embedding_timeout: float | None = None,
        llm_num_predict: int | None = None,
        llm_timeout: float | None = None,
        temperature: float | None = None,
        sampler_top_k: int | None = None,
        context_window: int | None = None,
        retrieval_candidate_k: int | None = None,
        retrieval_min_score: float | None = None,
        retrieval_relative_cutoff: float | None = None,
        context_token_fraction: float | None = None,
        web_search_enabled: bool = True,
        web_search_timeout: float | None = None,
        web_search_max_results: int | None = None,
        ollama_health_check_interval: float | None = None,
        ollama_max_lost_health_checks: int | None = None,
        system_prompt: str | None = None,
        planner_model: str | None = None,
        planner_enabled: bool = True,
        planner_max_queries: int | None = None,
        progress_enabled: bool = True,
    ):
        from src.local_rag import LocalQueryEngine

        self.local_engine = LocalQueryEngine(
            working_dir=working_dir,
            asset_dir=asset_dir,
            trust_path=trust_path,
            model=model,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_timeout=embedding_timeout,
            num_predict=llm_num_predict,
            llm_timeout=llm_timeout,
            temperature=temperature,
            sampler_top_k=sampler_top_k,
            context_window=context_window,
            retrieval_candidate_k=retrieval_candidate_k,
            retrieval_min_score=retrieval_min_score,
            retrieval_relative_cutoff=retrieval_relative_cutoff,
            context_token_fraction=context_token_fraction,
            web_search_enabled=web_search_enabled,
            web_search_timeout=web_search_timeout,
            web_search_max_results=web_search_max_results,
            ollama_health_check_interval=ollama_health_check_interval,
            ollama_max_lost_health_checks=ollama_max_lost_health_checks,
            system_prompt=system_prompt,
            planner_model=planner_model,
            planner_enabled=planner_enabled,
            planner_max_queries=planner_max_queries,
            progress_enabled=progress_enabled,
        )

    def ask(self, question: str):
        """
        Queries the local vector index and asks Ollama to answer from retrieved context.
        """
        logger.info("Querying local Ollama index: %s", question)
        return self.local_engine.ask(question)

    def ask_stream(self, question: str):
        """
        Streams a local vector-index answer from Ollama.
        """
        logger.info("Streaming local Ollama index query: %s", question)
        return self.local_engine.ask_stream(question)

    def ask_stream_events(self, question: str):
        """
        Streams local answer events with separate thinking and answer chunks.
        """
        logger.info("Streaming local Ollama index query events: %s", question)
        return self.local_engine.ask_stream_events(question)


if __name__ == "__main__":
    engine = QueryEngine()
    ans = engine.ask("What is the relationship between entropy and information theory?")
    print(ans)
