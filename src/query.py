import logging
from src.defaults import DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(
        self,
        working_dir="./db",
        model=DEFAULT_LLM_MODEL,
        embedding_model: str = "nomic-embed-text",
        embedding_batch_size: int | None = None,
        embedding_timeout: float | None = None,
        llm_num_predict: int | None = None,
        llm_timeout: float | None = None,
        temperature: float | None = None,
        sampler_top_k: int | None = None,
        context_window: int | None = None,
        progress_enabled: bool = True,
    ):
        from src.local_rag import LocalQueryEngine

        self.local_engine = LocalQueryEngine(
            working_dir=working_dir,
            model=model,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_timeout=embedding_timeout,
            num_predict=llm_num_predict,
            llm_timeout=llm_timeout,
            temperature=temperature,
            sampler_top_k=sampler_top_k,
            context_window=context_window,
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
