import logging
import asyncio
import sys
from src.defaults import DEFAULT_LLM_MODEL
from src.lightrag_compat import make_query_param

logger = logging.getLogger(__name__)


def _progress_status(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _use_local_backend(
    rag_backend: str,
    *,
    timeout_seconds: int,
    progress_enabled: bool,
) -> bool:
    backend = rag_backend.lower()
    if backend == "local":
        _progress_status("Using local vector RAG backend.", enabled=progress_enabled)
        return True
    if backend == "lightrag":
        _progress_status("Using LightRAG backend.", enabled=progress_enabled)
        return False
    if backend != "auto":
        raise ValueError("rag_backend must be one of: auto, lightrag, local")

    _progress_status(
        f"Checking LightRAG core import health ({timeout_seconds}s timeout)...",
        enabled=progress_enabled,
    )
    from src.lightrag_compat import lightrag_core_imports

    if lightrag_core_imports(timeout_seconds=timeout_seconds):
        _progress_status("LightRAG core import check passed.", enabled=progress_enabled)
        return False

    _progress_status(
        "LightRAG core import timed out or failed; falling back to local vector RAG backend.",
        enabled=progress_enabled,
    )
    return True


class QueryEngine:
    def __init__(
        self,
        working_dir="./db",
        model=DEFAULT_LLM_MODEL,
        embedding_backend: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        embedding_local_files_only: bool = True,
        embedding_batch_size: int | None = None,
        embedding_timeout: float | None = None,
        tokenizer_backend: str = "byte",
        rag_backend: str = "auto",
        lightrag_import_timeout: int = 10,
        progress_enabled: bool = True,
    ):
        self.local_engine = None
        if _use_local_backend(
            rag_backend,
            timeout_seconds=lightrag_import_timeout,
            progress_enabled=progress_enabled,
        ):
            from src.local_rag import LocalQueryEngine

            self.local_engine = LocalQueryEngine(
                working_dir=working_dir,
                model=model,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                embedding_local_files_only=embedding_local_files_only,
                embedding_batch_size=embedding_batch_size,
                embedding_timeout=embedding_timeout,
                progress_enabled=progress_enabled,
            )
            return

        from src.utils import create_lightrag_instance

        self.rag = create_lightrag_instance(
            working_dir=working_dir,
            model=model,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            embedding_local_files_only=embedding_local_files_only,
            embedding_batch_size=embedding_batch_size,
            embedding_timeout=embedding_timeout,
            tokenizer_backend=tokenizer_backend,
            progress_enabled=progress_enabled,
        )
        self._initialized = False

    async def _async_ask(self, question: str, mode: str):
        from src.embeddings import embedding_mode

        if not getattr(self, "_initialized", False) and hasattr(self.rag, "initialize_storages"):
            await self.rag.initialize_storages()
            self._initialized = True

        # Set prefix to "search_query: " for Nomic asymmetric retrieval
        embedding_mode.set("query")
        query_param = make_query_param(mode)
        if asyncio.iscoroutinefunction(self.rag.aquery):
            return await self.rag.aquery(question, param=query_param)
        else:
            return self.rag.query(question, param=query_param)

    def ask(self, question: str, mode: str = "hybrid"):
        """
        Queries the knowledge graph and vector DB.
        Modes: 'local', 'global', 'hybrid'
        """
        if self.local_engine is not None:
            return self.local_engine.ask(question, mode=mode)

        logger.info(f"Querying with mode '{mode}': {question}")
        response = asyncio.run(self._async_ask(question, mode))
        return response


if __name__ == "__main__":
    engine = QueryEngine()
    ans = engine.ask("What is the relationship between entropy and information theory?")
    print(ans)
