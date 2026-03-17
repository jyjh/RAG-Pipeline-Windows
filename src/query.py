import logging
import asyncio
from lightrag import QueryParam
from src.embeddings import embedding_mode
from src.utils import create_lightrag_instance

logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, working_dir="./db", model="deepseek-r1:32b"):
        self.rag = create_lightrag_instance(working_dir=working_dir, model=model)

    async def _async_ask(self, question: str, mode: str):
        # Set prefix to "search_query: " for Nomic asymmetric retrieval
        embedding_mode.set("query")
        if asyncio.iscoroutinefunction(self.rag.aquery):
            return await self.rag.aquery(question, param=QueryParam(mode=mode))
        else:
            return self.rag.query(question, param=QueryParam(mode=mode))

    def ask(self, question: str, mode: str = "hybrid"):
        """
        Queries the knowledge graph and vector DB.
        Modes: 'local', 'global', 'hybrid'
        """
        logger.info(f"Querying with mode '{mode}': {question}")
        response = asyncio.run(self._async_ask(question, mode))
        return response


if __name__ == "__main__":
    engine = QueryEngine()
    ans = engine.ask("What is the relationship between entropy and information theory?")
    print(ans)
