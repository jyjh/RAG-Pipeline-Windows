import asyncio
import os
import logging

import numpy as np
import ollama
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc
from transformers import AutoTokenizer

from src.embeddings import EmbeddingEngine, embedding_mode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Register LanceDB storage adapter with LightRAG before any instance is built
# ---------------------------------------------------------------------------
from lightrag.kg import STORAGES  # noqa: E402

STORAGES["LanceDBStorage"] = "src.lancedb_storage"


# ---------------------------------------------------------------------------
#  DeepSeek Qwen2 tokenizer shim
# ---------------------------------------------------------------------------
class _DeepSeekTokenizer:
    """
    Wraps HuggingFace AutoTokenizer for DeepSeek-R1-Distill-Qwen-32B into
    the interface LightRAG expects: .encode(str) -> list[int] and
    .decode(list[int]) -> str.

    This replaces the default tiktoken o200k_base (200K vocab) with the
    actual Qwen2 tokenizer (152K vocab) so that chunk token counts exactly
    match the executing LLM's token budget.
    """

    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"):
        logger.info(f"Loading DeepSeek tokenizer: {model_id}")
        self._tok = AutoTokenizer.from_pretrained(model_id)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tok.decode(tokens)


# Singleton — loaded once, reused by every LightRAG instance in this process.
_deepseek_tokenizer = None


def get_deepseek_tokenizer() -> _DeepSeekTokenizer:
    global _deepseek_tokenizer
    if _deepseek_tokenizer is None:
        _deepseek_tokenizer = _DeepSeekTokenizer()
    return _deepseek_tokenizer


# ---------------------------------------------------------------------------
#  VRAM management
# ---------------------------------------------------------------------------
def unload_model(model_name: str):
    """Unload a model from VRAM via Ollama (keep_alive="0" for immediate eviction)."""
    try:
        logger.info(f"Unloading model: {model_name}")
        ollama.generate(model=model_name, prompt=" ", keep_alive="0")
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")


def load_model(model_name: str):
    """Pre-load a model into VRAM via Ollama."""
    try:
        logger.info(f"Loading model: {model_name}")
        ollama.generate(model=model_name, prompt="Hello", keep_alive="30m")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")


def manage_vram(active_model: str, model_to_unload: str = None):
    """Switches between models to optimize VRAM usage."""
    if model_to_unload:
        unload_model(model_to_unload)
    load_model(active_model)


# ---------------------------------------------------------------------------
#  LightRAG instance factory
# ---------------------------------------------------------------------------
def create_lightrag_instance(working_dir="./db", model="deepseek-r1:32b", engine=None):
    """
    Creates a LightRAG instance configured with:
      - LanceDB vector storage (via custom adapter)
      - DeepSeek Qwen2 tokenizer (aligned to executing LLM)
      - Nomic MRL 768d embeddings on CPU with contextvars prefix switching
      - Ollama LLM backend for entity extraction

    Pass an existing EmbeddingEngine via `engine` to share the model and
    embedding cache across deduplication, indexing, and bridge edge computation.
    """
    os.makedirs(working_dir, exist_ok=True)

    if engine is None:
        engine = EmbeddingEngine()

    tokenizer = get_deepseek_tokenizer()

    async def mrl_embedding_func(texts: list[str]) -> np.ndarray:
        mode = embedding_mode.get()
        prefix = "search_query: " if mode == "query" else "search_document: "
        result = await asyncio.to_thread(
            engine.get_mrl_embeddings, texts, 768, prefix
        )
        return result

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=model,
        entity_extract_max_gleaning=1,
        chunk_token_size=1200,
        chunk_overlap_token_size=200,
        tokenizer=tokenizer,
        vector_storage="LanceDBStorage",
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": 0.2,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=mrl_embedding_func,
        ),
    )
