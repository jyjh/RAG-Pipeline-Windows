from __future__ import annotations

import asyncio
import os
import logging
import sys

logger = logging.getLogger(__name__)


def _progress_status(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


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
        from transformers import AutoTokenizer

        logger.info(f"Loading DeepSeek tokenizer: {model_id}")
        self._tok = AutoTokenizer.from_pretrained(model_id)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tok.decode(tokens)


class _ByteTokenizer:
    """Small reversible tokenizer used when Hugging Face tokenizer loading is not desired."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(int(token) for token in tokens if 0 <= int(token) <= 255).decode(
            "utf-8",
            errors="ignore",
        )


# Singleton — loaded once, reused by every LightRAG instance in this process.
_deepseek_tokenizer = None
_byte_tokenizer = None


def get_deepseek_tokenizer(tokenizer_backend: str = "byte"):
    global _deepseek_tokenizer, _byte_tokenizer
    backend = tokenizer_backend.lower()
    if backend in {"byte", "simple"}:
        if _byte_tokenizer is None:
            _byte_tokenizer = _ByteTokenizer()
        return _byte_tokenizer
    if backend not in {"deepseek", "huggingface", "hf"}:
        raise ValueError("tokenizer_backend must be one of: byte, deepseek")
    if _deepseek_tokenizer is None:
        _deepseek_tokenizer = _DeepSeekTokenizer()
    return _deepseek_tokenizer


# ---------------------------------------------------------------------------
#  VRAM management
# ---------------------------------------------------------------------------
def unload_model(model_name: str):
    """Unload a model from VRAM via Ollama (keep_alive="0" for immediate eviction)."""
    try:
        import ollama

        logger.info(f"Unloading model: {model_name}")
        ollama.generate(model=model_name, prompt=" ", keep_alive="0")
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")


def load_model(model_name: str):
    """Pre-load a model into VRAM via Ollama."""
    try:
        import ollama

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
def create_lightrag_instance(
    working_dir="./db",
    model=None,
    engine=None,
    embedding_backend: str = "ollama",
    embedding_model: str = "nomic-embed-text",
    embedding_local_files_only: bool = True,
    embedding_batch_size: int | None = None,
    embedding_timeout: float | None = None,
    tokenizer_backend: str = "byte",
    progress_enabled: bool = True,
):
    """
    Creates a LightRAG instance configured with:
      - LanceDB vector storage (via custom adapter)
      - DeepSeek Qwen2 tokenizer (aligned to executing LLM)
      - Nomic MRL 768d embeddings on CPU with contextvars prefix switching
      - Ollama LLM backend for entity extraction

    Pass an existing EmbeddingEngine via `engine` to share the model and
    embedding cache across deduplication, indexing, and bridge edge computation.
    """
    _progress_status("LightRAG setup: importing NumPy...", enabled=progress_enabled)
    import numpy as np

    _progress_status("LightRAG setup: importing project defaults and embedding components...", enabled=progress_enabled)
    from src.defaults import DEFAULT_LLM_MODEL
    from src.embeddings import EmbeddingEngine, embedding_mode
    _progress_status("LightRAG setup: importing compatibility layer...", enabled=progress_enabled)
    from src.lightrag_compat import load_lightrag_api, register_vector_storage

    if model is None:
        model = DEFAULT_LLM_MODEL

    _progress_status(f"LightRAG setup: ensuring working directory exists: {working_dir}", enabled=progress_enabled)
    os.makedirs(working_dir, exist_ok=True)
    _progress_status("LightRAG setup: loading LightRAG API...", enabled=progress_enabled)
    lightrag_api = load_lightrag_api()

    # Register LanceDB storage adapter before LightRAG validates storage names.
    _progress_status("LightRAG setup: registering LanceDB vector storage...", enabled=progress_enabled)
    register_vector_storage(lightrag_api, "LanceDBStorage", "src.lancedb_storage")

    if engine is None:
        _progress_status(
            f"LightRAG setup: creating embedding engine ({embedding_backend}, {embedding_model})...",
            enabled=progress_enabled,
        )
        engine = EmbeddingEngine(
            model_name=embedding_model,
            backend=embedding_backend,
            local_files_only=embedding_local_files_only,
            ollama_batch_size=embedding_batch_size,
            ollama_timeout=embedding_timeout,
        )
    else:
        _progress_status("LightRAG setup: reusing existing embedding engine.", enabled=progress_enabled)

    _progress_status(f"LightRAG setup: loading tokenizer backend: {tokenizer_backend}", enabled=progress_enabled)
    tokenizer = get_deepseek_tokenizer(tokenizer_backend)

    async def mrl_embedding_func(texts: list[str]) -> np.ndarray:
        mode = embedding_mode.get()
        prefix = "search_query: " if mode == "query" else "search_document: "
        result = await asyncio.to_thread(
            engine.get_mrl_embeddings, texts, 768, prefix
        )
        return result

    _progress_status(
        f"LightRAG setup: constructing LightRAG object with local Ollama LLM: {model}",
        enabled=progress_enabled,
    )
    rag = lightrag_api.LightRAG(
        working_dir=working_dir,
        llm_model_func=lightrag_api.ollama_model_complete,
        llm_model_name=model,
        entity_extract_max_gleaning=1,
        chunk_token_size=1200,
        chunk_overlap_token_size=200,
        tokenizer=tokenizer,
        vector_storage="LanceDBStorage",
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": 0.2,
        },
        embedding_func=lightrag_api.EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=mrl_embedding_func,
        ),
    )
    _progress_status("LightRAG setup: LightRAG object created.", enabled=progress_enabled)
    return rag
