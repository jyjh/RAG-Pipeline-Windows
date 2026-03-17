import contextvars
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Thread-safe flag for asymmetric Nomic prefix switching.
# Set to "query" before rag.query(), leave as "document" (default) during indexing.
# contextvars propagates correctly through asyncio.to_thread().
embedding_mode = contextvars.ContextVar("embedding_mode", default="document")


class EmbeddingEngine:
    """
    CPU-bound Nomic MRL embedding engine with an internal cache to avoid
    double-embedding during chunking_func dedup + LightRAG indexing.

    The cache is keyed on (text, truncate_dim, prefix) and stores the
    computed numpy vector. When the custom chunking_func embeds chunks for
    ANN dedup, results are cached. When LightRAG subsequently calls
    mrl_embedding_func on the same text, the cache returns the precomputed
    vector — zero redundant computation.
    """

    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5"):
        logger.info(f"Loading embedding model on CPU: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cpu")
        self._cache: dict[tuple[int, int, str], np.ndarray] = {}

    def _cache_key(self, text: str, truncate_dim: int, prefix: str) -> tuple[int, int, str]:
        return (hash(text), truncate_dim, prefix)

    def get_mrl_embeddings(
        self,
        texts: list[str],
        truncate_dim: int = 768,
        prefix: str = "search_document: ",
    ) -> np.ndarray:
        """
        Generates Matryoshka embeddings truncated to truncate_dim and L2-normalized.

        Cache-aware: checks _cache first per text; only computes misses.
        Returns a single (N, truncate_dim) array.
        """
        results = [None] * len(texts)
        to_compute_indices = []
        to_compute_texts = []

        for i, text in enumerate(texts):
            key = self._cache_key(text, truncate_dim, prefix)
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                to_compute_indices.append(i)
                to_compute_texts.append(text)

        if to_compute_texts:
            prefixed = [f"{prefix}{t}" for t in to_compute_texts]
            full = self.model.encode(prefixed, convert_to_numpy=True)
            truncated = full[:, :truncate_dim].copy()
            norms = np.linalg.norm(truncated, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            truncated = truncated / norms

            for j, idx in enumerate(to_compute_indices):
                key = self._cache_key(texts[idx], truncate_dim, prefix)
                self._cache[key] = truncated[j]
                results[idx] = truncated[j]

        return np.array(results)
