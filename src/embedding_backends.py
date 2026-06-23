from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass

from src.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    backend: str


class EmbeddingProvider:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.dim = config.embedding_dim
        self._model = None
        self.backend = "hash"

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model on CPU: %s", self.config.embedding_model)
            self._model = SentenceTransformer(
                self.config.embedding_model,
                trust_remote_code=True,
                device="cpu",
            )
            self.backend = "sentence-transformers"
            return self._model
        except Exception as exc:
            if not self.config.allow_hash_embeddings:
                raise
            logger.warning("Using deterministic hash embeddings: %s", exc)
            self.backend = "hash"
            self._model = False
            return None

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        return self._embed(texts, prefix="search_document: ")

    def embed_query(self, text: str) -> EmbeddingResult:
        return self._embed([text], prefix="search_query: ")

    def _embed(self, texts: list[str], prefix: str) -> EmbeddingResult:
        model = self._load_model()
        if model:
            vectors = model.encode([prefix + text for text in texts], convert_to_numpy=True)
            out = []
            for row in vectors:
                values = [float(x) for x in row[: self.dim]]
                out.append(_normalize(values))
            return EmbeddingResult(vectors=out, backend=self.backend)
        return EmbeddingResult(
            vectors=[_hash_embedding(prefix + text, self.dim) for text in texts],
            backend=self.backend,
        )


def _hash_embedding(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = -1.0 if digest[4] & 1 else 1.0
        vector[bucket] += sign
    return _normalize(vector)


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return values
    return [v / norm for v in values]

