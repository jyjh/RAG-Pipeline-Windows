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
        if self.config.embedding_model.lower() in {"hash", "deterministic-hash"}:
            self._model = False
            self.backend = "hash"
            return None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model, trust_remote_code=True, device="cpu")
            self.backend = "sentence-transformers"
            return self._model
        except Exception as exc:
            if not self.config.allow_hash_embeddings:
                raise
            logger.warning("Using deterministic hash embeddings: %s", exc)
            self._model = False
            self.backend = "hash"
            return None

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        return self._embed(texts, "search_document: ")

    def embed_query(self, text: str) -> EmbeddingResult:
        return self._embed([text], "search_query: ")

    def _embed(self, texts: list[str], prefix: str) -> EmbeddingResult:
        model = self._load_model()
        if model:
            rows = model.encode([prefix + t for t in texts], convert_to_numpy=True)
            return EmbeddingResult([_normalize([float(x) for x in row[: self.dim]]) for row in rows], self.backend)
        return EmbeddingResult([_hash_embedding(prefix + t, self.dim) for t in texts], self.backend)


def _hash_embedding(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(digest[:4], "little") % dim] += -1.0 if digest[4] & 1 else 1.0
    return _normalize(vector)


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return values if norm == 0.0 else [v / norm for v in values]
