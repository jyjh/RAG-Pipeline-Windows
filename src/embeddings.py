from __future__ import annotations

import contextvars
import json
import hashlib
import logging
import re
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _ollama_pull_command(model_name: str) -> str:
    executable = shutil.which("ollama")
    if executable is None:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidate = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
            if os.path.exists(candidate):
                executable = candidate
    if executable is None:
        executable = "ollama"
    if " " in executable:
        executable = f'"{executable}"'
    return f"{executable} pull {model_name}"


def _positive_int(value: int | str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _positive_float(value: float | str | None, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.1, parsed)


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host:
        return "http://127.0.0.1:11434"
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")
    return f"http://{host.rstrip('/')}"

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

    def __init__(
        self,
        model_name="nomic-embed-text",
        *,
        backend: str = "ollama",
        local_files_only: bool = True,
        ollama_batch_size: int | None = None,
        ollama_timeout: float | None = None,
    ):
        self.model_name = model_name
        self.backend = self._normalize_backend(backend)
        self.ollama_batch_size = _positive_int(
            ollama_batch_size or os.environ.get("OLLAMA_EMBED_BATCH_SIZE"),
            8,
        )
        self.ollama_timeout = _positive_float(
            ollama_timeout or os.environ.get("OLLAMA_EMBED_TIMEOUT"),
            30.0,
        )
        self.model = None
        self._cache: dict[tuple[int, int, str], Any] = {}

        if self.backend == "hash":
            logger.warning("Using deterministic hash embeddings; retrieval quality is degraded.")
            return
        if self.backend == "ollama":
            if model_name == "nomic-ai/nomic-embed-text-v1.5":
                model_name = "nomic-embed-text"
                self.model_name = model_name
            logger.info("Using Ollama embedding model: %s", model_name)
            _status(
                f"Using Ollama embedding model: {model_name} "
                f"(batch_size={self.ollama_batch_size}, timeout={self.ollama_timeout:g}s)"
            )
            return

        logger.info(f"Loading embedding model on CPU: {model_name}")
        logger.info("SentenceTransformer local_files_only=%s", local_files_only)
        model_source = model_name
        if local_files_only:
            _status(f"Resolving local Hugging Face snapshot for {model_name}...")
            from huggingface_hub import snapshot_download

            model_source = snapshot_download(repo_id=model_name, local_files_only=True)
            _status(f"Using local embedding snapshot: {model_source}")

        _status("Importing sentence_transformers package...")
        from sentence_transformers import SentenceTransformer

        _status("Constructing SentenceTransformer model...")
        self.model = SentenceTransformer(
            model_source,
            trust_remote_code=True,
            device="cpu",
            local_files_only=local_files_only,
        )
        _status("SentenceTransformer embedding model loaded.")

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        normalized = backend.lower().replace("_", "-")
        if normalized in {"sentence-transformer", "sentence-transformers", "st"}:
            return "sentence-transformers"
        if normalized == "ollama":
            return "ollama"
        if normalized in {"hash", "hashed"}:
            return "hash"
        raise ValueError("embedding backend must be one of: hash, ollama, sentence-transformers")

    def _cache_key(self, text: str, truncate_dim: int, prefix: str) -> tuple[int, int, str]:
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        return (int(digest[:16], 16), truncate_dim, prefix)

    def get_mrl_embeddings(
        self,
        texts: list[str],
        truncate_dim: int = 768,
        prefix: str = "search_document: ",
    ):
        """
        Generates Matryoshka embeddings truncated to truncate_dim and L2-normalized.

        Cache-aware: checks _cache first per text; only computes misses.
        Returns a single (N, truncate_dim) array.
        """
        import numpy as np

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
            if self.backend == "hash":
                truncated = np.array(
                    [
                        self._hash_embedding(f"{prefix}{text}", truncate_dim)
                        for text in to_compute_texts
                    ],
                    dtype=np.float32,
                )
            elif self.backend == "ollama":
                truncated = self._ollama_embeddings(
                    [f"{prefix}{text}" for text in to_compute_texts],
                    truncate_dim,
                )
            else:
                assert self.model is not None
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

    def _ollama_embeddings(self, texts: list[str], truncate_dim: int):
        import numpy as np

        if not texts:
            return np.empty((0, truncate_dim), dtype=np.float32)

        embeddings = []
        batch_size = self.ollama_batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start : start + batch_size]
            total_chars = sum(len(text) for text in batch)
            try:
                _status(
                    f"Requesting embeddings from Ollama model: {self.model_name} "
                    f"batch {batch_number}/{total_batches} "
                    f"({len(batch)} text(s), {total_chars} chars, timeout={self.ollama_timeout:g}s)"
                )
                payload_input = batch[0] if len(batch) == 1 else batch
                response = self._ollama_api(
                    "/api/embed",
                    {"model": self.model_name, "input": payload_input},
                )
                batch_embeddings = response.get("embeddings")
            except Exception as exc:
                raise RuntimeError(
                    f"Ollama embedding failed for model '{self.model_name}' "
                    f"on batch {batch_number}/{total_batches}. "
                    f"Run `{_ollama_pull_command(self.model_name)}` and ensure Ollama is running. "
                    f"Original error: {exc}"
                ) from exc

            if batch_embeddings is None:
                batch_embeddings = []
                for text_index, text in enumerate(batch, start=1):
                    _status(
                        f"Requesting single embedding fallback from Ollama model: "
                        f"{self.model_name} batch {batch_number}/{total_batches} "
                        f"text {text_index}/{len(batch)} ({len(text)} chars, timeout={self.ollama_timeout:g}s)"
                    )
                    response = self._ollama_api(
                        "/api/embeddings",
                        {"model": self.model_name, "prompt": text},
                    )
                    embedding = response.get("embedding")
                    if embedding is None:
                        raise RuntimeError("Ollama embedding response did not contain an embedding.")
                    batch_embeddings.append(embedding)

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(batch_embeddings)} embedding(s) for "
                    f"{len(batch)} input text(s) in batch {batch_number}/{total_batches}."
                )
            embeddings.extend(batch_embeddings)

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] < truncate_dim:
            padded = np.zeros((vectors.shape[0], truncate_dim), dtype=np.float32)
            padded[:, : vectors.shape[1]] = vectors
            vectors = padded
        else:
            vectors = vectors[:, :truncate_dim].copy()

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def _ollama_api(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{_ollama_host()}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.ollama_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.ollama_timeout:g}s at {url}."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed at {url}: {exc}") from exc

    @staticmethod
    def _hash_embedding(text: str, dim: int):
        import numpy as np

        vec = np.zeros(dim, dtype=np.float32)
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        if not tokens:
            tokens = [text]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = 1.0 + digest[5] / 255.0
            vec[index] += sign * weight

        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
