from __future__ import annotations

import json
import hashlib
import logging
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

class EmbeddingEngine:
    """
    Ollama-backed embedding engine with an internal cache.

    The cache is keyed on (text, truncate_dim, prefix) and stores the
    computed numpy vector. When the local index or query path requests
    the same text again, the cache returns the precomputed
    vector and avoids redundant computation.
    """

    def __init__(
        self,
        model_name="nomic-embed-text",
        *,
        ollama_batch_size: int | None = None,
        ollama_timeout: float | None = None,
    ):
        self.model_name = model_name
        self.ollama_batch_size = _positive_int(
            ollama_batch_size or os.environ.get("OLLAMA_EMBED_BATCH_SIZE"),
            8,
        )
        self.ollama_timeout = _positive_float(
            ollama_timeout or os.environ.get("OLLAMA_EMBED_TIMEOUT"),
            30.0,
        )
        self._cache: dict[tuple[int, int, str], Any] = {}

        if model_name == "nomic-ai/nomic-embed-text-v1.5":
            model_name = "nomic-embed-text"
            self.model_name = model_name
        logger.info("Using Ollama embedding model: %s", model_name)
        _status(
            f"Using Ollama embedding model: {model_name} "
            f"(batch_size={self.ollama_batch_size}, timeout={self.ollama_timeout:g}s)"
        )

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
        Generates Ollama embeddings truncated to truncate_dim and L2-normalized.

        Use `search_document: ` during indexing and `search_query: ` during retrieval
        with `nomic-embed-text`.
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
            truncated = self._ollama_embeddings(
                [f"{prefix}{text}" for text in to_compute_texts],
                truncate_dim,
            )

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

