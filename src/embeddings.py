from __future__ import annotations

import json
import hashlib
import logging
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
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


def _normalize_ollama_host(raw: str) -> str:
    """Normalize a single Ollama host string (scheme + trailing slash trim)."""
    host = raw.strip()
    if not host:
        return "http://127.0.0.1:11434"
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")
    return f"http://{host.rstrip('/')}"


def _resolve_ollama_hosts() -> list[str]:
    """Resolve the list of Ollama embedding endpoints.

    A comma-separated ``OLLAMA_EMBED_HOSTS`` env var enables multi-replica
    embedding: batches are round-robined across the listed hosts in parallel.
    When unset, falls back to the single ``OLLAMA_HOST`` (so the default is
    zero behavior change). Duplicates are removed while preserving order.

    Examples::

        OLLAMA_EMBED_HOSTS=http://gpu-a:11434,http://gpu-b:11434
        # -> ["http://gpu-a:11434", "http://gpu-b:11434"]
    """
    raw = os.environ.get("OLLAMA_EMBED_HOSTS", "").strip()
    if not raw:
        return [_ollama_host()]
    hosts: list[str] = []
    seen: set[str] = set()
    for piece in raw.split(","):
        normalized = _normalize_ollama_host(piece)
        if normalized not in seen:
            seen.add(normalized)
            hosts.append(normalized)
    return hosts or [_ollama_host()]


def _resolve_embed_concurrency() -> int:
    """Worker threads for parallel batch embedding.

    Default 1 = serial (preserves existing single-host behavior). When
    ``OLLAMA_EMBED_HOSTS`` lists multiple replicas the effective parallelism is
    ``max(concurrency, len(hosts))`` so every replica is used concurrently even
    if the operator left this at 1. Override with ``OLLAMA_EMBED_CONCURRENCY``
    to push more in-flight batches per host (e.g. with ``OLLAMA_NUM_PARALLEL>1``
    on the Ollama server).
    """
    raw = os.environ.get("OLLAMA_EMBED_CONCURRENCY", "").strip()
    try:
        value = int(raw) if raw else 1
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


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
        ollama_retries: int | None = None,
        max_cache_entries: int | None = None,
    ):
        self.model_name = model_name

        # Load the [embeddings] config once. Precedence for every knob below is:
        # explicit constructor arg > env var (OLLAMA_EMBED_*) > [embeddings]
        # config value > hardcoded default. The env vars remain the escape hatch
        # for ad-hoc overrides; surfacing them in config makes the dominant cost
        # at corpus scale (embedding throughput) discoverable and tunable.
        from src.config import load_config

        emb_cfg = load_config().embeddings

        self.ollama_batch_size = _positive_int(
            ollama_batch_size
            if ollama_batch_size is not None
            else os.environ.get("OLLAMA_EMBED_BATCH_SIZE"),
            emb_cfg.batch_size,
        )
        self._ollama_timeout_base = _positive_float(
            ollama_timeout
            if ollama_timeout is not None
            else os.environ.get("OLLAMA_EMBED_TIMEOUT"),
            emb_cfg.timeout_seconds,
        )
        # Batch-size-aware timeout. The configured ``timeout_seconds`` is
        # measured at ``timeout_batch_baseline`` batch size; larger batches get a
        # proportionally longer deadline (capped) so they don't silently trip
        # the retry loop on a slow GPU. With defaults (30s/128) this is a no-op
        # for batch_size<=128; batch_size=512 gets 120s. ``ollama_timeout`` is
        # the effective value used per request (read by ``_ollama_api``).
        baseline = max(1, int(emb_cfg.timeout_batch_baseline))
        scale = max(1.0, float(self.ollama_batch_size) / float(baseline))
        self.ollama_timeout = min(
            self._ollama_timeout_base * scale, float(emb_cfg.timeout_max_seconds)
        )
        # Bounded retry with exponential backoff for transient Ollama hiccups
        # (timeouts, brief connection resets). A single blip otherwise aborts an
        # entire multi-hour ingestion; the retry lets it ride out a transient
        # failure. Final-attempt failure still raises so callers can skip the
        # file (per-file isolation) rather than the whole corpus.
        self.ollama_retries = _positive_int(
            ollama_retries
            if ollama_retries is not None
            else os.environ.get("OLLAMA_EMBED_RETRIES"),
            emb_cfg.retries,
        )
        # Bounded LRU cache. At 100GB-scale cold indexing almost every chunk is
        # unique, so an unbounded dict would grow to hold every embedding for
        # the whole run (768 floats each) and OOM the process. A bounded LRU
        # caps the footprint while preserving the hit rate for the realistic
        # repeat cases (re-index reuse, repeated queries). Default 50k entries ≈
        # ~150MB at 768 float32 each; env-overridable.
        self.max_cache_entries = _positive_int(
            max_cache_entries
            if max_cache_entries is not None
            else os.environ.get("OLLAMA_EMBED_CACHE_MAX"),
            emb_cfg.cache_max_entries,
        )
        self._cache: OrderedDict[tuple[int, int, str], Any] = OrderedDict()

        # Persist the config-sourced hosts/concurrency so multi-replica
        # embedding is configurable without env vars. ``_resolve_ollama_hosts``
        # and ``_resolve_embed_concurrency`` still honor their env vars; we seed
        # the env from config only when the env var is unset, preserving the
        # explicit-arg > env > config > default precedence.
        if emb_cfg.hosts and not os.environ.get("OLLAMA_EMBED_HOSTS"):
            os.environ["OLLAMA_EMBED_HOSTS"] = ",".join(emb_cfg.hosts)
        if emb_cfg.concurrency and emb_cfg.concurrency != 1 and not os.environ.get(
            "OLLAMA_EMBED_CONCURRENCY"
        ):
            os.environ["OLLAMA_EMBED_CONCURRENCY"] = str(int(emb_cfg.concurrency))

        self.native_embeddings = load_config().models.native_embeddings
        self._native_model = None

        if model_name == "nomic-ai/nomic-embed-text-v1.5":
            model_name = "nomic-embed-text"
            self.model_name = model_name

        if self.native_embeddings:
            logger.info("Using Native embedding model (SentenceTransformers): %s", model_name)
            _status(f"Using Native embedding model (SentenceTransformers): {model_name}")
            self._init_native_model()
        else:
            logger.info("Using Ollama embedding model: %s", model_name)
            _status(
                f"Using Ollama embedding model: {model_name} "
                f"(batch_size={self.ollama_batch_size}, timeout={self.ollama_timeout:g}s)"
            )

    def _init_native_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._native_model = SentenceTransformer(self.model_name, trust_remote_code=True, device=device)
            _status(f"Native model loaded on device: {device}")
        except Exception as exc:
            _status(f"Failed to load native embedding model: {exc}. Falling back to Ollama.")
            logger.error("Native embedding load failed: %s", exc)
            self.native_embeddings = False

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
                # LRU: mark as recently used so the entry is not evicted while
                # still being queried.
                self._cache.move_to_end(key)
                results[i] = cached
            else:
                to_compute_indices.append(i)
                to_compute_texts.append(text)

        if to_compute_texts:
            prefixed = [f"{prefix}{text}" for text in to_compute_texts]
            if self.native_embeddings and self._native_model is not None:
                truncated = self._native_embeddings(prefixed, truncate_dim)
            else:
                truncated = self._ollama_embeddings(prefixed, truncate_dim)

            for j, idx in enumerate(to_compute_indices):
                key = self._cache_key(texts[idx], truncate_dim, prefix)
                self._cache[key] = truncated[j]
                # Enforce the cap after each insert. Eviction is FIFO by default;
                # combined with move_to_end on hit this is true LRU.
                while len(self._cache) > self.max_cache_entries:
                    self._cache.popitem(last=False)
                results[idx] = truncated[j]

        return np.array(results)

    def _native_embeddings(self, texts: list[str], truncate_dim: int):
        import numpy as np
        if not texts:
            return np.empty((0, truncate_dim), dtype=np.float32)
        
        vectors = self._native_model.encode(texts, batch_size=self.ollama_batch_size, convert_to_numpy=True)
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

    def _ollama_embeddings(self, texts: list[str], truncate_dim: int):
        import numpy as np

        if not texts:
            return np.empty((0, truncate_dim), dtype=np.float32)

        batch_size = self.ollama_batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # Build the list of (batch_number, batch_texts, target_host). Hosts are
        # round-robined across replicas so a multi-replica deployment shares
        # load evenly; with a single host every batch targets it (but may still
        # run concurrently if OLLAMA_EMBED_CONCURRENCY > 1, see below).
        hosts = _resolve_ollama_hosts()
        batches: list[tuple[int, list[str], str]] = []
        for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start : start + batch_size]
            host = hosts[(batch_number - 1) % len(hosts)]
            batches.append((batch_number, batch, host))

        # Decide concurrency. Default 1 = serial (preserves the historical
        # single-host, single-batch-at-a-time behavior exactly). With multiple
        # replicas we always parallelize at least len(hosts) ways; an explicit
        # OLLAMA_EMBED_CONCURRENCY can push higher (e.g. OLLAMA_NUM_PARALLEL>1).
        requested = _resolve_embed_concurrency()
        max_workers = max(requested, len(hosts)) if len(hosts) > 1 else requested
        use_pool = max_workers > 1 and len(batches) > 1

        # Ordered result slots filled by either the serial or the parallel path.
        per_batch: list[list[list[float]] | None] = [None] * len(batches)

        if use_pool:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(
                        self._embed_one_batch,
                        batch_number,
                        batch,
                        total_batches,
                        host,
                    ): index
                    for index, (batch_number, batch, host) in enumerate(batches)
                }
                # Reproduce the serial path's fail-fast semantics: the first
                # failing batch raises RuntimeError out of result() and the
                # remaining in-flight work is cancelled on context exit.
                for future, index in future_to_index.items():
                    per_batch[index] = future.result()
        else:
            for index, (batch_number, batch, host) in enumerate(batches):
                per_batch[index] = self._embed_one_batch(batch_number, batch, total_batches, host)

        embeddings: list[list[float]] = []
        for chunk in per_batch:
            assert chunk is not None
            embeddings.extend(chunk)

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

    def _embed_one_batch(
        self,
        batch_number: int,
        batch: list[str],
        total_batches: int,
        host: str,
    ) -> list[list[float]]:
        """Embed one batch against ``host``.

        Encapsulates the primary ``/api/embed`` call plus the one-by-one
        ``/api/embeddings`` fallback so the parallel dispatcher can submit
        whole batches as independent units of work. Raises RuntimeError on
        failure so the caller (serial loop or ``Future.result``) propagates it.
        """
        total_chars = sum(len(text) for text in batch)
        _status(
            f"Requesting embeddings from Ollama model: {self.model_name} "
            f"batch {batch_number}/{total_batches} "
            f"({len(batch)} text(s), {total_chars} chars, timeout={self.ollama_timeout}s) "
            f"-> {host}"
        )
        try:
            payload_input = batch[0] if len(batch) == 1 else batch
            response = self._ollama_api_with_retry(
                "/api/embed",
                {"model": self.model_name, "input": payload_input},
                host=host,
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
                    f"text {text_index}/{len(batch)} ({len(text)} chars, timeout={self.ollama_timeout}s)"
                )
                response = self._ollama_api_with_retry(
                    "/api/embeddings",
                    {"model": self.model_name, "prompt": text},
                    host=host,
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
        return batch_embeddings

    def _ollama_api(self, path: str, payload: dict[str, Any], host: str | None = None) -> dict[str, Any]:
        # ``host`` defaults to the single OLLAMA_HOST for backward compatibility
        # (existing callers and tests use the 2-arg form). The parallel batch
        # dispatcher passes the round-robin target host explicitly.
        base = host if host is not None else _ollama_host()
        url = f"{base}{path}"
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

    def _ollama_api_with_retry(
        self, path: str, payload: dict[str, Any], host: str | None = None
    ) -> dict[str, Any]:
        """Call :meth:`_ollama_api` with bounded exponential-backoff retry.

        Retries on the transient errors ``_ollama_api`` raises (RuntimeError
        wrapping timeouts/URL errors). The last attempt's exception propagates
        so a genuinely broken endpoint still fails the batch -- but only after
        a few rides through a transient blip.
        """
        import random

        attempts = max(1, int(self.ollama_retries))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._ollama_api(path, payload, host=host)
            except RuntimeError as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                # Exponential backoff: 0.5s, 1s, 2s, ... plus up to 25% jitter.
                backoff = (0.5 * (2 ** (attempt - 1))) * (1 + random.random() * 0.25)
                # Surface the retry in the server log (not just stderr) so a
                # slow-Ollama-induced retry storm is visible in logs/server.log
                # during a long ingest. The first retry of each batch warns.
                logger.warning(
                    "Ollama embedding retry %d/%d after %.2fs backoff: %s",
                    attempt,
                    attempts,
                    backoff,
                    exc,
                )
                _status(
                    f"Ollama request failed (attempt {attempt}/{attempts}); "
                    f"retrying in {backoff:.2f}s. Error: {exc}"
                )
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

