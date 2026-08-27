from __future__ import annotations

import json
import hashlib
import logging
import os
import socket
import urllib3
from collections import OrderedDict
from typing import Any

from src import llm_api
from src.coerce import as_positive_float, as_positive_int
from src.console import status as _status
from src.defaults import DEFAULT_EMBEDDING_BATCH_SIZE

_OLLAMA_POOL = urllib3.PoolManager(
    num_pools=4,
    maxsize=10,
    retries=False,  # we handle retries ourselves
)

logger = logging.getLogger(__name__)

# Shared host helpers live in llm_api (the backend module); kept as module
# aliases because tests patch these names on src.embeddings.
_normalize_ollama_host = llm_api.normalize_ollama_host
_ollama_pull_command = llm_api.ollama_pull_command


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    return _normalize_ollama_host(host)


_VALID_EMBEDDINGS_BACKENDS = {"", "soclaas", "ollama"}


def resolve_embeddings_backend() -> str:
    """Resolve the embeddings transport, independently of chat/vision.

    ``[embeddings].backend`` (env ``EMBEDDINGS_BACKEND``) lets a deployment
    keep chat/vision on the SoCLAaS API while embedding locally (the default:
    ``"ollama"``, a hosted nomic-embed-text), or vice versa. ``""`` follows
    ``[llm_api].backend``. ``[models].native_embeddings = true`` still takes
    precedence over both -- it is deliberately not a value here.

    Mirrors ``llm_api.active_backend`` semantics: an invalid env value raises;
    an invalid config value warns and is ignored.
    """
    env = os.environ.get("EMBEDDINGS_BACKEND", "").strip().lower()
    if env:
        if env not in _VALID_EMBEDDINGS_BACKENDS:
            raise ValueError(
                f"EMBEDDINGS_BACKEND={env!r}; must be 'soclaas', 'ollama', or unset"
            )
        return env
    from src.config import load_config

    value = (load_config().embeddings.backend or "").strip().lower()
    if value not in _VALID_EMBEDDINGS_BACKENDS:
        _status(
            f"Unknown [embeddings].backend={value!r}; ignoring it and following "
            "[llm_api].backend."
        )
        return ""
    return value


def embeddings_use_soclaas() -> bool:
    """Whether embeddings go to the SoCLAaS API (False = local Ollama)."""
    override = resolve_embeddings_backend()
    if override:
        return override == "soclaas"
    return llm_api.is_soclaas()


def configured_embedding_model() -> str:
    """Effective embedding model: ``[models].embedding_model`` > repo default.

    The repo default (nomic-embed-text) matches a fresh deployment, but an
    instance whose index was built with another model (e.g. all-minilm) must
    keep embedding queries with that same model or retrieval cosines collapse
    to noise. Query-side fallbacks use this instead of DEFAULT_EMBEDDING_MODEL.
    """
    try:
        from src.config import load_config

        value = str(load_config().models.embedding_model or "").strip()
    except Exception:
        value = ""
    return value or DEFAULT_EMBEDDING_MODEL


def resolve_embedding_dim(explicit: int | None = None) -> int:
    """Resolve the embedding dimension: explicit arg > ``[models].embedding_dim``
    > ``DEFAULT_EMBEDDING_DIM``.

    nomic-embed-text is 768-d (bge-m3 is 1024-d); a model/dim change
    invalidates an existing index (the indexer/query-engine reuse guards
    enforce a re-index).
    """
    if explicit is not None:
        return max(1, int(explicit))
    try:
        from src.config import load_config

        return max(1, int(load_config().models.embedding_dim))
    except Exception:
        from src.defaults import DEFAULT_EMBEDDING_DIM

        return DEFAULT_EMBEDDING_DIM


class EmbeddingSetup:
    """Resolved embedding configuration shared by the query engine and indexer.

    Centralizes model aliasing, dim resolution, the instruction prefix, and
    :class:`EmbeddingEngine` construction so both consumers stay consistent.
    """

    def __init__(
        self,
        model: str,
        *,
        embedding_dim: int | None = None,
        batch_size: int | str | None = None,
        timeout: float | None = None,
    ):
        self.model = (
            "nomic-embed-text" if model == "nomic-ai/nomic-embed-text-v1.5" else model
        )
        self.dim = resolve_embedding_dim(embedding_dim)
        self.engine = EmbeddingEngine(
            model_name=self.model,
            ollama_batch_size=batch_size,
            ollama_timeout=timeout,
        )
        self.batch_size = getattr(
            self.engine, "ollama_batch_size", None
        ) or as_positive_int(batch_size, DEFAULT_EMBEDDING_BATCH_SIZE)
        self.query_prefix = llm_api.resolve_embedding_prefix(self.model, "query")
        self.doc_prefix = llm_api.resolve_embedding_prefix(self.model, "doc")

    @property
    def backend_label(self) -> str:
        return "SoCLAaS API" if embeddings_use_soclaas() else "Ollama"


def _resolve_ollama_hosts(config_hosts: list[str] | None = None) -> list[str]:
    """Resolve the list of Ollama embedding endpoints.

    A comma-separated ``OLLAMA_EMBED_HOSTS`` env var enables multi-replica
    embedding: batches are round-robined across the listed hosts in parallel.
    When unset, ``[embeddings].hosts`` from config is used; when that is empty
    too, falls back to the single ``OLLAMA_HOST`` (so the default is zero
    behavior change). Duplicates are removed while preserving order.

    Examples::

        OLLAMA_EMBED_HOSTS=http://gpu-a:11434,http://gpu-b:11434
        # -> ["http://gpu-a:11434", "http://gpu-b:11434"]
    """
    raw = os.environ.get("OLLAMA_EMBED_HOSTS", "").strip()
    if not raw and config_hosts:
        raw = ",".join(str(host) for host in config_hosts)
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


def _resolve_embed_concurrency(config_concurrency: int | None = None) -> int:
    """Worker threads for parallel batch embedding.

    Default 1 = serial (preserves existing single-host behavior). When the
    host list has multiple replicas the effective parallelism is
    ``max(concurrency, len(hosts))`` so every replica is used concurrently even
    if the operator left this at 1. Override with ``OLLAMA_EMBED_CONCURRENCY``
    (or ``[embeddings].concurrency``) to push more in-flight batches per host
    (e.g. with ``OLLAMA_NUM_PARALLEL>1`` on the Ollama server).
    """
    raw = os.environ.get("OLLAMA_EMBED_CONCURRENCY", "").strip()
    try:
        value = int(raw) if raw else (int(config_concurrency) if config_concurrency else 1)
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

        self.ollama_batch_size = as_positive_int(
            ollama_batch_size
            if ollama_batch_size is not None
            else os.environ.get("OLLAMA_EMBED_BATCH_SIZE"),
            emb_cfg.batch_size,
        )
        self._ollama_timeout_base = as_positive_float(
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
        self.ollama_retries = as_positive_int(
            ollama_retries
            if ollama_retries is not None
            else os.environ.get("OLLAMA_EMBED_RETRIES"),
            emb_cfg.retries,
        )
        # Bounded LRU cache. At 100GB-scale cold indexing almost every chunk is
        # unique, so an unbounded dict would grow to hold every embedding for
        # the whole run and OOM the process. A bounded LRU caps the footprint
        # while preserving the hit rate for the realistic repeat cases
        # (re-index reuse, repeated queries). Default 50k entries of 1024-d
        # float32 is ~200MB; env-overridable.
        self.max_cache_entries = as_positive_int(
            max_cache_entries
            if max_cache_entries is not None
            else os.environ.get("OLLAMA_EMBED_CACHE_MAX"),
            emb_cfg.cache_max_entries,
        )
        self._cache: OrderedDict[tuple[int, int, str], Any] = OrderedDict()

        # Config-sourced multi-replica hosts/concurrency, passed explicitly to
        # the resolvers (env vars still take precedence inside them).
        self._config_hosts = list(emb_cfg.hosts or [])
        self._config_concurrency = int(emb_cfg.concurrency or 1)

        self.native_embeddings = load_config().models.native_embeddings
        self._native_model = None

        if model_name == "nomic-ai/nomic-embed-text-v1.5":
            model_name = "nomic-embed-text"
            self.model_name = model_name

        if self.native_embeddings:
            self._init_native_model()
        # Resolve the active backend once (native may fall back on load failure),
        # so the per-batch hot path branches on a cached string instead of
        # re-reading config/env for every batch. The embeddings transport may
        # differ from the chat/vision backend ([embeddings].backend).
        self._backend = (
            "native" if self.native_embeddings and self._native_model is not None
            else "soclaas" if embeddings_use_soclaas()
            else "ollama"
        )
        label = {"native": "Native (SentenceTransformers)",
                 "soclaas": "SoCLAaS API",
                 "ollama": "Ollama"}[self._backend]
        logger.info("Using %s embedding model: %s", label, model_name)
        _status(
            f"Using {label} embedding model: {model_name} "
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
            if self._backend == "native":
                truncated = self._native_embeddings(prefixed, truncate_dim)
            elif self._backend == "soclaas":
                truncated = self._soclaas_embeddings(prefixed, truncate_dim)
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
        raw = self._native_model.encode(
            texts, batch_size=self.ollama_batch_size, convert_to_numpy=True
        )
        return self._postprocess_vectors(raw, truncate_dim)

    def _ollama_embeddings(self, texts: list[str], truncate_dim: int):
        import numpy as np
        if not texts:
            return np.empty((0, truncate_dim), dtype=np.float32)
        # Multi-replica Ollama shards batches across hosts (round-robin, applied
        # per batch in _embed_one_batch); parallelize at least len(hosts) ways.
        hosts = _resolve_ollama_hosts(self._config_hosts)
        requested = _resolve_embed_concurrency(self._config_concurrency)
        max_workers = max(requested, len(hosts)) if len(hosts) > 1 else requested
        vectors = self._dispatch_embedding_batches(texts, max_workers, self._embed_one_batch)
        return self._postprocess_vectors(vectors, truncate_dim)

    def _soclaas_embeddings(self, texts: list[str], truncate_dim: int):
        import numpy as np
        if not texts:
            return np.empty((0, truncate_dim), dtype=np.float32)
        max_workers = _resolve_embed_concurrency(self._config_concurrency)
        vectors = self._dispatch_embedding_batches(texts, max_workers, self._soclaas_embed_one_batch)
        return self._postprocess_vectors(vectors, truncate_dim)

    def _dispatch_embedding_batches(self, texts, max_workers, batch_fn):
        """Split ``texts`` into batch_size chunks and run
        ``batch_fn(number, batch, total)`` serially or concurrently; return the
        concatenated vectors (pre-normalize). Shared by the Ollama and SoCLAaS
        backends. Fail-fast: the first failing batch raises and in-flight work
        is cancelled on context exit.
        """
        batch_size = self.ollama_batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size
        batches = [
            (n, texts[s:s + batch_size])
            for n, s in enumerate(range(0, len(texts), batch_size), start=1)
        ]
        per_batch = [None] * len(batches)
        if max_workers > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(batch_fn, n, b, total_batches): i
                    for i, (n, b) in enumerate(batches)
                }
                for future, index in future_to_index.items():
                    per_batch[index] = future.result()
        else:
            for index, (n, b) in enumerate(batches):
                per_batch[index] = batch_fn(n, b, total_batches)
        vectors: list[list[float]] = []
        for chunk in per_batch:
            vectors.extend(chunk)
        return vectors

    def _postprocess_vectors(self, embeddings, truncate_dim: int):
        """Truncate/pad to ``truncate_dim`` and L2-normalize (shared by all backends)."""
        import numpy as np

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

    def _soclaas_embed_one_batch(
        self,
        batch_number: int,
        batch: list[str],
        total_batches: int,
    ) -> list[list[float]]:
        """Embed one batch against the SoCLAaS ``/v1/embeddings`` endpoint."""
        total_chars = sum(len(text) for text in batch)
        _status(
            f"Requesting embeddings from SoCLAaS model: {self.model_name} "
            f"batch {batch_number}/{total_batches} "
            f"({len(batch)} text(s), {total_chars} chars)"
        )
        try:
            return llm_api.soclaas_embed(
                model=self.model_name, input_texts=batch, timeout=self.ollama_timeout
            )
        except Exception as exc:
            raise RuntimeError(
                f"SoCLAaS embedding failed for model '{self.model_name}' on "
                f"batch {batch_number}/{total_batches}. Verify the API key "
                f"(SOCLAAS_API_KEY) and base_url. Original error: {exc}"
            ) from exc

    def _embed_one_batch(
        self,
        batch_number: int,
        batch: list[str],
        total_batches: int,
    ) -> list[list[float]]:
        """Embed one batch against the round-robin Ollama host for its number.

        Tries ``/api/embed`` (batch), falling back to one-by-one ``/api/embeddings``.
        Raises RuntimeError on failure so the dispatcher (serial or ``Future.result``)
        propagates it.
        """
        hosts = _resolve_ollama_hosts(self._config_hosts)
        host = hosts[(batch_number - 1) % len(hosts)]
        _status(
            f"Requesting embeddings from Ollama model: {self.model_name} "
            f"batch {batch_number}/{total_batches} "
            f"({len(batch)} text(s), {sum(len(t) for t in batch)} chars, "
            f"timeout={self.ollama_timeout}s) -> {host}"
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
        try:
            response = _OLLAMA_POOL.request(
                "POST",
                url,
                body=body,
                headers={"Content-Type": "application/json"},
                timeout=urllib3.Timeout(total=self.ollama_timeout),
            )
            if response.status >= 400:
                raise RuntimeError(
                    f"Ollama request returned HTTP {response.status} at {url}: "
                    f"{response.data[:200].decode('utf-8', errors='replace')}"
                )
            return json.loads(response.data.decode("utf-8"))
        except (TimeoutError, socket.timeout, urllib3.exceptions.TimeoutError) as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.ollama_timeout:g}s at {url}."
            ) from exc
        except urllib3.exceptions.HTTPError as exc:
            raise RuntimeError(f"Ollama request failed at {url}: {exc}") from exc

    def _ollama_api_with_retry(
        self, path: str, payload: dict[str, Any], host: str | None = None
    ) -> dict[str, Any]:
        """Call :meth:`_ollama_api` with bounded exponential-backoff retry.

        Uses the shared retry engine in :mod:`src.llm_api` (announced on stderr
        so a slow-Ollama retry storm is visible during a long ingest). The last
        attempt's exception propagates so a genuinely broken endpoint still
        fails the batch -- but only after a few rides through a transient blip.
        """
        return llm_api.retry_with_backoff(
            lambda: self._ollama_api(path, payload, host=host),
            attempts=self.ollama_retries,
            description="Ollama request",
            retry_on=RuntimeError,
            announce=True,
        )

