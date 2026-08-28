"""SoCLAaS OpenAI-compatible LLM API client + backend selector.

Primary backend for chat (``gemma4:26b``), vision (``qwen3-vl:32b``), and
embeddings (``bge-m3``, 1024-d). Local Ollama is a dormant fallback
(``[llm_api].backend = "ollama"``). Centralises auth/retry/response-parsing for
the chat (``src/local_rag.py``), embedding (``src/embeddings.py``), and vision
(``src/ingestion.py``) transports. Stdlib ``urllib`` only -- no new dependency.
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

logger = logging.getLogger(__name__)

SOCLAAS = "soclaas"
OLLAMA = "ollama"
_VALID_BACKENDS = {SOCLAAS, OLLAMA}


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Config + backend selection
# --------------------------------------------------------------------------- #


def get_llm_api_config():
    """Return the ``[llm_api]`` config section (config.toml-discovered).

    Discovery goes through :func:`src.config.default_config_path` so the
    section reflects config.toml even when ``RAG_PIPELINE_CONFIG`` is unset
    (e.g. a bare ``python -m src.web_app`` start). A bare ``load_config()``
    here would silently read hardcoded defaults instead.
    """
    from src.config import default_config_path, load_config

    return load_config(default_config_path()).llm_api


def active_backend() -> str:
    """Effective backend: ``"soclaas"`` or ``"ollama"``.

    Selection order: env ``LLM_BACKEND`` > ``[llm_api].backend`` > ``"soclaas"``.
    When the selection is ``soclaas`` but no API key is resolvable, the backend
    falls back to ``ollama`` -- a missing key means the cloud is unavailable,
    and every transport (chat, vision, embeddings inheritance) should run
    locally instead of erroring. Set ``LLM_STRICT_BACKEND=1`` to keep the
    selected backend and fail loudly when the key is missing.
    """
    env = os.environ.get("LLM_BACKEND", "").strip().lower()
    if env:
        if env not in _VALID_BACKENDS:
            raise ValueError(f"LLM_BACKEND={env!r}; must be 'soclaas' or 'ollama'")
        backend = env
    else:
        backend = (get_llm_api_config().backend or SOCLAAS).strip().lower()
        if backend not in _VALID_BACKENDS:
            _status(f"Unknown [llm_api].backend={backend!r}; defaulting to 'soclaas'.")
            backend = SOCLAAS
    if backend == SOCLAAS and not _strict_backend() and not resolve_api_key():
        global _FALLBACK_NOTICE_PRINTED
        if not _FALLBACK_NOTICE_PRINTED:
            _FALLBACK_NOTICE_PRINTED = True
            _status(
                "SoCLAaS is selected but no API key is configured "
                f"({get_llm_api_config().key_env}/LLM_API_KEY or [llm_api].api_key); "
                "falling back to the local Ollama backend for chat/vision."
            )
        return OLLAMA
    return backend


_FALLBACK_NOTICE_PRINTED = False


def _strict_backend() -> bool:
    return os.environ.get("LLM_STRICT_BACKEND", "").strip().lower() in {"1", "true", "yes", "on"}


def soclaas_ready() -> bool:
    """True when SoCLAaS is the effective backend (selected AND usable)."""
    return active_backend() == SOCLAAS


def is_soclaas() -> bool:
    return soclaas_ready()


def resolve_api_key() -> str:
    """Resolve the SoCLAaS bearer token. Env ``SOCLAAS_API_KEY`` > ``LLM_API_KEY`` > config."""
    cfg = get_llm_api_config()
    for name in (cfg.key_env, "LLM_API_KEY"):
        env_val = os.environ.get(name or "", "").strip()
        if env_val:
            return env_val
    return (cfg.api_key or "").strip()


# --------------------------------------------------------------------------- #
# Local Ollama model resolution
# --------------------------------------------------------------------------- #
# Cloud-side model tags (e.g. "gemma4:26b", "qwen3-vl:32b") usually do not
# exist in the local Ollama registry ("gemma4:latest", "qwen2.5vl:7b"). When
# the backend falls back to local, map requested names onto installed models
# instead of failing every call with "model not found".

_TAGS_CACHE: dict[str, Any] = {}
# /api/tags is cheap but not free; a short TTL lets a model pulled mid-session
# become visible to substitution without waiting for a server restart.
_TAGS_CACHE_TTL_SECONDS = 60.0
_VISION_MODEL_HINTS = ("vl", "vision", "llava", "moondream", "minicpm-v", "cogvlm", "gemma3v")

# Ceiling for AUTOMATIC local model substitution (base-name/hint fallbacks).
# The configured pins ([models].local_llm_model / local_vision_model) are
# sized for a 4 GB GPU; the fallbacks must not silently route onto a 6-10 GB
# variant that thrashes such a GPU. Exact matches and the explicit pins are
# never ceiling-checked. 0 disables. Env-overridable.
_LOCAL_MODEL_DEFAULT_MAX_BYTES = 4 * 1024 ** 3


def _local_model_max_bytes() -> int:
    raw = os.environ.get("LOCAL_MODEL_MAX_BYTES", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return _LOCAL_MODEL_DEFAULT_MAX_BYTES


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    if gib >= 1:
        return f"{gib:.1f} GiB"
    return f"{num_bytes / (1024 ** 2):.0f} MiB"


def _ollama_tag_hosts() -> list[str]:
    """Candidate hosts for model enumeration, in chat-resolution order.

    Mirrors ``src.local_rag._get_ollama_candidate_hosts`` precedence
    (``OLLAMA_HOST`` env > ``[ollama].host`` > ``[ollama].hosts``) so model
    substitution enumerates the same Ollama that serves chat.
    """
    from src.config import default_config_path, load_config

    cfg = load_config(default_config_path()).ollama
    primary = os.environ.get("OLLAMA_HOST", "").strip() or str(getattr(cfg, "host", "") or "")
    candidates = [primary or "http://127.0.0.1:11434"]
    candidates.extend(str(host) for host in (getattr(cfg, "hosts", None) or []))
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_ollama_host(candidate)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _fetch_ollama_models() -> list[tuple[str, int | None]]:
    """Query ``/api/tags`` on each candidate host until one answers."""
    for host in _ollama_tag_hosts():
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
            models: list[tuple[str, int | None]] = []
            for item in (payload.get("models") or []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                size = item.get("size")
                models.append((
                    name,
                    int(size) if isinstance(size, (int, float)) and size > 0 else None,
                ))
            if models:
                return models
        except (OSError, ValueError, urllib.error.URLError) as exc:
            logger.debug("Could not list local Ollama models at %s: %s", host, exc)
    return []


def _ollama_models() -> list[tuple[str, int | None]]:
    """``(name, size_bytes)`` per installed local Ollama model (TTL-cached)."""
    cached = _TAGS_CACHE.get("models")
    fetched_at = _TAGS_CACHE.get("fetched_at")
    if cached is not None and fetched_at is not None:
        if (time.monotonic() - fetched_at) < _TAGS_CACHE_TTL_SECONDS:
            return cached
    models = _fetch_ollama_models()
    _TAGS_CACHE["models"] = models
    _TAGS_CACHE["sizes"] = {
        name.casefold(): size for name, size in models if size
    }
    _TAGS_CACHE["fetched_at"] = time.monotonic()
    return models


def _ollama_tags() -> list[str]:
    """Installed local Ollama model names (TTL-cached; empty when unreachable)."""
    return [name for name, _size in _ollama_models()]


def _ollama_model_sizes() -> dict[str, int]:
    """Folded model name -> on-disk size in bytes ({} when unknown).

    Populated only by a real ``/api/tags`` fetch. When ``_ollama_tags`` is
    stubbed (tests), sizes stay unknown and size-aware substitution is
    skipped rather than guessed.
    """
    return dict(_TAGS_CACHE.get("sizes") or {})


def reset_local_model_cache() -> None:
    """Forget the cached /api/tags result (test seam; also forces a refetch)."""
    _TAGS_CACHE.clear()


def exceeds_local_size_ceiling(model: str) -> bool:
    """True when ``model`` is installed locally and larger than the ceiling.

    Used to vet CLIENT-supplied model overrides: the configured
    ``[models].local_*`` pins are the operator's choice and bypass the
    ceiling, but a request field naming an exact installed tag (e.g. a
    9.6 GB chat model) would otherwise load a model the host cannot serve.
    Unknown sizes (older Ollama, stubbed tags) are never flagged.
    """
    name = str(model or "").strip()
    if not name:
        return False
    size = _ollama_model_sizes().get(name.casefold())
    if size is None:
        return False
    max_bytes = _local_model_max_bytes()
    return 0 < max_bytes < size


def resolve_local_model(preferred: str, *, vision: bool = False) -> str:
    """Map a (possibly cloud-named) model onto an installed local Ollama model.

    Order: exact tag match > the configured ``[models].local_llm_model`` /
    ``local_vision_model`` knob (default ``qwen3:4b-instruct`` for chat, a
    4 GB-GPU-sized substitute) when actually installed > same base name
    (``gemma4:26b`` -> ``gemma4``, ``gemma4:latest``) > for vision requests,
    any installed model whose name hints at vision capability
    (``qwen2.5vl``, ``llava``, ...). Returns ``preferred`` unchanged when
    Ollama is unreachable or nothing matches -- the subsequent call then
    fails with the model's real name in the error. Substitutions are logged
    so the quality difference is visible.

    Size-aware fallbacks: automatic substitutions prefer the SMALLEST
    installed candidate (by on-disk size from /api/tags) and refuse any
    candidate above ``LOCAL_MODEL_MAX_BYTES`` (default 4 GiB -- the sizing
    target of the local pins). A refused chat substitution returns
    ``preferred`` so the call fails loudly with a pull hint instead of
    loading a model the GPU cannot serve; a refused vision substitution
    degrades to the smallest over-ceiling vision model, because failing
    every figure description is worse than a slow one. Exact matches and the
    explicitly configured pins bypass the ceiling (the operator asked for
    them), and unknown sizes (older Ollama servers) keep the old behavior.
    """
    preferred = str(preferred or "").strip()
    tags = _ollama_tags()
    if not tags or not preferred:
        return preferred
    by_fold = {tag.casefold(): tag for tag in tags}
    if preferred.casefold() in by_fold:
        return preferred
    # Explicitly configured local substitute wins ([models].local_vision_model
    # / [models].local_llm_model) when it is actually installed.
    from src.config import default_config_path, load_config

    knob = "local_vision_model" if vision else "local_llm_model"
    configured = str(
        getattr(load_config(default_config_path()).models, knob, "") or ""
    ).strip()
    if configured and configured.casefold() in by_fold:
        logger.warning(
            "Using configured local model %s (cloud model %s not active)",
            configured, preferred,
        )
        return by_fold[configured.casefold()]

    sizes = _ollama_model_sizes()
    max_bytes = _local_model_max_bytes()

    def _eligible(candidates: list[str]) -> list[str]:
        """Candidates within the size ceiling; unknown sizes stay eligible."""
        if max_bytes <= 0:
            return list(candidates)
        return [
            tag for tag in candidates
            if tag.casefold() not in sizes or sizes[tag.casefold()] <= max_bytes
        ]

    base = preferred.split(":", 1)[0].casefold()
    base_matches = [
        tag for folded, tag in by_fold.items() if folded.split(":", 1)[0] == base
    ]
    eligible = _eligible(base_matches)
    if eligible:
        # Prefer the smallest installed variant (e.g. gemma4:4b over
        # gemma4:latest at 9.6 GiB).
        chosen = min(eligible, key=lambda t: sizes.get(t.casefold(), 0))
        logger.warning(
            "Local model substitution: %s not installed; using %s", preferred, chosen
        )
        return chosen
    if base_matches:
        _status(
            f"Refusing local substitution for {preferred}: installed variant(s) "
            f"{', '.join(base_matches)} exceed the {_format_bytes(max_bytes)} "
            "local-model ceiling (LOCAL_MODEL_MAX_BYTES) and cannot be served "
            "by this host. Install a smaller local model and set "
            f"[models].{knob} to its tag."
        )
        return preferred
    if vision:
        hint_matches = [
            tag for folded, tag in by_fold.items()
            if any(hint in folded.split(":", 1)[0] for hint in _VISION_MODEL_HINTS)
        ]
        if hint_matches:
            # Vision capability is required for figure enrichment; when every
            # candidate is over the ceiling, degrade to the smallest rather
            # than failing every image.
            eligible = _eligible(hint_matches) or hint_matches
            chosen = min(eligible, key=lambda t: sizes.get(t.casefold(), 0))
            if 0 < max_bytes < sizes.get(chosen.casefold(), 0):
                _status(
                    f"Local vision substitution {chosen} "
                    f"({_format_bytes(sizes.get(chosen.casefold(), 0))}) exceeds the "
                    f"{_format_bytes(max_bytes)} local-model ceiling; expecting slow "
                    "CPU offload. Pull a smaller vision model and set "
                    "[models].local_vision_model."
                )
            logger.warning(
                "Local vision model substitution: %s not installed; using %s",
                preferred, chosen,
            )
            return chosen
    return preferred


def require_api_key() -> str:
    """Return the API key or raise a clear, actionable error for SoCLAaS use."""
    key = resolve_api_key()
    if not key:
        cfg = get_llm_api_config()
        raise RuntimeError(
            "SoCLAaS API key not configured. Set the env var "
            f"{cfg.key_env} (or LLM_API_KEY) or [llm_api].api_key in config.toml, "
            'or switch to the dormant Ollama backend with [llm_api].backend = "ollama".'
        )
    return key


def _base_url() -> str:
    return (get_llm_api_config().base_url or "").rstrip("/")


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {require_api_key()}", "Content-Type": "application/json"}


def _timeout(timeout: float | None) -> float:
    return float(timeout) if timeout is not None else float(get_llm_api_config().request_timeout_seconds)


# --------------------------------------------------------------------------- #
# Ollama-options -> OpenAI-params translation
# --------------------------------------------------------------------------- #


def map_chat_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Translate Ollama-style ``options`` to OpenAI chat params.

    ``num_predict`` -> ``max_tokens``; ``temperature``/``top_p``/``top_k``/``seed``
    pass through; ``num_ctx`` (Ollama-only) is dropped (OpenAI servers reject it).
    """
    options = options or {}
    out = {k: options[k] for k in ("temperature", "top_p", "top_k", "seed") if k in options}
    if "num_predict" in options:
        out["max_tokens"] = options["num_predict"]
    return out


# --------------------------------------------------------------------------- #
# HTTP core (retry + error normalisation)
# --------------------------------------------------------------------------- #


class SoclaasError(RuntimeError):
    """A SoCLAaS request failed (HTTP error, timeout, or connection error)."""


def _request(url: str, *, headers: dict[str, str], body: Any = None,
             method: str = "POST", timeout: float):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise SoclaasError(f"SoCLAaS returned HTTP {exc.code} at {url}: {detail}") from exc
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        raise SoclaasError(f"SoCLAaS request failed at {url}: {exc}") from exc


def _is_transient(exc: SoclaasError) -> bool:
    msg = str(exc).lower()
    if "timed out" in msg or "timeout" in msg:
        return True
    if any(tag in msg for tag in ("urlopen", "connection", "reset", "unreachable", "refused")):
        return True
    # HTTP 429 (rate limit) and 5xx are retryable; 4xx (auth/shape) are not.
    return any(code in msg for code in ("http 429", "http 500", "http 502", "http 503", "http 504"))


def retry_with_backoff(
    fn,
    *,
    attempts: int,
    description: str,
    retry_on: type[Exception] | tuple[type[Exception], ...] = Exception,
    is_retryable=None,
    announce: bool = False,
):
    """Run ``fn`` with bounded exponential backoff (0.5s, 1s, 2s, ... +25% jitter).

    Shared by the SoCLAaS client (transient-error classification) and the
    Ollama embedding path. A non-retryable exception, or failure on the final
    attempt, propagates to the caller.
    """
    attempts = max(1, int(attempts))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            last_exc = exc
            if (is_retryable is not None and not is_retryable(exc)) or attempt >= attempts:
                raise
            backoff = (0.5 * (2 ** (attempt - 1))) * (1 + random.random() * 0.25)
            logger.warning(
                "%s retry %d/%d after %.2fs backoff: %s",
                description, attempt, attempts, backoff, exc,
            )
            if announce:
                _status(
                    f"{description} failed (attempt {attempt}/{attempts}); "
                    f"retrying in {backoff:.2f}s. Error: {exc}"
                )
            time.sleep(backoff)
    assert last_exc is not None
    raise last_exc


def _with_retry(fn, *, description: str):
    return retry_with_backoff(
        fn,
        attempts=get_llm_api_config().retries,
        description=f"SoCLAaS {description}",
        retry_on=SoclaasError,
        is_retryable=_is_transient,
    )


# --------------------------------------------------------------------------- #
# Chat completions
# --------------------------------------------------------------------------- #


def _chat_url() -> str:
    return _base_url() + get_llm_api_config().chat_path


def _build_chat_payload(model: str, messages: list[dict[str, Any]], *,
                        options: dict[str, Any] | None, tools, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": bool(stream)}
    payload.update(map_chat_options(options))
    if tools:
        payload["tools"] = tools
    return payload


def soclaas_chat_once(*, model: str, messages: list[dict[str, Any]],
                      options: dict[str, Any] | None = None, tools=None,
                      timeout: float | None = None) -> dict[str, Any]:
    """Non-streaming chat completion. Returns the raw OpenAI response dict."""
    url = _chat_url()
    payload = _build_chat_payload(model, messages, options=options, tools=tools, stream=False)

    def _do():
        with _request(url, headers=_auth_headers(), body=payload, timeout=_timeout(timeout)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return _with_retry(_do, description="chat")


def soclaas_chat_stream(*, model: str, messages: list[dict[str, Any]],
                        options: dict[str, Any] | None = None, tools=None,
                        timeout: float | None = None) -> Iterator[dict[str, Any]]:
    """Streaming chat completion; yields OpenAI chunk dicts (one per SSE ``data:`` line).

    Opening the stream is retried; a mid-stream drop propagates (consumer stops).
    """
    url = _chat_url()
    payload = _build_chat_payload(model, messages, options=options, tools=tools, stream=True)

    def _events():
        def _open():
            return _request(url, headers=_auth_headers(), body=payload, timeout=_timeout(timeout))

        response = _with_retry(_open, description="chat stream open")
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    return _events()


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


def soclaas_embed(*, model: str, input_texts: list[str],
                  timeout: float | None = None) -> list[list[float]]:
    """Embed ``input_texts`` via ``/v1/embeddings``. Returns vectors in input order."""
    if not input_texts:
        return []
    url = _base_url() + get_llm_api_config().embeddings_path
    payload = {"model": model, "input": list(input_texts)}

    def _do():
        with _request(url, headers=_auth_headers(), body=payload, timeout=_timeout(timeout)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    data = _with_retry(_do, description="embeddings")
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise SoclaasError(f"SoCLAaS embeddings response had no 'data' array: {str(data)[:200]}")
    # Preserve request order via the "index" field (OpenAI sends objects unordered).
    items_sorted = sorted(items, key=lambda d: d.get("index", 0) if isinstance(d, dict) else 0)
    vectors: list[list[float]] = []
    for item in items_sorted:
        emb = item.get("embedding") if isinstance(item, dict) else None
        if not emb:
            raise SoclaasError("SoCLAaS embeddings response entry missing 'embedding' field.")
        vectors.append([float(x) for x in emb])
    if len(vectors) != len(input_texts):
        raise SoclaasError(
            f"SoCLAaS returned {len(vectors)} embedding(s) for {len(input_texts)} input text(s)."
        )
    return vectors


# --------------------------------------------------------------------------- #
# Vision (chat-completions image_url parts)
# --------------------------------------------------------------------------- #


def extract_content(response: dict[str, Any]) -> str:
    """Best-effort ``choices[0].message.content`` extraction from an OpenAI response."""
    try:
        choices = response.get("choices") or []
        return str((choices[0] if choices else {}).get("message", {}).get("content") or "")
    except Exception:
        return ""


def soclaas_vision(*, model: str, prompt: str, images_b64: list[str],
                   media_type: str = "image/jpeg",
                   options: dict[str, Any] | None = None,
                   timeout: float | None = None) -> str:
    """Describe images via a vision chat completion. ``images_b64`` = raw base64 strings.

    Default media type is JPEG (the ingestion path re-encodes via
    ``_png_bytes_for_vision``). Returns ``choices[0].message.content``.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        })
    messages = [{"role": "user", "content": content}]
    response = soclaas_chat_once(model=model, messages=messages, options=options, tools=None, timeout=timeout)
    return extract_content(response)


# --------------------------------------------------------------------------- #
# Health probe + status snapshot (for /api/health, /api/metrics)
# --------------------------------------------------------------------------- #


def probe_soclaas(*, timeout: float = 5.0) -> tuple[bool, str]:
    """Probe ``/v1/models`` with the bearer key. Returns ``(reachable, detail)``."""
    if not resolve_api_key():
        return False, "api key not configured"
    url = _base_url() + get_llm_api_config().models_path
    try:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {resolve_api_key()}"}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            return status < 400, ("ok" if status < 400 else f"HTTP {status}")
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def soclaas_status_snapshot(*, timeout: float = 3.0) -> dict[str, Any]:
    """Backend-aware status for the web UI health/metrics endpoints."""
    cfg = get_llm_api_config()
    reachable, detail = probe_soclaas(timeout=timeout)
    return {
        "backend": "soclaas",
        "base_url": cfg.base_url,
        "api_key_configured": bool(resolve_api_key()),
        "reachable": reachable,
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# Dormant Ollama-fallback helpers (shared by local_rag.py and embeddings.py)
# --------------------------------------------------------------------------- #


def normalize_ollama_host(raw: str) -> str:
    """Normalize one Ollama host string (scheme + trailing-slash trim)."""
    host = (raw or "").strip()
    if not host:
        return "http://127.0.0.1:11434"
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")
    return f"http://{host.rstrip('/')}"


def ollama_pull_command(model: str) -> str:
    """Operator-facing ``ollama pull <model>`` hint, resolving the Windows path."""
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
    return f"{executable} pull {model}"


# --------------------------------------------------------------------------- #
# Embedding instruction-prefix resolution
# --------------------------------------------------------------------------- #


def resolve_embedding_prefix(model: str, kind: str) -> str:
    """Instruction prefix for ``model`` / ``kind`` ("doc" or "query").

    Config override (``[embeddings].doc_prefix``/``query_prefix``) wins; else
    nomic/e5 get ``"search_document: "``/``"search_query: "`` and instruction-free
    families (all-minilm, bge-m3, gte, jina, ...) get ``""``.
    """
    from src.config import default_config_path, load_config

    cfg = load_config(default_config_path()).embeddings
    configured = ((cfg.query_prefix if kind == "query" else cfg.doc_prefix) or "").strip()
    if configured:
        return configured
    name = (model or "").lower()
    if "nomic" in name or "e5" in name:
        return "search_query: " if kind == "query" else "search_document: "
    return ""
