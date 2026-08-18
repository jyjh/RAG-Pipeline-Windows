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
    """Return the ``[llm_api]`` config section."""
    from src.config import load_config

    return load_config().llm_api


def active_backend() -> str:
    """Active backend: ``"soclaas"`` or ``"ollama"`` (env ``LLM_BACKEND`` > config > ``"soclaas"``)."""
    env = os.environ.get("LLM_BACKEND", "").strip().lower()
    if env:
        if env not in _VALID_BACKENDS:
            raise ValueError(f"LLM_BACKEND={env!r}; must be 'soclaas' or 'ollama'")
        return env
    backend = (get_llm_api_config().backend or SOCLAAS).strip().lower()
    if backend not in _VALID_BACKENDS:
        _status(f"Unknown [llm_api].backend={backend!r}; defaulting to 'soclaas'.")
        return SOCLAAS
    return backend


def is_soclaas() -> bool:
    return active_backend() == SOCLAAS


def resolve_api_key() -> str:
    """Resolve the SoCLAaS bearer token. Env ``SOCLAAS_API_KEY`` > ``LLM_API_KEY`` > config."""
    cfg = get_llm_api_config()
    for name in (cfg.key_env, "LLM_API_KEY"):
        env_val = os.environ.get(name or "", "").strip()
        if env_val:
            return env_val
    return (cfg.api_key or "").strip()


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
    families (bge-m3, gte, jina, ...) get ``""``.
    """
    from src.config import load_config

    cfg = load_config().embeddings
    configured = ((cfg.query_prefix if kind == "query" else cfg.doc_prefix) or "").strip()
    if configured:
        return configured
    name = (model or "").lower()
    if "nomic" in name or "e5" in name:
        return "search_query: " if kind == "query" else "search_document: "
    return ""
