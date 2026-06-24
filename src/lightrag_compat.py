from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


INSTALL_HINT = (
    "This project requires the HKUDS LightRAG package API, which exposes "
    "`LightRAG`, `QueryParam`, token chunking, and storage registries. "
    "The installed `lightrag` package is incompatible. Run: "
    "`pip uninstall -y lightrag && pip install lightrag-hku`."
)


class LightRAGDependencyError(ImportError):
    """Raised when the installed LightRAG package is not the expected API."""


@dataclass(frozen=True)
class LightRAGApi:
    LightRAG: Any
    QueryParam: Any
    EmbeddingFunc: Any
    STORAGES: dict[str, str]
    STORAGE_IMPLEMENTATIONS: dict[str, Any]
    STORAGE_ENV_REQUIREMENTS: dict[str, list[str]]
    chunking_by_token_size: Any
    compute_mdhash_id: Any
    ollama_model_complete: Any


def _installed_details() -> str:
    for distribution in ("lightrag-hku", "lightrag"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue

        metadata = importlib.metadata.metadata(distribution)
        summary = metadata.get("Summary") or "no summary"
        return f"Installed `{distribution}` distribution: version {version} ({summary})."

    return "No installed LightRAG distribution was found."


def _dependency_error(exc: BaseException) -> LightRAGDependencyError:
    error = LightRAGDependencyError(f"{INSTALL_HINT} {_installed_details()}")
    error.__cause__ = exc
    return error


def lightrag_core_imports(timeout_seconds: int = 10) -> bool:
    try:
        subprocess.run(
            [sys.executable, "-B", "-c", "import lightrag.lightrag"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _import_module(name: str):
    _status(f"LightRAG API: importing {name}...")
    started = time.perf_counter()
    done = threading.Event()

    def heartbeat() -> None:
        while not done.wait(10):
            elapsed = time.perf_counter() - started
            _status(f"LightRAG API: still importing {name} after {elapsed:.0f}s...")

    watcher = threading.Thread(target=heartbeat, daemon=True)
    watcher.start()
    try:
        module = importlib.import_module(name)
    finally:
        done.set()
    elapsed = time.perf_counter() - started
    _status(f"LightRAG API: imported {name} in {elapsed:.2f}s.")
    return module


def _response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def _message_content(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", "") or ""


def _normalize_response_format(kwargs: dict[str, Any], *, force_json: bool) -> None:
    response_format = kwargs.pop("response_format", None)
    if kwargs.get("format") is not None:
        return
    if force_json:
        kwargs["format"] = "json"
        return
    if response_format is None:
        return
    if isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            kwargs["format"] = "json"
            return
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema")
            if isinstance(schema, dict):
                kwargs["format"] = schema.get("schema", schema)
                return
    kwargs["format"] = response_format


def _move_ollama_options(kwargs: dict[str, Any]) -> None:
    option_keys = {
        "temperature",
        "top_k",
        "top_p",
        "num_ctx",
        "num_predict",
        "repeat_penalty",
        "seed",
        "stop",
    }
    options = dict(kwargs.pop("options", None) or {})
    max_tokens = kwargs.pop("max_tokens", None)
    if max_tokens is not None:
        options.setdefault("num_predict", max_tokens)
    for key in list(kwargs):
        if key in option_keys:
            options[key] = kwargs.pop(key)
    if options:
        kwargs["options"] = options


def _model_name_from_lightrag_kwargs(kwargs: dict[str, Any]) -> str:
    hashing_kv = kwargs.pop("hashing_kv", None)
    global_config = getattr(hashing_kv, "global_config", None) or {}
    return global_config.get("llm_model_name") or kwargs.pop("model", None) or "gemma4"


async def _stream_chat_content(response: Any):
    async for part in response:
        content = _message_content(_response_value(part, "message"))
        if content:
            yield content


async def local_ollama_model_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
):
    """Small local Ollama adapter used to avoid importing lightrag.llm.ollama at startup."""
    import ollama

    model_name = _model_name_from_lightrag_kwargs(kwargs)
    force_json = bool(keyword_extraction or entity_extraction)
    force_json = force_json or bool(kwargs.pop("keyword_extraction", False))
    force_json = force_json or bool(kwargs.pop("entity_extraction", False))
    _normalize_response_format(kwargs, force_json=force_json)
    _move_ollama_options(kwargs)

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    if timeout == 0:
        timeout = None
    api_key = kwargs.pop("api_key", None)
    kwargs.pop("enable_cot", None)

    headers = None
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    image_inputs = kwargs.pop("image_inputs", None)
    if image_inputs:
        user_message["images"] = image_inputs
    messages.append(user_message)

    stream = bool(kwargs.pop("stream", False))
    client_kwargs: dict[str, Any] = {}
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    if headers is not None:
        client_kwargs["headers"] = headers

    client = ollama.AsyncClient(host=host, **client_kwargs)
    response = await client.chat(
        model=model_name,
        messages=messages,
        stream=stream,
        **kwargs,
    )
    if stream:
        return _stream_chat_content(response)
    return _message_content(_response_value(response, "message"))


@lru_cache(maxsize=1)
def load_lightrag_api() -> LightRAGApi:
    try:
        _import_module("lightrag")
        lightrag_core = _import_module("lightrag.lightrag")
        LightRAG = getattr(lightrag_core, "LightRAG")
        QueryParam = getattr(lightrag_core, "QueryParam")

        utils = _import_module("lightrag.utils")
        EmbeddingFunc = getattr(utils, "EmbeddingFunc")
        compute_mdhash_id = getattr(utils, "compute_mdhash_id")

        kg = _import_module("lightrag.kg")
        STORAGES = getattr(kg, "STORAGES")
        STORAGE_IMPLEMENTATIONS = getattr(kg, "STORAGE_IMPLEMENTATIONS")
        STORAGE_ENV_REQUIREMENTS = getattr(kg, "STORAGE_ENV_REQUIREMENTS")

        try:
            token_size = _import_module("lightrag.chunker.token_size")
            chunking_by_token_size = getattr(token_size, "chunking_by_token_size")
        except (ImportError, AttributeError):
            operate = _import_module("lightrag.operate")
            chunking_by_token_size = getattr(operate, "chunking_by_token_size")

        _status("LightRAG API: using local Ollama adapter.")
        ollama_model_complete = local_ollama_model_complete
    except (ImportError, AttributeError) as exc:
        raise _dependency_error(exc)

    return LightRAGApi(
        LightRAG=LightRAG,
        QueryParam=QueryParam,
        EmbeddingFunc=EmbeddingFunc,
        STORAGES=STORAGES,
        STORAGE_IMPLEMENTATIONS=STORAGE_IMPLEMENTATIONS,
        STORAGE_ENV_REQUIREMENTS=STORAGE_ENV_REQUIREMENTS,
        chunking_by_token_size=chunking_by_token_size,
        compute_mdhash_id=compute_mdhash_id,
        ollama_model_complete=ollama_model_complete,
    )


def register_vector_storage(api: LightRAGApi, storage_name: str, import_path: str) -> None:
    api.STORAGES[storage_name] = import_path
    api.STORAGE_ENV_REQUIREMENTS.setdefault(storage_name, [])

    vector_storage = api.STORAGE_IMPLEMENTATIONS.get("VECTOR_STORAGE")
    if not vector_storage:
        return

    implementations = vector_storage.setdefault("implementations", [])
    if storage_name not in implementations:
        implementations.append(storage_name)


def make_query_param(mode: str):
    return load_lightrag_api().QueryParam(mode=mode)
