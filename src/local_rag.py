from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from src.sectioning import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TARGET_TOKENS,
    build_section_records,
)
from src.vector_store import LanceDBVectorStore, default_store

QUERY_TEMPERATURE = 0.9
DEFAULT_NUM_PREDICT = 4096
DEFAULT_SAMPLER_TOP_K = 40
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_LLM_TIMEOUT = 120.0
DEFAULT_RETRIEVAL_CANDIDATE_K = 80
DEFAULT_RETRIEVAL_MIN_SCORE = 0.50
DEFAULT_RETRIEVAL_RELATIVE_CUTOFF = 0.72
DEFAULT_CONTEXT_TOKEN_FRACTION = 0.60
DEFAULT_TOOL_MAX_ROUNDS = 4
DEFAULT_TOOL_MAX_CALLS = 8
DEFAULT_WEB_SEARCH_TIMEOUT = 8.0
DEFAULT_WEB_SEARCH_MAX_RESULTS = 5
DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL = 5.0
DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS = 5
CONTEXT_METADATA_TOKEN_OVERHEAD = 64
CONTEXT_TRUNCATION_NOTICE = "\n\n[Context truncated to fit prompt budget.]"
MIN_CONTEXT_BLOCK_TOKENS = 96
DEFAULT_QUERY_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Before answering, you must call "
    "search_local_context at least once. You may call search_local_context again "
    "with a narrower query when more evidence is needed. {web_instruction} "
    "Use only information returned by tools. In the final answer, cite every "
    "factual claim with source IDs exactly as provided, such as [S1] or [W1]. "
    "If the available sources are insufficient, say what is missing."
)


def _status(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _tqdm():
    from tqdm import tqdm

    return tqdm


def _iter_with_progress(
    iterable,
    *,
    enabled: bool,
    total: int | None,
    desc: str,
    unit: str,
):
    if not enabled:
        return iterable
    return _tqdm()(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        ascii=True,
    )


def _ollama_pull_command(model: str) -> str:
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


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host:
        return "http://127.0.0.1:11434"
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")
    return f"http://{host.rstrip('/')}"


def _ollama_chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, int | float],
    stream: bool,
    timeout: float | None = None,
    health_check_interval: float = DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL,
    max_lost_health_checks: int = DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS,
    tools: list[dict[str, Any]] | None = None,
):
    payload = {
        "model": model,
        "messages": messages,
        "options": options,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
    if stream:
        return _ollama_chat_stream(
            payload,
            health_check_interval=health_check_interval,
            max_lost_health_checks=max_lost_health_checks,
        )
    return _ollama_chat_once(
        payload,
        health_check_interval=health_check_interval,
        max_lost_health_checks=max_lost_health_checks,
    )


def _ollama_chat_request(payload: dict[str, Any]):
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        f"{_ollama_host()}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _ollama_health_request():
    return urllib.request.Request(f"{_ollama_host()}/api/version", method="GET")


def _ollama_server_healthy(*, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(_ollama_health_request(), timeout=timeout) as response:
            return int(getattr(response, "status", 200) or 200) < 500
    except Exception:
        return False


def _wait_for_ollama_recovery(
    *,
    health_check_interval: float,
    max_lost_health_checks: int,
) -> bool:
    interval = max(0.1, float(health_check_interval))
    cycles = max(1, int(max_lost_health_checks))
    health_timeout = min(max(interval, 0.1), 10.0)
    for _ in range(cycles):
        time.sleep(interval)
        if _ollama_server_healthy(timeout=health_timeout):
            return True
    return False


def _connection_lost_error(kind: str, exc: BaseException, *, max_lost_health_checks: int) -> RuntimeError:
    return RuntimeError(
        f"Ollama {kind} lost connection to {_ollama_host()} and did not recover "
        f"within {max_lost_health_checks} health check cycle(s). Original error: {exc}"
    )


def _ollama_chat_once(
    payload: dict[str, Any],
    *,
    health_check_interval: float,
    max_lost_health_checks: int,
) -> dict[str, Any]:
    retries_after_recovery = 0
    while True:
        try:
            with urllib.request.urlopen(_ollama_chat_request(payload), timeout=None) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout, urllib.error.URLError, OSError) as exc:
            recovered = _wait_for_ollama_recovery(
                health_check_interval=health_check_interval,
                max_lost_health_checks=max_lost_health_checks,
            )
            if not recovered:
                raise _connection_lost_error("chat request", exc, max_lost_health_checks=max_lost_health_checks) from exc
            retries_after_recovery += 1
            if retries_after_recovery > max(1, int(max_lost_health_checks)):
                raise RuntimeError(
                    f"Ollama chat request repeatedly failed even though {_ollama_host()} recovered. "
                    f"Original error: {exc}"
                ) from exc


def _ollama_chat_stream(
    payload: dict[str, Any],
    *,
    health_check_interval: float,
    max_lost_health_checks: int,
):
    def events():
        retries_after_recovery = 0
        while True:
            try:
                with urllib.request.urlopen(_ollama_chat_request(payload), timeout=None) as response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        event = json.loads(line)
                        if event.get("error"):
                            raise RuntimeError(str(event["error"]))
                        yield event
                    return
            except (TimeoutError, socket.timeout, urllib.error.URLError, OSError) as exc:
                recovered = _wait_for_ollama_recovery(
                    health_check_interval=health_check_interval,
                    max_lost_health_checks=max_lost_health_checks,
                )
                if not recovered:
                    raise _connection_lost_error(
                        "chat stream",
                        exc,
                        max_lost_health_checks=max_lost_health_checks,
                    ) from exc
                retries_after_recovery += 1
                if retries_after_recovery > max(1, int(max_lost_health_checks)):
                    raise RuntimeError(
                        f"Ollama chat stream repeatedly failed even though {_ollama_host()} recovered. "
                        f"Original error: {exc}"
                    ) from exc

    return events()


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
    return max(0.0, parsed)


def _ollama_response_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return message.get("content") or ""
        return getattr(message, "content", "") or ""

    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", "") or ""


def _ollama_response_thinking(response: Any) -> str:
    fields = ("thinking", "reasoning", "reasoning_content")
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            for field in fields:
                if message.get(field):
                    return str(message[field])
        for field in fields:
            if response.get(field):
                return str(response[field])
        return ""

    message = getattr(response, "message", None)
    if isinstance(message, dict):
        for field in fields:
            if message.get(field):
                return str(message[field])
    else:
        for field in fields:
            value = getattr(message, field, None)
            if value:
                return str(value)

    for field in fields:
        value = getattr(response, field, None)
        if value:
            return str(value)
    return ""


def _ollama_done_reason(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("done_reason") or "")
    return str(getattr(response, "done_reason", "") or "")


def _ollama_response_message(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        message = response.get("message") or {}
        return dict(message) if isinstance(message, dict) else {}
    message = getattr(response, "message", None)
    return dict(message) if isinstance(message, dict) else {}


def _ollama_tool_calls(response: Any) -> list[dict[str, Any]]:
    message = _ollama_response_message(response)
    calls = message.get("tool_calls") or []
    return [dict(call) for call in calls if isinstance(call, dict)]


def _split_think_tag_events(content: str, *, in_thinking: bool) -> tuple[list[dict[str, str]], bool]:
    events: list[dict[str, str]] = []
    text = content
    while text:
        if in_thinking:
            end = text.find("</think>")
            if end == -1:
                if text:
                    events.append({"type": "thinking", "text": text})
                return events, True
            if end > 0:
                events.append({"type": "thinking", "text": text[:end]})
            text = text[end + len("</think>") :]
            in_thinking = False
            continue

        start = text.find("<think>")
        if start == -1:
            if text:
                events.append({"type": "answer", "text": text})
            return events, False
        if start > 0:
            events.append({"type": "answer", "text": text[:start]})
        text = text[start + len("<think>") :]
        in_thinking = True
    return events, in_thinking


def chunk_markdown(text: str, *, max_chars: int = 3000, overlap: int = 400) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush_current()
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(start + 1, end - overlap)
            continue

        projected_len = current_len + len(paragraph) + (2 if current else 0)
        if current and projected_len > max_chars:
            flush_current()
        current.append(paragraph)
        current_len += len(paragraph) + (2 if current_len else 0)

    flush_current()
    return chunks


def estimate_context_tokens(text: str) -> int:
    return estimate_text_tokens(text) + CONTEXT_METADATA_TOKEN_OVERHEAD


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text or "") / 4)


def estimate_json_tokens(value: Any) -> int:
    return estimate_context_tokens(json.dumps(value, ensure_ascii=False))


def estimate_prompt_tokens(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> int:
    total = 0
    for message in messages:
        total += CONTEXT_METADATA_TOKEN_OVERHEAD
        for key in ("role", "content", "name", "tool_name"):
            if key in message:
                total += estimate_text_tokens(str(message.get(key) or ""))
        if message.get("tool_calls"):
            total += estimate_text_tokens(json.dumps(message["tool_calls"], ensure_ascii=False))
    if tools:
        total += estimate_json_tokens(tools)
    return total


def _fit_text_to_context_budget(text: str, token_budget: int) -> tuple[str, bool, int]:
    token_budget = max(0, int(token_budget))
    current_tokens = estimate_context_tokens(text)
    if current_tokens <= token_budget:
        return text, False, current_tokens
    if token_budget < MIN_CONTEXT_BLOCK_TOKENS:
        return "", True, 0

    source = text or ""
    best = ""
    best_tokens = 0
    low = 0
    high = len(source)
    while low <= high:
        middle = (low + high) // 2
        candidate = source[:middle].rstrip()
        if candidate:
            candidate = f"{candidate}{CONTEXT_TRUNCATION_NOTICE}"
        candidate_tokens = estimate_context_tokens(candidate)
        if candidate and candidate_tokens <= token_budget:
            best = candidate
            best_tokens = candidate_tokens
            low = middle + 1
        else:
            high = middle - 1
    return best, True, best_tokens


def _fit_text_field_to_json_budget(
    item: dict[str, Any],
    field: str,
    token_budget: int,
    *,
    truncated_field: str,
) -> tuple[dict[str, Any] | None, bool, int]:
    token_budget = max(0, int(token_budget))
    current_tokens = estimate_json_tokens(item)
    if current_tokens <= token_budget:
        return item, False, current_tokens
    if token_budget < MIN_CONTEXT_BLOCK_TOKENS:
        return None, True, 0

    source = str(item.get(field) or "")
    best: dict[str, Any] | None = None
    best_tokens = 0
    low = 0
    high = len(source)
    while low <= high:
        middle = (low + high) // 2
        candidate_text = source[:middle].rstrip()
        if candidate_text:
            candidate_text = f"{candidate_text}{CONTEXT_TRUNCATION_NOTICE}"
        candidate = dict(item)
        candidate[field] = candidate_text
        candidate[truncated_field] = True
        candidate_tokens = estimate_json_tokens(candidate)
        if candidate_text and candidate_tokens <= token_budget:
            best = candidate
            best_tokens = candidate_tokens
            low = middle + 1
        else:
            high = middle - 1
    return best, True, best_tokens


def _source_pdf_name(record: dict[str, Any]) -> str:
    name = str(record.get("source_pdf_name") or "").strip()
    if name:
        return name
    source_path = str(record.get("source_pdf_path") or "").strip()
    if source_path:
        return Path(source_path).name
    file_path = str(record.get("file_path") or "").strip()
    return Path(file_path).name if file_path else "source"


def _page_label(record: dict[str, Any]) -> str:
    page_start = int(record.get("page_start") or 0)
    page_end = int(record.get("page_end") or 0)
    if page_start and page_end and page_start != page_end:
        return f"pages {page_start}-{page_end}"
    if page_start:
        return f"page {page_start}"
    return ""


def _short_snippet(text: str, *, limit: int = 360) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "..."


class CitationRegistry:
    def __init__(self):
        self._local_by_record_id: dict[str, dict[str, Any]] = {}
        self._web_by_url: dict[str, dict[str, Any]] = {}
        self._local_sources: list[dict[str, Any]] = []
        self._web_sources: list[dict[str, Any]] = []

    def _local_source(self, record: dict[str, Any], index: int) -> dict[str, Any]:
        record_id = str(record.get("id") or "")
        source_hash = str(record.get("source_hash") or "")
        return {
            "id": f"S{index}",
            "kind": "local",
            "label": f"[S{index}]",
            "chunk_id": record_id,
            "doc_id": str(record.get("doc_id") or ""),
            "source_hash": source_hash,
            "source_pdf_name": _source_pdf_name(record),
            "source_pdf_path": str(record.get("source_pdf_path") or ""),
            "file_path": str(record.get("file_path") or ""),
            "section_path": str(record.get("section_path") or ""),
            "page_start": int(record.get("page_start") or 0),
            "page_end": int(record.get("page_end") or 0),
            "page_label": _page_label(record),
            "score": round(float(record.get("score") or 0.0), 4),
            "snippet": _short_snippet(str(record.get("content") or "")),
            "download_url": f"/api/pdfs/{source_hash}/download" if source_hash else "",
        }

    def preview_local(self, record: dict[str, Any]) -> dict[str, Any]:
        record_id = str(record.get("id") or "")
        if record_id in self._local_by_record_id:
            return self._local_by_record_id[record_id]
        return self._local_source(record, len(self._local_sources) + 1)

    def add_local(self, record: dict[str, Any]) -> dict[str, Any]:
        record_id = str(record.get("id") or "")
        if record_id in self._local_by_record_id:
            return self._local_by_record_id[record_id]
        source = self._local_source(record, len(self._local_sources) + 1)
        self._local_by_record_id[record_id] = source
        self._local_sources.append(source)
        return source

    def _web_source(self, result: dict[str, Any], index: int) -> dict[str, Any]:
        url = str(result.get("url") or "")
        return {
            "id": f"W{index}",
            "kind": "web",
            "label": f"[W{index}]",
            "title": str(result.get("title") or url),
            "url": url,
            "snippet": _short_snippet(str(result.get("snippet") or ""), limit=300),
            "provider": str(result.get("provider") or "duckduckgo_lite"),
        }

    def preview_web(self, result: dict[str, Any]) -> dict[str, Any]:
        url = str(result.get("url") or "")
        if url in self._web_by_url:
            return self._web_by_url[url]
        return self._web_source(result, len(self._web_sources) + 1)

    def add_web(self, result: dict[str, Any]) -> dict[str, Any]:
        url = str(result.get("url") or "")
        if url in self._web_by_url:
            return self._web_by_url[url]
        source = self._web_source(result, len(self._web_sources) + 1)
        self._web_by_url[url] = source
        self._web_sources.append(source)
        return source

    def local_record_ids(self) -> set[str]:
        return set(self._local_by_record_id)

    def all_sources(self) -> list[dict[str, Any]]:
        return [*self._local_sources, *self._web_sources]

    def valid_labels(self) -> set[str]:
        return {source["label"] for source in self.all_sources()}


class DuckDuckGoLiteParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._link: dict[str, Any] | None = None
        self._snippet: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key.lower(): value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "a" and attr.get("href") and (
            "result-link" in classes
            or "result__a" in classes
            or "result-title-a" in classes
            or "uddg=" in attr.get("href", "")
        ):
            self._link = {"href": attr["href"], "text": []}
            return
        if "result-snippet" in classes or "result__snippet" in classes:
            self._snippet = []

    def handle_data(self, data: str) -> None:
        if self._link is not None:
            self._link["text"].append(data)
        if self._snippet is not None:
            self._snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._link is not None:
            title = " ".join("".join(self._link["text"]).split())
            url = normalize_search_url(str(self._link["href"]))
            if title and url and not any(item["url"] == url for item in self.results):
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._link = None
            return
        if self._snippet is not None:
            snippet = " ".join("".join(self._snippet).split())
            if snippet and self.results and not self.results[-1].get("snippet"):
                self.results[-1]["snippet"] = snippet
            self._snippet = None


def normalize_search_url(url: str) -> str:
    raw = url.strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urllib.parse.urlparse(raw)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return urllib.parse.unquote(query["uddg"][0])
    return raw


def web_search_duckduckgo_lite(
    query: str,
    *,
    max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS,
    timeout: float = DEFAULT_WEB_SEARCH_TIMEOUT,
) -> dict[str, Any]:
    import requests

    params = urllib.parse.urlencode({"q": query})
    url = f"https://lite.duckduckgo.com/lite/?{params}"
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "local-rag-pipeline/1.0"},
        )
        response.raise_for_status()
    except Exception as exc:
        return {"results": [], "error": f"Web search failed: {exc}", "provider": "duckduckgo_lite"}

    parser = DuckDuckGoLiteParser()
    parser.feed(response.text)
    results = []
    for result in parser.results:
        if not result.get("url"):
            continue
        results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "provider": "duckduckgo_lite",
            }
        )
        if len(results) >= max(1, max_results):
            break
    return {"results": results, "error": "", "provider": "duckduckgo_lite"}


class LocalVectorIndexer:
    def __init__(
        self,
        working_dir: str = "./db",
        *,
        embedding_model: str = "nomic-embed-text",
        embedding_batch_size: int | None = None,
        embedding_timeout: float | None = None,
        index_backend: str = "lancedb",
        summary_mode: str = "hybrid",
        chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
        chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        progress_enabled: bool = True,
    ):
        from src.embeddings import EmbeddingEngine

        self.working_dir = working_dir
        self.progress_enabled = progress_enabled
        self.index_backend = index_backend
        self.summary_mode = summary_mode
        self.chunk_target_tokens = _positive_int(chunk_target_tokens, DEFAULT_CHUNK_TARGET_TOKENS)
        self.chunk_overlap_tokens = _positive_int(chunk_overlap_tokens, DEFAULT_CHUNK_OVERLAP_TOKENS)
        self.embedding_model = (
            "nomic-embed-text"
            if embedding_model == "nomic-ai/nomic-embed-text-v1.5"
            else embedding_model
        )
        self.embedding_batch_size = _positive_int(
            embedding_batch_size or os.environ.get("LOCAL_RAG_EMBED_BATCH_SIZE"),
            8,
        )
        _status(
            f"Local index: using Ollama embedding model {self.embedding_model} "
            f"(batch_size={self.embedding_batch_size})",
            enabled=progress_enabled,
        )
        self.engine = EmbeddingEngine(
            model_name=self.embedding_model,
            ollama_batch_size=self.embedding_batch_size,
            ollama_timeout=embedding_timeout,
        )

    def _preflight_embeddings(self) -> None:
        _status("Local index: checking Ollama embedding endpoint...", enabled=self.progress_enabled)
        try:
            self.engine.get_mrl_embeddings(
                ["embedding health check"],
                truncate_dim=8,
                prefix="search_document: ",
            )
        except Exception as exc:
            raise RuntimeError(
                "Ollama embedding preflight failed. Restart Ollama, then retry. "
                f"Original error: {exc}"
            ) from exc
        _status("Local index: Ollama embedding endpoint responded.", enabled=self.progress_enabled)

    def _embed_texts(self, texts: list[str], *, file_name: str):
        import numpy as np

        vectors = []
        total_batches = (len(texts) + self.embedding_batch_size - 1) // self.embedding_batch_size
        for batch_number, start in enumerate(
            range(0, len(texts), self.embedding_batch_size),
            start=1,
        ):
            batch = texts[start : start + self.embedding_batch_size]
            end = start + len(batch)
            _status(
                f"Local index: embedding batch {batch_number}/{total_batches} "
                f"for {file_name} records {start + 1}-{end}/{len(texts)} "
                f"({len(batch)} record(s), {sum(len(text) for text in batch)} chars)",
                enabled=self.progress_enabled,
            )
            batch_vectors = self.engine.get_mrl_embeddings(
                batch,
                truncate_dim=768,
                prefix="search_document: ",
            )
            vectors.extend(batch_vectors)
        return np.asarray(vectors)

    def index_markdown(self, markdown_dir: str) -> None:
        os.makedirs(self.working_dir, exist_ok=True)
        self._preflight_embeddings()
        files = sorted(Path(markdown_dir).glob("*.md"))
        _status(f"Local index: found {len(files)} Markdown file(s).", enabled=self.progress_enabled)

        records: list[dict[str, Any]] = []
        for file_path in _iter_with_progress(
            files,
            enabled=self.progress_enabled,
            total=len(files),
            desc="Local index",
            unit="doc",
        ):
            _status(f"Local index: reading {file_path.name}", enabled=self.progress_enabled)
            if not file_path.read_text(encoding="utf-8").strip():
                _status(f"Local index: skipping empty Markdown file {file_path.name}", enabled=self.progress_enabled)
                continue
            file_records = build_section_records(
                file_path,
                source_root=Path.cwd(),
                summary_mode=self.summary_mode,
                chunk_target_tokens=self.chunk_target_tokens,
                chunk_overlap_tokens=self.chunk_overlap_tokens,
            )
            if not file_records:
                continue

            _status(
                f"Local index: embedding {len(file_records)} structured record(s) from {file_path.name}",
                enabled=self.progress_enabled,
            )
            vectors = self._embed_texts(
                [record["content"] for record in file_records],
                file_name=file_path.name,
            )
            for record, vector in zip(file_records, vectors):
                record["vector"] = vector.tolist()
                records.append(record)

        backend = (self.index_backend or "lancedb").lower()
        if backend != "lancedb":
            raise ValueError("index_backend must be lancedb; JSON chunk storage has been removed.")

        store = LanceDBVectorStore(self.working_dir)
        output_target = store.db_path / "chunks"
        store.write_records(
            records,
            embedding_model=self.embedding_model,
            embedding_dim=768,
        )
        _status(
            f"Local index: wrote {len(records)} structured record(s) to {output_target}",
            enabled=self.progress_enabled,
        )


class LocalQueryEngine:
    def __init__(
        self,
        working_dir: str = "./db",
        *,
        model: str = "gemma4",
        embedding_model: str = "nomic-embed-text",
        embedding_batch_size: int | None = None,
        embedding_timeout: float | None = None,
        progress_enabled: bool = True,
        top_k: int | None = None,
        num_predict: int | None = None,
        llm_timeout: float | None = None,
        temperature: float | None = None,
        sampler_top_k: int | None = None,
        context_window: int | None = None,
        retrieval_candidate_k: int | None = None,
        retrieval_min_score: float | None = None,
        retrieval_relative_cutoff: float | None = None,
        context_token_fraction: float | None = None,
        web_search_enabled: bool = True,
        web_search_timeout: float | None = None,
        web_search_max_results: int | None = None,
        tool_max_rounds: int = DEFAULT_TOOL_MAX_ROUNDS,
        tool_max_calls: int = DEFAULT_TOOL_MAX_CALLS,
        ollama_health_check_interval: float | None = None,
        ollama_max_lost_health_checks: int | None = None,
        system_prompt: str | None = None,
    ):
        from src.embeddings import EmbeddingEngine

        self.working_dir = working_dir
        self.model = model
        self.progress_enabled = progress_enabled
        self.top_k = _positive_int(top_k, 5) if top_k is not None else 5
        self.num_predict = _positive_int(
            num_predict or os.environ.get("LOCAL_RAG_NUM_PREDICT"),
            DEFAULT_NUM_PREDICT,
        )
        self.temperature = _positive_float(
            temperature if temperature is not None else os.environ.get("LOCAL_RAG_TEMPERATURE"),
            QUERY_TEMPERATURE,
        )
        self.sampler_top_k = _positive_int(
            sampler_top_k or os.environ.get("LOCAL_RAG_SAMPLER_TOP_K"),
            DEFAULT_SAMPLER_TOP_K,
        )
        self.context_window = _positive_int(
            context_window or os.environ.get("LOCAL_RAG_NUM_CTX"),
            DEFAULT_CONTEXT_WINDOW,
        )
        self.llm_timeout = _positive_float(
            llm_timeout if llm_timeout is not None else os.environ.get("LOCAL_RAG_LLM_TIMEOUT"),
            DEFAULT_LLM_TIMEOUT,
        )
        self.ollama_health_check_interval = _positive_float(
            ollama_health_check_interval
            if ollama_health_check_interval is not None
            else os.environ.get("LOCAL_RAG_OLLAMA_HEALTH_CHECK_INTERVAL"),
            DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL,
        )
        self.ollama_max_lost_health_checks = _positive_int(
            ollama_max_lost_health_checks or os.environ.get("LOCAL_RAG_OLLAMA_MAX_LOST_HEALTH_CHECKS"),
            DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS,
        )
        self.system_prompt = str(
            system_prompt
            if system_prompt is not None
            else os.environ.get("LOCAL_RAG_SYSTEM_PROMPT") or DEFAULT_QUERY_SYSTEM_PROMPT
        ).strip()
        self.retrieval_candidate_k = _positive_int(
            retrieval_candidate_k or os.environ.get("LOCAL_RAG_RETRIEVAL_CANDIDATE_K"),
            DEFAULT_RETRIEVAL_CANDIDATE_K,
        )
        self.retrieval_min_score = _positive_float(
            retrieval_min_score
            if retrieval_min_score is not None
            else os.environ.get("LOCAL_RAG_RETRIEVAL_MIN_SCORE"),
            DEFAULT_RETRIEVAL_MIN_SCORE,
        )
        self.retrieval_relative_cutoff = _positive_float(
            retrieval_relative_cutoff
            if retrieval_relative_cutoff is not None
            else os.environ.get("LOCAL_RAG_RETRIEVAL_RELATIVE_CUTOFF"),
            DEFAULT_RETRIEVAL_RELATIVE_CUTOFF,
        )
        self.context_token_fraction = _positive_float(
            context_token_fraction
            if context_token_fraction is not None
            else os.environ.get("LOCAL_RAG_CONTEXT_TOKEN_FRACTION"),
            DEFAULT_CONTEXT_TOKEN_FRACTION,
        )
        self.web_search_enabled = bool(web_search_enabled)
        self.web_search_timeout = _positive_float(
            web_search_timeout
            if web_search_timeout is not None
            else os.environ.get("LOCAL_RAG_WEB_SEARCH_TIMEOUT"),
            DEFAULT_WEB_SEARCH_TIMEOUT,
        )
        self.web_search_max_results = _positive_int(
            web_search_max_results or os.environ.get("LOCAL_RAG_WEB_SEARCH_MAX_RESULTS"),
            DEFAULT_WEB_SEARCH_MAX_RESULTS,
        )
        self.tool_max_rounds = _positive_int(tool_max_rounds, DEFAULT_TOOL_MAX_ROUNDS)
        self.tool_max_calls = _positive_int(tool_max_calls, DEFAULT_TOOL_MAX_CALLS)
        self.engine = EmbeddingEngine(
            model_name=embedding_model,
            ollama_batch_size=embedding_batch_size,
            ollama_timeout=embedding_timeout,
        )
        self.store = default_store(working_dir, prefer_lancedb=True)
        self.record_count = self._load_record_count()

    def _ollama_options(self) -> dict[str, int | float]:
        return {
            "temperature": self.temperature,
            "top_k": self.sampler_top_k,
            "num_ctx": self.context_window,
            "num_predict": self.num_predict,
        }

    def _tool_messages(self, question: str) -> list[dict[str, Any]]:
        web_instruction = (
            "Use web_search only when local context is insufficient or the user asks for current/external facts, "
            "and only while the prompt remains under the input-context budget."
            if self.web_search_enabled
            else "Do not use web_search; it is disabled for this request."
        )
        system_prompt = self.system_prompt or DEFAULT_QUERY_SYSTEM_PROMPT
        if "{web_instruction}" in system_prompt:
            system_prompt = system_prompt.replace("{web_instruction}", web_instruction)
        else:
            system_prompt = f"{system_prompt}\n\n{web_instruction}"
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": question},
        ]

    def _tool_definitions(self, *, include_web_search: bool | None = None) -> list[dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_local_context",
                    "description": (
                        "Search the local vectorized PDF index for source chunks relevant to the question. "
                        "Call this before answering and again if additional local evidence is needed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The focused retrieval query.",
                            },
                            "exclude_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional local chunk IDs that should not be returned again.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        web_search_available = self.web_search_enabled if include_web_search is None else include_web_search
        if web_search_available:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": (
                            "Search the public web for current or external facts that are not available "
                            "in local PDF context."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The web search query.",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum result count to return.",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            )
        return tools

    def _load_record_count(self) -> int:
        if not self.store.exists():
            _status(
                f"Local query: LanceDB index not found in {self.working_dir}",
                enabled=self.progress_enabled,
            )
            return 0
        count = self.store.count()
        _status(f"Local query: loaded {count} structured record(s).", enabled=self.progress_enabled)
        return count

    def _context_token_budget(self) -> int:
        effective_fraction = min(self.context_token_fraction, DEFAULT_CONTEXT_TOKEN_FRACTION)
        return max(1, int(math.floor(self.context_window * effective_fraction)))

    def _prompt_token_count(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        return estimate_prompt_tokens(messages, tools=tools)

    def _final_answer_instruction(self) -> str:
        return (
            "Write the final answer now using only the tool results above. "
            "Cite every factual claim with the source IDs in square brackets. "
            "If the sources are insufficient, say what information is missing."
        )

    def _final_answer_reserve_tokens(self) -> int:
        return estimate_prompt_tokens([{"role": "user", "content": self._final_answer_instruction()}])

    def _tool_result_token_budget(self, messages: list[dict[str, Any]] | None) -> int:
        if messages is None:
            return self._context_token_budget()
        remaining = (
            self._context_token_budget()
            - self._final_answer_reserve_tokens()
            - self._prompt_token_count(messages)
        )
        return max(0, remaining)

    def _web_search_allowed(self, messages: list[dict[str, Any]]) -> bool:
        if not self.web_search_enabled:
            return False
        tools_with_web = self._tool_definitions(include_web_search=True)
        return self._prompt_token_count(messages, tools=tools_with_web) <= self._context_token_budget()

    def _retrieve(
        self,
        question: str,
        *,
        exclude_ids: set[str] | None = None,
        token_budget: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.record_count:
            return []
        exclude_ids = exclude_ids or set()

        query_vector = self.engine.get_mrl_embeddings(
            [question],
            truncate_dim=768,
            prefix="search_query: ",
        )[0]
        candidates = self.store.search(
            query_vector.tolist(),
            top_k=self.retrieval_candidate_k,
        )
        if not candidates:
            return []

        best_score = max(float(candidate.get("score") or 0.0) for candidate in candidates)
        score_cutoff = max(self.retrieval_min_score, best_score * self.retrieval_relative_cutoff)
        token_budget = self._context_token_budget() if token_budget is None else max(0, int(token_budget))
        if token_budget <= 0:
            return []
        used_tokens = 0
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        def maybe_add(record: dict[str, Any], *, inherited_score: float | None = None) -> bool:
            nonlocal used_tokens
            record_id = str(record.get("id") or "")
            if not record_id or record_id in seen or record_id in exclude_ids:
                return False
            score = float(record.get("score") or inherited_score or 0.0)
            if score < score_cutoff:
                return False
            content = str(record.get("content") or "")
            tokens = estimate_context_tokens(content)
            original_tokens = tokens
            content_truncated = False
            remaining_tokens = token_budget - used_tokens
            if remaining_tokens <= 0:
                return False
            if tokens > remaining_tokens:
                content, content_truncated, tokens = _fit_text_to_context_budget(content, remaining_tokens)
                if not content:
                    return False
            item = dict(record)
            item["content"] = content
            item["score"] = score
            item["estimated_tokens"] = tokens
            if content_truncated:
                item["content_truncated"] = True
                item["original_estimated_tokens"] = original_tokens
            seen.add(record_id)
            results.append(item)
            used_tokens += tokens
            return True

        for candidate in candidates:
            score = float(candidate.get("score") or 0.0)
            if score < score_cutoff:
                continue
            node_type = str(candidate.get("node_type", "chunk"))
            if node_type == "chunk":
                maybe_add(candidate)
            else:
                for child in self.store.child_chunks(candidate, limit=self.retrieval_candidate_k):
                    if not maybe_add(child, inherited_score=score) and results:
                        if used_tokens >= token_budget:
                            break
            if used_tokens >= token_budget:
                break
        return results

    def _local_tool_result(
        self,
        *,
        query: str,
        exclude_ids: set[str],
        citations: CitationRegistry,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        token_budget = self._context_token_budget() if token_budget is None else max(0, int(token_budget))
        result = {
            "tool": "search_local_context",
            "query": query,
            "result_count": 0,
            "context_token_budget": self._context_token_budget(),
            "results": [],
            "instructions": (
                "Use these source IDs for citations. Do not cite local context that is not listed here."
            ),
        }
        base_tokens = estimate_json_tokens(result)
        if token_budget <= base_tokens:
            result["error"] = "Local context skipped because the prompt token budget is exhausted."
            return result

        result_budget = token_budget - base_tokens
        matches = self._retrieve(query, exclude_ids=exclude_ids, token_budget=result_budget)
        context_blocks = []
        used_tokens = 0
        truncated = False
        for match in matches:
            source = citations.preview_local(match)
            location = " :: ".join(
                part
                for part in [
                    source.get("source_pdf_name", ""),
                    source.get("section_path", ""),
                    source.get("page_label", ""),
                ]
                if part
            )
            block = {
                "source_id": source["id"],
                "citation": source["label"],
                "chunk_id": source["chunk_id"],
                "score": source["score"],
                "location": location,
                "content": str(match.get("content") or ""),
            }
            remaining_tokens = result_budget - used_tokens
            fitted, block_truncated, block_tokens = _fit_text_field_to_json_budget(
                block,
                "content",
                remaining_tokens,
                truncated_field="content_truncated",
            )
            if fitted is None:
                if context_blocks:
                    break
                continue
            citations.add_local(match)
            if bool(match.get("content_truncated")) and not block_truncated:
                fitted["content_truncated"] = True
            context_blocks.append(fitted)
            used_tokens += block_tokens
            truncated = truncated or block_truncated or bool(match.get("content_truncated"))
            if used_tokens >= result_budget:
                break
        result["results"] = context_blocks
        result["result_count"] = len(context_blocks)
        if truncated:
            result["context_truncated"] = True
        if not context_blocks and matches:
            result["error"] = "Retrieved local context did not fit within the prompt token budget."
        return result

    def search_local_context(
        self,
        *,
        query: str,
        relevance_floor: float | None = None,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        previous_min_score = self.retrieval_min_score
        if relevance_floor is not None:
            try:
                self.retrieval_min_score = max(0.0, float(relevance_floor))
            except (TypeError, ValueError):
                self.retrieval_min_score = previous_min_score
        try:
            return self._local_tool_result(
                query=query,
                exclude_ids=set(),
                citations=CitationRegistry(),
                token_budget=token_budget,
            )
        finally:
            self.retrieval_min_score = previous_min_score

    def _web_tool_result(
        self,
        *,
        query: str,
        max_results: int | None,
        citations: CitationRegistry,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        if not self.web_search_enabled:
            return {"tool": "web_search", "query": query, "result_count": 0, "error": "Web search is disabled."}
        token_budget = self._context_token_budget() if token_budget is None else max(0, int(token_budget))
        result = {
            "tool": "web_search",
            "query": query,
            "result_count": 0,
            "provider": "duckduckgo_lite",
            "error": "",
            "results": [],
            "instructions": "Use these web source IDs for citations. Do not cite URLs that are not listed here.",
        }
        base_tokens = estimate_json_tokens(result)
        if token_budget <= base_tokens:
            result["error"] = "Web search skipped because the prompt token budget is exhausted."
            return result

        response = web_search_duckduckgo_lite(
            query,
            max_results=_positive_int(max_results, self.web_search_max_results),
            timeout=self.web_search_timeout,
        )
        results = []
        used_tokens = 0
        result_budget = token_budget - base_tokens
        truncated = False
        for raw_result in response.get("results", []):
            source = citations.preview_web(raw_result)
            block = {
                "source_id": source["id"],
                "citation": source["label"],
                "title": source["title"],
                "url": source["url"],
                "snippet": source["snippet"],
                "provider": source["provider"],
            }
            remaining_tokens = result_budget - used_tokens
            fitted, snippet_truncated, block_tokens = _fit_text_field_to_json_budget(
                block,
                "snippet",
                remaining_tokens,
                truncated_field="snippet_truncated",
            )
            if fitted is None:
                if results:
                    break
                continue
            citation_result = dict(raw_result)
            citation_result["snippet"] = fitted.get("snippet", "")
            citations.add_web(citation_result)
            results.append(fitted)
            used_tokens += block_tokens
            truncated = truncated or snippet_truncated
            if used_tokens >= result_budget:
                break
        result = {
            "tool": "web_search",
            "query": query,
            "result_count": len(results),
            "provider": response.get("provider", "duckduckgo_lite"),
            "error": response.get("error", ""),
            "results": results,
            "instructions": "Use these web source IDs for citations. Do not cite URLs that are not listed here.",
        }
        if truncated:
            result["context_truncated"] = True
        if not results and response.get("results") and not result.get("error"):
            result["error"] = "Web results did not fit within the prompt token budget."
        return result

    def _execute_tool_call(
        self,
        call: dict[str, Any],
        *,
        question: str,
        citations: CitationRegistry,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        if name == "search_local_context":
            query = str(arguments.get("query") or question).strip() or question
            raw_excludes = arguments.get("exclude_ids", [])
            requested_excludes = {str(value) for value in raw_excludes} if isinstance(raw_excludes, list) else set()
            exclude_ids = citations.local_record_ids() | requested_excludes
            result = self._local_tool_result(
                query=query,
                exclude_ids=exclude_ids,
                citations=citations,
                token_budget=self._tool_result_token_budget(messages),
            )
            text = f"Retrieved {result['result_count']} local source chunk(s)."
            if result.get("error"):
                text = str(result["error"])
            return name, result, text

        if name == "web_search":
            query = str(arguments.get("query") or question).strip() or question
            if messages is not None:
                current_tokens = self._prompt_token_count(
                    messages,
                    tools=self._tool_definitions(include_web_search=True),
                )
                context_budget = self._context_token_budget()
                if current_tokens > context_budget:
                    result = {
                        "tool": "web_search",
                        "query": query,
                        "result_count": 0,
                        "error": (
                            "Web search skipped because the current prompt is already "
                            f"estimated at {current_tokens} token(s), above the {context_budget} token input-context budget."
                        ),
                    }
                    return name, result, str(result["error"])
            result = self._web_tool_result(
                query=query,
                max_results=arguments.get("max_results"),
                citations=citations,
                token_budget=self._tool_result_token_budget(messages),
            )
            text = f"Retrieved {result['result_count']} web result(s)."
            if result.get("error"):
                text = str(result["error"])
            return name, result, text

        return name or "unknown", {"error": f"Unknown tool: {name or 'unknown'}"}, "Unknown tool call."

    def _append_tool_result(
        self,
        messages: list[dict[str, Any]],
        *,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    @staticmethod
    def _tool_result_event(tool_name: str, result: dict[str, Any], text: str) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool": tool_name,
            "text": text,
            "result": result,
            "content": json.dumps(result, ensure_ascii=False),
        }

    def _forced_local_tool_call(
        self,
        messages: list[dict[str, Any]],
        *,
        question: str,
        citations: CitationRegistry,
    ) -> tuple[dict[str, Any], str]:
        call = {
            "function": {
                "name": "search_local_context",
                "arguments": {"query": question, "exclude_ids": []},
            }
        }
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        tool_name, result, text = self._execute_tool_call(
            call,
            question=question,
            citations=citations,
            messages=messages,
        )
        self._append_tool_result(messages, tool_name=tool_name, result=result)
        return result, text

    def _run_tool_rounds(self, question: str):
        citations = CitationRegistry()
        messages = self._tool_messages(question)
        local_search_used = False
        tool_calls_used = 0

        for _ in range(self.tool_max_rounds):
            tools = self._tool_definitions(include_web_search=self._web_search_allowed(messages))
            response = _ollama_chat(
                model=self.model,
                messages=messages,
                options=self._ollama_options(),
                stream=False,
                timeout=self.llm_timeout,
                health_check_interval=self.ollama_health_check_interval,
                max_lost_health_checks=self.ollama_max_lost_health_checks,
                tools=tools,
            )
            thinking = _ollama_response_thinking(response)
            if thinking:
                yield {"type": "thinking", "text": thinking}

            tool_calls = _ollama_tool_calls(response)
            if not tool_calls:
                if not local_search_used:
                    yield {
                        "type": "notice",
                        "text": "The model did not call the required local search tool, so local context was retrieved before answering.",
                    }
                    result, text = self._forced_local_tool_call(
                        messages,
                        question=question,
                        citations=citations,
                    )
                    local_search_used = True
                    tool_calls_used += 1
                    yield self._tool_result_event("search_local_context", result, text)
                    yield {"type": "sources", "sources": citations.all_sources()}
                    if not result.get("results"):
                        break
                    continue
                break

            message = _ollama_response_message(response)
            messages.append(
                {
                    "role": "assistant",
                    "content": str(message.get("content") or ""),
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                if tool_calls_used >= self.tool_max_calls:
                    yield {
                        "type": "notice",
                        "text": "Tool-call limit reached; answering from the context already retrieved.",
                    }
                    break
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                tool_name = str(function.get("name") or "unknown")
                yield {"type": "tool_call", "tool": tool_name, "text": f"Running {tool_name}..."}
                tool_name, result, text = self._execute_tool_call(
                    call,
                    question=question,
                    citations=citations,
                    messages=messages,
                )
                tool_calls_used += 1
                if tool_name == "search_local_context":
                    local_search_used = True
                self._append_tool_result(messages, tool_name=tool_name, result=result)
                yield self._tool_result_event(tool_name, result, text)
                yield {"type": "sources", "sources": citations.all_sources()}
            if tool_calls_used >= self.tool_max_calls:
                break

        if not local_search_used:
            result, text = self._forced_local_tool_call(
                messages,
                question=question,
                citations=citations,
            )
            yield self._tool_result_event("search_local_context", result, text)
            yield {"type": "sources", "sources": citations.all_sources()}

        messages.append(
            {
                "role": "user",
                "content": self._final_answer_instruction(),
            }
        )
        yield {
            "type": "_tool_state",
            "messages": messages,
            "citations": citations,
        }

    def ask(self, question: str) -> str:
        return "".join(self.ask_stream(question)).strip()

    def ask_stream(self, question: str):
        for event in self.ask_stream_events(question):
            if event["type"] == "answer":
                yield event["text"]

    def ask_stream_events(self, question: str):
        yield {"type": "notice", "text": "Planning retrieval tool calls..."}
        try:
            tool_state: dict[str, Any] | None = None
            for event in self._run_tool_rounds(question):
                if event.get("type") == "_tool_state":
                    tool_state = event
                    continue
                yield event
            if tool_state is None:
                yield {"type": "answer", "text": "No local context could be retrieved. Run index mode first."}
                return

            messages = tool_state["messages"]
            citations: CitationRegistry = tool_state["citations"]
            _status(
                f"Local query: streaming answer from Ollama model {self.model} "
                f"(lost_connection_cycles={self.ollama_max_lost_health_checks}, "
                f"health_interval={self.ollama_health_check_interval:g}s)",
                enabled=self.progress_enabled,
            )
            stream = _ollama_chat(
                model=self.model,
                messages=messages,
                options=self._ollama_options(),
                stream=True,
                timeout=self.llm_timeout,
                health_check_interval=self.ollama_health_check_interval,
                max_lost_health_checks=self.ollama_max_lost_health_checks,
            )
            emitted = False
            answer_emitted = False
            thinking_emitted = False
            in_thinking = False
            done_reason = ""
            answer_text = ""
            for chunk in stream:
                done_reason = _ollama_done_reason(chunk) or done_reason
                thinking = _ollama_response_thinking(chunk)
                if thinking:
                    emitted = True
                    thinking_emitted = True
                    yield {"type": "thinking", "text": thinking}

                content = _ollama_response_content(chunk)
                if content:
                    events, in_thinking = _split_think_tag_events(
                        content,
                        in_thinking=in_thinking,
                    )
                    for event in events:
                        emitted = True
                        if event["type"] == "thinking":
                            thinking_emitted = True
                        if event["type"] == "answer":
                            answer_emitted = True
                            answer_text += event["text"]
                        yield event
        except Exception as exc:
            raise RuntimeError(
                f"Local Ollama streaming query failed for model '{self.model}'. "
                f"Run `{_ollama_pull_command(self.model)}` and ensure Ollama is running. "
                f"Original error: {exc}"
            ) from exc

        if not emitted:
            raise RuntimeError(
                f"Ollama returned an empty streamed answer for model '{self.model}'. "
                "Check the model response in Ollama logs or retry with a different --llm_model."
            )
        if thinking_emitted and not answer_emitted:
            if done_reason == "length":
                yield {
                    "type": "notice",
                    "text": (
                        "The model reached its token limit while still producing its thinking. "
                        "The token budget has been raised for future requests; retry the question "
                        "or increase LOCAL_RAG_NUM_PREDICT for longer answers."
                    ),
                }
            else:
                yield {
                    "type": "notice",
                    "text": "The model ended before producing final answer text.",
                }
        elif done_reason == "length":
            yield {
                "type": "notice",
                "text": "The model reached its token limit, so the answer may be incomplete.",
            }
        invalid = sorted(set(re.findall(r"\[[SW]\d+\]", answer_text)) - citations.valid_labels())
        if invalid:
            yield {
                "type": "notice",
                "text": f"The answer cited unknown source ID(s): {', '.join(invalid)}.",
            }
