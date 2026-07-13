from __future__ import annotations

import json
import hashlib
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
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from src.atomic_io import write_json_atomic
from src.sectioning import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TARGET_TOKENS,
    build_section_records,
)
from src.vector_store import LanceDBVectorStore, default_store

from src._class_module_support import import_split_class

_CLASS_MODULE_PROXY_FUNCTIONS = (
    "build_section_records",
    "_status",
    "_tqdm",
    "_iter_with_progress",
    "_ollama_pull_command",
    "_ollama_host",
    "_ollama_chat",
    "_ollama_chat_request",
    "_ollama_health_request",
    "_ollama_server_healthy",
    "_wait_for_ollama_recovery",
    "_connection_lost_error",
    "_ollama_chat_once",
    "_ollama_chat_stream",
    "_positive_int",
    "_positive_float",
    "_bounded_float",
    "_ollama_response_content",
    "_ollama_response_thinking",
    "_ollama_done_reason",
    "_ollama_response_message",
    "_ollama_tool_calls",
    "generate_search_queries",
    "_split_think_tag_events",
    "chunk_markdown",
    "estimate_context_tokens",
    "estimate_text_tokens",
    "estimate_json_tokens",
    "estimate_prompt_tokens",
    "_fit_text_to_context_budget",
    "_fit_text_field_to_json_budget",
    "_source_pdf_name",
    "_page_label",
    "_short_snippet",
    "_pdf_source_url",
    "_citation_labels",
    "_claim_keywords",
    "_record_text_for_lexical_score",
    "_lexical_relevance",
    "_bm25_scores",
    "_rrf_fuse",
    "_claim_fragments",
    "_tool_source_texts",
    "citation_support_warnings",
    "_manifest_source_key",
    "index_record_content_hash",
    "_build_index_manifest",
    "_empty_manifest",
    "_merge_records_into_manifest",
    "write_index_manifest",
    "write_index_manifest_payload",
    "load_content_hash_sidecar",
    "_content_hash_sidecar_path",
    "_content_hash_sidecar_dir",
    "_write_content_hash_sidecar",
    "normalize_search_url",
    "web_search_duckduckgo_lite",
)

QUERY_TEMPERATURE = 0.3
INDEX_MANIFEST_FILENAME = "index_manifest.json"
DEFAULT_NUM_PREDICT = 4096
DEFAULT_SAMPLER_TOP_K = 40
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_LLM_TIMEOUT = 120.0
DEFAULT_RETRIEVAL_CANDIDATE_K = 80
DEFAULT_RETRIEVAL_MIN_SCORE = 0.50
DEFAULT_RETRIEVAL_RELATIVE_CUTOFF = 0.72
DEFAULT_RETRIEVAL_LEXICAL_WEIGHT = 0.20
DEFAULT_RRF_K = 60
DEFAULT_CONTEXT_TOKEN_FRACTION = 0.60
DEFAULT_TOOL_MAX_ROUNDS = 4
DEFAULT_TOOL_MAX_CALLS = 8
DEFAULT_WEB_SEARCH_TIMEOUT = 8.0
DEFAULT_WEB_SEARCH_MAX_RESULTS = 5
DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL = 5.0
DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS = 5
DEFAULT_PLANNER_MODEL = "qwen2.5:1.5b"
DEFAULT_PLANNER_MAX_QUERIES = 3
DEFAULT_PLANNER_TIMEOUT = 30.0
DEFAULT_PLANNER_TEMPERATURE = 0.0
DEFAULT_PLANNER_NUM_PREDICT = 256
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
# Variant used when the eager planner has already retrieved local context and
# injected it as a completed search_local_context tool call before the model's
# first turn. The model should answer from that pre-fetched context and only
# call tools again if the evidence is insufficient.
DEFAULT_EAGER_QUERY_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Relevant local context has already "
    "been retrieved for you and is provided in the search_local_context tool "
    "result below; you do not need to call search_local_context before answering. "
    "Answer directly from that pre-fetched context. You may call search_local_context "
    "again with a narrower query, or web_search, only if the provided context is "
    "insufficient to answer. {web_instruction} Use only information returned by "
    "tools. In the final answer, cite every factual claim with source IDs exactly "
    "as provided, such as [S1] or [W1]. If the available sources are insufficient, "
    "say what is missing."
)
# Short directive appended to a user-supplied custom system_prompt when eager
# retrieval is active, so a customized prompt still steers the model to treat the
# pre-fetched tool result as already-provided context.
EAGER_CONTEXT_SUFFIX = (
    "\n\nNote: relevant local context has already been retrieved and is provided "
    "in the search_local_context tool result. Answer directly from it; only call "
    "search_local_context again (narrower query) or web_search if it is insufficient."
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
            timeout=timeout,
            health_check_interval=health_check_interval,
            max_lost_health_checks=max_lost_health_checks,
        )
    return _ollama_chat_once(
        payload,
        timeout=timeout,
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
    timeout: float | None,
    health_check_interval: float,
    max_lost_health_checks: int,
) -> dict[str, Any]:
    retries_after_recovery = 0
    while True:
        try:
            with urllib.request.urlopen(_ollama_chat_request(payload), timeout=timeout) as response:
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
    timeout: float | None,
    health_check_interval: float,
    max_lost_health_checks: int,
):
    def events():
        retries_after_recovery = 0
        while True:
            try:
                with urllib.request.urlopen(_ollama_chat_request(payload), timeout=timeout) as response:
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


def _bounded_float(value: float | str | None, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, _positive_float(value, default)))


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


def _parse_query_list(raw: str, *, max_queries: int) -> list[str]:
    """Parse a planner response into a list of search queries.

    Tries JSON first (handling ```json fences), then falls back to one
    query per non-empty line. Empty results are dropped; the caller is
    responsible for guaranteeing at least the original question is searched.
    """
    text = (raw or "").strip()
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text
    queries: list[str] = []
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            queries = [str(item).strip() for item in parsed if str(item).strip()]
        elif isinstance(parsed, str):
            queries = [parsed.strip()] if parsed.strip() else []
    except (json.JSONDecodeError, ValueError):
        for line in candidate.splitlines():
            line = line.strip().strip('"').strip("'").rstrip(",").strip()
            if line:
                queries.append(line)
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(query)
        if len(unique) >= max_queries:
            break
    return unique


def generate_search_queries(
    question: str,
    *,
    model: str,
    max_queries: int,
    timeout: float,
    temperature: float = DEFAULT_PLANNER_TEMPERATURE,
) -> list[str]:
    """Expand a user question into several diverse retrieval queries.

    Uses a small planner model (``model``) to emit paraphrases, synonyms, and
    key terms. Always returns at least ``[question]`` and never raises: if the
    planner model is unavailable or returns malformed output, the original
    question is used so the flow degrades gracefully to single-query retrieval.
    """
    question = (question or "").strip()
    if not question:
        return []
    if max_queries <= 0:
        return [question]
    system_prompt = (
        "You are a search-query planner for a retrieval-augmented assistant. "
        "Given a user question, produce alternative search queries that would "
        "surface the most relevant source chunks. Use paraphrases, synonyms, "
        "and the key technical terms from the question. Do not answer the "
        'question. Reply with ONLY a JSON array of short query strings.'
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Return up to {max_queries} diverse search queries as a JSON array of "
        'strings, e.g. ["query one", "query two"]. No prose, no explanation.'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    options = {"temperature": temperature, "num_predict": DEFAULT_PLANNER_NUM_PREDICT}
    try:
        response = _ollama_chat(
            model=model,
            messages=messages,
            options=options,
            stream=False,
            timeout=timeout,
        )
        raw = _ollama_response_content(response)
    except Exception as exc:  # planner is best-effort; never break the chat flow
        _status(f"Planner model '{model}' failed; using original question only: {exc}")
        return [question]
    planned = _parse_query_list(raw, max_queries=max_queries)
    if not planned:
        _status(f"Planner model '{model}' returned no usable queries; using original question only.")
        return [question]
    # Guarantee the literal question is always searched, then keep it first.
    planned_lower = {q.lower() for q in planned}
    if question.lower() not in planned_lower:
        return [question, *planned][: max_queries]
    return [question] + [q for q in planned if q.lower() != question.lower()][: max_queries - 1]


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


def _pdf_source_url(source_hash: str, *, mode: str, page: int = 0) -> str:
    if not source_hash:
        return ""
    encoded = urllib.parse.quote(source_hash, safe="")
    url = f"/api/pdfs/{encoded}/{mode}"
    if mode == "view" and page:
        return f"{url}#page={page}"
    return url


CitationRegistry = import_split_class("src.local_rag_classes.citation_registry", "CitationRegistry")
CitationRegistry.__module__ = __name__


SUPPORT_STOPWORDS = {
    "about",
    "above",
    "after",
    "also",
    "because",
    "before",
    "being",
    "between",
    "could",
    "every",
    "from",
    "have",
    "into",
    "more",
    "only",
    "same",
    "should",
    "source",
    "sources",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "using",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def _citation_labels(text: str) -> list[str]:
    return re.findall(r"\[[SW]\d+\]", text or "")


def _claim_keywords(text: str) -> set[str]:
    cleaned = re.sub(r"\[[SW]\d+\]", " ", text or "")
    words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", cleaned)
        if word.lower() not in SUPPORT_STOPWORDS
    }
    return words


def _record_text_for_lexical_score(record: dict[str, Any]) -> str:
    tags = record.get("tags") or []
    if isinstance(tags, (list, tuple)):
        tag_text = " ".join(str(tag) for tag in tags)
    else:
        tag_text = str(tags)
    return " ".join(
        str(record.get(field) or "")
        for field in ("title", "section_path", "summary", "content")
    ) + f" {tag_text}"


def _lexical_relevance(
    query_terms: set[str],
    record: dict[str, Any],
    *,
    keyword_cache: dict[str, tuple[set[str], set[str]]] | None = None,
) -> float:
    if not query_terms:
        return 0.0
    # Per-record keyword sets are recomputed via regex on every call, but
    # ``_retrieve`` ranks candidates and then re-ranks their children, so the
    # same record can be scored twice. Memoize on the record id within one
    # retrieval pass when a cache is supplied.
    record_terms: set[str]
    title_terms: set[str]
    record_id = str(record.get("id") or "")
    if keyword_cache is not None and record_id:
        cached_terms = keyword_cache.get(record_id)
        if cached_terms is not None:
            record_terms, title_terms = cached_terms
        else:
            record_terms = _claim_keywords(_record_text_for_lexical_score(record))
            title_terms = _claim_keywords(
                " ".join(str(record.get(field) or "") for field in ("title", "section_path"))
            )
            keyword_cache[record_id] = (record_terms, title_terms)
    else:
        record_terms = _claim_keywords(_record_text_for_lexical_score(record))
        title_terms = _claim_keywords(
            " ".join(str(record.get(field) or "") for field in ("title", "section_path"))
        )
    if not record_terms:
        return 0.0
    overlap = len(query_terms & record_terms) / max(len(query_terms), 1)
    title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1) if title_terms else 0.0
    return min(1.0, overlap + (0.15 * title_overlap))


def _bm25_scores(
    query_terms: set[str],
    records: list[dict[str, Any]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """Compute Okapi BM25 scores for ``records`` against ``query_terms``.

    A proper lexical ranker over the candidate pool: term-frequency saturation
    (k1) and document-length normalization (b) that the Jaccard overlap score
    lacks. Returns ``{record_id: bm25_score}`` (un-normalized; callers map to
    ranks for RRF). Built over the small candidate set (vector top-k + children)
    so the corpus statistics are local -- not a global BM25 over the whole index,
    but a far stronger lexical signal than set-overlap.
    """
    if not query_terms or not records:
        return {}

    # Tokenize each document once (reuse the same keyword extractor as the
    # Jaccard path so tokenization is consistent across the two lexical scores).
    doc_terms: list[set[str]] = []
    doc_ids: list[str] = []
    for record in records:
        record_id = str(record.get("id") or "")
        if not record_id:
            continue
        terms = _claim_keywords(_record_text_for_lexical_score(record))
        if not terms:
            terms = set()
        doc_terms.append(terms)
        doc_ids.append(record_id)
    if not doc_ids:
        return {}

    n = len(doc_ids)
    avgdl = sum(len(terms) for terms in doc_terms) / n if n else 0.0
    if avgdl <= 0:
        avgdl = 1.0

    # Document frequency per query term across the candidate pool.
    df: dict[str, int] = {}
    for term in query_terms:
        df[term] = sum(1 for terms in doc_terms if term in terms)

    scores: dict[str, float] = {}
    for idx, record_id in enumerate(doc_ids):
        terms = doc_terms[idx]
        dl = len(terms)
        score = 0.0
        for term in query_terms:
            if term not in terms:
                continue
            term_df = df.get(term, 0)
            if term_df <= 0:
                continue
            # BM25 IDF (with +1 smoothing to keep it non-negative for in-pool df).
            idf = math.log(1.0 + (n - term_df + 0.5) / (term_df + 0.5))
            tf = 1  # binary tf within the keyword set (each term counted once)
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * (tf * (k1 + 1.0)) / denom
        scores[record_id] = score
    return scores


def _rrf_fuse(
    vector_ranked: list[str],
    lexical_ranked: list[str],
    *,
    rrf_k: int = 60,
) -> dict[str, float]:
    """Reciproical Rank Fusion of two ranked id lists.

    Combines a vector-search ranking and a lexical (BM25) ranking into a single
    fused score: ``score(id) = 1/(rrf_k + rank_vector) + 1/(rrf_k + rank_lex)``
    (1-indexed ranks; ids absent from one list contribute 0 from that list).
    ``rrf_k`` (default 60) dampens the influence of highly-ranked items so a
    single dominant list can't swamp the other. This is the standard RRF used in
    hybrid retrieval; the ``rrf_k`` config knob has been present but unused.
    """
    scores: dict[str, float] = {}
    for rank, record_id in enumerate(vector_ranked, start=1):
        scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, record_id in enumerate(lexical_ranked, start=1):
        scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def _claim_fragments(answer_text: str) -> list[str]:
    fragments = []
    for line in re.split(r"\n+", answer_text or ""):
        line = line.strip()
        if not line:
            continue
        fragments.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
    return fragments


def _tool_source_texts(messages: list[dict[str, Any]]) -> dict[str, str]:
    texts: dict[str, list[str]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(str(message.get("content") or "{}"))
        except json.JSONDecodeError:
            continue
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("citation") or item.get("source_id") or "").strip()
            if label and not label.startswith("["):
                label = f"[{label}]"
            if not label:
                continue
            text = str(item.get("content") or item.get("snippet") or "")
            if text:
                texts.setdefault(label, []).append(text)
    return {label: "\n".join(parts) for label, parts in texts.items()}


def citation_support_warnings(answer_text: str, messages: list[dict[str, Any]]) -> list[str]:
    source_texts = _tool_source_texts(messages)
    weak_claim_labels: set[str] = set()
    for fragment in _claim_fragments(answer_text):
        labels = _citation_labels(fragment)
        if not labels:
            continue
        keywords = _claim_keywords(fragment)
        if len(keywords) < 4:
            continue
        supported = False
        for label in labels:
            source_text = source_texts.get(label, "").lower()
            if not source_text:
                continue
            overlap = sum(1 for keyword in keywords if keyword in source_text)
            if overlap / max(len(keywords), 1) >= 0.25:
                supported = True
                break
        if not supported:
            weak_claim_labels.update(labels)
    if not weak_claim_labels:
        return []
    labels = ", ".join(sorted(weak_claim_labels))
    return [
        "Citation support check found weak lexical support for cited claim(s) "
        f"using {labels}. Review the Sources panel before relying on those claim(s)."
    ]


def _manifest_source_key(record: dict[str, Any]) -> str:
    source_hash = str(record.get("source_hash") or "").strip()
    if source_hash:
        return source_hash
    source_path = str(record.get("source_pdf_path") or record.get("file_path") or "").strip()
    return source_path or str(record.get("doc_id") or "unknown")


def index_record_content_hash(record: dict[str, Any]) -> str:
    content = str(record.get("content") or "")
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def _build_index_manifest(records: list[dict[str, Any]], *, embedding_model: str, embedding_dim: int) -> dict[str, Any]:
    manifest = _empty_manifest(embedding_model, embedding_dim)
    _merge_records_into_manifest(manifest, records)
    return manifest


def _empty_manifest(embedding_model: str, embedding_dim: int) -> dict[str, Any]:
    """Return a fresh, empty manifest shell ready for :func:`_merge_records_into_manifest`.

    Split from :func:`_build_index_manifest` so the streaming indexer can build
    a manifest incrementally (one file's records merged at a time) instead of
    holding the whole corpus's records in memory before computing the manifest.
    """
    return {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": str(embedding_model),
        "embedding_dim": int(embedding_dim),
        "total_records": 0,
        "embedded_records": 0,
        "reused_records": 0,
        "documents": {},
    }


def _merge_records_into_manifest(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    """Fold ``records`` into ``manifest`` in place, updating running counts.

    Safe to call repeatedly (once per file during streaming indexing). For
    re-indexing a source that already has an entry, pass only that file's
    records and delete the existing document key first.

    Per-record content hashes are written to a per-source sidecar file
    (``hashes/<source_key>.json``) rather than embedded in the manifest, so the
    monolithic ``index_manifest.json`` stays small at scale (millions of chunks
    would otherwise make it hundreds of MB and re-serialize on every write).
    The hashes are consumed by tests and available for future vector-reuse
    bookkeeping; the runtime reuse path computes its own in-memory hashes.
    """
    documents: dict[str, dict[str, Any]] = manifest.setdefault("documents", {})
    embedded = int(manifest.get("embedded_records") or 0)
    reused = int(manifest.get("reused_records") or 0)
    total = int(manifest.get("total_records") or 0)
    # Accumulate per-source content hashes for the sidecar write.
    sidecar_hashes: dict[str, dict[str, str]] = {}
    for record in records:
        key = _manifest_source_key(record)
        document = documents.setdefault(
            key,
            {
                "source_hash": str(record.get("source_hash") or ""),
                "source_pdf_name": str(record.get("source_pdf_name") or ""),
                "source_pdf_path": str(record.get("source_pdf_path") or ""),
                "file_path": str(record.get("file_path") or ""),
                "record_count": 0,
                "chunk_count": 0,
                "summary_count": 0,
                "content_char_count": 0,
                "page_start": 0,
                "page_end": 0,
            },
        )
        node_type = str(record.get("node_type") or "")
        content = str(record.get("content") or "")
        record_id = str(record.get("id") or "")
        if record_id:
            sidecar_hashes.setdefault(key, {})[record_id] = index_record_content_hash(record)
        document["record_count"] += 1
        document["content_char_count"] += len(content)
        if node_type == "chunk":
            document["chunk_count"] += 1
        elif node_type.endswith("summary"):
            document["summary_count"] += 1
        page_start = int(record.get("page_start") or 0)
        page_end = int(record.get("page_end") or 0)
        if page_start:
            current = int(document.get("page_start") or 0)
            document["page_start"] = page_start if not current else min(current, page_start)
        if page_end:
            document["page_end"] = max(int(document.get("page_end") or 0), page_end)
        if record.get("vector_reused"):
            reused += 1
        else:
            embedded += 1
        total += 1
    manifest["total_records"] = total
    manifest["embedded_records"] = embedded
    manifest["reused_records"] = reused
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Write the per-source content-hash sidecars (one file per source touched in
    # this merge). Cheap: each sidecar is ~one source's worth of hashes.
    working_dir = manifest.get("_working_dir")
    if isinstance(working_dir, str) and sidecar_hashes:
        for source_key, hashes in sidecar_hashes.items():
            _write_content_hash_sidecar(working_dir, source_key, hashes)


def _content_hash_sidecar_dir(working_dir: str | Path) -> Path:
    return Path(working_dir) / "hashes"


def _content_hash_sidecar_path(working_dir: str | Path, source_key: str) -> Path:
    # The source_key can be a path (e.g. "processed_docs/doc.md") which is not a
    # safe filename. Hash it to a stable, filesystem-safe basename.
    safe = hashlib.sha256(source_key.encode("utf-8", errors="ignore")).hexdigest()[:32]
    return _content_hash_sidecar_dir(working_dir) / f"{safe}.json"


def _write_content_hash_sidecar(
    working_dir: str | Path, source_key: str, hashes: dict[str, str]
) -> None:
    """Atomically write one source's content hashes to its sidecar file."""
    sidecar_dir = _content_hash_sidecar_dir(working_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    payload = {"source_key": source_key, "content_hashes": hashes}
    path = _content_hash_sidecar_path(working_dir, source_key)
    write_json_atomic(path, payload)


def load_content_hash_sidecar(
    working_dir: str | Path, source_key: str
) -> dict[str, str]:
    """Read one source's content-hash sidecar, or ``{}`` if absent/corrupt."""
    path = _content_hash_sidecar_path(working_dir, source_key)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    hashes = payload.get("content_hashes") if isinstance(payload, dict) else None
    return dict(hashes) if isinstance(hashes, dict) else {}


def _remove_source_from_manifest(manifest: dict[str, Any], source_key: str) -> None:
    """Drop a document entry from ``manifest`` and recompute running totals.

    Also removes the per-source content-hash sidecar if the manifest carries a
    ``_working_dir`` (so a source delete cleans up its hashes file too).
    """
    documents: dict[str, dict[str, Any]] = manifest.setdefault("documents", {})
    document = documents.pop(source_key, None)
    if not document:
        return
    delta = int(document.get("record_count") or 0)
    manifest["total_records"] = max(0, int(manifest.get("total_records") or 0) - delta)
    # Embedded/reused counts are not recoverable per-source from the stored
    # document shape, so recompute them as best-effort: subtract the doc's own
    # record count proportionally only if we tracked it. We did not store
    # per-doc embedded/reused, so leave the totals as upper-bound-ish and let
    # the final write be authoritative for a full rebuild. For incremental
    # re-index the merged records re-add the correct counts.
    # Best-effort sidecar cleanup.
    working_dir = manifest.get("_working_dir")
    if isinstance(working_dir, str):
        try:
            sidecar = _content_hash_sidecar_path(working_dir, source_key)
            if sidecar.exists():
                sidecar.unlink()
        except OSError:
            pass


def write_index_manifest_payload(
    working_dir: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write a pre-built manifest dict to ``working_dir`` atomically."""
    # Strip the internal _working_dir key (used only to route sidecar writes
    # during _merge_records_into_manifest) so it never lands in the JSON file.
    payload = {k: v for k, v in manifest.items() if k != "_working_dir"}
    path = Path(working_dir) / INDEX_MANIFEST_FILENAME
    write_json_atomic(path, payload)
    return payload


def write_index_manifest(
    working_dir: str | Path,
    records: list[dict[str, Any]],
    *,
    embedding_model: str,
    embedding_dim: int,
) -> dict[str, Any]:
    # Build the manifest shell, set the working dir so the merge writes
    # per-source content-hash sidecars next to the manifest, then merge once.
    manifest = _empty_manifest(embedding_model, embedding_dim)
    manifest["_working_dir"] = str(working_dir)
    _merge_records_into_manifest(manifest, records)
    return write_index_manifest_payload(working_dir, manifest)


DuckDuckGoLiteParser = import_split_class("src.local_rag_classes.duck_duck_go_lite_parser", "DuckDuckGoLiteParser")
DuckDuckGoLiteParser.__module__ = __name__


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


LocalVectorIndexer = import_split_class("src.local_rag_classes.local_vector_indexer", "LocalVectorIndexer")
LocalVectorIndexer.__module__ = __name__


LocalQueryEngine = import_split_class("src.local_rag_classes.local_query_engine", "LocalQueryEngine")
LocalQueryEngine.__module__ = __name__
