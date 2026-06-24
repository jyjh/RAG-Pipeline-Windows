from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import sys
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
DEFAULT_RETRIEVAL_MIN_SCORE = 0.36
DEFAULT_RETRIEVAL_RELATIVE_CUTOFF = 0.72
DEFAULT_CONTEXT_TOKEN_FRACTION = 0.60
DEFAULT_TOOL_MAX_ROUNDS = 4
DEFAULT_TOOL_MAX_CALLS = 8
DEFAULT_WEB_SEARCH_TIMEOUT = 8.0
DEFAULT_WEB_SEARCH_MAX_RESULTS = 5
CONTEXT_METADATA_TOKEN_OVERHEAD = 64


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
    timeout: float,
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
        return _ollama_chat_stream(payload, timeout=timeout)
    return _ollama_chat_once(payload, timeout=timeout)


def _ollama_chat_request(payload: dict[str, Any]):
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        f"{_ollama_host()}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _ollama_chat_once(payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(_ollama_chat_request(payload), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"Ollama chat request timed out after {timeout:g}s at {_ollama_host()}/api/chat."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama chat request failed at {_ollama_host()}/api/chat: {exc}") from exc


def _ollama_chat_stream(payload: dict[str, Any], *, timeout: float):
    def events():
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
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"Ollama chat stream timed out after {timeout:g}s at {_ollama_host()}/api/chat."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama chat stream failed at {_ollama_host()}/api/chat: {exc}") from exc

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
    return math.ceil(len(text or "") / 4) + CONTEXT_METADATA_TOKEN_OVERHEAD


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

    def add_local(self, record: dict[str, Any]) -> dict[str, Any]:
        record_id = str(record.get("id") or "")
        if record_id in self._local_by_record_id:
            return self._local_by_record_id[record_id]
        source_hash = str(record.get("source_hash") or "")
        source = {
            "id": f"S{len(self._local_sources) + 1}",
            "kind": "local",
            "label": f"[S{len(self._local_sources) + 1}]",
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
        self._local_by_record_id[record_id] = source
        self._local_sources.append(source)
        return source

    def add_web(self, result: dict[str, Any]) -> dict[str, Any]:
        url = str(result.get("url") or "")
        if url in self._web_by_url:
            return self._web_by_url[url]
        source = {
            "id": f"W{len(self._web_sources) + 1}",
            "kind": "web",
            "label": f"[W{len(self._web_sources) + 1}]",
            "title": str(result.get("title") or url),
            "url": url,
            "snippet": _short_snippet(str(result.get("snippet") or ""), limit=300),
            "provider": str(result.get("provider") or "duckduckgo_lite"),
        }
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
            "Use web_search only when local context is insufficient or the user asks for current/external facts."
            if self.web_search_enabled
            else "Do not use web_search; it is disabled for this request."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-augmented assistant. Before answering, you must call "
                    "search_local_context at least once. You may call search_local_context again "
                    "with a narrower query when more evidence is needed. "
                    f"{web_instruction} "
                    "Use only information returned by tools. In the final answer, cite every "
                    "factual claim with source IDs exactly as provided, such as [S1] or [W1]. "
                    "If the available sources are insufficient, say what is missing."
                ),
            },
            {"role": "user", "content": question},
        ]

    def _tool_definitions(self) -> list[dict[str, Any]]:
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
        if self.web_search_enabled:
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
        return max(256, int(math.floor(self.context_window * min(self.context_token_fraction, 0.95))))

    def _retrieve(self, question: str, *, exclude_ids: set[str] | None = None) -> list[dict[str, Any]]:
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
        token_budget = self._context_token_budget()
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
            tokens = estimate_context_tokens(str(record.get("content") or ""))
            if results and used_tokens + tokens > token_budget:
                return False
            item = dict(record)
            item["score"] = score
            item["estimated_tokens"] = tokens
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
    ) -> dict[str, Any]:
        matches = self._retrieve(query, exclude_ids=exclude_ids)
        context_blocks = []
        for match in matches:
            source = citations.add_local(match)
            location = " :: ".join(
                part
                for part in [
                    source.get("source_pdf_name", ""),
                    source.get("section_path", ""),
                    source.get("page_label", ""),
                ]
                if part
            )
            context_blocks.append(
                {
                    "source_id": source["id"],
                    "citation": source["label"],
                    "chunk_id": source["chunk_id"],
                    "score": source["score"],
                    "location": location,
                    "content": str(match.get("content") or ""),
                }
            )
        return {
            "tool": "search_local_context",
            "query": query,
            "result_count": len(context_blocks),
            "context_token_budget": self._context_token_budget(),
            "results": context_blocks,
            "instructions": (
                "Use these source IDs for citations. Do not cite local context that is not listed here."
            ),
        }

    def _web_tool_result(
        self,
        *,
        query: str,
        max_results: int | None,
        citations: CitationRegistry,
    ) -> dict[str, Any]:
        if not self.web_search_enabled:
            return {"tool": "web_search", "query": query, "result_count": 0, "error": "Web search is disabled."}
        response = web_search_duckduckgo_lite(
            query,
            max_results=_positive_int(max_results, self.web_search_max_results),
            timeout=self.web_search_timeout,
        )
        results = []
        for result in response.get("results", []):
            source = citations.add_web(result)
            results.append(
                {
                    "source_id": source["id"],
                    "citation": source["label"],
                    "title": source["title"],
                    "url": source["url"],
                    "snippet": source["snippet"],
                    "provider": source["provider"],
                }
            )
        return {
            "tool": "web_search",
            "query": query,
            "result_count": len(results),
            "provider": response.get("provider", "duckduckgo_lite"),
            "error": response.get("error", ""),
            "results": results,
            "instructions": "Use these web source IDs for citations. Do not cite URLs that are not listed here.",
        }

    def _execute_tool_call(
        self,
        call: dict[str, Any],
        *,
        question: str,
        citations: CitationRegistry,
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
            result = self._local_tool_result(query=query, exclude_ids=exclude_ids, citations=citations)
            text = f"Retrieved {result['result_count']} local source chunk(s)."
            return name, result, text

        if name == "web_search":
            query = str(arguments.get("query") or question).strip() or question
            result = self._web_tool_result(
                query=query,
                max_results=arguments.get("max_results"),
                citations=citations,
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
        tool_name, result, text = self._execute_tool_call(call, question=question, citations=citations)
        self._append_tool_result(messages, tool_name=tool_name, result=result)
        return result, text

    def _run_tool_rounds(self, question: str):
        citations = CitationRegistry()
        messages = self._tool_messages(question)
        tools = self._tool_definitions()
        local_search_used = False
        tool_calls_used = 0

        for _ in range(self.tool_max_rounds):
            response = _ollama_chat(
                model=self.model,
                messages=messages,
                options=self._ollama_options(),
                stream=False,
                timeout=self.llm_timeout,
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
                    yield {"type": "tool_result", "tool": "search_local_context", "text": text}
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
                )
                tool_calls_used += 1
                if tool_name == "search_local_context":
                    local_search_used = True
                self._append_tool_result(messages, tool_name=tool_name, result=result)
                yield {"type": "tool_result", "tool": tool_name, "text": text}
                yield {"type": "sources", "sources": citations.all_sources()}
            if tool_calls_used >= self.tool_max_calls:
                break

        if not local_search_used:
            result, text = self._forced_local_tool_call(
                messages,
                question=question,
                citations=citations,
            )
            yield {"type": "tool_result", "tool": "search_local_context", "text": text}
            yield {"type": "sources", "sources": citations.all_sources()}

        messages.append(
            {
                "role": "user",
                "content": (
                    "Write the final answer now using only the tool results above. "
                    "Cite every factual claim with the source IDs in square brackets. "
                    "If the sources are insufficient, say what information is missing."
                ),
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
                f"(timeout={self.llm_timeout:g}s)",
                enabled=self.progress_enabled,
            )
            stream = _ollama_chat(
                model=self.model,
                messages=messages,
                options=self._ollama_options(),
                stream=True,
                timeout=self.llm_timeout,
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
