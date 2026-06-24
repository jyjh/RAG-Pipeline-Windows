from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.sectioning import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TARGET_TOKENS,
    build_section_records,
)
from src.vector_store import INDEX_FILENAME, JsonVectorStore, LanceDBVectorStore, default_store

QUERY_TEMPERATURE = 0.9
DEFAULT_NUM_PREDICT = 4096
DEFAULT_SAMPLER_TOP_K = 40
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_LLM_TIMEOUT = 120.0


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
    messages: list[dict[str, str]],
    options: dict[str, int | float],
    stream: bool,
    timeout: float,
):
    payload = {
        "model": model,
        "messages": messages,
        "options": options,
        "stream": stream,
    }
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


def _index_path(working_dir: str) -> Path:
    return Path(working_dir) / INDEX_FILENAME


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
        if backend == "json":
            store = JsonVectorStore(self.working_dir)
            output_target = _index_path(self.working_dir)
        elif backend == "lancedb":
            store = LanceDBVectorStore(self.working_dir)
            output_target = store.db_path / "chunks"
        else:
            raise ValueError("index_backend must be one of: lancedb, json")

        try:
            store.write_records(
                records,
                embedding_model=self.embedding_model,
                embedding_dim=768,
            )
        except RuntimeError as exc:
            if backend != "lancedb" or "Access is denied" not in str(exc):
                raise
            _status(
                "Local index: LanceDB write was denied by the filesystem; "
                "falling back to the legacy JSON store for this run.",
                enabled=self.progress_enabled,
            )
            store = JsonVectorStore(self.working_dir)
            output_target = _index_path(self.working_dir)
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
        top_k: int = 5,
        num_predict: int | None = None,
        llm_timeout: float | None = None,
        temperature: float | None = None,
        sampler_top_k: int | None = None,
        context_window: int | None = None,
    ):
        from src.embeddings import EmbeddingEngine

        self.working_dir = working_dir
        self.model = model
        self.progress_enabled = progress_enabled
        self.top_k = top_k
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

    def _chat_messages(self, question: str, matches: list[dict[str, Any]]) -> list[dict[str, str]]:
        context_blocks = []
        for i, match in enumerate(matches, start=1):
            section_path = match.get("section_path") or f"chunk-{match.get('chunk_index')}"
            page_start = int(match.get("page_start") or 0)
            page_end = int(match.get("page_end") or 0)
            page_label = ""
            if page_start and page_end and page_start != page_end:
                page_label = f" pages {page_start}-{page_end}"
            elif page_start:
                page_label = f" page {page_start}"
            context_blocks.append(
                f"[{i}] {match.get('file_path', '')} :: {section_path}{page_label}\n"
                f"{match.get('content', '')}"
            )
        prompt = (
            "Answer the question using only the context below. "
            "Cite source numbers like [1] when relevant. "
            "If the context is insufficient, say what is missing.\n\n"
            f"Question: {question}\n\n"
            "Context:\n"
            + "\n\n".join(context_blocks)
        )
        return [
            {
                "role": "system",
                "content": (
                    "You answer from the supplied local context only. "
                    "Keep any thinking concise, then write a complete final answer "
                    "in plain text and cite source numbers."
                ),
            },
            {"role": "user", "content": prompt},
        ]

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

    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        if not self.record_count:
            return []

        query_vector = self.engine.get_mrl_embeddings(
            [question],
            truncate_dim=768,
            prefix="search_query: ",
        )[0]
        candidates = self.store.search(
            query_vector.tolist(),
            top_k=max(self.top_k * 4, self.top_k + 8),
        )
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            node_type = str(candidate.get("node_type", "chunk"))
            if node_type == "chunk":
                if candidate.get("id") not in seen:
                    seen.add(str(candidate.get("id")))
                    results.append(candidate)
            else:
                for child in self.store.child_chunks(candidate, limit=self.top_k):
                    if child.get("id") in seen:
                        continue
                    seen.add(str(child.get("id")))
                    results.append(child)
                    if len(results) >= self.top_k:
                        break
            if len(results) >= self.top_k:
                break
        return results

    def ask(self, question: str) -> str:
        matches = self._retrieve(question)
        if not matches:
            return "No local index records were found. Run index mode first."

        try:
            _status(
                f"Local query: requesting answer from Ollama model {self.model} "
                f"(timeout={self.llm_timeout:g}s)",
                enabled=self.progress_enabled,
            )
            response = _ollama_chat(
                model=self.model,
                messages=self._chat_messages(question, matches),
                options=self._ollama_options(),
                stream=False,
                timeout=self.llm_timeout,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Local Ollama query failed for model '{self.model}'. "
                f"Run `{_ollama_pull_command(self.model)}` and ensure Ollama is running. "
                f"Original error: {exc}"
            ) from exc

        content = _ollama_response_content(response).strip()
        if not content:
            raise RuntimeError(
                f"Ollama returned an empty answer for model '{self.model}'. "
                "Check the model response in Ollama logs or retry with a different --llm_model."
            )
        return content

    def ask_stream(self, question: str):
        for event in self.ask_stream_events(question):
            if event["type"] == "answer":
                yield event["text"]

    def ask_stream_events(self, question: str):
        yield {"type": "notice", "text": "Embedding query and retrieving context..."}
        matches = self._retrieve(question)
        if not matches:
            yield {"type": "answer", "text": "No local index records were found. Run index mode first."}
            return

        yield {
            "type": "notice",
            "text": (
                f"Retrieved {len(matches)} context chunk(s). "
                f"Requesting answer from {self.model}..."
            ),
        }

        try:
            _status(
                f"Local query: streaming answer from Ollama model {self.model} "
                f"(timeout={self.llm_timeout:g}s)",
                enabled=self.progress_enabled,
            )
            stream = _ollama_chat(
                model=self.model,
                messages=self._chat_messages(question, matches),
                options=self._ollama_options(),
                stream=True,
                timeout=self.llm_timeout,
            )
            emitted = False
            answer_emitted = False
            thinking_emitted = False
            in_thinking = False
            done_reason = ""
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
