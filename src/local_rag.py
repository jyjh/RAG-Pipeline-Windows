from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


INDEX_FILENAME = "local_vector_index.json"


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


def _positive_int(value: int | str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


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
        embedding_backend: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        embedding_local_files_only: bool = True,
        embedding_batch_size: int | None = None,
        embedding_timeout: float | None = None,
        progress_enabled: bool = True,
    ):
        from src.embeddings import EmbeddingEngine

        self.working_dir = working_dir
        self.progress_enabled = progress_enabled
        self.embedding_backend = embedding_backend.lower()
        self.embedding_batch_size = _positive_int(
            embedding_batch_size or os.environ.get("LOCAL_RAG_EMBED_BATCH_SIZE"),
            8,
        )
        _status(
            f"Local index: using embedding backend {embedding_backend} with model {embedding_model} "
            f"(batch_size={self.embedding_batch_size})",
            enabled=progress_enabled,
        )
        self.engine = EmbeddingEngine(
            model_name=embedding_model,
            backend=embedding_backend,
            local_files_only=embedding_local_files_only,
            ollama_batch_size=self.embedding_batch_size,
            ollama_timeout=embedding_timeout,
        )

    def _preflight_embeddings(self) -> None:
        if self.embedding_backend != "ollama":
            return
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
                "To bypass Ollama embeddings, rerun with `--embedding_backend hash`. "
                f"Original error: {exc}"
            ) from exc
        _status("Local index: Ollama embedding endpoint responded.", enabled=self.progress_enabled)

    def _embed_chunks(self, chunks: list[str], *, file_name: str):
        import numpy as np

        vectors = []
        total_batches = (len(chunks) + self.embedding_batch_size - 1) // self.embedding_batch_size
        for batch_number, start in enumerate(
            range(0, len(chunks), self.embedding_batch_size),
            start=1,
        ):
            batch = chunks[start : start + self.embedding_batch_size]
            end = start + len(batch)
            _status(
                f"Local index: embedding batch {batch_number}/{total_batches} "
                f"for {file_name} chunks {start + 1}-{end}/{len(chunks)} "
                f"({len(batch)} chunk(s), {sum(len(chunk) for chunk in batch)} chars)",
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
            content = file_path.read_text(encoding="utf-8")
            chunks = chunk_markdown(content)
            if not chunks:
                continue

            _status(
                f"Local index: embedding {len(chunks)} chunk(s) from {file_path.name}",
                enabled=self.progress_enabled,
            )
            vectors = self._embed_chunks(chunks, file_name=file_path.name)
            for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors)):
                records.append(
                    {
                        "id": f"{file_path.name}:{chunk_index}",
                        "file_path": str(file_path),
                        "chunk_index": chunk_index,
                        "content": chunk,
                        "vector": vector.tolist(),
                    }
                )

        payload = {
            "backend": "local_vector",
            "embedding_dim": 768,
            "records": records,
        }
        output_path = _index_path(self.working_dir)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _status(
            f"Local index: wrote {len(records)} chunk(s) to {output_path}",
            enabled=self.progress_enabled,
        )


class LocalQueryEngine:
    def __init__(
        self,
        working_dir: str = "./db",
        *,
        model: str = "gemma4",
        embedding_backend: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        embedding_local_files_only: bool = True,
        embedding_batch_size: int | None = None,
        embedding_timeout: float | None = None,
        progress_enabled: bool = True,
        top_k: int = 5,
    ):
        from src.embeddings import EmbeddingEngine

        self.working_dir = working_dir
        self.model = model
        self.progress_enabled = progress_enabled
        self.top_k = top_k
        self.engine = EmbeddingEngine(
            model_name=embedding_model,
            backend=embedding_backend,
            local_files_only=embedding_local_files_only,
            ollama_batch_size=embedding_batch_size,
            ollama_timeout=embedding_timeout,
        )
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        path = _index_path(self.working_dir)
        if not path.exists():
            _status(f"Local query: index not found at {path}", enabled=self.progress_enabled)
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        _status(f"Local query: loaded {len(records)} chunk(s).", enabled=self.progress_enabled)
        return records

    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        if not self.records:
            return []

        import numpy as np

        query_vector = self.engine.get_mrl_embeddings(
            [question],
            truncate_dim=768,
            prefix="search_query: ",
        )[0]
        vectors = np.asarray([record["vector"] for record in self.records], dtype=np.float32)
        scores = vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        results: list[dict[str, Any]] = []
        for index in top_indices:
            record = dict(self.records[int(index)])
            record["score"] = float(scores[int(index)])
            results.append(record)
        return results

    def ask(self, question: str, mode: str = "hybrid") -> str:
        matches = self._retrieve(question)
        if not matches:
            return "No local index records were found. Run index mode first."

        context_blocks = []
        for i, match in enumerate(matches, start=1):
            context_blocks.append(
                f"[{i}] {match['file_path']}#chunk-{match['chunk_index']}\n{match['content']}"
            )
        prompt = (
            "Answer the question using only the context below. "
            "Cite source numbers like [1] when relevant. "
            "If the context is insufficient, say what is missing.\n\n"
            f"Question: {question}\n\n"
            "Context:\n"
            + "\n\n".join(context_blocks)
        )

        try:
            import ollama

            _status(
                f"Local query: requesting answer from Ollama model {self.model}",
                enabled=self.progress_enabled,
            )
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer from the supplied local context only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Local Ollama query failed for model '{self.model}'. "
                f"Run `{_ollama_pull_command(self.model)}` and ensure Ollama is running. "
                f"Original error: {exc}"
            ) from exc

        if isinstance(response, dict):
            message = response.get("message") or {}
            return message.get("content") or ""
        message = getattr(response, "message", None)
        if isinstance(message, dict):
            return message.get("content") or ""
        return getattr(message, "content", "") or ""
