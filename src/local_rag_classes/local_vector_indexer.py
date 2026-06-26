from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.local_rag as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


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

    def _reuse_candidates(self, store: LanceDBVectorStore) -> dict[str, dict[str, Any]]:
        if not store.exists():
            return {}
        try:
            model, dim = store.metadata()
            if model != self.embedding_model or dim != 768:
                return {}
            candidates: dict[str, dict[str, Any]] = {}
            for record in store.all_records():
                record_id = str(record.get("id") or "")
                vector = record.get("vector")
                if not record_id or vector is None:
                    continue
                if len(vector) != 768:
                    continue
                candidates[record_id] = {
                    "content_hash": index_record_content_hash(record),
                    "vector": [float(value) for value in vector],
                }
            return candidates
        except Exception as exc:
            _status(
                f"Local index: existing index could not be inspected for vector reuse: {exc}",
                enabled=self.progress_enabled,
            )
            return {}

    def _attach_vectors(
        self,
        records: list[dict[str, Any]],
        *,
        store: LanceDBVectorStore,
    ) -> tuple[int, int]:
        reuse_candidates = self._reuse_candidates(store)
        pending: list[tuple[int, dict[str, Any]]] = []
        reused = 0
        for index, record in enumerate(records):
            record_id = str(record.get("id") or "")
            candidate = reuse_candidates.get(record_id)
            if candidate and candidate["content_hash"] == index_record_content_hash(record):
                record["vector"] = list(candidate["vector"])
                record["vector_reused"] = True
                reused += 1
            else:
                pending.append((index, record))

        if not pending:
            _status(
                f"Local index: reused all {reused} existing vector(s); no embedding batches needed.",
                enabled=self.progress_enabled,
            )
            return reused, 0

        self._preflight_embeddings()
        embedded = 0
        for file_name, grouped in self._group_pending_by_file(pending).items():
            _status(
                f"Local index: embedding {len(grouped)} changed/new record(s) from {file_name}",
                enabled=self.progress_enabled,
            )
            vectors = self._embed_texts(
                [record["content"] for _, record in grouped],
                file_name=file_name,
            )
            for (record_index, _), vector in zip(grouped, vectors):
                records[record_index]["vector"] = vector.tolist()
                embedded += 1

        return reused, embedded

    @staticmethod
    def _group_pending_by_file(
        pending: list[tuple[int, dict[str, Any]]],
    ) -> dict[str, list[tuple[int, dict[str, Any]]]]:
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for item in pending:
            file_name = Path(str(item[1].get("file_path") or "records")).name
            grouped.setdefault(file_name, []).append(item)
        return grouped

    def index_markdown(self, markdown_dir: str) -> None:
        os.makedirs(self.working_dir, exist_ok=True)
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
                f"Local index: prepared {len(file_records)} structured record(s) from {file_path.name}",
                enabled=self.progress_enabled,
            )
            records.extend(file_records)

        backend = (self.index_backend or "lancedb").lower()
        if backend != "lancedb":
            raise ValueError("index_backend must be lancedb; JSON chunk storage has been removed.")

        store = LanceDBVectorStore(self.working_dir)
        reused, embedded = self._attach_vectors(records, store=store)
        output_target = store.db_path / "chunks"
        store.write_records(
            records,
            embedding_model=self.embedding_model,
            embedding_dim=768,
        )
        write_index_manifest(
            self.working_dir,
            records,
            embedding_model=self.embedding_model,
            embedding_dim=768,
        )
        _status(
            f"Local index: wrote {len(records)} structured record(s) to {output_target} "
            f"({embedded} embedded, {reused} reused)",
            enabled=self.progress_enabled,
        )

LocalVectorIndexer.__module__ = _source_module.__name__
finalize_split_class(_source_module, LocalVectorIndexer)

