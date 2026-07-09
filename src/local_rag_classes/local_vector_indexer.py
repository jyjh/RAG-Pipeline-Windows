from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.local_rag as _source_module
from src.index_overrides import (
    apply_index_overrides,
    apply_overrides_to_records,
    copy_index_overrides,
    load_index_overrides,
)

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
        reuse_db_dir: str | None = None,
        summary_mode: str = "hybrid",
        chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
        chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        progress_enabled: bool = True,
    ):
        from src.embeddings import EmbeddingEngine

        self.working_dir = working_dir
        self.reuse_db_dir = reuse_db_dir
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

    def _reuse_candidates_for_source(
        self,
        store: LanceDBVectorStore,
        source_hash: str,
    ) -> dict[str, dict[str, Any]]:
        """Vector-reuse candidates for a single source only.

        Unlike :meth:`_reuse_candidates` (which loads the *entire* reuse index
        into RAM), this reads only the target source's rows via
        :meth:`records_by_source_hash`, so peak memory stays ~one file during
        streaming indexing instead of ~the whole corpus.
        """
        source_hash = str(source_hash or "")
        if not source_hash or not store.exists():
            return {}
        try:
            model, dim = store.metadata()
            if model != self.embedding_model or dim != 768:
                return {}
            candidates: dict[str, dict[str, Any]] = {}
            for record in store.vectors_by_source_hash([source_hash]):
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

    def _attach_vectors_for_file(
        self,
        records: list[dict[str, Any]],
        *,
        source_hash: str,
        store: LanceDBVectorStore,
    ) -> tuple[int, int]:
        """Attach vectors to a single source's records (streaming indexer).

        Reuses existing vectors by record id + content hash and embeds only the
        pending records. When ``source_hash`` is present, reuse candidates are
        read for that source only (one file's rows, constant memory). When it is
        absent (records not attributable to a source), falls back to a full
        reuse scan of ``store`` -- this is an edge case that does not occur in
        production (every record carries a source_hash) but preserves the old
        whole-index reuse semantics for callers that build records directly.
        Returns ``(reused, embedded)``.
        """
        if source_hash:
            reuse_candidates = self._reuse_candidates_for_source(store, source_hash)
        else:
            reuse_candidates = self._reuse_candidates(store)
        pending: list[dict[str, Any]] = []
        reused = 0
        for record in records:
            record_id = str(record.get("id") or "")
            candidate = reuse_candidates.get(record_id)
            if candidate and candidate["content_hash"] == index_record_content_hash(record):
                record["vector"] = list(candidate["vector"])
                record["vector_reused"] = True
                reused += 1
            else:
                pending.append(record)

        embedded = 0
        if pending:
            file_name = Path(str(records[0].get("file_path") or "records")).name if records else "records"
            _status(
                f"Local index: embedding {len(pending)} changed/new record(s) from {file_name}",
                enabled=self.progress_enabled,
            )
            vectors = self._embed_texts(
                [record["content"] for record in pending],
                file_name=file_name,
            )
            for record, vector in zip(pending, vectors):
                record["vector"] = vector.tolist()
                embedded += 1

        return reused, embedded

    @staticmethod
    def _group_records_by_source(
        records: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group records by their ``source_hash`` (empty string bucket for none).

        In production each Markdown file maps to exactly one source, so this
        yields a single group. Grouping keeps per-file processing correct when a
        file's records span sources (e.g. test fixtures) and lets each source be
        reused + replaced independently.
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            key = str(record.get("source_hash") or "")
            groups.setdefault(key, []).append(record)
        return groups

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
        """Build the index one Markdown file at a time.

        Streaming model: each file is sectioned, embedded, and written
        immediately via :meth:`replace_records_by_source_hash`, then dropped
        from memory before the next file. Peak RAM is ~one file instead of the
        whole corpus. A single file that fails (e.g. an embedding error after
        retries) is recorded in the index summary and skipped; only an all-fail
        run aborts, so the staged build's failure keeps the live index intact.
        """
        os.makedirs(self.working_dir, exist_ok=True)
        files = sorted(Path(markdown_dir).glob("*.md"))
        _status(f"Local index: found {len(files)} Markdown file(s).", enabled=self.progress_enabled)

        backend = (self.index_backend or "lancedb").lower()
        if backend != "lancedb":
            raise ValueError("index_backend must be lancedb; JSON chunk storage has been removed.")

        # Load persisted overrides once and apply them per-file. This replaces
        # the old whole-corpus apply_index_overrides call, which needed every
        # record in memory at once.
        override_db_dir = self.reuse_db_dir or self.working_dir
        overrides = load_index_overrides(override_db_dir)

        store = LanceDBVectorStore(self.working_dir)
        reuse_store = LanceDBVectorStore(self.reuse_db_dir) if self.reuse_db_dir else store

        # Fail fast on an unreachable embedding service before doing any work.
        # Kept outside the per-file try/except so a dead Ollama surfaces as a
        # clear preflight error rather than an "all files failed" summary.
        self._preflight_embeddings()

        # Create the (possibly empty) table up front so the publish
        # precondition (staged lancedb exists) holds even for an empty corpus.
        # append_records creates the table on first write (mode="overwrite"),
        # which is correct for a fresh staged dir; for the in-place rebuild path
        # (no reuse_db_dir, reuse_store IS store) the existing table is left in
        # place so per-source reuse can read it before each replace.
        store.append_records(
            [],
            embedding_model=self.embedding_model,
            embedding_dim=768,
        )

        manifest = _empty_manifest(self.embedding_model, 768)
        total_reused = 0
        total_embedded = 0
        written_records = 0
        processed_files = 0
        failed_files = 0
        index_errors: list[dict[str, Any]] = []

        for file_path in _iter_with_progress(
            files,
            enabled=self.progress_enabled,
            total=len(files),
            desc="Local index",
            unit="doc",
        ):
            file_name = file_path.name
            try:
                _status(f"Local index: reading {file_name}", enabled=self.progress_enabled)
                if not file_path.read_text(encoding="utf-8").strip():
                    _status(
                        f"Local index: skipping empty Markdown file {file_name}",
                        enabled=self.progress_enabled,
                    )
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

                before_overrides = len(file_records)
                file_records = apply_overrides_to_records(file_records, overrides)
                removed_by_overrides = before_overrides - len(file_records)
                if removed_by_overrides:
                    _status(
                        f"Local index: skipped {removed_by_overrides} record(s) hidden by "
                        f"persisted index overrides in {file_name}.",
                        enabled=self.progress_enabled,
                    )
                if not file_records:
                    continue

                _status(
                    f"Local index: prepared {len(file_records)} structured record(s) from {file_name}",
                    enabled=self.progress_enabled,
                )

                # Process each source in the file independently so reuse and the
                # per-source replace stay scoped. In production a file maps to
                # exactly one source, so this loop runs once.
                for source_hash, source_records in self._group_records_by_source(file_records).items():
                    reused, embedded = self._attach_vectors_for_file(
                        source_records,
                        source_hash=source_hash,
                        store=reuse_store,
                    )
                    total_reused += reused
                    total_embedded += embedded

                    if source_hash:
                        store.replace_records_by_source_hash(
                            source_hash=source_hash,
                            records=source_records,
                            embedding_model=self.embedding_model,
                            embedding_dim=768,
                        )
                    else:
                        store.append_records(
                            source_records,
                            embedding_model=self.embedding_model,
                            embedding_dim=768,
                        )
                    _merge_records_into_manifest(manifest, source_records)
                    written_records += len(source_records)

                processed_files += 1
                _status(
                    f"Local index: wrote {len(file_records)} record(s) from {file_name} "
                    f"({total_embedded} embedded, {total_reused} reused cumulative)",
                    enabled=self.progress_enabled,
                )
                # Drop this file's records before reading the next one.
                del file_records
            except Exception as exc:
                failed_files += 1
                index_errors.append({"file": file_name, "error": str(exc)})
                _status(
                    f"Local index: failed to index {file_name}: {exc}. Continuing.",
                    enabled=self.progress_enabled,
                )

        write_index_manifest_payload(self.working_dir, manifest)
        if self.reuse_db_dir:
            copy_index_overrides(self.reuse_db_dir, self.working_dir)
        output_target = store.db_path / "chunks"

        # Persist a structured index run-summary (mirrors the ingest result) so
        # the job queue can surface a processed/failed breakdown and disk usage
        # in the job log, and so a post-mortem is possible after a crash.
        try:
            from src.disk_space import estimate_dir_bytes
            from src.job_logging import write_run_summary

            disk_used = estimate_dir_bytes(self.db_path)
            write_run_summary(
                Path(self.working_dir) / ".index_result.json",
                phase="index",
                files_processed=processed_files,
                files_failed=failed_files,
                elapsed_s=0.0,
                disk_used_bytes=disk_used,
                errors=index_errors or None,
            )
            _source_module.log_event(  # type: ignore[attr-defined]
                "index_run_complete",
                files_processed=processed_files,
                files_failed=failed_files,
                records=written_records,
                embedded=total_embedded,
                reused=total_reused,
                disk_used_bytes=disk_used,
            )
        except Exception:
            # Diagnostics must never break a successful index build.
            pass

        # An all-failed run leaves no useful staged index: abort so the caller
        # skips publishing and the live index is preserved. A partial-success
        # run still publishes the valid subset.
        if processed_files == 0 and files:
            summary = (
                f"Local index: no files were indexed successfully "
                f"({failed_files} failed of {len(files)}). "
            )
            if index_errors:
                summary += "First error: " + index_errors[0]["error"]
            raise RuntimeError(summary)

        _status(
            f"Local index: wrote {written_records} structured record(s) to {output_target} "
            f"from {processed_files} file(s) ({total_embedded} embedded, {total_reused} reused"
            + (f", {failed_files} failed" if failed_files else "")
            + ")",
            enabled=self.progress_enabled,
        )

LocalVectorIndexer.__module__ = _source_module.__name__
finalize_split_class(_source_module, LocalVectorIndexer)

