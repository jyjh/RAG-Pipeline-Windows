from __future__ import annotations

import json

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.local_rag as _source_module
from src.index_overrides import (
    apply_index_overrides,
    apply_overrides_to_records,
    copy_index_overrides,
    load_index_overrides,
)
from src.job_logging import RunTimer
from src.progress_protocol import emit_progress

# Crash-resume checkpoint for staged full builds. Written atomically into the
# staged dir every CHECKPOINT_INTERVAL_FILES (or sooner on the last file) so a
# hard crash/OOM/kill mid-build can resume from the last completed file instead
# of restarting the whole (potentially multi-day) embedding pass. Resume is an
# optimization, not a correctness requirement: replace_records_by_source_hash is
# idempotent (delete-then-add per source), so re-running without --resume is
# safe but wasteful. See index_markdown(resume=...).
CHECKPOINT_FILENAME = ".index_build_checkpoint.json"
CHECKPOINT_INTERVAL_FILES = 250
CHECKPOINT_INTERVAL_SECONDS = 900.0  # 15 minutes

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
            128,
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

    def _reuse_candidates(
        self,
        store: LanceDBVectorStore,
        *,
        max_records: int = 100_000,
    ) -> dict[str, dict[str, Any]]:
        # Safety ceiling: loading the entire index into RAM (one Python dict
        # entry per record, each holding a 768-float vector) is only safe for
        # small corpora. At 100GB-scale this would hold gigabytes of vectors and
        # OOM the process. The streaming indexer uses
        # :meth:`_reuse_candidates_for_source` (one source at a time) instead;
        # this whole-index path is an edge-case fallback kept for callers that
        # build records without a source_hash. Guard it so an accidental large
        # index fails loudly instead of silently exhausting memory.
        if not store.exists():
            return {}
        try:
            total = store.count()
            if total > max_records:
                _status(
                    f"Local index: whole-index reuse scan refused for {total} records "
                    f"(cap {max_records}); skipping vector reuse for this batch.",
                    enabled=self.progress_enabled,
                )
                return {}
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

    def _write_file_records(
        self,
        store: LanceDBVectorStore,
        source_groups: list[tuple[str, list[dict[str, Any]]]],
        manifest: dict[str, Any],
    ) -> None:
        """Write one file's already-embedded records to the store (writer thread).

        Called from a background ThreadPoolExecutor thread so the LanceDB write
        overlaps the next file's embedding on the main thread. Each source is
        replaced/appended independently and merged into the shared manifest.
        Writes are serialized (one writer thread, ordered by submission), so
        LanceDB never sees concurrent writes to the same table from this path.
        """
        for source_hash, source_records in source_groups:
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

    def _checkpoint_path(self) -> Path:
        return Path(self.working_dir) / CHECKPOINT_FILENAME

    def _load_checkpoint(self) -> dict[str, Any] | None:
        """Load a valid resume checkpoint from the staged dir, or None.

        A checkpoint is valid only if it was written for the same set of input
        files (same file names + count). A mismatch (e.g. the corpus changed
        between the crash and the resume) invalidates it so resume never skips
        a file that should be (re)processed. Returns None on any issue so the
        caller falls back to a fresh build.
        """
        path = self._checkpoint_path()
        if not path.exists():
            return None
        try:
            import json as _json

            payload = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _write_checkpoint(
        self,
        *,
        completed_files: list[str],
        processed_files: int,
        failed_files: int,
        written_records: int,
        total_embedded: int,
        total_reused: int,
        manifest: dict[str, Any],
    ) -> None:
        """Atomically persist resume state.

        Captures the set of completed markdown filenames plus a snapshot of the
        running manifest so a resumed run can (a) skip re-sectioning/embedding
        completed files and (b) reconstruct the manifest without re-reading the
        already-written LanceDB rows. Atomic via temp+replace so a crash during
        the write never leaves a half-written checkpoint.
        """
        from src.atomic_io import write_json_atomic

        # Strip the non-persisted _working_dir routing key (same as
        # write_index_manifest_payload) so it never lands in the checkpoint.
        manifest_snapshot = {k: v for k, v in manifest.items() if k != "_working_dir"}
        payload = {
            "completed_files": sorted(set(completed_files)),
            "processed_files": int(processed_files),
            "failed_files": int(failed_files),
            "written_records": int(written_records),
            "total_embedded": int(total_embedded),
            "total_reused": int(total_reused),
            "manifest": manifest_snapshot,
            "version": 1,
        }
        try:
            write_json_atomic(self._checkpoint_path(), payload)
        except Exception:
            # Checkpoint is best-effort: a write failure must never abort the
            # build. The next checkpoint attempt will overwrite this one.
            pass

    def index_markdown(self, markdown_dir: str, *, resume: bool = False) -> None:
        """Build the index one Markdown file at a time.

        Streaming model: each file is sectioned, embedded, and written
        immediately via :meth:`replace_records_by_source_hash`, then dropped
        from memory before the next file. Peak RAM is ~one file instead of the
        whole corpus. A single file that fails (e.g. an embedding error after
        retries) is recorded in the index summary and skipped; only an all-fail
        run aborts, so the staged build's failure keeps the live index intact.

        When ``resume=True`` and a valid checkpoint exists in the working dir
        (written by a prior interrupted run of this same build), files already
        completed are skipped -- their LanceDB rows and manifest entries are
        already in place. This turns a crash mid-way through a multi-day build
        into a resume-from-last-checkpoint rather than a restart-from-zero.
        Resume is an optimization: ``replace_records_by_source_hash`` is
        idempotent, so re-running without resume is correct but wasteful.
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
        # Stash the working dir so _merge_records_into_manifest can write
        # per-source content-hash sidecars next to the manifest.
        manifest["_working_dir"] = str(self.working_dir)
        total_reused = 0
        total_embedded = 0
        written_records = 0
        processed_files = 0
        failed_files = 0
        index_errors: list[dict[str, Any]] = []

        # --- Crash-resume: skip files already completed in a prior run. ---
        # A checkpoint is only honored when resume=True AND it was written for
        # exactly this set of input files (so a corpus change between crash and
        # resume can't silently skip new/changed files). On a valid hit, seed
        # the manifest + counters from the checkpoint and filter the file list.
        completed_file_names: set[str] = set()
        checkpoint = self._load_checkpoint() if resume else None
        resuming = False
        if checkpoint is not None:
            cp_files = checkpoint.get("completed_files") or []
            cp_file_set = {str(name) for name in cp_files}
            current_names = {f.name for f in files}
            # Only resume if the checkpoint's completed set is a subset of the
            # current inputs (no missing files) AND non-empty. A superset or a
            # mismatch means the corpus changed -- start fresh.
            if cp_file_set and cp_file_set.issubset(current_names):
                completed_file_names = cp_file_set
                # Rebuild the manifest from the checkpoint so the final publish
                # has correct document entries + embedded/reused totals for the
                # skipped files without re-reading their LanceDB rows.
                cp_manifest = checkpoint.get("manifest") or {}
                if isinstance(cp_manifest, dict):
                    for key, value in cp_manifest.items():
                        if key != "_working_dir":
                            manifest[key] = value
                processed_files = int(checkpoint.get("processed_files") or 0)
                failed_files = int(checkpoint.get("failed_files") or 0)
                written_records = int(checkpoint.get("written_records") or 0)
                total_embedded = int(checkpoint.get("total_embedded") or 0)
                total_reused = int(checkpoint.get("total_reused") or 0)
                resuming = True
                _status(
                    f"Local index: resuming from checkpoint -- skipping "
                    f"{len(completed_file_names)} already-completed file(s).",
                    enabled=self.progress_enabled,
                )
            else:
                _status(
                    "Local index: checkpoint present but inputs changed; starting a fresh build.",
                    enabled=self.progress_enabled,
                )
        files_to_process = [f for f in files if f.name not in completed_file_names]
        # Total file count is known up front; used for the progress ETA. The
        # record-level total is unknown until sectioning, so we emit file-level
        # progress here and embed the cumulative record totals as `extra` for the
        # UI to display alongside. The done count includes previously-completed
        # files so the progress bar reflects overall position, not just this run.
        index_timer = RunTimer()
        total_files_to_index = len(files)

        # Two-stage pipeline: the main thread reads + sections + embeds (GPU-
        # bound), while a single background thread writes the previous file's
        # records to LanceDB (CPU/disk-bound). This overlaps the GPU and I/O so
        # the GPU is never idle waiting for a write. At most one write is in
        # flight at a time, so peak memory stays ~2 files instead of ~1.
        from concurrent.futures import ThreadPoolExecutor

        write_executor = ThreadPoolExecutor(max_workers=1)
        pending_write = None  # Future of the in-flight write, if any

        def _await_pending_write() -> None:
            nonlocal pending_write
            if pending_write is not None:
                pending_write.result()  # propagate write errors synchronously
                pending_write = None

        try:
            # Track the file currently in-flight on the writer thread so a
            # checkpoint can record it as complete only once its write is
            # durable (confirmed by the next _await_pending_write or the final
            # flush). This is the key to crash-safety: a file recorded in the
            # checkpoint is guaranteed to have its LanceDB rows + manifest entry.
            pending_write_file: str | None = None
            last_checkpoint_write = 0.0
            files_done_this_run = 0
            for file_path in _iter_with_progress(
                files_to_process,
                enabled=self.progress_enabled,
                total=len(files_to_process),
                desc="Local index",
                unit="doc",
            ):
                file_name = file_path.name
                try:
                    _status(f"Local index: reading {file_name}", enabled=self.progress_enabled)
                    file_records = build_section_records(
                        file_path,
                        source_root=Path.cwd(),
                        summary_mode=self.summary_mode,
                        chunk_target_tokens=self.chunk_target_tokens,
                        chunk_overlap_tokens=self.chunk_overlap_tokens,
                    )
                    if not file_records:
                        # No records: nothing to write, but count it as processed
                        # so progress is accurate. It is safe to record as
                        # complete immediately (no write to await).
                        processed_files += 1
                        completed_file_names.add(file_name)
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
                        processed_files += 1
                        completed_file_names.add(file_name)
                        continue

                    _status(
                        f"Local index: prepared {len(file_records)} structured record(s) from {file_name}",
                        enabled=self.progress_enabled,
                    )

                    # Embed all sources for this file on the main thread (GPU).
                    source_groups = self._group_records_by_source(file_records)
                    file_reused = 0
                    file_embedded = 0
                    for source_hash, source_records in source_groups.items():
                        reused, embedded = self._attach_vectors_for_file(
                            source_records,
                            source_hash=source_hash,
                            store=reuse_store,
                        )
                        file_reused += reused
                        file_embedded += embedded
                    total_reused += file_reused
                    total_embedded += file_embedded

                    # Wait for the previous write to finish before queueing this
                    # one (bounds memory to ~2 files and preserves write order).
                    # The previous file's write is now durable -- record it as
                    # completed for the checkpoint.
                    _await_pending_write()
                    if pending_write_file is not None:
                        completed_file_names.add(pending_write_file)
                        pending_write_file = None

                    # Snapshot what the background writer needs, then queue it.
                    # manifest merging happens in the writer to keep the main
                    # thread free for the next file's embedding.
                    groups_snapshot = [
                        (sh, [dict(r) for r in recs])
                        for sh, recs in source_groups.items()
                    ]
                    pending_write = write_executor.submit(
                        self._write_file_records,
                        store,
                        groups_snapshot,
                        manifest,
                    )
                    pending_write_file = file_name
                    written_records += len(file_records)
                    processed_files += 1
                    files_done_this_run += 1
                    _status(
                        f"Local index: wrote {len(file_records)} record(s) from {file_name} "
                        f"({total_embedded} embedded, {total_reused} reused cumulative)",
                        enabled=self.progress_enabled,
                    )
                    # Structured progress for the web UI. Emitted unconditionally
                    # (not gated by progress_enabled) so a multi-day index run
                    # surfaced via the job queue still reports done/total/rate.
                    # `done` includes files completed in a prior run (resume) so
                    # the bar reflects overall position.
                    elapsed_min = max(index_timer.elapsed() / 60.0, 1e-9)
                    done_count = len(completed_file_names) + files_done_this_run
                    emit_progress(
                        phase="indexing",
                        done=done_count,
                        total=total_files_to_index,
                        unit="files",
                        rate_per_min=files_done_this_run / elapsed_min,
                        extra={
                            "records_written": written_records,
                            "records_embedded": total_embedded,
                            "records_reused": total_reused,
                            "failed_files": failed_files,
                            "resumed": resuming,
                        },
                    )
                    # Periodic checkpoint so a crash loses at most ~one checkpoint
                    # interval of work. Written only after the in-flight write for
                    # the previous file is confirmed durable above.
                    now = time.monotonic()
                    if (
                        files_done_this_run % CHECKPOINT_INTERVAL_FILES == 0
                        or (now - last_checkpoint_write) >= CHECKPOINT_INTERVAL_SECONDS
                    ):
                        self._write_checkpoint(
                            completed_files=sorted(completed_file_names),
                            processed_files=processed_files,
                            failed_files=failed_files,
                            written_records=written_records,
                            total_embedded=total_embedded,
                            total_reused=total_reused,
                            manifest=manifest,
                        )
                        last_checkpoint_write = now
                    # Drop this file's records before reading the next one.
                    del file_records
                except Exception as exc:
                    failed_files += 1
                    index_errors.append({"file": file_name, "error": str(exc)})
                    _status(
                        f"Local index: failed to index {file_name}: {exc}. Continuing.",
                        enabled=self.progress_enabled,
                    )
                    elapsed_min = max(index_timer.elapsed() / 60.0, 1e-9)
                    done_count = len(completed_file_names) + files_done_this_run
                    emit_progress(
                        phase="indexing",
                        done=done_count,
                        total=total_files_to_index,
                        unit="files",
                        rate_per_min=files_done_this_run / elapsed_min,
                        extra={
                            "records_written": written_records,
                            "records_embedded": total_embedded,
                            "records_reused": total_reused,
                            "failed_files": failed_files,
                            "resumed": resuming,
                        },
                    )
        finally:
            # Flush the last in-flight write before continuing to ANN build.
            _await_pending_write()
            if pending_write_file is not None:
                completed_file_names.add(pending_write_file)
                pending_write_file = None
            write_executor.shutdown(wait=True)


        write_index_manifest_payload(self.working_dir, manifest)
        # The build succeeded: the durable manifest supersedes the resume
        # checkpoint, so remove it. A leftover checkpoint in the staged dir
        # would otherwise be published into the live index (harmless but messy)
        # and could mislead a future resume of a different build. Best-effort.
        try:
            self._checkpoint_path().unlink(missing_ok=True)
        except Exception:
            pass
        if self.reuse_db_dir:
            copy_index_overrides(self.reuse_db_dir, self.working_dir)
        output_target = store.db_path / "chunks"

        # Build the ANN vector index on the staged table so the published index
        # is already query-accelerated. Below ANN_MIN_ROWS this is a no-op (flat
        # scan is faster there); above it, IVF_PQ turns every query from an O(N)
        # scan into a sublinear lookup. Best-effort: an index-build failure must
        # not abort an otherwise-successful index build (search falls back to
        # flat scan when no index is present).
        #
        # The IVF_PQ retrain on a large corpus can take minutes-to-tens-of-minutes
        # with no other output, so emit a progress line first so the UI shows the
        # job is training the ANN index rather than hung.
        emit_progress(
            phase="building_ann_index",
            done=processed_files,
            total=total_files_to_index,
            unit="files",
            extra={
                "records_written": written_records,
                "records_embedded": total_embedded,
                "records_reused": total_reused,
            },
        )
        try:
            ann_result = store.create_vector_index()
            if ann_result.get("built"):
                _status(
                    f"Local index: built {ann_result.get('index_type')} ANN vector index "
                    f"({ann_result.get('num_partitions')} partitions, {ann_result.get('rows')} rows).",
                    enabled=self.progress_enabled,
                )
            else:
                _status(
                    f"Local index: skipped ANN index build ({ann_result.get('reason')}).",
                    enabled=self.progress_enabled,
                )
            emit_progress(
                phase="ann_index_complete",
                done=processed_files,
                total=total_files_to_index,
                unit="files",
                extra={"ann_built": bool(ann_result.get("built"))},
            )
        except Exception as exc:
            from src.vector_store import ANN_MIN_ROWS

            if store.count() >= ANN_MIN_ROWS:
                raise RuntimeError(
                    "ANN index construction failed for a large corpus; refusing to publish "
                    f"a flat-scan index ({store.count()} records). Original error: {exc}"
                ) from exc
            _status(
                f"Local index: ANN index build failed (queries will use flat scan): {exc}",
                enabled=self.progress_enabled,
            )

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

    def index_markdown_sources(self, markdown_dir: str, source_hashes: list[str] | set[str]) -> None:
        """Replace only selected sources in an existing live LanceDB table.

        This is the upload/reindex-source path. A missing table falls back to a
        full build because there is no unaffected corpus to preserve.
        """
        requested = {str(value).strip() for value in source_hashes if str(value).strip()}
        if not requested:
            return

        store = LanceDBVectorStore(self.working_dir)
        if not store.exists():
            self.index_markdown(markdown_dir)
            return

        self._preflight_embeddings()
        overrides = load_index_overrides(self.working_dir)
        manifest_path = Path(self.working_dir) / _source_module.INDEX_MANIFEST_FILENAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Incremental indexing requires a readable index manifest. "
                "Run a full rebuild to recreate it."
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("Incremental indexing found an invalid index manifest. Run a full rebuild.")
        documents = manifest.setdefault("documents", {})
        if not isinstance(documents, dict):
            documents = {}
            manifest["documents"] = documents

        from src.pdf_registry import source_entry_for_markdown

        files_by_hash: dict[str, Path] = {}
        for file_path in sorted(Path(markdown_dir).glob("*.md")):
            entry = source_entry_for_markdown(file_path)
            source_hash = str(entry.get("source_hash") or "").strip()
            if source_hash in requested:
                files_by_hash[source_hash] = file_path

        had_ann = store.has_vector_index()
        changed_rows = 0
        seen: set[str] = set()
        source_updates: dict[str, list[dict[str, Any]]] = {}

        for source_hash, file_path in files_by_hash.items():
            _status(
                f"Local incremental index: reading {file_path.name}",
                enabled=self.progress_enabled,
            )
            records = build_section_records(
                file_path,
                source_root=Path.cwd(),
                summary_mode=self.summary_mode,
                chunk_target_tokens=self.chunk_target_tokens,
                chunk_overlap_tokens=self.chunk_overlap_tokens,
            )
            records = apply_overrides_to_records(records, overrides)
            if records:
                self._attach_vectors_for_file(records, source_hash=source_hash, store=store)
            store.replace_records_by_source_hash(
                source_hash=source_hash,
                records=records,
                embedding_model=self.embedding_model,
                embedding_dim=768,
            )
            source_updates[source_hash] = records
            changed_rows += len(records)
            seen.add(source_hash)

        # Forced re-upload cleanup removes the old Markdown before this method
        # runs. Remove any requested source with no replacement as well.
        missing = requested - seen
        if missing:
            store.delete_records_by_source_hash(source_hashes=sorted(missing))
            for source_hash in missing:
                source_updates[source_hash] = []

        # Update only the affected manifest entries. The manifest is already a
        # compact document-level summary; records remain in LanceDB.
        _source_module.update_index_manifest_sources(
            self.working_dir,
            source_updates,
            embedding_model=self.embedding_model,
            embedding_dim=768,
        )

        if had_ann:
            # Retraining IVF_PQ from scratch is expensive at scale (minutes for
            # millions of vectors). For small deltas the stale centroids still
            # return good results (recall degrades gracefully, not incorrectly),
            # so only retrain when the changed-row fraction crosses the
            # configured threshold. Operators can force a retrain via the
            # 'rebuild_vector_index' job.
            total = store.count()
            threshold = max(0.0, float(_source_module.ANN_RETRAIN_THRESHOLD))
            retrain = not total or changed_rows / total >= threshold
            if retrain:
                ann_result = store.create_vector_index()
                if not ann_result.get("built"):
                    raise RuntimeError(
                        "Incremental indexing changed a corpus with an ANN index, "
                        f"but ANN rebuild did not complete: {ann_result}"
                    )
            else:
                _status(
                    f"Local incremental index: skipping ANN retrain "
                    f"({changed_rows}/{total} rows changed < {threshold:.0%} threshold). "
                    f"Run 'rebuild_vector_index' to retrain manually.",
                    enabled=self.progress_enabled,
                )

        # Re-indexing uses logical delete (tombstones) then add, so repeated
        # incremental runs accumulate dead fragments that raise scan cost and
        # disk usage. compact() runs LanceDB optimize() to merge fragments and
        # reclaim space -- safe to call any time. Auto-compacting here keeps the
        # table healthy without a separate operator action.
        if changed_rows and store.count():
            try:
                compact_result = store.compact()
                if compact_result.get("compacted"):
                    reclaimed = int(compact_result.get("bytes_reclaimed", 0))
                    _status(
                        f"Local incremental index: compacted table "
                        f"(reclaimed {reclaimed/1e6:.1f}MB).",
                        enabled=self.progress_enabled,
                    )
            except Exception as exc:
                # Compaction is maintenance; a failure must not abort a
                # successful incremental index. The operator can retry via the
                # 'compact' job.
                _status(
                    f"Local incremental index: auto-compact skipped ({exc}).",
                    enabled=self.progress_enabled,
                )
        _status(
            f"Local incremental index: replaced {len(source_updates)} source(s), "
            f"{changed_rows} record(s).",
            enabled=self.progress_enabled,
        )

LocalVectorIndexer.__module__ = _source_module.__name__
finalize_split_class(_source_module, LocalVectorIndexer)

