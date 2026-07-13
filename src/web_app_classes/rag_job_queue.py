from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


def _human_bytes(value: int | float | None) -> str:
    """Format a byte count as a short human-readable string (e.g. '1.2 GB')."""
    if value is None:
        return "0 B"
    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


class RagJobQueue:
    def __init__(
        self,
        *,
        upload_root: Path = UPLOAD_DIR,
        processed_dir: Path = PROCESSED_DIR,
        db_dir: Path = DB_DIR,
        registry_path: Path = PDF_REGISTRY_PATH,
        job_ledger_path: Path = JOB_LEDGER_PATH,
        run_ingestion_func=None,
        run_indexing_func=None,
        max_workers: int = 1,
    ):
        self.upload_root = Path(upload_root)
        self.processed_dir = Path(processed_dir)
        self.db_dir = Path(db_dir)
        self.registry = PdfRegistry(registry_path)
        # Durable ledger for non-upload jobs (reindex/rebuild/backup/restore).
        # Upload jobs recover via the PDF registry; the ledger covers the rest.
        from src.job_ledger import JobLedger

        self.ledger = JobLedger(job_ledger_path)
        self._run_ingestion_func = run_ingestion_func
        self._run_indexing_func = run_indexing_func
        self._condition = threading.Condition(threading.RLock())
        self._jobs: dict[str, QueueJob] = {}
        self._queue: deque[str] = deque()
        # Multiple worker threads pull from the shared deque (each pop is atomic
        # under the condition lock, so no two workers get the same job). This
        # lets ingestion-only phases of different upload jobs overlap. Index-
        # mutating phases are serialized via ``_index_write_lock`` below.
        self._max_workers = max(1, int(max_workers))
        self._workers: list[threading.Thread] = []
        # Serializes all index-mutating operations (reindex, rebuild, backup,
        # restore, rebuild_vector_index, and the upload job's indexing phase).
        # These all write to the same ``db/`` dir and share the cross-process
        # index lock; only one may run at a time. Ingestion-only phases (parsing
        # PDFs into processed_docs/) do NOT take this lock, so they overlap.
        self._index_write_lock = threading.Lock()
        self.active_query_count = 0

    def begin_query(self) -> None:
        with self._condition:
            self.active_query_count += 1
            self._condition.notify_all()

    def finish_query(self) -> None:
        with self._condition:
            self.active_query_count = max(0, self.active_query_count - 1)
            self._condition.notify_all()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in {"done", "failed", "cancelled"}:
                raise ValueError(f"Job {job_id} is already {job.status}.")

            job.cancel_requested = True
            job._cancel_event.set()
            if job.status == "queued":
                self._queue = deque(existing_id for existing_id in self._queue if existing_id != job.id)
                self._mark_cancelled_locked(job, "Job cancelled by user.")
                payload = job.to_dict()
            else:
                job.error = "Cancellation requested."
                payload = job.to_dict()
            self._condition.notify_all()

        if payload["status"] == "cancelled":
            self._record_interrupted(job, "Job cancelled by user.")
        return payload

    def _mark_cancelled_locked(self, job: QueueJob, message: str) -> None:
        job.status = "cancelled"
        job.phase = "cancelled"
        job.error = message
        job.finished_at = _utcnow()

    def _record_interrupted(self, job: QueueJob, message: str) -> None:
        if job.kind == "upload" and job.uploads:
            self.registry.mark_job_status(
                job_id=job.id,
                files=job.uploads,
                status="interrupted",
                error=message,
            )
        if job.source_hashes:
            self.registry.mark_sources_interrupted(
                job_id=job.id,
                source_hashes=job.source_hashes,
                error=message,
            )

    def _raise_if_cancelled(self, job: QueueJob | None) -> None:
        if job is not None and (job.cancel_requested or job._cancel_event.is_set()):
            raise JobCancelled("Job cancelled by user.")

    def _append_job_log(self, job: QueueJob, text: str) -> None:
        line = str(text or "").rstrip()
        if not line:
            return
        with self._condition:
            job.log_line_count += 1
            job.log_tail.append(line)
            if len(job.log_tail) > JOB_LOG_TAIL_LINES:
                del job.log_tail[: len(job.log_tail) - JOB_LOG_TAIL_LINES]
            self._condition.notify_all()

    def enqueue_upload(
        self,
        *,
        staging_dir: Path,
        filenames: list[str],
        uploads: list[dict[str, Any]] | None = None,
        force_duplicate_hashes: list[str] | None = None,
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        resume_status: str | None = None,
        recovered: bool = False,
        created_at: str | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="upload",
            filenames=filenames,
            uploads=uploads or [],
            force_duplicate_hashes=force_duplicate_hashes or [],
            staging_dir=str(staging_dir),
            resume_status=resume_status,
            recovered=recovered,
            options=options or {},
            created_at=created_at or _utcnow(),
        )
        return self._enqueue(job, auto_start=auto_start)

    def enqueue_reindex(
        self,
        *,
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="reindex",
            options=options or {},
        )
        return self._enqueue(job, auto_start=auto_start)

    def enqueue_reindex_source(
        self,
        *,
        source_hashes: list[str],
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="reindex_source",
            source_hashes=[str(value) for value in source_hashes if value],
            options=options or {},
        )
        return self._enqueue(job, auto_start=auto_start)

    def enqueue_backup(
        self,
        *,
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="backup",
            options=options or {},
        )
        return self._enqueue(job, auto_start=auto_start)

    def enqueue_rebuild(
        self,
        *,
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="rebuild",
            options=options or {},
        )
        return self._enqueue(job, auto_start=auto_start)

    def enqueue_rebuild_vector_index(
        self,
        *,
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        """Re-train the ANN vector index on the live DB without re-embedding.

        Useful after many incremental reindex_source operations leave the IVF
        partitions suboptimal. Reads existing vectors, drops the old index, and
        rebuilds in place under the index lock.
        """
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="rebuild_vector_index",
            options=options or {},
        )
        return self._enqueue(job, auto_start=auto_start)

    def enqueue_restore(
        self,
        *,
        backup_name: str,
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="restore",
            backup_name=str(backup_name),
            options=options or {},
        )
        return self._enqueue(job, auto_start=auto_start)

    def enqueue_compact(
        self,
        *,
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        """Compact the live LanceDB table, reclaiming space from reindex cycles.

        Re-indexing uses logical delete + add, so dead fragments accumulate and
        the on-disk table grows (and scans slow) until a compaction runs. Safe
        to run any time; recommended after a large reindex or periodically.
        """
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="compact",
            options=options or {},
        )
        return self._enqueue(job, auto_start=auto_start)

    def _enqueue(self, job: QueueJob, *, auto_start: bool) -> QueueJob:
        with self._condition:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            if auto_start:
                self._ensure_worker_locked()
            self._condition.notify_all()
        # Persist non-upload jobs so they survive a crash. Done outside the
        # condition lock because the ledger has its own lock and is best-effort.
        self._ledger_record(job)
        return job

    def _ensure_worker_locked(self) -> None:
        # Spawn enough workers to reach ``_max_workers`` as long as there is
        # queued work. Each worker exits when the queue empties (see
        # ``_worker_loop``), so the pool naturally winds down to zero when idle.
        alive = sum(1 for w in self._workers if w.is_alive())
        while alive < self._max_workers and self._queue:
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._workers.append(worker)
            worker.start()
            alive += 1

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                if not self._queue:
                    # Remove this worker from the pool; it will be re-spawned by
                    # ``_ensure_worker_locked`` on the next enqueue if needed.
                    current = threading.current_thread()
                    self._workers = [w for w in self._workers if w is not current]
                    return
                job = self._jobs[self._queue.popleft()]
                if job.cancel_requested:
                    self._mark_cancelled_locked(job, "Job cancelled by user.")
                    self._condition.notify_all()
                    cancelled_before_start = True
                else:
                    cancelled_before_start = False
                    job.status = "running"
                    job.phase = "starting"
                    job.started_at = job.started_at or _utcnow()

            if cancelled_before_start:
                self._record_interrupted(job, "Job cancelled by user.")
                self._ledger_remove(job)
                continue

            try:
                self._raise_if_cancelled(job)
                self._run_job(job)
            except JobCancelled as exc:
                message = str(exc) or "Job cancelled by user."
                self._record_interrupted(job, message)
                with self._condition:
                    self._mark_cancelled_locked(job, message)
                    self._condition.notify_all()
            except Exception as exc:
                if job.cancel_requested:
                    message = "Job cancelled by user."
                    self._record_interrupted(job, message)
                    with self._condition:
                        self._mark_cancelled_locked(job, message)
                        self._condition.notify_all()
                    self._ledger_remove(job)
                    continue
                if job.kind == "upload" and job.uploads:
                    self.registry.mark_job_status(
                        job_id=job.id,
                        files=job.uploads,
                        status="failed",
                        error=str(exc),
                    )
                with self._condition:
                    job.status = "failed"
                    job.phase = "failed"
                    job.error = str(exc)
                    job.finished_at = _utcnow()
                    self._condition.notify_all()
            else:
                with self._condition:
                    if job.cancel_requested:
                        self._mark_cancelled_locked(job, "Job cancelled by user.")
                        record_cancelled = True
                    else:
                        job.status = "done"
                        job.phase = "done"
                        job.finished_at = _utcnow()
                        record_cancelled = False
                    self._condition.notify_all()
                if record_cancelled:
                    self._record_interrupted(job, "Job cancelled by user.")
            # Any terminal state (done/failed/cancelled) removes the job from
            # the durable ledger so it is not re-enqueued on the next startup.
            self._ledger_remove(job)

    def _ledger_record(self, job: QueueJob) -> None:
        """Persist a tracked (non-upload) job so it survives a crash.

        Only the fields needed to reconstruct the job on recovery are stored;
        transient state (log tail, status) is not, since the recovered job
        restarts fresh.
        """
        if job.kind not in {
            "reindex",
            "reindex_source",
            "rebuild",
            "backup",
            "restore",
            "rebuild_vector_index",
            "compact",
        }:
            return
        try:
            self.ledger.record(
                job.id,
                kind=job.kind,
                source_hashes=list(job.source_hashes or []),
                backup_name=job.backup_name,
                options=dict(job.options or {}),
            )
        except Exception:
            # Ledger is best-effort: never let it block an enqueue.
            pass

    def _ledger_remove(self, job: QueueJob) -> None:
        """Remove a job from the durable ledger once it reaches a terminal state."""
        if job.kind not in {
            "reindex",
            "reindex_source",
            "rebuild",
            "backup",
            "restore",
            "rebuild_vector_index",
            "compact",
        }:
            return
        try:
            self.ledger.remove(job.id)
        except Exception:
            pass

    def _run_job(self, job: QueueJob) -> None:
        if job.kind == "upload":
            self._raise_if_cancelled(job)
            self._run_upload_job(job)
            return

        # All non-upload job kinds mutate the live index and must be serialized
        # so only one writes to ``db/`` at a time. The lock is held for the
        # entire job (including the query-wait) so a second index job doesn't
        # start its query-wait while the first is still publishing.
        with self._index_write_lock:
            if job.kind == "reindex":
                self._wait_for_no_queries(job, "indexing")
                self._raise_if_cancelled(job)
                self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options, job=job)
                return

            if job.kind == "reindex_source":
                self._wait_for_no_queries(job, "indexing")
                self._raise_if_cancelled(job)
                if self._run_indexing_func is not None:
                    # Keep custom/injected indexers on the historical contract.
                    self._delete_source_index_records(job.source_hashes)
                    self._raise_if_cancelled(job)
                    self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options, job=job)
                else:
                    incremental_options = dict(job.options or {})
                    incremental_options["source_hashes"] = list(job.source_hashes)
                    self._run_incremental_indexing(
                        str(self.processed_dir),
                        str(self.db_dir),
                        incremental_options,
                        job=job,
                    )
                return

            if job.kind == "backup":
                self._wait_for_no_queries(job, "backing_up")
                self._raise_if_cancelled(job)
                snapshot = _source_module.create_index_backup(self.db_dir)
                self._append_job_log(
                    job,
                    f"Backed up LanceDB index to {snapshot.get('name')} ({snapshot.get('record_count')} records).",
                )
                return

            if job.kind == "rebuild":
                self._wait_for_no_queries(job, "indexing")
                self._raise_if_cancelled(job)
                self._drop_live_index()
                self._raise_if_cancelled(job)
                rebuild_options = dict(job.options or {})
                # Fresh rebuild: never reuse vectors from the (possibly corrupt)
                # live index we just dropped.
                rebuild_options["reuse_db_dir"] = None
                self._run_indexing(str(self.processed_dir), str(self.db_dir), rebuild_options, job=job)
                return

            if job.kind == "restore":
                self._wait_for_no_queries(job, "restoring")
                self._raise_if_cancelled(job)
                if not job.backup_name:
                    raise ValueError("Restore job is missing a backup name.")
                result = _source_module.restore_index_from_backup(self.db_dir, job.backup_name)
                self._append_job_log(
                    job,
                    f"Restored LanceDB index from backup {job.backup_name}.",
                )
                if result.get("safety_backup"):
                    self._append_job_log(
                        job,
                        f"Pre-restore index rolled back to backup {result['safety_backup'].get('name')}.",
                    )
                return

            if job.kind == "rebuild_vector_index":
                self._wait_for_no_queries(job, "rebuilding_index")
                self._raise_if_cancelled(job)
                result = self._rebuild_vector_index()
                if result.get("built"):
                    self._append_job_log(
                        job,
                        f"Rebuilt {result.get('index_type')} ANN vector index "
                        f"({result.get('num_partitions')} partitions, {result.get('rows')} rows).",
                    )
                else:
                    self._append_job_log(
                        job,
                        f"ANN index rebuild skipped: {result.get('reason')} "
                        f"(rows={result.get('rows')}).",
                    )
                return

            if job.kind == "compact":
                self._wait_for_no_queries(job, "compacting")
                self._raise_if_cancelled(job)
                result = self._compact_index()
                if result.get("compacted"):
                    reclaimed = result.get("bytes_reclaimed", 0)
                    self._append_job_log(
                        job,
                        f"Compacted LanceDB index: reclaimed {_human_bytes(reclaimed)} "
                        f"({_human_bytes(result.get('bytes_before', 0))} -> "
                        f"{_human_bytes(result.get('bytes_after', 0))}).",
                    )
                else:
                    self._append_job_log(
                        job,
                        f"Compaction skipped: {result.get('reason')}.",
                    )
                return

        raise ValueError(f"Unknown job kind: {job.kind}")

    def _run_upload_job(self, job: QueueJob) -> None:
        resume_status = job.resume_status or "queued"
        if resume_status == "ingested" and self._job_processed_paths_exist(job):
            # Resume directly into the indexing phase (ingestion already done).
            self._run_upload_indexing_phase(job)
            return

        # --- Ingestion phase (no index lock) ---
        # Saving uploads and parsing PDFs only touches data/uploads/ and
        # processed_docs/. Multiple upload jobs can ingest concurrently since
        # each writes to its own job-scoped upload dir and per-file markdown.
        upload_dir = self._ensure_upload_dir(job)
        self._wait_for_no_queries(job, "ingesting")
        self._raise_if_cancelled(job)
        self.registry.mark_job_status(job_id=job.id, files=job.uploads, status="ingesting")
        self._prepare_for_forced_duplicates(job)
        self._raise_if_cancelled(job)
        self._run_ingestion(str(upload_dir), str(self.processed_dir), job.options, job=job)
        self._raise_if_cancelled(job)
        self._mark_processed_paths(job)
        self.registry.mark_job_status(job_id=job.id, files=job.uploads, status="ingested")

        # --- Indexing phase (serialized with other index-mutating jobs) ---
        self._run_upload_indexing_phase(job)

    def _run_upload_indexing_phase(self, job: QueueJob) -> None:
        """The indexing + publish phase of an upload job.

        Serialized via ``_index_write_lock`` so only one job writes to ``db/``
        at a time. The query-wait happens inside the lock so a second indexing
        job doesn't start waiting while the first is still publishing.
        """
        with self._index_write_lock:
            self._wait_for_no_queries(job, "indexing")
            self._raise_if_cancelled(job)
            if self._run_indexing_func is not None:
                self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options, job=job)
            else:
                incremental_options = dict(job.options or {})
                incremental_options["source_hashes"] = [
                    str(item.get("hash") or "") for item in job.uploads if item.get("hash")
                ]
                self._run_incremental_indexing(
                    str(self.processed_dir),
                    str(self.db_dir),
                    incremental_options,
                    job=job,
                )
            self.registry.mark_job_status(job_id=job.id, files=job.uploads, status="indexed")

    def _wait_for_no_queries(self, job: QueueJob, phase: str) -> None:
        with self._condition:
            self._raise_if_cancelled(job)
            job.phase = phase
            while self.active_query_count > 0:
                self._raise_if_cancelled(job)
                job.status = "paused_for_queries"
                job.phase = phase
                self._condition.notify_all()
                self._condition.wait(timeout=0.2)
            self._raise_if_cancelled(job)
            job.status = "running"
            job.phase = phase
            self._condition.notify_all()

    def _save_staged_uploads(self, job: QueueJob) -> Path:
        self._wait_for_no_queries(job, "saving_uploads")
        if not job.staging_dir:
            raise ValueError("Upload job has no staging directory.")

        staging_dir = Path(job.staging_dir)
        upload_dir = self.upload_root / job.id
        upload_dir.mkdir(parents=True, exist_ok=True)
        # Saving uploads copies the staged PDFs into the upload dir, doubling
        # their footprint. Refuse before copying if the volume is too full.
        required = _source_module.estimate_dir_bytes(staging_dir)
        if required > 0:
            _source_module.check_disk_space(upload_dir, required)
        if not staging_dir.exists():
            self._populate_existing_upload_paths(job, upload_dir)
            self._require_upload_files(job, upload_dir)
            job.upload_dir = str(upload_dir)
            return upload_dir

        for filename in job.filenames:
            source = staging_dir / filename
            destination = upload_dir / source.name
            if source.exists():
                try:
                    same_file = source.resolve() == destination.resolve()
                except OSError:
                    same_file = False
                if not same_file:
                    shutil.copy2(source, destination)
            elif not destination.exists():
                continue
            for item in job.uploads:
                if item.get("filename") == filename:
                    item["upload_path"] = str(destination)
            if source.exists():
                try:
                    source.unlink()
                except PermissionError:
                    pass
        shutil.rmtree(staging_dir, ignore_errors=True)
        job.upload_dir = str(upload_dir)
        self._require_upload_files(job, upload_dir)
        self.registry.mark_job_status(job_id=job.id, files=job.uploads, status="saving_uploads")
        return upload_dir

    def _ensure_upload_dir(self, job: QueueJob) -> Path:
        upload_dir = Path(job.upload_dir) if job.upload_dir else self.upload_root / job.id
        staging_dir = Path(job.staging_dir) if job.staging_dir else None
        if staging_dir and staging_dir.exists():
            return self._save_staged_uploads(job)

        self._populate_existing_upload_paths(job, upload_dir)
        self._require_upload_files(job, upload_dir)
        job.upload_dir = str(upload_dir)
        return upload_dir

    def _populate_existing_upload_paths(self, job: QueueJob, upload_dir: Path) -> None:
        for item in job.uploads:
            raw_upload_path = str(item.get("upload_path") or "")
            upload_path = Path(raw_upload_path) if raw_upload_path else None
            if upload_path is not None and upload_path.exists():
                continue
            filename = str(item.get("filename") or "")
            candidate = upload_dir / filename if filename else None
            if candidate is not None and candidate.exists():
                item["upload_path"] = str(candidate)

    def _require_upload_files(self, job: QueueJob, upload_dir: Path) -> None:
        missing: list[str] = []
        for item in job.uploads:
            filename = str(item.get("filename") or "")
            raw_upload_path = str(item.get("upload_path") or "")
            candidates = [Path(raw_upload_path)] if raw_upload_path else []
            if filename:
                candidates.append(upload_dir / filename)
            existing = next((path for path in candidates if path.exists() and path.suffix.lower() == ".pdf"), None)
            if existing is None:
                missing.append(filename or str(item.get("hash") or "unknown PDF"))
            else:
                item["upload_path"] = str(existing)
        if missing:
            raise FileNotFoundError(f"Uploaded PDF file(s) missing for job {job.id}: {', '.join(missing)}")

    def _processed_path_for_upload(self, item: dict[str, Any]) -> Path:
        raw_path = str(item.get("processed_markdown_path") or "")
        if raw_path:
            path = Path(raw_path)
            return path if path.is_absolute() else self.processed_dir / path.name
        filename = str(item.get("filename") or "")
        return self.processed_dir / f"{Path(filename).stem}.md"

    def _job_processed_paths_exist(self, job: QueueJob) -> bool:
        if not job.uploads:
            return False
        return all(self._processed_path_for_upload(item).exists() for item in job.uploads)

    def _prepare_for_forced_duplicates(self, job: QueueJob) -> None:
        hashes = {value for value in job.force_duplicate_hashes if value}
        if not hashes:
            return
        from src.asset_store import ImageAssetStore

        asset_store = ImageAssetStore(_resolve_root_path(job.options.get("asset_dir") or _source_module.ASSET_DIR))
        for source_hash in hashes:
            asset_store.remove_source_assets(source_hash)
        clear_overrides_for_sources(self.db_dir, hashes)
        removed_entries = remove_source_entries_by_hash(self.processed_dir, hashes)
        legacy_paths: list[str] = []
        legacy_doc_ids: list[str] = []
        for entry in removed_entries:
            for key in ("processed_markdown_path", "source_pdf_path", "source_pdf_name"):
                value = str(entry.get(key, ""))
                if value:
                    legacy_paths.append(value)
            markdown_path = Path(str(entry.get("processed_markdown_path", "")))
            if markdown_path.name:
                from src.sectioning import stable_id

                legacy_doc_ids.append(stable_id("doc", markdown_path.stem))
                self._delete_processed_markdown(markdown_path)

        with INDEX_LOCK:
            store = _index_store(self.db_dir)
            if store.exists():
                store.delete_records_by_source_hash(
                    source_hashes=list(hashes),
                    legacy_file_paths=legacy_paths,
                    legacy_doc_ids=legacy_doc_ids,
                )

    def _delete_processed_markdown(self, markdown_path: Path) -> None:
        processed_root = self.processed_dir.resolve()
        candidate = markdown_path if markdown_path.is_absolute() else self.processed_dir / markdown_path.name
        try:
            resolved = candidate.resolve()
        except OSError:
            return
        try:
            resolved.relative_to(processed_root)
        except ValueError:
            return
        try:
            resolved.unlink(missing_ok=True)
        except PermissionError:
            try:
                resolved.write_text("", encoding="utf-8")
            except OSError:
                pass
        try:
            sidecar = resolved.with_suffix(".pages.json")
            sidecar.relative_to(processed_root)
            sidecar.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass

    def _mark_processed_paths(self, job: QueueJob) -> None:
        for item in job.uploads:
            filename = str(item.get("filename", ""))
            if filename:
                item["processed_markdown_path"] = str(self.processed_dir / f"{Path(filename).stem}.md")

    def _run_pipeline_subprocess(self, command: list[str], *, job: QueueJob | None = None) -> None:
        worker_threads = _source_module._background_worker_threads()
        if job is None:
            _source_module._run_job_subprocess(command, worker_threads=worker_threads)
            return
        self._raise_if_cancelled(job)
        try:
            _source_module._run_job_subprocess(
                command,
                worker_threads=worker_threads,
                cancel_event=job._cancel_event,
                log_callback=lambda line: self._append_job_log(job, line),
            )
        except TypeError as exc:
            if "cancel_event" not in str(exc) and "log_callback" not in str(exc):
                raise
            self._raise_if_cancelled(job)
            _source_module._run_job_subprocess(command, worker_threads=worker_threads)
        self._raise_if_cancelled(job)

    def _run_ingestion(
        self,
        input_dir: str,
        output_dir: str,
        options: dict[str, Any],
        *,
        job: QueueJob | None = None,
    ) -> None:
        run_ingestion_func = self._run_ingestion_func
        resolved = {
            "parser_mode": options.get("parser_mode", INGESTION_CONFIG["parser_mode"]),
            "accelerator": options.get("accelerator", INGESTION_CONFIG["accelerator"]),
            "num_threads": options.get("num_threads", INGESTION_CONFIG["num_threads"]),
            "asset_triggers": options.get("asset_triggers", INGESTION_CONFIG["asset_triggers"]),
            "asset_dir": str(_resolve_root_path(options.get("asset_dir", INGESTION_CONFIG["asset_dir"]))),
            "code_enrichment": options.get("code_enrichment", INGESTION_CONFIG["code_enrichment"]),
            "formula_enrichment": options.get("formula_enrichment", INGESTION_CONFIG["formula_enrichment"]),
            "vision_model": options.get("vision_model", INGESTION_CONFIG["vision_model"]),
            "vision_enabled": options.get("vision_enabled", INGESTION_CONFIG["vision_enabled"]),
            "ocr_backend": options.get("ocr_backend", INGESTION_CONFIG["ocr_backend"]),
            "ocr_langs": options.get("ocr_langs", INGESTION_CONFIG["ocr_langs"]),
            "ocr_force_full_page": options.get("ocr_force_full_page", INGESTION_CONFIG["ocr_force_full_page"]),
            "ocr_bitmap_area_threshold": options.get(
                "ocr_bitmap_area_threshold",
                INGESTION_CONFIG["ocr_bitmap_area_threshold"],
            ),
            "rapidocr_backend": options.get("rapidocr_backend", INGESTION_CONFIG["rapidocr_backend"]),
            "tesseract_cmd": options.get("tesseract_cmd", INGESTION_CONFIG["tesseract_cmd"]),
            "tesseract_data_path": options.get("tesseract_data_path", INGESTION_CONFIG["tesseract_data_path"]),
            "tesseract_psm": options.get("tesseract_psm", INGESTION_CONFIG["tesseract_psm"]),
            "ingestion_workers": options.get(
                "ingestion_workers", INGESTION_CONFIG.get("ingestion_workers", 1)
            ),
            "max_pages_whole_doc": options.get(
                "max_pages_whole_doc", INGESTION_CONFIG.get("max_pages_whole_doc", 50)
            ),
            "progress_enabled": options.get("progress_enabled", False),
        }
        if run_ingestion_func is not None:
            self._raise_if_cancelled(job)
            run_ingestion_func(input_dir, output_dir, **resolved)
            self._raise_if_cancelled(job)
            self._summarize_ingest_result(job, output_dir)
            return

        command = [
            sys.executable,
            str(ROOT_DIR / "main.py"),
            "--mode",
            "ingest",
            "--data_dir",
            input_dir,
            "--md_dir",
            output_dir,
        ]
        _append_cli_option(command, "--parser_mode", resolved["parser_mode"])
        _append_cli_option(command, "--asset_dir", resolved["asset_dir"])
        _append_cli_option(command, "--accelerator", resolved["accelerator"])
        _append_cli_option(command, "--num_threads", resolved["num_threads"])
        _append_cli_option(command, "--asset_triggers", resolved["asset_triggers"])
        _append_cli_option(command, "--code_enrichment", resolved["code_enrichment"])
        _append_cli_option(command, "--formula_enrichment", resolved["formula_enrichment"])
        _append_cli_option(command, "--vision_model", resolved["vision_model"])
        _append_cli_option(command, "--vision_enabled", resolved["vision_enabled"])
        _append_cli_option(command, "--ocr_backend", resolved["ocr_backend"])
        _append_cli_option(command, "--ocr_langs", resolved["ocr_langs"])
        _append_cli_option(command, "--ocr_force_full_page", resolved["ocr_force_full_page"])
        _append_cli_option(command, "--ocr_bitmap_area_threshold", resolved["ocr_bitmap_area_threshold"])
        _append_cli_option(command, "--rapidocr_backend", resolved["rapidocr_backend"])
        _append_cli_option(command, "--tesseract_cmd", resolved["tesseract_cmd"])
        _append_cli_option(command, "--tesseract_data_path", resolved["tesseract_data_path"])
        _append_cli_option(command, "--tesseract_psm", resolved["tesseract_psm"])
        _append_cli_option(command, "--ingestion_workers", resolved["ingestion_workers"])
        _append_cli_option(command, "--max_pages_whole_doc", resolved["max_pages_whole_doc"])
        command.append("--no_progress")
        self._run_pipeline_subprocess(command, job=job)
        self._summarize_ingest_result(job, output_dir)

    def _summarize_ingest_result(self, job: QueueJob | None, output_dir: str) -> None:
        """Attach the per-file ingest breakdown to the job log.

        ``run_ingestion`` writes ``.ingest_result.json`` with processed/skipped/
        failed lists. The subprocess boundary means the registry can't see per-
        file outcomes, so we surface them here in the job log the API returns.
        """
        if job is None:
            return
        from src.ingestion import INGEST_RESULT_FILENAME

        result_path = Path(output_dir) / INGEST_RESULT_FILENAME
        if not result_path.exists():
            return
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        processed = payload.get("processed") or []
        skipped = payload.get("skipped") or []
        failed = payload.get("failed") or []
        if not isinstance(processed, list):
            processed = []
        if not isinstance(skipped, list):
            skipped = []
        if not isinstance(failed, list):
            failed = []
        self._append_job_log(
            job,
            f"Ingest summary: {len(processed)} processed, {len(skipped)} skipped, "
            f"{len(failed)} failed.",
        )
        for entry in failed:
            if isinstance(entry, dict):
                self._append_job_log(
                    job,
                    f"  failed: {entry.get('file', '?')} - {entry.get('error', '')}",
                )

    def _summarize_index_result(self, job: QueueJob | None, staged_db_dir: str) -> None:
        """Attach the index run-summary to the job log.

        The indexer writes ``.index_result.json`` into the staged dir with
        processed/failed counts and disk usage. Read it before the staged dir
        is removed so a partial-failure indexing run still reports which files
        failed in the job log the API returns.
        """
        if job is None:
            return
        result_path = Path(staged_db_dir) / ".index_result.json"
        if not result_path.exists():
            return
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        processed = int(payload.get("files_processed") or 0)
        failed_entries = payload.get("errors") or []
        if not isinstance(failed_entries, list):
            failed_entries = []
        disk_used = payload.get("disk_used_bytes")
        self._append_job_log(
            job,
            f"Index summary: {processed} file(s) indexed, {len(failed_entries)} failed"
            + (f", {disk_used} bytes on disk." if disk_used is not None else "."),
        )
        for entry in failed_entries:
            if isinstance(entry, dict):
                self._append_job_log(
                    job,
                    f"  failed: {entry.get('file', '?')} - {entry.get('error', '')}",
                )

    def _run_indexing(
        self,
        md_dir: str,
        db_dir: str,
        options: dict[str, Any],
        *,
        job: QueueJob | None = None,
    ) -> None:
        run_indexing_func = self._run_indexing_func
        resolved = {
            "progress_enabled": options.get("progress_enabled", False),
            "embedding_model": options.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            "embedding_batch_size": options.get("embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE),
            "embedding_timeout": options.get("embedding_timeout", DEFAULT_EMBEDDING_TIMEOUT),
            "index_backend": options.get("index_backend", DEFAULT_INDEX_BACKEND),
            "summary_mode": options.get("summary_mode", DEFAULT_SUMMARY_MODE),
            "chunk_target_tokens": options.get("chunk_target_tokens", DEFAULT_CHUNK_TARGET_TOKENS),
            "chunk_overlap_tokens": options.get("chunk_overlap_tokens", DEFAULT_CHUNK_OVERLAP_TOKENS),
            "source_hashes": options.get("source_hashes") or [],
        }
        if run_indexing_func is not None:
            self._raise_if_cancelled(job)
            run_indexing_func(md_dir, db_dir, **resolved)
            self._raise_if_cancelled(job)
            return

        staged_db_dir = _source_module._staged_index_dir(db_dir)
        # The staged build writes a second full copy of the index alongside the
        # live one, and the publish step then creates a third copy (a safety
        # backup of the live index before swapping). Budget for the full
        # footprint: staged index (~existing index size, since it reuses vectors
        # and rewrites all records) + the Markdown corpus it rebuilds from + a
        # publish-time backup of the live index, plus a safety factor. See
        # disk_space.py for why the tripling matters at scale.
        existing_index = _source_module.estimate_dir_bytes(_source_module.lancedb_path(db_dir))
        corpus_bytes = _source_module.estimate_dir_bytes(md_dir)
        # The staged index approximates the live one; the publish safety backup
        # is another full copy. So peak transient cost is ~2× existing index +
        # corpus. Add a configurable safety factor for headroom.
        safety_factor = _source_module.SERVER_CONFIG.get("disk_safety_factor", 1.15)
        try:
            safety_factor = float(safety_factor)
        except (TypeError, ValueError):
            safety_factor = 1.15
        safety_factor = max(1.0, safety_factor)
        required_bytes = int((existing_index * 2 + corpus_bytes) * safety_factor)
        try:
            _source_module.check_disk_space(db_dir, required_bytes)
        except Exception as exc:
            raise RuntimeError(
                f"Not enough free disk space to stage the index build. "
                f"Estimated peak: staged index (~{existing_index/1e6:.1f}MB) + "
                f"publish backup (~{existing_index/1e6:.1f}MB) + "
                f"corpus ({corpus_bytes/1e6:.1f}MB) "
                f"×{safety_factor:g} ≈ {required_bytes/1e9:.2f}GB. {exc}"
            ) from exc
        command = [
            sys.executable,
            str(ROOT_DIR / "main.py"),
            "--mode",
            "index",
            "--md_dir",
            md_dir,
            "--db_dir",
            str(staged_db_dir),
            "--reuse_db_dir",
            db_dir,
        ]
        _append_cli_option(command, "--embedding_model", resolved["embedding_model"])
        _append_cli_option(command, "--embedding_batch_size", resolved["embedding_batch_size"])
        _append_cli_option(command, "--embedding_timeout", resolved["embedding_timeout"])
        _append_cli_option(command, "--index_backend", resolved["index_backend"])
        _append_cli_option(command, "--summary_mode", resolved["summary_mode"])
        _append_cli_option(command, "--chunk_target_tokens", resolved["chunk_target_tokens"])
        _append_cli_option(command, "--chunk_overlap_tokens", resolved["chunk_overlap_tokens"])
        _append_cli_option(command, "--source_hashes", resolved["source_hashes"])
        command.append("--no_progress")

        try:
            self._run_pipeline_subprocess(command, job=job)
            # Read the staged index run-summary before publish deletes the dir,
            # so per-file failures and disk usage can be surfaced in the job log.
            self._summarize_index_result(job, staged_db_dir)
            if job is not None:
                self._wait_for_no_queries(job, "publishing_index")
            with INDEX_LOCK:
                _source_module._publish_staged_index(staged_db_dir, db_dir)
        finally:
            shutil.rmtree(staged_db_dir, ignore_errors=True)

    def _run_incremental_indexing(
        self,
        md_dir: str,
        db_dir: str,
        options: dict[str, Any],
        *,
        job: QueueJob | None = None,
    ) -> None:
        """Replace selected source rows directly in the live index."""
        hashes = [str(value) for value in options.get("source_hashes") or [] if value]
        if not hashes:
            return
        from src.indexing import run_indexing

        self._raise_if_cancelled(job)
        with _source_module.acquire_index_lock(db_dir):
            run_indexing(
                md_dir,
                db_dir,
                progress_enabled=bool(options.get("progress_enabled", False)),
                embedding_model=options.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
                embedding_batch_size=options.get("embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE),
                embedding_timeout=options.get("embedding_timeout", DEFAULT_EMBEDDING_TIMEOUT),
                index_backend=options.get("index_backend", DEFAULT_INDEX_BACKEND),
                summary_mode=options.get("summary_mode", DEFAULT_SUMMARY_MODE),
                chunk_target_tokens=options.get("chunk_target_tokens", DEFAULT_CHUNK_TARGET_TOKENS),
                chunk_overlap_tokens=options.get("chunk_overlap_tokens", DEFAULT_CHUNK_OVERLAP_TOKENS),
                source_hashes=hashes,
            )
        self._raise_if_cancelled(job)

    def _delete_source_index_records(self, source_hashes: list[str]) -> None:
        """Drop existing vectors for the given sources so re-indexing rebuilds them.

        Markdown and source-map entries are left intact; only the vector store
        records are removed. Re-indexing rebuilds a fresh staged index from all
        processed Markdown, reusing every other source's vectors.
        """
        hashes = {str(value) for value in source_hashes or () if value}
        if not hashes:
            return
        with INDEX_LOCK:
            store = _index_store(self.db_dir)
            if store.exists():
                store.delete_records_by_source_hash(source_hashes=sorted(hashes))

    def _drop_live_index(self) -> None:
        """Remove the live LanceDB directory so a rebuild starts from scratch.

        Used when the index is corrupted: the staged rebuild that follows cannot
        reuse vectors that are not there, so the next ``_run_indexing`` builds a
        clean table from the processed Markdown in ``processed_dir``.
        """
        with INDEX_LOCK:
            with _source_module.acquire_index_lock(self.db_dir):
                live_lancedb = lancedb_path(self.db_dir)
                if live_lancedb.exists():
                    _source_module._remove_path(live_lancedb)
                store = _index_store(self.db_dir)
                store._invalidate_table()

    def _rebuild_vector_index(self) -> dict[str, Any]:
        """Re-train the ANN vector index on the live DB in place.

        Drops any existing ANN index and rebuilds it from the current vectors
        (no re-embedding). Done under the cross-process index lock so concurrent
        queries/edits don't race. Returns the ``create_vector_index`` diagnostics.
        """
        with INDEX_LOCK:
            with _source_module.acquire_index_lock(self.db_dir):
                store = _index_store(self.db_dir)
                if not store.exists():
                    return {"built": False, "reason": "table_missing"}
                # Drop the old index first so the rebuild trains fresh partitions
                # on the current data distribution (incremental reindex_source
                # jobs can leave the old IVF partitions suboptimal).
                store.drop_vector_index()
                result = store.create_vector_index()
                _source_module._invalidate_index_caches(self.db_dir)
                return result

    def _compact_index(self) -> dict[str, Any]:
        """Compact the live LanceDB table in place.

        Re-indexing uses logical delete (deletion vectors) + add, so repeated
        reindex_source cycles leave dead fragments that grow the on-disk table
        and slow scans. LanceDB's optimize() merges fragments and drops the
        tombstones. Done under the cross-process index lock so concurrent
        queries/edits don't race.
        """
        with INDEX_LOCK:
            with _source_module.acquire_index_lock(self.db_dir):
                store = _index_store(self.db_dir)
                result = store.compact()
                _source_module._invalidate_index_caches(self.db_dir)
                return result

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._condition:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._condition:
            return [job.to_dict() for job in reversed(list(self._jobs.values()))]

    def recover_pending_uploads(self, *, auto_start: bool = True) -> dict[str, Any]:
        jobs = self._recovery_jobs_from_registry()
        recovered: list[QueueJob] = []
        with self._condition:
            for job in jobs:
                if job.id in self._jobs:
                    continue
                self._jobs[job.id] = job
                self._queue.append(job.id)
                recovered.append(job)
            # Re-enqueue non-upload jobs (reindex/rebuild/backup/restore) that
            # were interrupted by a crash and persisted in the durable ledger.
            # These are not in the PDF registry, so without the ledger they'd be
            # silently lost on every restart.
            for job in self._recovery_jobs_from_ledger():
                if job.id in self._jobs:
                    continue
                self._jobs[job.id] = job
                self._queue.append(job.id)
                recovered.append(job)
                # Re-record so a second crash before completion still recovers
                # the job. Done under the condition lock is fine (best-effort).
                self._ledger_record(job)
            if auto_start and recovered:
                self._ensure_worker_locked()
            if recovered:
                self._condition.notify_all()
        return {
            "recovered": len(recovered),
            "jobs": [job.to_dict() for job in recovered],
        }

    def _recovery_jobs_from_ledger(self) -> list[QueueJob]:
        """Reconstruct tracked jobs left in the ledger by a crashed run.

        Each ledger entry carries the kind plus the options needed to rebuild
        the job. Recovered jobs are marked ``recovered`` so the UI can surface
        that they were resumed rather than freshly enqueued. We re-record them
        on enqueue (so a second crash still recovers them) and they're removed
        when they next reach a terminal state.
        """
        jobs: list[QueueJob] = []
        try:
            entries = self.ledger.pending_entries()
        except Exception:
            return jobs
        for job_id, entry in entries:
            kind = str(entry.get("kind") or "")
            options = dict(entry.get("options") or {}) if isinstance(entry.get("options"), dict) else {}
            source_hashes = [
                str(value)
                for value in (entry.get("source_hashes") or [])
                if value
            ]
            backup_name = entry.get("backup_name")
            jobs.append(
                QueueJob(
                    id=str(job_id),
                    kind=kind,
                    phase="recovered",
                    source_hashes=source_hashes,
                    backup_name=str(backup_name) if backup_name else None,
                    options=options,
                    recovered=True,
                    created_at=str(entry.get("recorded_at") or _utcnow()),
                )
            )
        return jobs

    def _recovery_jobs_from_registry(self) -> list[QueueJob]:
        payload = self.registry.load()
        groups: dict[str, dict[str, Any]] = {}
        for source_hash, entry in payload.get("pdfs", {}).items():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status") or "")
            if status not in RECOVERABLE_UPLOAD_STATUSES:
                continue
            job_id = str(entry.get("job_id") or f"recovered-{source_hash}")
            group = groups.setdefault(
                job_id,
                {
                    "job_id": job_id,
                    "entries": [],
                    "resume_status": status,
                    "created_at": str(entry.get("created_at") or _utcnow()),
                    "options": dict(entry.get("options") or {}) if isinstance(entry.get("options"), dict) else {},
                },
            )
            group["entries"].append((str(source_hash), entry))
            if UPLOAD_RESUME_STATUS_ORDER[status] < UPLOAD_RESUME_STATUS_ORDER[str(group["resume_status"])]:
                group["resume_status"] = status
            if not group["options"] and isinstance(entry.get("options"), dict):
                group["options"] = dict(entry["options"])

        jobs: list[QueueJob] = []
        for group in sorted(groups.values(), key=lambda item: str(item.get("created_at") or "")):
            uploads: list[dict[str, Any]] = []
            filenames: list[str] = []
            staging_dirs: list[Path] = []
            force_duplicate_hashes: list[str] = []
            upload_dir = ""
            for source_hash, entry in group["entries"]:
                filename = str(entry.get("filename") or Path(str(entry.get("upload_path") or "")).name)
                staging_path = str(entry.get("staging_path") or "")
                if staging_path:
                    staging_dirs.append(Path(staging_path).parent)
                upload_path = str(entry.get("upload_path") or "")
                if upload_path:
                    upload_dir = upload_dir or str(Path(upload_path).parent)
                if isinstance(entry.get("previous_entry"), dict):
                    force_duplicate_hashes.append(source_hash)
                filenames.append(filename)
                uploads.append(
                    {
                        "filename": filename,
                        "hash": source_hash,
                        "staging_path": staging_path,
                        "upload_path": upload_path,
                        "processed_markdown_path": str(entry.get("processed_markdown_path") or ""),
                    }
                )
            staging_dir = staging_dirs[0] if staging_dirs else None
            jobs.append(
                QueueJob(
                    id=str(group["job_id"]),
                    kind="upload",
                    phase="recovered",
                    filenames=filenames,
                    uploads=uploads,
                    force_duplicate_hashes=sorted(set(force_duplicate_hashes)),
                    staging_dir=str(staging_dir) if staging_dir is not None else None,
                    upload_dir=upload_dir or None,
                    resume_status=str(group["resume_status"]),
                    recovered=True,
                    options=dict(group["options"]),
                    created_at=str(group["created_at"]),
                )
            )
        return jobs

    def summary(self) -> dict[str, Any]:
        with self._condition:
            running = [
                job.id
                for job in self._jobs.values()
                if job.status in {"running", "paused_for_queries"}
            ]
            indexing = [
                job.id
                for job in self._jobs.values()
                if job.status in {"running", "paused_for_queries"}
                and job.phase in {"indexing", "publishing_index"}
            ]
            return {
                "active_query_count": self.active_query_count,
                "queued_count": len(self._queue),
                "running_job_ids": running,
                "indexing_job_ids": indexing,
                "active_job_count": len(running) + len(self._queue),
                "job_count": len(self._jobs),
            }

RagJobQueue.__module__ = _source_module.__name__
finalize_split_class(_source_module, RagJobQueue)

