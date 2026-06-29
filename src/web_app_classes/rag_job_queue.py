from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class RagJobQueue:
    def __init__(
        self,
        *,
        upload_root: Path = UPLOAD_DIR,
        processed_dir: Path = PROCESSED_DIR,
        db_dir: Path = DB_DIR,
        registry_path: Path = PDF_REGISTRY_PATH,
        run_ingestion_func=None,
        run_indexing_func=None,
    ):
        self.upload_root = Path(upload_root)
        self.processed_dir = Path(processed_dir)
        self.db_dir = Path(db_dir)
        self.registry = PdfRegistry(registry_path)
        self._run_ingestion_func = run_ingestion_func
        self._run_indexing_func = run_indexing_func
        self._condition = threading.Condition(threading.RLock())
        self._jobs: dict[str, QueueJob] = {}
        self._queue: deque[str] = deque()
        self._worker: threading.Thread | None = None
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

    def _enqueue(self, job: QueueJob, *, auto_start: bool) -> QueueJob:
        with self._condition:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            if auto_start:
                self._ensure_worker_locked()
            self._condition.notify_all()
            return job

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                if not self._queue:
                    self._worker = None
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

    def _run_job(self, job: QueueJob) -> None:
        if job.kind == "upload":
            self._raise_if_cancelled(job)
            self._run_upload_job(job)
            return

        if job.kind == "reindex":
            self._wait_for_no_queries(job, "indexing")
            self._raise_if_cancelled(job)
            self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options, job=job)
            return

        if job.kind == "reindex_source":
            self._wait_for_no_queries(job, "indexing")
            self._raise_if_cancelled(job)
            self._delete_source_index_records(job.source_hashes)
            self._raise_if_cancelled(job)
            self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options, job=job)
            return

        raise ValueError(f"Unknown job kind: {job.kind}")

    def _run_upload_job(self, job: QueueJob) -> None:
        resume_status = job.resume_status or "queued"
        if resume_status == "ingested" and self._job_processed_paths_exist(job):
            self._wait_for_no_queries(job, "indexing")
            self._raise_if_cancelled(job)
            self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options, job=job)
            self.registry.mark_job_status(job_id=job.id, files=job.uploads, status="indexed")
            return

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
        self._wait_for_no_queries(job, "indexing")
        self._raise_if_cancelled(job)
        self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options, job=job)
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
            _source_module._run_job_subprocess(command, worker_threads=worker_threads, cancel_event=job._cancel_event)
        except TypeError as exc:
            if "cancel_event" not in str(exc):
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
            "progress_enabled": options.get("progress_enabled", False),
        }
        if run_ingestion_func is not None:
            self._raise_if_cancelled(job)
            run_ingestion_func(input_dir, output_dir, **resolved)
            self._raise_if_cancelled(job)
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
        command.append("--no_progress")
        self._run_pipeline_subprocess(command, job=job)

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
        }
        if run_indexing_func is not None:
            self._raise_if_cancelled(job)
            run_indexing_func(md_dir, db_dir, **resolved)
            self._raise_if_cancelled(job)
            return

        staged_db_dir = _source_module._staged_index_dir(db_dir)
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
        command.append("--no_progress")

        try:
            self._run_pipeline_subprocess(command, job=job)
            if job is not None:
                self._wait_for_no_queries(job, "publishing_index")
            with INDEX_LOCK:
                _source_module._publish_staged_index(staged_db_dir, db_dir)
        finally:
            shutil.rmtree(staged_db_dir, ignore_errors=True)

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
            if auto_start and recovered:
                self._ensure_worker_locked()
            if recovered:
                self._condition.notify_all()
        return {
            "recovered": len(recovered),
            "jobs": [job.to_dict() for job in recovered],
        }

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

