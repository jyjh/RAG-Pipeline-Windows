from __future__ import annotations

import html
import hashlib
import json
import re
import shutil
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

from src.defaults import DEFAULT_LLM_MODEL
from src.local_rag import (
    DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL,
    DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS,
    DEFAULT_QUERY_SYSTEM_PROMPT,
)
from src.pdf_registry import PdfRegistry, load_source_map, remove_source_entries_by_hash
from src.vector_store import default_store, lancedb_path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STAGING_DIR = DATA_DIR / ".upload_queue"
PDF_REGISTRY_PATH = DATA_DIR / ".pdf_upload_registry.json"
PROCESSED_DIR = ROOT_DIR / "processed_docs"
DB_DIR = ROOT_DIR / "db"
WEB_DIR = ROOT_DIR / "web"

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_EMBEDDING_TIMEOUT = 30.0
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_K = 40
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_LLM_NUM_PREDICT = 4096
DEFAULT_LLM_TIMEOUT = 120.0
DEFAULT_RETRIEVAL_CANDIDATE_K = 80
DEFAULT_RETRIEVAL_MIN_SCORE = 0.50
DEFAULT_RETRIEVAL_RELATIVE_CUTOFF = 0.72
DEFAULT_CONTEXT_TOKEN_FRACTION = 0.60
DEFAULT_WEB_SEARCH_ENABLED = True
DEFAULT_WEB_SEARCH_TIMEOUT = 8.0
DEFAULT_WEB_SEARCH_MAX_RESULTS = 5
DEFAULT_SYSTEM_PROMPT = DEFAULT_QUERY_SYSTEM_PROMPT
DEFAULT_OLLAMA_CHAT_HEALTH_CHECK_INTERVAL = DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL
DEFAULT_OLLAMA_CHAT_MAX_LOST_HEALTH_CHECKS = DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS
DEFAULT_INDEX_BACKEND = "lancedb"
DEFAULT_SUMMARY_MODE = "hybrid"
DEFAULT_CHUNK_TARGET_TOKENS = 900
DEFAULT_CHUNK_OVERLAP_TOKENS = 120
DEFAULT_HEALTH_POLL_INTERVAL_MS = 60_000
DEFAULT_JOBS_POLL_INTERVAL_MS = 60_000

INDEX_LOCK = threading.RLock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return name or f"upload-{uuid.uuid4().hex}.pdf"


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _load_toml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        import tomllib

        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_server_config(config_path: Path | None = None) -> dict[str, int]:
    config_path = config_path or (ROOT_DIR / "config.toml")
    payload = _load_toml_config(config_path)
    server_config = payload.get("server", {}) if isinstance(payload.get("server"), dict) else {}

    return {
        "health_poll_interval_ms": _positive_int(
            server_config.get("health_poll_interval_ms"),
            DEFAULT_HEALTH_POLL_INTERVAL_MS,
        ),
        "jobs_poll_interval_ms": _positive_int(
            server_config.get("jobs_poll_interval_ms"),
            DEFAULT_JOBS_POLL_INTERVAL_MS,
        ),
    }


def _load_chat_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or (ROOT_DIR / "config.toml")
    payload = _load_toml_config(config_path)
    chat_config = payload.get("chat", {}) if isinstance(payload.get("chat"), dict) else {}
    retrieval_config = payload.get("retrieval", {}) if isinstance(payload.get("retrieval"), dict) else {}
    ollama_config = payload.get("ollama", {}) if isinstance(payload.get("ollama"), dict) else {}

    return {
        "system_prompt": str(chat_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT),
        "context_window": _positive_int(
            chat_config.get("context_window"),
            DEFAULT_CONTEXT_WINDOW,
        ),
        "llm_num_predict": _positive_int(
            chat_config.get("llm_num_predict"),
            DEFAULT_LLM_NUM_PREDICT,
        ),
        "retrieval_min_score": _positive_float(
            retrieval_config.get("min_relevance_score"),
            DEFAULT_RETRIEVAL_MIN_SCORE,
        ),
        "ollama_health_check_interval": _positive_float(
            ollama_config.get("chat_health_check_interval_seconds"),
            DEFAULT_OLLAMA_CHAT_HEALTH_CHECK_INTERVAL,
        ),
        "ollama_max_lost_health_checks": _positive_int(
            ollama_config.get("chat_max_lost_health_checks"),
            DEFAULT_OLLAMA_CHAT_MAX_LOST_HEALTH_CHECKS,
        ),
    }


SERVER_CONFIG = _load_server_config()
CHAT_CONFIG = _load_chat_config()


def _index_store(db_dir: Path | None = None):
    return default_store(db_dir or DB_DIR)


def list_index_rows(
    *,
    offset: int = 0,
    limit: int = 50,
    search: str = "",
    db_dir: Path | None = None,
) -> dict[str, Any]:
    offset = max(0, int(offset))
    limit = min(max(1, int(limit)), 200)
    with INDEX_LOCK:
        return _index_store(db_dir).list_records(offset=offset, limit=limit, search=search)


def update_index_record(
    *,
    record_id: str,
    content: str,
    embedding_model: str | None = None,
    embedding_batch_size: int | None = None,
    embedding_timeout: float | None = None,
    db_dir: Path | None = None,
) -> dict[str, Any]:
    if not content.strip():
        raise ValueError("Record content cannot be empty.")

    with INDEX_LOCK:
        store = _index_store(db_dir)
        record = store.get_record(record_id)
        model = embedding_model or record.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
        embedding_dim = int(record.get("embedding_dim") or len(record.get("vector") or []) or 768)

        from src.embeddings import EmbeddingEngine

        engine = EmbeddingEngine(
            model_name=model,
            ollama_batch_size=embedding_batch_size,
            ollama_timeout=embedding_timeout,
        )
        vector = engine.get_mrl_embeddings(
            [content],
            truncate_dim=embedding_dim,
            prefix="search_document: ",
        )[0]

        return store.update_record(
            record_id=record_id,
            content=content,
            vector=vector.tolist(),
            embedding_model=model,
            embedding_dim=embedding_dim,
        )


def delete_index_records(*, record_ids: list[str], db_dir: Path | None = None) -> dict[str, Any]:
    ids = {record_id for record_id in record_ids if record_id}
    if not ids:
        raise ValueError("At least one record ID is required.")

    with INDEX_LOCK:
        return _index_store(db_dir).delete_records(record_ids=list(ids))


def _resolve_pdf_path(raw_path: str, *, root_dir: Path = ROOT_DIR, data_dir: Path = DATA_DIR) -> Path:
    if not raw_path:
        raise FileNotFoundError("PDF path is missing.")
    path = Path(raw_path)
    candidate = path if path.is_absolute() else root_dir / path
    resolved = candidate.resolve()
    data_root = data_dir.resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError as exc:
        raise PermissionError(f"PDF path is outside the data directory: {resolved}") from exc
    if resolved.suffix.lower() != ".pdf":
        raise PermissionError(f"Download path is not a PDF: {resolved}")
    return resolved


def _pdf_download_url(source_hash: str) -> str:
    return f"/api/pdfs/{source_hash}/download" if source_hash else ""


def _pdf_entry_for_response(
    *,
    source_hash: str,
    filename: str,
    status: str = "",
    upload_path: str = "",
    source_pdf_path: str = "",
    processed_markdown_path: str = "",
    updated_at: str = "",
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> dict[str, Any]:
    raw_path = upload_path or source_pdf_path
    can_download = False
    path_error = ""
    if raw_path:
        try:
            can_download = _resolve_pdf_path(raw_path, root_dir=root_dir, data_dir=data_dir).exists()
        except FileNotFoundError:
            path_error = "missing_path"
        except PermissionError:
            path_error = "unsafe_path"
        except OSError:
            path_error = "invalid_path"
    return {
        "hash": source_hash,
        "filename": filename or Path(raw_path).name,
        "status": status,
        "upload_path": upload_path,
        "source_pdf_path": source_pdf_path,
        "processed_markdown_path": processed_markdown_path,
        "updated_at": updated_at,
        "can_download": can_download,
        "download_url": _pdf_download_url(source_hash) if can_download else "",
        "path_error": path_error,
    }


def list_pdf_documents(
    *,
    search: str = "",
    registry_path: Path | None = None,
    processed_dir: Path | None = None,
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> dict[str, Any]:
    registry_path = registry_path or PDF_REGISTRY_PATH
    processed_dir = processed_dir or PROCESSED_DIR
    entries: dict[str, dict[str, Any]] = {}

    payload = PdfRegistry(registry_path).load()
    for source_hash, entry in payload.get("pdfs", {}).items():
        if not isinstance(entry, dict):
            continue
        entries[str(source_hash)] = _pdf_entry_for_response(
            source_hash=str(source_hash),
            filename=str(entry.get("filename", "")),
            status=str(entry.get("status", "")),
            upload_path=str(entry.get("upload_path", "")),
            processed_markdown_path=str(entry.get("processed_markdown_path", "")),
            updated_at=str(entry.get("updated_at", "")),
            root_dir=root_dir,
            data_dir=data_dir,
        )

    source_map = load_source_map(processed_dir)
    for entry in source_map.get("documents", {}).values():
        if not isinstance(entry, dict):
            continue
        source_hash = str(entry.get("source_hash", ""))
        if not source_hash:
            continue
        current = entries.get(source_hash, {})
        entries[source_hash] = _pdf_entry_for_response(
            source_hash=source_hash,
            filename=str(current.get("filename") or entry.get("source_pdf_name", "")),
            status=str(current.get("status") or "indexed"),
            upload_path=str(current.get("upload_path") or ""),
            source_pdf_path=str(entry.get("source_pdf_path", "")),
            processed_markdown_path=str(entry.get("processed_markdown_path", "")),
            updated_at=str(entry.get("updated_at", current.get("updated_at", ""))),
            root_dir=root_dir,
            data_dir=data_dir,
        )

    rows = sorted(entries.values(), key=lambda item: (item.get("filename", ""), item.get("hash", "")))
    query = search.strip().lower()
    if query:
        rows = [
            item
            for item in rows
            if query in str(item.get("filename", "")).lower()
            or query in str(item.get("hash", "")).lower()
        ]
    return {"pdfs": rows, "total": len(rows)}


def resolve_pdf_download_path(
    source_hash: str,
    *,
    registry_path: Path | None = None,
    processed_dir: Path | None = None,
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> tuple[Path, str]:
    documents = list_pdf_documents(
        search="",
        registry_path=registry_path,
        processed_dir=processed_dir,
        root_dir=root_dir,
        data_dir=data_dir,
    )["pdfs"]
    match = next((item for item in documents if item.get("hash") == source_hash), None)
    if match is None:
        raise FileNotFoundError(f"PDF not found for source hash: {source_hash}")
    raw_path = str(match.get("upload_path") or match.get("source_pdf_path") or "")
    path = _resolve_pdf_path(raw_path, root_dir=root_dir, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"PDF file is missing for source hash: {source_hash}")
    return path, str(match.get("filename") or path.name)


@dataclass
class QueueJob:
    id: str
    kind: str
    status: str = "queued"
    phase: str = "queued"
    filenames: list[str] = field(default_factory=list)
    uploads: list[dict[str, Any]] = field(default_factory=list)
    force_duplicate_hashes: list[str] = field(default_factory=list)
    staging_dir: str | None = None
    upload_dir: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(default_factory=_utcnow)
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "phase": self.phase,
            "filenames": list(self.filenames),
            "uploads": [dict(item) for item in self.uploads],
            "force_duplicate_hashes": list(self.force_duplicate_hashes),
            "staging_dir": self.staging_dir,
            "upload_dir": self.upload_dir,
            "options": dict(self.options),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


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

    def enqueue_upload(
        self,
        *,
        staging_dir: Path,
        filenames: list[str],
        uploads: list[dict[str, Any]] | None = None,
        force_duplicate_hashes: list[str] | None = None,
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="upload",
            filenames=filenames,
            uploads=uploads or [],
            force_duplicate_hashes=force_duplicate_hashes or [],
            staging_dir=str(staging_dir),
            options=options or {},
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
                job.status = "running"
                job.phase = "starting"
                job.started_at = job.started_at or _utcnow()

            try:
                self._run_job(job)
            except Exception as exc:
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
                    job.status = "done"
                    job.phase = "done"
                    job.finished_at = _utcnow()
                    self._condition.notify_all()

    def _run_job(self, job: QueueJob) -> None:
        if job.kind == "upload":
            upload_dir = self._save_staged_uploads(job)
            self._wait_for_no_queries(job, "ingesting")
            self.registry.mark_job_status(job_id=job.id, files=job.uploads, status="ingesting")
            self._prepare_for_forced_duplicates(job)
            self._run_ingestion(str(upload_dir), str(self.processed_dir), job.options)
            self._mark_processed_paths(job)
            self.registry.mark_job_status(job_id=job.id, files=job.uploads, status="ingested")
            self._wait_for_no_queries(job, "indexing")
            self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options)
            self.registry.mark_job_status(job_id=job.id, files=job.uploads, status="indexed")
            return

        if job.kind == "reindex":
            self._wait_for_no_queries(job, "indexing")
            self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options)
            return

        raise ValueError(f"Unknown job kind: {job.kind}")

    def _wait_for_no_queries(self, job: QueueJob, phase: str) -> None:
        with self._condition:
            job.phase = phase
            while self.active_query_count > 0:
                job.status = "paused_for_queries"
                job.phase = phase
                self._condition.wait(timeout=0.2)
            job.status = "running"
            job.phase = phase

    def _save_staged_uploads(self, job: QueueJob) -> Path:
        self._wait_for_no_queries(job, "saving_uploads")
        if not job.staging_dir:
            raise ValueError("Upload job has no staging directory.")

        staging_dir = Path(job.staging_dir)
        if not staging_dir.exists():
            raise FileNotFoundError(f"Upload staging directory not found: {staging_dir}")

        upload_dir = self.upload_root / job.id
        upload_dir.mkdir(parents=True, exist_ok=True)
        for filename in job.filenames:
            source = staging_dir / filename
            if not source.exists():
                continue
            destination = upload_dir / source.name
            destination.write_bytes(source.read_bytes())
            for item in job.uploads:
                if item.get("filename") == filename:
                    item["upload_path"] = str(destination)
            try:
                source.unlink()
            except PermissionError:
                pass
        shutil.rmtree(staging_dir, ignore_errors=True)
        job.upload_dir = str(upload_dir)
        self.registry.mark_job_status(job_id=job.id, files=job.uploads, status="saving_uploads")
        return upload_dir

    def _prepare_for_forced_duplicates(self, job: QueueJob) -> None:
        hashes = {value for value in job.force_duplicate_hashes if value}
        if not hashes:
            return
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

    def _run_ingestion(self, input_dir: str, output_dir: str, options: dict[str, Any]) -> None:
        run_ingestion_func = self._run_ingestion_func
        if run_ingestion_func is None:
            from src.ingestion import run_ingestion as run_ingestion_func

        run_ingestion_func(
            input_dir,
            output_dir,
            parser_mode=options.get("parser_mode", "hybrid"),
            accelerator=options.get("accelerator", "auto"),
            asset_triggers=options.get("asset_triggers", "none"),
            progress_enabled=options.get("progress_enabled", False),
        )

    def _run_indexing(self, md_dir: str, db_dir: str, options: dict[str, Any]) -> None:
        run_indexing_func = self._run_indexing_func
        if run_indexing_func is None:
            from src.indexing import run_indexing as run_indexing_func

        run_indexing_func(
            md_dir,
            db_dir,
            progress_enabled=options.get("progress_enabled", False),
            embedding_model=options.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            embedding_batch_size=options.get("embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE),
            embedding_timeout=options.get("embedding_timeout", DEFAULT_EMBEDDING_TIMEOUT),
            index_backend=options.get("index_backend", DEFAULT_INDEX_BACKEND),
            summary_mode=options.get("summary_mode", DEFAULT_SUMMARY_MODE),
            chunk_target_tokens=options.get("chunk_target_tokens", DEFAULT_CHUNK_TARGET_TOKENS),
            chunk_overlap_tokens=options.get("chunk_overlap_tokens", DEFAULT_CHUNK_OVERLAP_TOKENS),
        )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._condition:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._condition:
            return [job.to_dict() for job in reversed(list(self._jobs.values()))]

    def summary(self) -> dict[str, Any]:
        with self._condition:
            running = [
                job.id
                for job in self._jobs.values()
                if job.status in {"running", "paused_for_queries"}
            ]
            return {
                "active_query_count": self.active_query_count,
                "queued_count": len(self._queue),
                "running_job_ids": running,
                "job_count": len(self._jobs),
            }


class ChatRequest(BaseModel):
    question: str
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int | None = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_timeout: float | None = DEFAULT_EMBEDDING_TIMEOUT
    temperature: float | None = DEFAULT_TEMPERATURE
    max_k: int | None = DEFAULT_MAX_K
    context_window: int | None = CHAT_CONFIG["context_window"]
    llm_num_predict: int | None = CHAT_CONFIG["llm_num_predict"]
    llm_timeout: float | None = DEFAULT_LLM_TIMEOUT
    web_search_enabled: bool = DEFAULT_WEB_SEARCH_ENABLED
    retrieval_candidate_k: int | None = DEFAULT_RETRIEVAL_CANDIDATE_K
    retrieval_min_score: float | None = CHAT_CONFIG["retrieval_min_score"]
    retrieval_relative_cutoff: float | None = DEFAULT_RETRIEVAL_RELATIVE_CUTOFF
    context_token_fraction: float | None = DEFAULT_CONTEXT_TOKEN_FRACTION
    web_search_timeout: float | None = DEFAULT_WEB_SEARCH_TIMEOUT
    web_search_max_results: int | None = DEFAULT_WEB_SEARCH_MAX_RESULTS
    ollama_health_check_interval: float | None = CHAT_CONFIG["ollama_health_check_interval"]
    ollama_max_lost_health_checks: int | None = CHAT_CONFIG["ollama_max_lost_health_checks"]
    system_prompt: str | None = CHAT_CONFIG["system_prompt"]


class IndexUpdateRequest(BaseModel):
    record_id: str
    content: str
    embedding_model: str | None = None
    embedding_batch_size: int | None = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_timeout: float | None = DEFAULT_EMBEDDING_TIMEOUT


class IndexDeleteRequest(BaseModel):
    record_ids: list[str]


class ReindexRequest(BaseModel):
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int | None = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_timeout: float | None = DEFAULT_EMBEDDING_TIMEOUT
    index_backend: str = DEFAULT_INDEX_BACKEND
    summary_mode: str = DEFAULT_SUMMARY_MODE
    chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS


class RenderRequest(BaseModel):
    text: str


def render_markdown_text(text: str) -> str:
    from latex2mathml.converter import convert as latex_to_mathml
    from markdown_it import MarkdownIt

    math_blocks: list[str] = []

    def replace_math(match: re.Match[str]) -> str:
        latex = next(group for group in match.groups() if group is not None)
        display = "block" if match.group(1) is not None or match.group(2) is not None else "inline"
        placeholder = f"@@RAG_MATH_{len(math_blocks)}@@"
        try:
            math_blocks.append(latex_to_mathml(latex.strip(), display=display))
        except Exception:
            math_blocks.append(html.escape(match.group(0)))
        return placeholder

    protected = re.sub(
        r"(?s)\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|(?<!\\)\$(?!\s)(.+?)(?<!\s)(?<!\\)\$",
        replace_math,
        text,
    )
    markdown = MarkdownIt("commonmark", {"html": False, "linkify": False})
    rendered = markdown.render(protected)
    for index, math_html in enumerate(math_blocks):
        rendered = rendered.replace(f"@@RAG_MATH_{index}@@", math_html)
    return rendered


job_queue = RagJobQueue()
app = FastAPI(title="Local FSAE RAG Pipeline")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    path = lancedb_path(DB_DIR)
    store = _index_store()
    record_count = 0
    index_exists = store.exists()
    if index_exists:
        try:
            with INDEX_LOCK:
                record_count = store.count()
        except Exception:
            record_count = 0
    return {
        "ok": True,
        "paths": {
            "data_dir": str(DATA_DIR),
            "upload_dir": str(UPLOAD_DIR),
            "processed_dir": str(PROCESSED_DIR),
            "db_dir": str(DB_DIR),
            "index_file": str(path),
        },
        "index_exists": index_exists,
        "record_count": record_count,
        "server": dict(SERVER_CONFIG),
        "chat": {
            "context_window": CHAT_CONFIG["context_window"],
            "llm_num_predict": CHAT_CONFIG["llm_num_predict"],
            "retrieval_min_score": CHAT_CONFIG["retrieval_min_score"],
        },
        "queue": job_queue.summary(),
    }


@app.post("/api/render")
def render_markdown(payload: RenderRequest):
    return {"html": render_markdown_text(payload.text)}


@app.post("/api/uploads")
async def upload_files(request: Request):
    job_id = uuid.uuid4().hex
    staging_dir = STAGING_DIR / job_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    filenames: list[str] = []
    uploads: list[dict[str, Any]] = []
    used_names: set[str] = set()
    queued = False
    registered = False

    try:
        form = await request.form()
        force_duplicates = str(form.get("force_duplicates") or "").lower() in {"1", "true", "yes", "on"}
        for _, value in form.multi_items():
            if not isinstance(value, StarletteUploadFile):
                continue

            filename = _safe_filename(value.filename or "")
            if Path(filename).suffix.lower() != ".pdf":
                raise HTTPException(status_code=400, detail=f"Only PDF uploads are supported: {filename}")

            base = filename
            counter = 1
            while filename.lower() in used_names:
                stem = Path(base).stem
                filename = f"{stem}-{counter}.pdf"
                counter += 1
            used_names.add(filename.lower())

            destination = staging_dir / filename
            digest = hashlib.sha256()
            with destination.open("wb") as handle:
                while True:
                    chunk = await value.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    handle.write(chunk)
            await value.close()
            filenames.append(filename)
            uploads.append(
                {
                    "filename": filename,
                    "hash": digest.hexdigest(),
                    "staging_path": str(destination),
                }
            )

        if not filenames:
            raise HTTPException(status_code=400, detail="No PDF files were uploaded.")

        batch_duplicate_hashes = {
            item["hash"]
            for item in uploads
            if sum(1 for candidate in uploads if candidate["hash"] == item["hash"]) > 1
        }
        if batch_duplicate_hashes:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Duplicate PDFs were selected in the same upload batch.",
                    "can_force": False,
                    "duplicates": [
                        {
                            "filename": item["filename"],
                            "hash": item["hash"],
                            "status": "selected_twice",
                        }
                        for item in uploads
                        if item["hash"] in batch_duplicate_hashes
                    ],
                },
            )

        registry = PdfRegistry(PDF_REGISTRY_PATH)
        duplicate_entries = registry.blocking_duplicates(uploads)
        if duplicate_entries and not force_duplicates:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "One or more PDFs have already been uploaded or queued.",
                    "can_force": True,
                    "duplicates": duplicate_entries,
                },
            )

        forced_hashes = {entry["hash"] for entry in duplicate_entries} if force_duplicates else set()
        registry.register_queued(job_id=job_id, files=uploads, forced_hashes=forced_hashes)
        registered = True
        job = job_queue.enqueue_upload(
            staging_dir=staging_dir,
            filenames=filenames,
            uploads=uploads,
            force_duplicate_hashes=sorted(forced_hashes),
            job_id=job_id,
            options={
                "parser_mode": str(form.get("parser_mode") or "hybrid"),
                "accelerator": str(form.get("accelerator") or "auto"),
                "asset_triggers": str(form.get("asset_triggers") or "none"),
                "embedding_model": str(form.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
                "embedding_batch_size": int(form.get("embedding_batch_size") or DEFAULT_EMBEDDING_BATCH_SIZE),
                "embedding_timeout": float(form.get("embedding_timeout") or DEFAULT_EMBEDDING_TIMEOUT),
                "index_backend": str(form.get("index_backend") or DEFAULT_INDEX_BACKEND),
                "summary_mode": str(form.get("summary_mode") or DEFAULT_SUMMARY_MODE),
                "chunk_target_tokens": int(form.get("chunk_target_tokens") or DEFAULT_CHUNK_TARGET_TOKENS),
                "chunk_overlap_tokens": int(form.get("chunk_overlap_tokens") or DEFAULT_CHUNK_OVERLAP_TOKENS),
                "progress_enabled": False,
            },
        )
        queued = True
        return job.to_dict()
    except Exception:
        if not queued:
            if registered:
                PdfRegistry(PDF_REGISTRY_PATH).mark_job_status(
                    job_id=job_id,
                    files=uploads,
                    status="failed",
                    error="Upload was not queued.",
                )
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


async def _optional_json(request: Request) -> dict[str, Any]:
    body = await request.body()
    if not body:
        return {}
    try:
        value = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return value


@app.post("/api/reindex")
async def reindex(request: Request):
    payload = ReindexRequest(**await _optional_json(request))
    job = job_queue.enqueue_reindex(options=payload.dict())
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": job_queue.list_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/api/pdfs")
def pdf_documents(search: str = ""):
    return list_pdf_documents(search=search)


@app.get("/api/pdfs/{source_hash}/download")
def download_pdf(source_hash: str):
    try:
        path, filename = resolve_pdf_download_path(source_hash)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="application/pdf", filename=filename)


@app.get("/api/index")
def index_rows(offset: int = 0, limit: int = 50, search: str = ""):
    try:
        return list_index_rows(offset=offset, limit=limit, search=search)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/index/update")
def update_index(payload: IndexUpdateRequest):
    try:
        row = update_index_record(
            record_id=payload.record_id,
            content=payload.content,
            embedding_model=payload.embedding_model,
            embedding_batch_size=payload.embedding_batch_size,
            embedding_timeout=payload.embedding_timeout,
        )
        return {"row": row}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Record not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/index/delete")
def delete_index(payload: IndexDeleteRequest):
    try:
        return delete_index_records(record_ids=payload.record_ids)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Record not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    job_queue.begin_query()

    def generate():
        def encode_event(event: dict[str, Any]) -> str:
            payload = dict(event)
            payload["type"] = str(payload.get("type") or "answer")
            return json.dumps(payload, ensure_ascii=False) + "\n"

        try:
            from src.query import QueryEngine

            engine = QueryEngine(
                working_dir=str(DB_DIR),
                model=payload.llm_model,
                embedding_model=payload.embedding_model,
                embedding_batch_size=payload.embedding_batch_size,
                embedding_timeout=payload.embedding_timeout,
                llm_num_predict=payload.llm_num_predict,
                llm_timeout=payload.llm_timeout,
                temperature=payload.temperature,
                sampler_top_k=payload.max_k,
                context_window=payload.context_window,
                retrieval_candidate_k=payload.retrieval_candidate_k,
                retrieval_min_score=payload.retrieval_min_score,
                retrieval_relative_cutoff=payload.retrieval_relative_cutoff,
                context_token_fraction=payload.context_token_fraction,
                web_search_enabled=payload.web_search_enabled,
                web_search_timeout=payload.web_search_timeout,
                web_search_max_results=payload.web_search_max_results,
                ollama_health_check_interval=payload.ollama_health_check_interval,
                ollama_max_lost_health_checks=payload.ollama_max_lost_health_checks,
                system_prompt=payload.system_prompt,
                progress_enabled=False,
            )
            if hasattr(engine, "ask_stream_events"):
                for event in engine.ask_stream_events(question):
                    if event.get("text") or event.get("sources"):
                        yield encode_event(event)
            else:
                for chunk in engine.ask_stream(question):
                    if chunk:
                        yield encode_event({"type": "answer", "text": chunk})
        except Exception as exc:
            yield encode_event({"type": "error", "text": str(exc)})
        finally:
            job_queue.finish_query()

    return StreamingResponse(generate(), media_type="application/x-ndjson; charset=utf-8")
