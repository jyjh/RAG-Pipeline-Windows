from __future__ import annotations

import html
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
from src.vector_store import default_store, lancedb_path, vector_index_path as json_index_path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STAGING_DIR = DATA_DIR / ".upload_queue"
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


def _load_server_config(config_path: Path | None = None) -> dict[str, int]:
    config_path = config_path or (ROOT_DIR / "config.toml")
    server_config: dict[str, Any] = {}
    if config_path.exists():
        try:
            import tomllib

            with config_path.open("rb") as handle:
                payload = tomllib.load(handle)
            if isinstance(payload.get("server"), dict):
                server_config = payload["server"]
        except Exception:
            server_config = {}

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


SERVER_CONFIG = _load_server_config()


def vector_index_path(db_dir: Path | None = None) -> Path:
    return json_index_path(db_dir or DB_DIR)


def _index_store(db_dir: Path | None = None):
    return default_store(db_dir or DB_DIR, prefer_lancedb=True)


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


@dataclass
class QueueJob:
    id: str
    kind: str
    status: str = "queued"
    phase: str = "queued"
    filenames: list[str] = field(default_factory=list)
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
        run_ingestion_func=None,
        run_indexing_func=None,
    ):
        self.upload_root = Path(upload_root)
        self.processed_dir = Path(processed_dir)
        self.db_dir = Path(db_dir)
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
        job_id: str | None = None,
        options: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> QueueJob:
        job = QueueJob(
            id=job_id or uuid.uuid4().hex,
            kind="upload",
            filenames=filenames,
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
            self._run_ingestion(str(upload_dir), str(self.processed_dir), job.options)
            self._wait_for_no_queries(job, "indexing")
            self._run_indexing(str(self.processed_dir), str(self.db_dir), job.options)
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
            try:
                source.unlink()
            except PermissionError:
                pass
        shutil.rmtree(staging_dir, ignore_errors=True)
        job.upload_dir = str(upload_dir)
        return upload_dir

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
    context_window: int | None = DEFAULT_CONTEXT_WINDOW
    llm_num_predict: int | None = DEFAULT_LLM_NUM_PREDICT
    llm_timeout: float | None = DEFAULT_LLM_TIMEOUT


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
            "legacy_json_index_file": str(vector_index_path()),
        },
        "index_exists": index_exists,
        "record_count": record_count,
        "server": dict(SERVER_CONFIG),
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
    used_names: set[str] = set()
    queued = False

    try:
        form = await request.form()
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
            with destination.open("wb") as handle:
                while True:
                    chunk = await value.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            await value.close()
            filenames.append(filename)

        if not filenames:
            raise HTTPException(status_code=400, detail="No PDF files were uploaded.")

        job = job_queue.enqueue_upload(
            staging_dir=staging_dir,
            filenames=filenames,
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
        def encode_event(event_type: str, text: str) -> str:
            return json.dumps({"type": event_type, "text": text}, ensure_ascii=False) + "\n"

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
                progress_enabled=False,
            )
            if hasattr(engine, "ask_stream_events"):
                for event in engine.ask_stream_events(question):
                    event_type = str(event.get("type", "answer"))
                    text = str(event.get("text", ""))
                    if text:
                        yield encode_event(event_type, text)
            else:
                for chunk in engine.ask_stream(question):
                    if chunk:
                        yield encode_event("answer", chunk)
        except Exception as exc:
            yield encode_event("error", str(exc))
        finally:
            job_queue.finish_query()

    return StreamingResponse(generate(), media_type="application/x-ndjson; charset=utf-8")
