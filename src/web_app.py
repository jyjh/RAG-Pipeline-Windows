from __future__ import annotations

import base64
import hmac
import html
import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.parse
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile

from src.defaults import (
    DEFAULT_ASSET_DIR,
    DEFAULT_ASSET_TRIGGERS,
    DEFAULT_CODE_ENRICHMENT,
    DEFAULT_DOCLING_ACCELERATOR,
    DEFAULT_FORMULA_ENRICHMENT,
    DEFAULT_LLM_MODEL,
    DEFAULT_OCR_BACKEND,
    DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
    DEFAULT_OCR_FORCE_FULL_PAGE,
    DEFAULT_OCR_LANGS,
    DEFAULT_PDF_PARSER_MODE,
    DEFAULT_RAPIDOCR_BACKEND,
    DEFAULT_TESSERACT_CMD,
    DEFAULT_TESSERACT_DATA_PATH,
    DEFAULT_TESSERACT_PSM,
    DEFAULT_VISION_ENABLED,
    DEFAULT_VISION_MODEL,
)
from src.local_rag import (
    DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL,
    DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS,
    DEFAULT_QUERY_SYSTEM_PROMPT,
    INDEX_MANIFEST_FILENAME,
    write_index_manifest,
)
from src.index_overrides import (
    clear_overrides_for_sources,
    edited_record_ids,
    index_overrides_path,
    persist_index_deletions,
    persist_index_edit,
)
from src.pdf_registry import PdfRegistry, load_source_map, remove_source_entries_by_hash, sha256_file, source_map_path
from src.reliability import (
    SOURCE_GROUP_UNGROUPED,
    normalize_source_group,
    source_group_weight,
    valid_assignable_source_groups,
)
from src.vector_store import LANCEDB_DIRNAME, default_store, lancedb_path, record_matches, record_row

from src._class_module_support import import_split_class

_CLASS_MODULE_PROXY_FUNCTIONS = (
    "_resolve_root_path",
    "_utcnow",
    "_safe_filename",
    "_positive_int",
    "_positive_float",
    "_nonempty_str",
    "_bool_value",
    "_optional_int",
    "_page_slice",
    "_string_list",
    "_load_toml_config",
    "_load_server_config",
    "_load_chat_config",
    "_load_ingestion_config",
    "_index_store",
    "_background_worker_threads",
    "_bool_cli_value",
    "_append_cli_option",
    "_job_subprocess_env",
    "_job_subprocess_creationflags",
    "_process_output_tail",
    "_run_job_subprocess",
    "_staged_index_dir",
    "_remove_path",
    "_publish_staged_index",
    "_index_mutation_blocker",
    "_git_target_valid",
    "_run_git",
    "_git_failure_message",
    "_git_output",
    "_git_is_ancestor",
    "_update_target",
    "_update_response",
    "_active_update_blocker",
    "get_update_status",
    "_is_loopback_host",
    "_is_unspecified_host",
    "_is_configured_bind_host",
    "_is_local_update_host",
    "_require_local_update_request",
    "_spawn_restart_helper",
    "_schedule_process_exit",
    "apply_available_update",
    "list_index_rows",
    "_index_records_snapshot",
    "_is_index_summary",
    "_index_top_summary_rows",
    "_index_descends_from",
    "_index_child_candidates",
    "_index_section_depth",
    "_index_sort_key",
    "_asset_url",
    "_with_index_assets",
    "_with_index_assets_for_rows",
    "_with_index_hierarchy_metadata",
    "list_index_summary_rows",
    "list_index_child_rows",
    "_index_stream_batch_size",
    "iter_index_row_events",
    "update_index_record",
    "delete_index_records",
    "vector_search_index_rows",
    "index_backup_root",
    "list_index_backups",
    "create_index_backup",
    "restore_index_from_backup",
    "_resolve_pdf_path",
    "_pdf_download_url",
    "delete_pdf_document",
    "_load_trust_registry",
    "_write_trust_registry",
    "_default_trust_entry",
    "_normalize_trust_entry",
    "_parse_expiry",
    "_trust_warnings",
    "update_document_trust",
    "_resolve_workspace_path",
    "_load_index_manifest",
    "_derive_index_manifest",
    "_index_manifest_stats",
    "_markdown_quality",
    "_document_quality",
    "_pdf_entry_for_response",
    "list_pdf_documents",
    "list_job_rows",
    "resolve_pdf_download_path",
    "_reprocess_options_for_source",
    "enqueue_source_reprocess",
    "enqueue_full_reingest",
    "_model_dump",
    "render_markdown_text",
    "recover_pending_upload_jobs_on_startup",
    "lifespan",
    "root",
    "health",
    "update_status",
    "update_apply",
    "render_markdown",
    "_upload_options_from_form",
    "_upload_hashes",
    "_duplicate_hashes",
    "_force_upload_token",
    "_decode_force_token_part",
    "_force_upload_token_valid",
    "_duplicate_response_detail",
    "_indexed_source_duplicate_entries",
    "_vector_store_duplicate_entries",
    "_path_is_relative_to",
    "_data_pdf_duplicate_entries",
    "_blocking_duplicate_entries",
    "upload_files",
    "_optional_json",
    "reindex",
    "list_jobs",
    "get_job",
    "cancel_job",
    "pdf_documents",
    "image_asset",
    "download_pdf",
    "view_pdf",
    "delete_pdf",
    "reprocess_pdf",
    "update_pdf_trust",
    "index_rows",
    "index_summary_rows",
    "index_child_rows",
    "index_rows_stream",
    "update_index",
    "delete_index",
    "vector_search_index",
    "chat_stream",
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STAGING_DIR = DATA_DIR / ".upload_queue"
PDF_REGISTRY_PATH = DATA_DIR / ".pdf_upload_registry.json"
DOCUMENT_TRUST_PATH = DATA_DIR / ".document_trust.json"
PROCESSED_DIR = ROOT_DIR / "processed_docs"
DB_DIR = ROOT_DIR / "db"
WEB_DIR = ROOT_DIR / "web"


def _resolve_root_path(raw_path: Any, *, default: str | Path | None = None) -> Path:
    value = raw_path if raw_path not in (None, "") else default
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT_DIR / path


DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_EMBEDDING_TIMEOUT = 30.0
DEFAULT_TEMPERATURE = 0.3
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
DEFAULT_PLANNER_MODEL = "qwen2.5:1.5b"
DEFAULT_PLANNER_ENABLED = True
DEFAULT_PLANNER_MAX_QUERIES = 3
DEFAULT_INDEX_BACKEND = "lancedb"
DEFAULT_SUMMARY_MODE = "hybrid"
DEFAULT_CHUNK_TARGET_TOKENS = 900
DEFAULT_CHUNK_OVERLAP_TOKENS = 120
DEFAULT_HEALTH_POLL_INTERVAL_MS = 60_000
DEFAULT_JOBS_POLL_INTERVAL_MS = 60_000
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_BIND_ALL_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8000
DEFAULT_BACKGROUND_WORKER_THREADS = min(4, max(1, (os.cpu_count() or 2) // 2))
DEFAULT_UPDATE_REMOTE = "origin"
DEFAULT_UPDATE_BRANCH = "main"
JOB_LOG_TAIL_LINES = 200
GIT_TIMEOUT_SECONDS = 30.0
GIT_PULL_TIMEOUT_SECONDS = 300.0
INDEX_STREAM_DEFAULT_BATCH_SIZE = 250
INDEX_STREAM_MAX_BATCH_SIZE = 1_000
INDEX_STREAM_WORKERS = min(8, max(2, os.cpu_count() or 2))
INDEX_SUMMARY_NODE_TYPES = {"document_summary", "section_summary"}
INDEX_CHILD_DEFAULT_LIMIT = 100
INDEX_CHILD_MAX_LIMIT = 500
DEFAULT_INDEX_VECTOR_RELEVANCE_FLOOR = 0.70
FORCE_UPLOAD_TOKEN_TTL_SECONDS = 15 * 60
UPLOAD_FORCE_TOKEN_SECRET = os.urandom(32)
RECOVERABLE_UPLOAD_STATUSES = {"queued", "saving_uploads", "ingesting", "ingested"}
UPLOAD_RESUME_STATUS_ORDER = {
    "queued": 0,
    "saving_uploads": 1,
    "ingesting": 2,
    "ingested": 3,
}

INDEX_LOCK = threading.RLock()
TRUST_LOCK = threading.RLock()
LANCEDB_BACKUP_DIRNAME = "backups"
LANCEDB_BACKUP_KEEP = 5
INDEX_BACKUP_COMPONENTS = (LANCEDB_DIRNAME, INDEX_MANIFEST_FILENAME, "index_overrides.json")
TRUST_REVIEW_STATUSES = {"unreviewed", "approved", "rejected", "stale"}
TRUST_SOURCE_TYPES = {
    "unknown",
    "textbook",
    "research_paper",
    "lecture_notes",
    "student_project",
    "team_report",
    "standard",
    "web",
}
TRUST_SOURCE_GROUPS = set(valid_assignable_source_groups())


class JobCancelled(RuntimeError):
    """Raised when a queued background job is cancelled by the user."""


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


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _nonempty_str(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _optional_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _page_slice(rows: list[dict[str, Any]], *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    total = len(rows)
    offset = max(0, int(offset))
    if limit is None:
        page = rows[offset:]
        resolved_limit = len(page)
    else:
        resolved_limit = min(max(1, int(limit)), 100)
        page = rows[offset : offset + resolved_limit]
    return {
        "rows": page,
        "total": total,
        "offset": offset,
        "limit": resolved_limit,
    }


def _string_list(value: Any, default: tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or list(default)
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return parts or list(default)
    return list(default)


_TOML_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def _load_toml_config(config_path: Path) -> dict[str, Any]:
    cache_key = str(config_path)
    cached = _TOML_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not config_path.exists():
        result: dict[str, Any] = {}
        _TOML_CONFIG_CACHE[cache_key] = result
        return result
    try:
        import tomllib

        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
        result = payload if isinstance(payload, dict) else {}
    except Exception:
        result = {}
    _TOML_CONFIG_CACHE[cache_key] = result
    return result


def _normalize_server_host(raw_host: Any, *, bind_all: bool) -> str:
    if bind_all:
        return DEFAULT_BIND_ALL_HOST
    host = _nonempty_str(raw_host, DEFAULT_SERVER_HOST).strip()
    if host.lower() in {"*", "all"}:
        return DEFAULT_BIND_ALL_HOST
    return host


def _server_bind_all_enabled(server_config: dict[str, Any], host: str) -> bool:
    requested = _bool_value(
        server_config.get("bind_all"),
        _bool_value(server_config.get("lan"), False),
    )
    return requested or host.strip().lower() in {DEFAULT_BIND_ALL_HOST, "::"}


def _load_server_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or (ROOT_DIR / "config.toml")
    payload = _load_toml_config(config_path)
    server_config = payload.get("server", {}) if isinstance(payload.get("server"), dict) else {}
    bind_all_requested = _bool_value(
        server_config.get("bind_all"),
        _bool_value(server_config.get("lan"), False),
    )
    host = _normalize_server_host(server_config.get("host"), bind_all=bind_all_requested)
    bind_all = _server_bind_all_enabled(server_config, host)

    return {
        "host": host,
        "bind_all": bind_all,
        "port": _positive_int(server_config.get("port"), DEFAULT_SERVER_PORT),
        "health_poll_interval_ms": _positive_int(
            server_config.get("health_poll_interval_ms"),
            DEFAULT_HEALTH_POLL_INTERVAL_MS,
        ),
        "jobs_poll_interval_ms": _positive_int(
            server_config.get("jobs_poll_interval_ms"),
            DEFAULT_JOBS_POLL_INTERVAL_MS,
        ),
        "background_worker_threads": _positive_int(
            server_config.get("background_worker_threads"),
            DEFAULT_BACKGROUND_WORKER_THREADS,
        ),
        "update_remote": _nonempty_str(server_config.get("update_remote"), DEFAULT_UPDATE_REMOTE),
        "update_branch": _nonempty_str(server_config.get("update_branch"), DEFAULT_UPDATE_BRANCH),
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
        "planner_model": str(chat_config.get("planner_model") or DEFAULT_PLANNER_MODEL),
        "planner_enabled": _bool_value(chat_config.get("planner_enabled"), DEFAULT_PLANNER_ENABLED),
        "planner_max_queries": _positive_int(
            chat_config.get("planner_max_queries"),
            DEFAULT_PLANNER_MAX_QUERIES,
        ),
    }


def _load_ingestion_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or (ROOT_DIR / "config.toml")
    payload = _load_toml_config(config_path)
    ingestion = payload.get("ingestion", {}) if isinstance(payload.get("ingestion"), dict) else {}
    models = payload.get("models", {}) if isinstance(payload.get("models"), dict) else {}
    paths = payload.get("paths", {}) if isinstance(payload.get("paths"), dict) else {}

    return {
        "asset_dir": _nonempty_str(paths.get("asset_dir"), DEFAULT_ASSET_DIR),
        "parser_mode": _nonempty_str(ingestion.get("parser_mode"), DEFAULT_PDF_PARSER_MODE),
        "accelerator": _nonempty_str(ingestion.get("accelerator"), DEFAULT_DOCLING_ACCELERATOR),
        "num_threads": _positive_int(ingestion.get("num_threads"), 8),
        "asset_triggers": _nonempty_str(ingestion.get("asset_triggers"), DEFAULT_ASSET_TRIGGERS),
        "code_enrichment": _bool_value(ingestion.get("code_enrichment"), DEFAULT_CODE_ENRICHMENT),
        "formula_enrichment": _bool_value(ingestion.get("formula_enrichment"), DEFAULT_FORMULA_ENRICHMENT),
        "vision_model": _nonempty_str(models.get("vision_model"), DEFAULT_VISION_MODEL),
        "vision_enabled": _bool_value(ingestion.get("vision_enabled"), DEFAULT_VISION_ENABLED),
        "ocr_backend": _nonempty_str(ingestion.get("ocr_backend"), DEFAULT_OCR_BACKEND),
        "ocr_langs": _string_list(ingestion.get("ocr_langs"), DEFAULT_OCR_LANGS),
        "ocr_force_full_page": _bool_value(ingestion.get("ocr_force_full_page"), DEFAULT_OCR_FORCE_FULL_PAGE),
        "ocr_bitmap_area_threshold": _positive_float(
            ingestion.get("ocr_bitmap_area_threshold"),
            DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
        ),
        "rapidocr_backend": _nonempty_str(ingestion.get("rapidocr_backend"), DEFAULT_RAPIDOCR_BACKEND),
        "tesseract_cmd": _nonempty_str(ingestion.get("tesseract_cmd"), DEFAULT_TESSERACT_CMD),
        "tesseract_data_path": str(ingestion.get("tesseract_data_path") or DEFAULT_TESSERACT_DATA_PATH),
        "tesseract_psm": _optional_int(ingestion.get("tesseract_psm"), DEFAULT_TESSERACT_PSM),
    }


SERVER_CONFIG = _load_server_config()
CHAT_CONFIG = _load_chat_config()
INGESTION_CONFIG = _load_ingestion_config()
ASSET_DIR = _resolve_root_path(INGESTION_CONFIG["asset_dir"], default=DEFAULT_ASSET_DIR)


_INDEX_STORE_CACHE: dict = {}
_INDEX_RECORDS_SNAPSHOT_CACHE: dict[str, tuple[str, tuple[list[dict[str, Any]], str, int]]] = {}


def _index_store(db_dir: Path | None = None):
    resolved = str(db_dir or DB_DIR)
    store = _INDEX_STORE_CACHE.get(resolved)
    if store is None:
        store = default_store(resolved)
        _INDEX_STORE_CACHE[resolved] = store
    return store


def _background_worker_threads() -> int:
    return _positive_int(
        SERVER_CONFIG.get("background_worker_threads"),
        DEFAULT_BACKGROUND_WORKER_THREADS,
    )


def _bool_cli_value(value: Any) -> str:
    return "true" if _bool_value(value, False) else "false"


def _append_cli_option(command: list[str], name: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, bool):
        value = _bool_cli_value(value)
    elif isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    command.extend([name, str(value)])


def _job_subprocess_env(worker_threads: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    thread_cap = (
        _positive_int(worker_threads, _background_worker_threads())
        if worker_threads
        else _background_worker_threads()
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        current = env.get(key)
        try:
            current_value = int(current) if current is not None else thread_cap
        except ValueError:
            current_value = thread_cap
        env[key] = str(max(1, min(current_value, thread_cap)))
    return env


def _job_subprocess_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000))


def _process_output_tail(result: subprocess.CompletedProcess[str], *, limit: int = 4000) -> str:
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    if len(output) <= limit:
        return output
    return output[-limit:]


def _run_job_subprocess(
    command: list[str],
    *,
    worker_threads: int | None = None,
    cancel_event: threading.Event | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "cwd": ROOT_DIR,
        "env": _job_subprocess_env(worker_threads),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    creationflags = _job_subprocess_creationflags()
    if creationflags:
        kwargs["creationflags"] = creationflags
    if cancel_event is None:
        result = subprocess.run(command, **kwargs, check=False)
        if result.returncode == 0:
            return
        detail = _process_output_tail(result)
        command_text = " ".join(command)
        message = f"Background pipeline command failed with exit code {result.returncode}: {command_text}"
        if detail:
            message = f"{message}\n{detail}"
        raise RuntimeError(message)

    if cancel_event.is_set():
        raise JobCancelled("Job cancelled by user.")

    kwargs["stderr"] = subprocess.STDOUT

    process = subprocess.Popen(command, **kwargs)
    output_lines: list[str] = []
    reader_done = threading.Event()

    def read_output() -> None:
        try:
            stream = process.stdout
            if stream is None:
                return
            for line in stream:
                text = str(line).rstrip()
                if text:
                    output_lines.append(text)
                    if len(output_lines) > JOB_LOG_TAIL_LINES:
                        del output_lines[: len(output_lines) - JOB_LOG_TAIL_LINES]
                    if log_callback is not None:
                        log_callback(text)
        finally:
            reader_done.set()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    while True:
        returncode = process.poll()
        if returncode is not None:
            reader_done.wait(timeout=5)
            break
        if cancel_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader_done.wait(timeout=5)
            raise JobCancelled("Job cancelled by user.")
        reader_done.wait(timeout=0.2)

    stdout = "\n".join(output_lines)
    result = subprocess.CompletedProcess(command, process.returncode, stdout=stdout, stderr="")
    if result.returncode == 0:
        return
    detail = _process_output_tail(result)
    command_text = " ".join(command)
    message = f"Background pipeline command failed with exit code {result.returncode}: {command_text}"
    if detail:
        message = f"{message}\n{detail}"
    raise RuntimeError(message)


def _staged_index_dir(live_db_dir: str | Path) -> Path:
    live_path = Path(live_db_dir)
    return live_path.parent / f".index_build_{uuid.uuid4().hex}"


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _publish_staged_index(staged_db_dir: str | Path, live_db_dir: str | Path) -> None:
    staged_db = Path(staged_db_dir)
    live_db = Path(live_db_dir)
    staged_lancedb = lancedb_path(staged_db)
    staged_manifest = staged_db / INDEX_MANIFEST_FILENAME
    staged_overrides = index_overrides_path(staged_db)
    live_lancedb = lancedb_path(live_db)
    live_manifest = live_db / INDEX_MANIFEST_FILENAME
    live_overrides = index_overrides_path(live_db)

    if not staged_lancedb.exists():
        raise RuntimeError(f"Indexing did not create a LanceDB directory at {staged_lancedb}")
    if not staged_manifest.exists():
        raise RuntimeError(f"Indexing did not create an index manifest at {staged_manifest}")

    live_db.mkdir(parents=True, exist_ok=True)
    backup_dir = live_db.parent / f".index_backup_{uuid.uuid4().hex}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_lancedb = backup_dir / LANCEDB_DIRNAME
    backup_manifest = backup_dir / INDEX_MANIFEST_FILENAME
    backup_overrides = backup_dir / index_overrides_path(live_db).name

    try:
        if live_lancedb.exists():
            shutil.move(str(live_lancedb), str(backup_lancedb))
        if live_manifest.exists():
            shutil.move(str(live_manifest), str(backup_manifest))
        if live_overrides.exists():
            shutil.move(str(live_overrides), str(backup_overrides))
        shutil.move(str(staged_lancedb), str(live_lancedb))
        shutil.move(str(staged_manifest), str(live_manifest))
        if staged_overrides.exists():
            shutil.move(str(staged_overrides), str(live_overrides))
        elif backup_overrides.exists():
            shutil.move(str(backup_overrides), str(live_overrides))
    except Exception:
        _remove_path(live_lancedb)
        _remove_path(live_manifest)
        _remove_path(live_overrides)
        if backup_lancedb.exists():
            shutil.move(str(backup_lancedb), str(live_lancedb))
        if backup_manifest.exists():
            shutil.move(str(backup_manifest), str(live_manifest))
        if backup_overrides.exists():
            shutil.move(str(backup_overrides), str(live_overrides))
        raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)


def index_backup_root(db_dir: str | Path) -> Path:
    return Path(db_dir) / LANCEDB_BACKUP_DIRNAME


def _index_backup_component_paths(db_dir: str | Path) -> list[Path]:
    base = Path(db_dir)
    overrides = index_overrides_path(base)
    return [lancedb_path(base), base / INDEX_MANIFEST_FILENAME, overrides]


def _backup_created_at_from_name(name: str) -> str:
    # Snapshot dir names are ``YYYYMMDD-HHMMSS-<hex>``. Parse the leading
    # timestamp back into an ISO-8601 string for display; fall back to the
    # raw name when the prefix does not match (manual/imported backups).
    match = re.match(r"^(\d{8})-(\d{6})", name)
    if not match:
        return name
    date_part, time_part = match.groups()
    try:
        parsed = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
    except ValueError:
        return name
    return parsed.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")


def _backup_size_bytes(backup_dir: Path) -> int:
    total = 0
    for entry in backup_dir.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _invalidate_index_caches(db_dir: str | Path) -> None:
    resolved = str(Path(db_dir).resolve())
    _INDEX_RECORDS_SNAPSHOT_CACHE.pop(resolved, None)
    store = _INDEX_STORE_CACHE.get(resolved)
    if store is not None:
        store._invalidate_table()


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    """Sort snapshots newest-first by creation mtime, then name as a tiebreaker.

    Two backups created within the same wall-clock second share the timestamp
    prefix in their directory name, so lexicographic name ordering alone is not
    enough to recover creation order. The directory mtime is monotonic across
    freshly-created snapshots and disambiguates them.
    """
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return (mtime, path.name)


def list_index_backups(db_dir: str | Path) -> list[dict[str, Any]]:
    """List LanceDB index backup snapshots under ``db/backups/`` (newest first)."""
    backup_root = index_backup_root(db_dir)
    if not backup_root.exists():
        return []
    children = sorted(
        (child for child in backup_root.iterdir() if child.is_dir()),
        key=_snapshot_sort_key,
        reverse=True,
    )
    entries: list[dict[str, Any]] = []
    for child in children:
        lancedb_dir = child / LANCEDB_DIRNAME
        manifest = child / INDEX_MANIFEST_FILENAME
        record_count: int | None
        try:
            store = default_store(child)
            record_count = store.count() if lancedb_dir.exists() else None
        except Exception:
            record_count = None
        entries.append(
            {
                "name": child.name,
                "created_at": _backup_created_at_from_name(child.name),
                "path": str(child),
                "lancedb_present": lancedb_dir.exists(),
                "manifest_present": manifest.exists(),
                "record_count": record_count,
                "size_bytes": _backup_size_bytes(child),
            }
        )
    return entries


def create_index_backup(db_dir: str | Path) -> dict[str, Any]:
    """Snapshot the live LanceDB index, manifest, and overrides under ``db/backups/``.

    Backups older than ``LANCEDB_BACKUP_KEEP`` are pruned so disk usage stays
    bounded. Callers must hold the index-mutation guard (no active indexing) so
    the live table is not mid-write while it is copied.
    """
    resolved_db_dir = Path(db_dir)
    live_lancedb = lancedb_path(resolved_db_dir)
    if not live_lancedb.exists():
        raise FileNotFoundError(
            f"Nothing to back up: LanceDB directory not found at {live_lancedb}"
        )
    backup_root = index_backup_root(resolved_db_dir)
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    snapshot_dir = backup_root / f"{stamp}-{uuid.uuid4().hex[:6]}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    components = _index_backup_component_paths(resolved_db_dir)
    try:
        for source in components:
            if not source.exists():
                continue
            destination = snapshot_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
    except Exception:
        _remove_path(snapshot_dir)
        raise

    _prune_index_backups(backup_root)
    _invalidate_index_caches(resolved_db_dir)
    snapshot_entries = list_index_backups(resolved_db_dir)
    return next(
        (entry for entry in snapshot_entries if entry["name"] == snapshot_dir.name),
        {
            "name": snapshot_dir.name,
            "created_at": _backup_created_at_from_name(snapshot_dir.name),
            "path": str(snapshot_dir),
            "lancedb_present": True,
            "manifest_present": (snapshot_dir / INDEX_MANIFEST_FILENAME).exists(),
            "record_count": None,
            "size_bytes": _backup_size_bytes(snapshot_dir),
        },
    )


def _prune_index_backups(backup_root: Path) -> None:
    if LANCEDB_BACKUP_KEEP <= 0:
        return
    snapshots = sorted(
        (child for child in backup_root.iterdir() if child.is_dir()),
        key=_snapshot_sort_key,
        reverse=True,
    )
    for stale in snapshots[LANCEDB_BACKUP_KEEP:]:
        _remove_path(stale)


def _swap_index_into(source_db_dir: str | Path, live_db_dir: str | Path) -> None:
    """Atomically swap the LanceDB/manifest/overrides from ``source`` into ``live``.

    The current live components are moved aside into a ``.index_rollover_<hex>``
    directory first so a failed swap can roll back, mirroring the safe-swap logic
    already used by ``_publish_staged_index``.
    """
    source_db = Path(source_db_dir)
    live_db = Path(live_db_dir)
    source_components = _index_backup_component_paths(source_db)
    live_components = _index_backup_component_paths(live_db)

    live_db.mkdir(parents=True, exist_ok=True)
    rollover = live_db.parent / f".index_rollover_{uuid.uuid4().hex}"
    rollover.mkdir(parents=True, exist_ok=False)
    rollover_targets = [rollover / component.name for component in live_components]
    try:
        for live_component, rollover_target in zip(live_components, rollover_targets):
            if live_component.exists():
                shutil.move(str(live_component), str(rollover_target))
        for source_component, live_component in zip(source_components, live_components):
            if source_component.exists():
                shutil.move(str(source_component), str(live_component))
    except Exception:
        for live_component, rollover_target in zip(live_components, rollover_targets):
            _remove_path(live_component)
            if rollover_target.exists():
                shutil.move(str(rollover_target), str(live_component))
        _remove_path(rollover)
        raise
    finally:
        _remove_path(rollover)

    _invalidate_index_caches(live_db)


def restore_index_from_backup(db_dir: str | Path, backup_name: str) -> dict[str, Any]:
    """Restore a named backup snapshot over the live LanceDB index.

    A one-shot backup of the current live index is taken first into
    ``db/backups/`` so a bad restore is itself recoverable. ``backup_name`` is
    validated to be a bare directory name under the backup root to prevent path
    traversal.
    """
    resolved_db_dir = Path(db_dir)
    raw_name = str(backup_name or "").strip()
    # Reject anything that is not a bare directory name. Path separators (in
    # either direction), drive letters, parent traversal, and absolute paths are
    # all blocked so ``backup_root / name`` cannot escape the backup directory.
    if (
        not raw_name
        or raw_name in {".", ".."}
        or "/" in raw_name
        or "\\" in raw_name
        or ":" in raw_name
        or Path(raw_name).is_absolute()
    ):
        raise ValueError("Invalid backup name.")
    name = raw_name
    backup_root = index_backup_root(resolved_db_dir)
    snapshot_dir = backup_root / name
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Backup not found: {name}")
    backup_lancedb = snapshot_dir / LANCEDB_DIRNAME
    if not backup_lancedb.exists():
        raise FileNotFoundError(f"Backup is missing its LanceDB directory: {name}")

    safety_backup = None
    if lancedb_path(resolved_db_dir).exists():
        try:
            safety_backup = create_index_backup(resolved_db_dir)
        except Exception:
            safety_backup = None

    _swap_index_into(snapshot_dir, resolved_db_dir)
    return {
        "restored": True,
        "backup_name": name,
        "safety_backup": safety_backup,
    }


def _index_mutation_blocker() -> str:
    summary = job_queue.summary()
    if summary.get("indexing_job_ids"):
        return "Index edits are disabled while indexing is running. Try again after the job finishes."
    return ""


def _asset_url(asset_id: str) -> str:
    encoded = urllib.parse.quote(str(asset_id or ""), safe="")
    return f"/api/assets/{encoded}" if encoded else ""


def _with_index_assets(row: dict[str, Any], *, asset_store: Any | None = None) -> dict[str, Any]:
    item = dict(row)
    content = str(item.get("content") or "")
    if "[Image Asset:" not in content:
        return item
    if asset_store is None:
        from src.asset_store import ImageAssetStore

        asset_store = ImageAssetStore(ASSET_DIR)
    assets = asset_store.assets_for_text(content, url_for=_asset_url)
    if assets:
        item["assets"] = assets
    return item


def _with_index_assets_for_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any("[Image Asset:" in str(row.get("content") or "") for row in rows):
        return rows

    from src.asset_store import ImageAssetStore

    asset_store = ImageAssetStore(ASSET_DIR)
    return [_with_index_assets(row, asset_store=asset_store) for row in rows]


def _with_index_override_metadata_for_rows(
    rows: list[dict[str, Any]],
    *,
    db_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    try:
        edited_ids = edited_record_ids(Path(db_dir or DB_DIR))
    except Exception:
        edited_ids = set()
    if not edited_ids:
        return rows
    marked: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if str(item.get("id") or "") in edited_ids:
            item["edited"] = True
        marked.append(item)
    return marked


def _git_target_valid(value: str) -> bool:
    if not value or value.startswith(("-", "/", "\\")):
        return False
    if value.endswith(("/", ".", ".lock")):
        return False
    if ".." in value or "@{" in value or "\\" in value:
        return False
    return re.fullmatch(r"[A-Za-z0-9._/-]+", value) is not None


def _run_git(args: list[str], *, timeout: float = GIT_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _git_failure_message(args: list[str], result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    if detail:
        return detail
    return f"git {' '.join(args)} failed with exit code {result.returncode}"


def _git_output(args: list[str], *, timeout: float = GIT_TIMEOUT_SECONDS) -> str:
    result = _run_git(args, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(_git_failure_message(args, result))
    return result.stdout.strip()


def _git_is_ancestor(ancestor_sha: str, descendant_sha: str) -> bool:
    result = _run_git(["merge-base", "--is-ancestor", ancestor_sha, descendant_sha])
    if result.returncode in {0, 1}:
        return result.returncode == 0
    raise RuntimeError(_git_failure_message(["merge-base", "--is-ancestor", ancestor_sha, descendant_sha], result))


def _update_target() -> tuple[str, str]:
    return str(SERVER_CONFIG["update_remote"]), str(SERVER_CONFIG["update_branch"])


def _update_response(
    *,
    state: str,
    message: str,
    current_sha: str = "",
    latest_sha: str = "",
    current_branch: str = "",
    can_update: bool = False,
) -> dict[str, Any]:
    remote, branch = _update_target()
    return {
        "state": state,
        "can_update": can_update,
        "current_sha": current_sha,
        "latest_sha": latest_sha,
        "current_branch": current_branch,
        "target_remote": remote,
        "target_branch": branch,
        "message": message,
    }


def _active_update_blocker() -> str:
    summary = job_queue.summary()
    if int(summary.get("active_query_count") or 0) > 0:
        return "An active chat query is running. Try updating after it finishes."
    if summary.get("running_job_ids"):
        return "An ingestion or indexing job is running. Try updating after it finishes."
    if int(summary.get("queued_count") or 0) > 0:
        return "Queued ingestion or indexing work is pending. Try updating after the queue finishes."
    return ""


def get_update_status(*, fetch: bool = True) -> dict[str, Any]:
    remote, branch = _update_target()
    current_branch = ""
    current_sha = ""
    latest_sha = ""

    if not _git_target_valid(remote) or not _git_target_valid(branch):
        return _update_response(
            state="blocked",
            message="Update remote or branch contains unsupported characters.",
        )

    try:
        current_branch = _git_output(["branch", "--show-current"])
        current_sha = _git_output(["rev-parse", "HEAD"])

        if current_branch != branch:
            return _update_response(
                state="blocked",
                current_branch=current_branch,
                current_sha=current_sha,
                message=f"Auto-update is configured for {remote}/{branch}, but the server is running {current_branch or 'detached HEAD'}.",
            )

        dirty = _git_output(["status", "--porcelain", "--untracked-files=no"])
        if dirty:
            return _update_response(
                state="blocked",
                current_branch=current_branch,
                current_sha=current_sha,
                message="Tracked files have local changes. Commit or stash them before updating.",
            )

        blocker = _active_update_blocker()
        if blocker:
            return _update_response(
                state="blocked",
                current_branch=current_branch,
                current_sha=current_sha,
                message=blocker,
            )

        remote_ref = f"refs/remotes/{remote}/{branch}"
        if fetch:
            refspec = f"+refs/heads/{branch}:{remote_ref}"
            fetch_result = _run_git(["fetch", "--quiet", remote, refspec])
            if fetch_result.returncode != 0:
                return _update_response(
                    state="error",
                    current_branch=current_branch,
                    current_sha=current_sha,
                    message=f"Unable to fetch {remote}/{branch}: {_git_failure_message(['fetch', remote, branch], fetch_result)}",
                )

        latest_sha = _git_output(["rev-parse", "--verify", remote_ref])
        if current_sha == latest_sha:
            return _update_response(
                state="current",
                current_branch=current_branch,
                current_sha=current_sha,
                latest_sha=latest_sha,
                message=f"Already on the latest {remote}/{branch} commit.",
            )

        if _git_is_ancestor(current_sha, latest_sha):
            return _update_response(
                state="available",
                can_update=True,
                current_branch=current_branch,
                current_sha=current_sha,
                latest_sha=latest_sha,
                message=f"Update available from {current_sha[:7]} to {latest_sha[:7]}.",
            )

        if _git_is_ancestor(latest_sha, current_sha):
            return _update_response(
                state="current",
                current_branch=current_branch,
                current_sha=current_sha,
                latest_sha=latest_sha,
                message=f"Local {branch} is ahead of {remote}/{branch}; no update is required.",
            )

        return _update_response(
            state="blocked",
            current_branch=current_branch,
            current_sha=current_sha,
            latest_sha=latest_sha,
            message=f"Local {branch} and {remote}/{branch} have diverged. Resolve Git history manually.",
        )
    except subprocess.TimeoutExpired:
        return _update_response(
            state="error",
            current_branch=current_branch,
            current_sha=current_sha,
            latest_sha=latest_sha,
            message="Timed out while checking for updates.",
        )
    except Exception as exc:
        return _update_response(
            state="error",
            current_branch=current_branch,
            current_sha=current_sha,
            latest_sha=latest_sha,
            message=str(exc),
        )


def _is_loopback_host(host: str | None) -> bool:
    value = str(host or "").strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _is_unspecified_host(host: str | None) -> bool:
    value = str(host or "").strip().lower()
    if value in {"", "*"}:
        return True
    try:
        return ipaddress.ip_address(value).is_unspecified
    except ValueError:
        return False


def _is_configured_bind_host(host: str | None) -> bool:
    bind_host = str(SERVER_CONFIG.get("host") or "").strip().lower()
    if _is_unspecified_host(bind_host):
        return False

    value = str(host or "").strip().lower()
    if value == bind_host:
        return True

    try:
        return ipaddress.ip_address(value) == ipaddress.ip_address(bind_host)
    except ValueError:
        return False


def _is_local_update_host(host: str | None) -> bool:
    return _is_loopback_host(host) or _is_configured_bind_host(host)


def _require_local_update_request(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    if not _is_local_update_host(client_host):
        raise HTTPException(status_code=403, detail="Updates can only be started from the local machine.")


def _spawn_restart_helper(*, old_pid: int, host: str, port: int) -> None:
    command = [
        sys.executable,
        "-m",
        "src.restart_server",
        "--root",
        str(ROOT_DIR),
        "--old-pid",
        str(old_pid),
        "--host",
        host,
        "--port",
        str(port),
        "--app",
        "src.web_app:app",
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
    )


def _schedule_process_exit(delay_seconds: float = 0.5) -> None:
    timer = threading.Timer(delay_seconds, lambda: os._exit(0))
    timer.daemon = True
    timer.start()


def apply_available_update() -> dict[str, Any]:
    status = get_update_status(fetch=True)
    if not status.get("can_update"):
        raise HTTPException(status_code=409, detail=status)

    remote = str(status["target_remote"])
    branch = str(status["target_branch"])
    previous_sha = str(status["current_sha"])

    pull_result = _run_git(["pull", "--ff-only", remote, branch], timeout=GIT_PULL_TIMEOUT_SECONDS)
    if pull_result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=_update_response(
                state="error",
                current_branch=str(status.get("current_branch") or ""),
                current_sha=previous_sha,
                latest_sha=str(status.get("latest_sha") or ""),
                message=f"Unable to pull {remote}/{branch}: {_git_failure_message(['pull', '--ff-only', remote, branch], pull_result)}",
            ),
        )

    current_sha = _git_output(["rev-parse", "HEAD"])
    _spawn_restart_helper(
        old_pid=os.getpid(),
        host=str(SERVER_CONFIG["host"]),
        port=int(SERVER_CONFIG["port"]),
    )
    return {
        "state": "restarting",
        "previous_sha": previous_sha,
        "current_sha": current_sha,
        "target_remote": remote,
        "target_branch": branch,
        "message": f"Updated to {current_sha[:7]}. Restarting server.",
    }


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
        payload = _index_store(db_dir).list_records(offset=offset, limit=limit, search=search)
    rows = _with_index_override_metadata_for_rows(list(payload.get("rows") or []), db_dir=db_dir)
    payload["rows"] = _with_index_assets_for_rows(rows)
    return payload


def _index_records_snapshot(db_dir: Path | None = None) -> tuple[list[dict[str, Any]], str, int]:
    resolved_db_dir = Path(db_dir or DB_DIR)
    store = _index_store(resolved_db_dir)
    if not store.exists():
        raise FileNotFoundError(f"LanceDB table not found at {lancedb_path(resolved_db_dir) / 'chunks'}")
    cache_key = str(resolved_db_dir.resolve())
    signature = _file_signature(
        store.table_version_hint_path(),
        resolved_db_dir / INDEX_MANIFEST_FILENAME,
        index_overrides_path(resolved_db_dir),
    )
    cached = _INDEX_RECORDS_SNAPSHOT_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        rows, model, dim = cached[1]
        return [dict(row) for row in rows], model, dim
    count = store.count()
    if count <= 0:
        model, dim = store.metadata()
        result = ([], model, dim)
        _INDEX_RECORDS_SNAPSHOT_CACHE[cache_key] = (signature, result)
        return result
    payload = store.list_records(offset=0, limit=count, search="")
    result = (
        list(payload.get("rows") or []),
        str(payload.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
        int(payload.get("embedding_dim") or 768),
    )
    _INDEX_RECORDS_SNAPSHOT_CACHE[cache_key] = (signature, result)
    return [dict(row) for row in result[0]], result[1], result[2]


def _is_index_summary(row: dict[str, Any]) -> bool:
    return str(row.get("node_type") or "chunk") in INDEX_SUMMARY_NODE_TYPES


def _index_top_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc_summaries = [row for row in rows if str(row.get("node_type") or "") == "document_summary"]
    if doc_summaries:
        return doc_summaries

    summary_ids = {str(row.get("id") or "") for row in rows if _is_index_summary(row)}
    root_summaries = [
        row
        for row in rows
        if _is_index_summary(row) and str(row.get("parent_id") or "") not in summary_ids
    ]
    if root_summaries:
        return root_summaries

    root_chunks = [row for row in rows if not str(row.get("parent_id") or "")]
    return root_chunks or rows


def _index_descends_from(row: dict[str, Any], parent_id: str, by_id: dict[str, dict[str, Any]]) -> bool:
    current_id = str(row.get("parent_id") or "")
    seen: set[str] = set()
    while current_id and current_id not in seen:
        if current_id == parent_id:
            return True
        seen.add(current_id)
        current_id = str(by_id.get(current_id, {}).get("parent_id") or "")
    return False


def _index_child_candidates(
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_id = str(parent.get("id") or "")
    if not parent_id:
        return []

    if str(parent.get("node_type") or "") == "document_summary":
        doc_id = str(parent.get("doc_id") or "")
        return [
            row
            for row in rows
            if str(row.get("id") or "") != parent_id
            and str(row.get("doc_id") or "") == doc_id
        ]

    return [
        row
        for row in rows
        if str(row.get("id") or "") != parent_id
        and _index_descends_from(row, parent_id, by_id)
    ]


def _index_section_depth(row: dict[str, Any], parent: dict[str, Any]) -> int:
    parent_parts = [part.strip() for part in str(parent.get("section_path") or "").split(">") if part.strip()]
    row_parts = [part.strip() for part in str(row.get("section_path") or "").split(">") if part.strip()]
    if parent_parts and row_parts[: len(parent_parts)] == parent_parts:
        return max(1, len(row_parts) - len(parent_parts))
    return 1


def _index_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    node_type = str(row.get("node_type") or "chunk")
    node_order = {"document_summary": 0, "section_summary": 1, "chunk": 2}.get(node_type, 3)
    chunk_index = row.get("chunk_index")
    try:
        chunk_number = int(chunk_index)
    except (TypeError, ValueError):
        chunk_number = -1
    return (
        str(row.get("source_pdf_name") or row.get("file_path") or ""),
        str(row.get("section_path") or ""),
        node_order,
        chunk_number,
        str(row.get("id") or ""),
    )


def _with_index_hierarchy_metadata(
    row: dict[str, Any],
    *,
    children: list[dict[str, Any]],
    parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = dict(row)
    item["child_count"] = len(children)
    item["summary_count"] = sum(1 for child in children if _is_index_summary(child))
    item["detail_count"] = sum(1 for child in children if str(child.get("node_type") or "") == "chunk")
    item["node_level"] = 0 if parent is None else min(_index_section_depth(row, parent), 6)
    return item


def list_index_summary_rows(
    *,
    offset: int = 0,
    limit: int = 20,
    search: str = "",
    db_dir: Path | None = None,
) -> dict[str, Any]:
    offset = max(0, int(offset))
    resolved_limit = None if int(limit) <= 0 else min(max(1, int(limit)), 200)
    query = str(search or "").strip()

    with INDEX_LOCK:
        rows, embedding_model, embedding_dim = _index_records_snapshot(db_dir)

    by_id = {str(row.get("id") or ""): row for row in rows if row.get("id")}
    top_rows = _index_top_summary_rows(rows)
    if query:
        top_rows = [
            row
            for row in top_rows
            if record_matches(row, query)
            or any(record_matches(child, query) for child in _index_child_candidates(row, rows, by_id))
        ]
    top_rows = sorted(top_rows, key=_index_sort_key)
    page = _page_slice(top_rows, offset=offset, limit=resolved_limit)
    page_rows = [
        _with_index_hierarchy_metadata(
            row,
            children=[
                child
                for child in _index_child_candidates(row, rows, by_id)
                if not query or record_matches(child, query)
            ],
        )
        for row in page["rows"]
    ]
    page_rows = _with_index_override_metadata_for_rows(page_rows, db_dir=db_dir)
    page_rows = _with_index_assets_for_rows(page_rows)
    return {
        "offset": page["offset"],
        "limit": page["limit"],
        "total": page["total"],
        "rows": page_rows,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "view": "hierarchy",
    }


def list_index_child_rows(
    *,
    parent_id: str,
    offset: int = 0,
    limit: int = INDEX_CHILD_DEFAULT_LIMIT,
    search: str = "",
    db_dir: Path | None = None,
) -> dict[str, Any]:
    parent_id = str(parent_id or "").strip()
    if not parent_id:
        raise KeyError(parent_id)
    offset = max(0, int(offset))
    resolved_limit = min(max(1, int(limit)), INDEX_CHILD_MAX_LIMIT)
    query = str(search or "").strip()

    with INDEX_LOCK:
        rows, embedding_model, embedding_dim = _index_records_snapshot(db_dir)

    by_id = {str(row.get("id") or ""): row for row in rows if row.get("id")}
    parent = by_id.get(parent_id)
    if parent is None:
        raise KeyError(parent_id)

    children = _index_child_candidates(parent, rows, by_id)
    if query:
        children = [row for row in children if record_matches(row, query)]
    children = sorted(children, key=_index_sort_key)
    page = _page_slice(children, offset=offset, limit=resolved_limit)
    return {
        "parent_id": parent_id,
        "offset": page["offset"],
        "limit": page["limit"],
        "total": page["total"],
        "rows": _with_index_assets_for_rows(_with_index_override_metadata_for_rows([
            _with_index_hierarchy_metadata(row, children=[], parent=parent)
            for row in page["rows"]
        ], db_dir=db_dir)),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "view": "hierarchy_children",
    }


def _index_stream_batch_size(value: int) -> int:
    return min(max(1, int(value)), INDEX_STREAM_MAX_BATCH_SIZE)


def iter_index_row_events(
    *,
    batch_size: int = INDEX_STREAM_DEFAULT_BATCH_SIZE,
    search: str = "",
    db_dir: Path | None = None,
) -> Iterator[dict[str, Any]]:
    batch_size = _index_stream_batch_size(batch_size)
    search = str(search or "")
    store = _index_store(db_dir)
    if not store.exists():
        raise FileNotFoundError(f"LanceDB table not found at {lancedb_path(db_dir or DB_DIR) / 'chunks'}")

    total = None if search else store.count()
    embedding_model, embedding_dim = store.metadata()
    yield {
        "type": "metadata",
        "batch_size": batch_size,
        "total": total,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }

    received = 0
    for rows in store.iter_record_batches(
        batch_size=batch_size,
        search=search,
        workers=INDEX_STREAM_WORKERS,
    ):
        received += len(rows)
        yield {
            "type": "rows",
            "rows": _with_index_assets_for_rows(_with_index_override_metadata_for_rows(rows, db_dir=db_dir)),
            "received": received,
        }

    yield {
        "type": "done",
        "received": received,
        "total": received if search else total,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }


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

    resolved_db_dir = Path(db_dir or DB_DIR)
    with INDEX_LOCK:
        store = _index_store(resolved_db_dir)
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

    with INDEX_LOCK:
        store = _index_store(resolved_db_dir)
        row = store.update_record(
            record_id=record_id,
            content=content,
            vector=vector.tolist(),
            embedding_model=model,
            embedding_dim=embedding_dim,
        )
        persist_index_edit(resolved_db_dir, record, content)
        manifest_model, manifest_dim = store.metadata()
        write_index_manifest(
            resolved_db_dir,
            store.all_records(),
            embedding_model=manifest_model,
            embedding_dim=manifest_dim,
        )
        row["edited"] = True
        return row


def delete_index_records(*, record_ids: list[str], db_dir: Path | None = None) -> dict[str, Any]:
    ids = {record_id for record_id in record_ids if record_id}
    if not ids:
        raise ValueError("At least one record ID is required.")

    with INDEX_LOCK:
        resolved_db_dir = Path(db_dir or DB_DIR)
        store = _index_store(resolved_db_dir)
        records_to_tombstone: list[dict[str, Any]] = []
        for record_id in sorted(ids):
            try:
                records_to_tombstone.append(store.get_record(record_id))
            except KeyError:
                continue
        result = store.delete_records(record_ids=list(ids))
        persist_index_deletions(resolved_db_dir, records_to_tombstone)
        model, dim = store.metadata()
        write_index_manifest(
            resolved_db_dir,
            store.all_records(),
            embedding_model=model,
            embedding_dim=dim,
        )
        return result


def vector_search_index_rows(
    *,
    query: str,
    relevance_floor: float = DEFAULT_INDEX_VECTOR_RELEVANCE_FLOOR,
    embedding_model: str | None = None,
    embedding_batch_size: int | None = None,
    embedding_timeout: float | None = None,
    db_dir: Path | None = None,
) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        raise ValueError("Vector search query cannot be empty.")
    relevance_floor = min(max(0.0, float(relevance_floor)), 1.0)

    from src.local_rag import LocalQueryEngine

    engine = LocalQueryEngine(
        working_dir=str(db_dir or DB_DIR),
        asset_dir=str(ASSET_DIR),
        trust_path=str(DOCUMENT_TRUST_PATH),
        embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        embedding_batch_size=embedding_batch_size,
        embedding_timeout=embedding_timeout,
        retrieval_candidate_k=DEFAULT_RETRIEVAL_CANDIDATE_K,
        retrieval_min_score=relevance_floor,
        retrieval_relative_cutoff=DEFAULT_RETRIEVAL_RELATIVE_CUTOFF,
        context_token_fraction=DEFAULT_CONTEXT_TOKEN_FRACTION,
        web_search_enabled=False,
        progress_enabled=False,
    )
    result = engine.search_local_context(query=query, relevance_floor=relevance_floor)
    rows: list[dict[str, Any]] = []
    for item in result.get("results") or []:
        if not isinstance(item, dict):
            continue
        record_id = str(item.get("chunk_id") or "")
        if not record_id:
            continue
        try:
            row = record_row(engine.store.get_record(record_id))
        except Exception:
            row = {
                "id": record_id,
                "content": str(item.get("content") or ""),
                "node_type": "chunk",
                "file_path": str(item.get("location") or ""),
                "chunk_index": "",
            }
        row["score"] = float(item.get("score") or 0.0)
        row["vector_score"] = float(item.get("vector_score") or 0.0)
        row["lexical_score"] = float(item.get("lexical_score") or 0.0)
        row["hybrid_score"] = float(item.get("hybrid_score") or 0.0)
        row["reliability_modifier"] = float(item.get("reliability_modifier") or 0.0)
        row["source_group"] = str(item.get("source_group") or SOURCE_GROUP_UNGROUPED)
        row["citation"] = str(item.get("citation") or "")
        row["source_id"] = str(item.get("source_id") or "")
        row["location"] = str(item.get("location") or "")
        row["vector_query"] = query
        rows.append(row)

    rows = _with_index_override_metadata_for_rows(rows, db_dir=db_dir)
    rows = [_with_index_assets(row, asset_store=getattr(engine, "asset_store", None)) for row in rows]

    return {
        "query": query,
        "relevance_floor": relevance_floor,
        "rows": rows,
        "total": len(rows),
        "tool_result": result,
    }


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


def _load_trust_registry(path: Path | None = None) -> dict[str, Any]:
    trust_path = path or DOCUMENT_TRUST_PATH
    payload = _cached_json_load(trust_path, default={"version": 1, "documents": {}})
    payload.setdefault("version", 1)
    payload.setdefault("documents", {})
    if not isinstance(payload["documents"], dict):
        payload["documents"] = {}
    return payload


def _write_trust_registry(payload: dict[str, Any], path: Path | None = None) -> None:
    trust_path = path or DOCUMENT_TRUST_PATH
    trust_path.parent.mkdir(parents=True, exist_ok=True)
    trust_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _default_trust_entry(source_hash: str) -> dict[str, Any]:
    return {
        "source_hash": source_hash,
        "review_status": "unreviewed",
        "source_type": "unknown",
        "source_group": SOURCE_GROUP_UNGROUPED,
        "reliability_weight": source_group_weight(SOURCE_GROUP_UNGROUPED),
        "publication_year": None,
        "expires_at": "",
        "reviewed_by": "",
        "reviewed_at": "",
        "notes": "",
        "updated_at": "",
    }


def _normalize_trust_entry(source_hash: str, entry: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = _default_trust_entry(source_hash)
    if isinstance(entry, dict):
        normalized.update(
            {
                key: entry.get(key, value)
                for key, value in normalized.items()
                if key != "reliability_weight"
            }
        )
    normalized["source_hash"] = source_hash
    for key in (
        "review_status",
        "source_type",
        "source_group",
        "expires_at",
        "reviewed_by",
        "reviewed_at",
        "notes",
        "updated_at",
    ):
        normalized[key] = str(normalized.get(key) or "").strip()
    normalized["reviewed_by"] = normalized["reviewed_by"][:80]
    if normalized["review_status"] not in TRUST_REVIEW_STATUSES:
        normalized["review_status"] = "unreviewed"
    if normalized["source_type"] not in TRUST_SOURCE_TYPES:
        normalized["source_type"] = "unknown"
    normalized["source_group"] = normalize_source_group(normalized.get("source_group"))
    normalized["reliability_weight"] = source_group_weight(normalized["source_group"])
    try:
        year = normalized.get("publication_year")
        normalized["publication_year"] = int(year) if year not in {None, ""} else None
    except (TypeError, ValueError):
        normalized["publication_year"] = None
    return normalized


def _parse_expiry(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if "T" not in text:
            text = f"{text}T23:59:59+00:00"
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _trust_warnings(trust: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    status = str(trust.get("review_status") or "unreviewed")
    if status == "unreviewed":
        warnings.append("unreviewed_source")
    if status == "rejected":
        warnings.append("rejected_source")
    if status == "stale":
        warnings.append("marked_stale")
    expires_at = _parse_expiry(str(trust.get("expires_at") or ""))
    if expires_at is not None and expires_at < datetime.now(timezone.utc):
        warnings.append("review_expired")
    return warnings


def update_document_trust(
    source_hash: str,
    updates: dict[str, Any],
    *,
    trust_path: Path | None = None,
) -> dict[str, Any]:
    if not source_hash:
        raise ValueError("source_hash is required.")
    with TRUST_LOCK:
        payload = _load_trust_registry(trust_path)
        documents = payload.setdefault("documents", {})
        current = _normalize_trust_entry(source_hash, documents.get(source_hash))
        next_entry = dict(current)
        if "review_status" in updates and updates["review_status"] not in TRUST_REVIEW_STATUSES:
            choices = ", ".join(sorted(TRUST_REVIEW_STATUSES))
            raise ValueError(f"review_status must be one of: {choices}")
        if "source_type" in updates and updates["source_type"] not in TRUST_SOURCE_TYPES:
            choices = ", ".join(sorted(TRUST_SOURCE_TYPES))
            raise ValueError(f"source_type must be one of: {choices}")
        if "source_group" in updates and updates["source_group"] not in TRUST_SOURCE_GROUPS:
            choices = ", ".join(sorted(TRUST_SOURCE_GROUPS))
            raise ValueError(f"source_group must be one of: {choices}")
        for key in (
            "review_status",
            "source_type",
            "source_group",
            "publication_year",
            "expires_at",
            "reviewed_by",
            "notes",
        ):
            if key in updates:
                next_entry[key] = updates[key]
        next_entry = _normalize_trust_entry(source_hash, next_entry)
        if "review_status" in updates:
            next_entry["reviewed_at"] = _utcnow()
        elif next_entry["review_status"] == "approved" and not next_entry.get("reviewed_at"):
            next_entry["reviewed_at"] = _utcnow()
        next_entry["updated_at"] = _utcnow()
        documents[source_hash] = next_entry
        _write_trust_registry(payload, trust_path)
        return dict(next_entry)


def _resolve_workspace_path(raw_path: str, *, root_dir: Path = ROOT_DIR) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    try:
        path = Path(text)
        return path if path.is_absolute() else root_dir / path
    except (OSError, ValueError):
        return None


_UNSET_SENTINEL: Any = object()
_JSON_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}


def _file_signature(*paths: Path) -> str:
    """Cheap opaque ETag seed derived from the (mtime_ns, size) of the given files.

    Used to short-circuit semi-static GET endpoints (PDF list, index rows) with a
    conditional 304 so the expensive rebuild behind them only runs when the
    underlying registry/manifest/source-map files actually changed.
    """
    parts: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            parts.append(f"{path}:missing")
            continue
        parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _signature_from_parts(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _scoped_signature(seed: str, **params: Any) -> str:
    param_parts = [f"{key}={params[key]}" for key in sorted(params)]
    return _signature_from_parts(seed, *param_parts)


def _payload_signature(payload: Any, *parts: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return _signature_from_parts(*parts, serialized)


def _not_modified_or_etag(request: Request, seed: str) -> Response | None:
    """Return a 304 response if the client already holds this ETag, else None."""
    etag = f'"{seed}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
    return None


def _etagged_json(request: Request, payload: Any, seed: str) -> JSONResponse:
    etag = f'"{seed}"'
    return JSONResponse(content=payload, headers={"ETag": etag, "Cache-Control": "no-cache"})


def _conditional_json(request: Request, payload: Any, seed: str) -> Response:
    if not_modified := _not_modified_or_etag(request, seed):
        return not_modified
    return _etagged_json(request, payload, seed)


def _cached_json_load(
    path: Path,
    default: Any,
    *,
    on_decode_error: Any = _UNSET_SENTINEL,
) -> Any:
    """Load and JSON-parse ``path``, cached on its ``(mtime_ns, size)`` signature.

    The PDF/trust/registry/manifest files are read on every PDF-listing request
    but only change when ingestion/indexing writes them. Re-parsing the same
    payload every request is pure waste, so we keep a small memo keyed on the
    filesystem signature and only re-read when the file actually changes.
    """
    try:
        stat = path.stat()
    except OSError:
        return default
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _JSON_CACHE.get(str(path))
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = on_decode_error if on_decode_error is not _UNSET_SENTINEL else default
        _JSON_CACHE[str(path)] = (signature, value)
        return value
    value = payload if isinstance(payload, dict) else default
    _JSON_CACHE[str(path)] = (signature, value)
    return value


def _load_index_manifest(db_dir: Path | None = None) -> dict[str, Any]:
    resolved_db_dir = Path(db_dir or DB_DIR)
    path = resolved_db_dir / INDEX_MANIFEST_FILENAME
    if not path.exists():
        return _derive_index_manifest(resolved_db_dir)
    payload = _cached_json_load(path, default=None)
    if payload is None:
        return _derive_index_manifest(resolved_db_dir)
    return payload


def _derive_index_manifest(db_dir: Path) -> dict[str, Any]:
    try:
        store = _index_store(db_dir)
        if not store.exists():
            return {}
        model, dim = store.metadata()
        records: list[dict[str, Any]] = []
        for batch in store.iter_record_batches(batch_size=INDEX_STREAM_DEFAULT_BATCH_SIZE):
            records.extend(batch)
        return write_index_manifest(db_dir, records, embedding_model=model, embedding_dim=dim)
    except Exception:
        return {}


def _index_manifest_stats(
    entry: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    documents = manifest.get("documents", {}) if isinstance(manifest.get("documents"), dict) else {}
    keys = [
        str(entry.get("hash") or ""),
        str(entry.get("source_pdf_path") or ""),
        str(entry.get("upload_path") or ""),
    ]
    for key in keys:
        stats = documents.get(key)
        if isinstance(stats, dict):
            return dict(stats)
    return {}


_MARKDOWN_QUALITY_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def _markdown_quality(processed_markdown_path: str, *, root_dir: Path = ROOT_DIR) -> dict[str, Any]:
    path = _resolve_workspace_path(processed_markdown_path, root_dir=root_dir)
    if path is None:
        return {"markdown_exists": False, "markdown_char_count": 0, "page_markers": 0, "enrichment_markers": 0}
    try:
        stat = path.stat()
    except OSError:
        return {"markdown_exists": False, "markdown_char_count": 0, "page_markers": 0, "enrichment_markers": 0}
    signature = (stat.st_mtime_ns, stat.st_size)
    cache_key = str(path)
    cached = _MARKDOWN_QUALITY_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return dict(cached[1])
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {"markdown_exists": False, "markdown_char_count": 0, "page_markers": 0, "enrichment_markers": 0}
    result = {
        "markdown_exists": True,
        "markdown_char_count": len(text.strip()),
        "page_markers": len(re.findall(r"(?m)^##\s+Page\s+\d+", text)),
        "enrichment_markers": text.count("[Vision Analysis]") + text.count("[Page Image Analysis]"),
        "table_markers": text.count("|"),
        "equation_markers": len(re.findall(r"(?<!\\)\$[^$\n]{1,160}(?<!\\)\$", text)),
    }
    _MARKDOWN_QUALITY_CACHE[cache_key] = (signature, dict(result))
    return result


def _document_quality(
    entry: dict[str, Any],
    *,
    manifest: dict[str, Any],
    trust: dict[str, Any],
    root_dir: Path = ROOT_DIR,
) -> dict[str, Any]:
    stats = _index_manifest_stats(entry, manifest)
    markdown = _markdown_quality(str(entry.get("processed_markdown_path") or ""), root_dir=root_dir)
    status = str(entry.get("status") or "")
    warnings: list[str] = []
    if not markdown["markdown_exists"]:
        warnings.append("missing_markdown")
    elif int(markdown["markdown_char_count"]) < 500:
        warnings.append("low_extracted_text")
    if status != "indexed":
        warnings.append("not_indexed")
    if status == "interrupted" or str(entry.get("last_interrupted_at") or ""):
        warnings.append("job_interrupted")
    if not stats:
        warnings.append("missing_index_manifest")
    elif int(stats.get("chunk_count") or 0) <= 0:
        warnings.append("no_chunks")
    warnings.extend(_trust_warnings(trust))

    if not warnings:
        label = "ready"
    elif "not_indexed" in warnings or "no_chunks" in warnings or "rejected_source" in warnings:
        label = "not_ready"
    else:
        label = "review"

    return {
        "label": label,
        "warnings": warnings,
        "record_count": int(stats.get("record_count") or 0),
        "chunk_count": int(stats.get("chunk_count") or 0),
        "summary_count": int(stats.get("summary_count") or 0),
        "content_char_count": int(stats.get("content_char_count") or 0),
        "page_start": int(stats.get("page_start") or 0),
        "page_end": int(stats.get("page_end") or 0),
        **markdown,
    }


def _pdf_entry_for_response(
    *,
    source_hash: str,
    filename: str,
    status: str = "",
    upload_path: str = "",
    source_pdf_path: str = "",
    processed_markdown_path: str = "",
    updated_at: str = "",
    last_interrupted_at: str = "",
    last_interrupted_job_id: str = "",
    last_error: str = "",
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
        "last_interrupted_at": last_interrupted_at,
        "last_interrupted_job_id": last_interrupted_job_id,
        "last_error": last_error,
        "can_download": can_download,
        "download_url": _pdf_download_url(source_hash) if can_download else "",
        "path_error": path_error,
    }


def list_pdf_documents(
    *,
    search: str = "",
    offset: int = 0,
    limit: int | None = None,
    registry_path: Path | None = None,
    processed_dir: Path | None = None,
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> dict[str, Any]:
    registry_path = registry_path or PDF_REGISTRY_PATH
    processed_dir = processed_dir or PROCESSED_DIR
    entries: dict[str, dict[str, Any]] = {}
    manifest = _load_index_manifest(DB_DIR)
    trust_payload = _load_trust_registry()
    trust_documents = trust_payload.get("documents", {}) if isinstance(trust_payload.get("documents"), dict) else {}

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
            last_interrupted_at=str(entry.get("last_interrupted_at", "")),
            last_interrupted_job_id=str(entry.get("last_interrupted_job_id", "")),
            last_error=str(entry.get("last_error", "")),
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
            last_interrupted_at=str(current.get("last_interrupted_at", "")),
            last_interrupted_job_id=str(current.get("last_interrupted_job_id", "")),
            last_error=str(current.get("last_error", "")),
            root_dir=root_dir,
            data_dir=data_dir,
        )

    rows = sorted(entries.values(), key=lambda item: (item.get("filename", ""), item.get("hash", "")))
    for item in rows:
        if str(item.get("status") or "") == "indexed" and not _index_manifest_stats(item, manifest):
            item["status"] = "not_indexed"
        trust = _normalize_trust_entry(str(item.get("hash") or ""), trust_documents.get(str(item.get("hash") or "")))
        item["trust"] = trust
        item["quality"] = _document_quality(item, manifest=manifest, trust=trust, root_dir=root_dir)
    rows.sort(
        key=lambda item: (
            0 if str(item.get("trust", {}).get("source_group") or "") == SOURCE_GROUP_UNGROUPED else 1,
            str(item.get("filename") or ""),
            str(item.get("hash") or ""),
        )
    )
    query = search.strip().lower()
    if query:
        rows = [
            item
            for item in rows
            if query in str(item.get("filename", "")).lower()
            or query in str(item.get("hash", "")).lower()
        ]
    page = _page_slice(rows, offset=offset, limit=limit)
    return {
        "pdfs": page["rows"],
        "total": page["total"],
        "offset": page["offset"],
        "limit": page["limit"],
    }


def list_job_rows(*, offset: int = 0, limit: int | None = 10, search: str = "") -> dict[str, Any]:
    jobs = job_queue.list_jobs()
    active_count = sum(
        1
        for job in jobs
        if str(job.get("status") or "") in {"queued", "running", "paused_for_queries"}
    )
    query = search.strip().lower()
    if query:
        jobs = [
            job
            for job in jobs
            if query in str(job.get("id", "")).lower()
            or query in str(job.get("status", "")).lower()
            or query in str(job.get("phase", "")).lower()
            or query in str(job.get("error", "")).lower()
            or query in ", ".join(str(name) for name in (job.get("filenames") or [])).lower()
        ]
    page = _page_slice(jobs, offset=offset, limit=limit)
    return {
        "jobs": page["rows"],
        "total": page["total"],
        "offset": page["offset"],
        "limit": page["limit"],
        "active_count": active_count,
    }


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


def _pdf_document_by_hash(
    source_hash: str,
    *,
    registry_path: Path | None = None,
    processed_dir: Path | None = None,
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    documents = list_pdf_documents(
        search="",
        offset=0,
        limit=None,
        registry_path=registry_path,
        processed_dir=processed_dir,
        root_dir=root_dir,
        data_dir=data_dir,
    )["pdfs"]
    match = next((item for item in documents if item.get("hash") == source_hash), None)
    if match is None:
        raise FileNotFoundError(f"PDF not found for source hash: {source_hash}")
    return match, documents


def _legacy_delete_identifiers(
    source_hash: str,
    document: dict[str, Any],
    source_entries: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    legacy_paths: list[str] = []
    legacy_doc_ids: list[str] = []
    for value in (
        document.get("processed_markdown_path"),
        document.get("source_pdf_path"),
        document.get("upload_path"),
        document.get("filename"),
    ):
        text = str(value or "")
        if text:
            legacy_paths.append(text)
    for entry in source_entries:
        for key in ("processed_markdown_path", "source_pdf_path", "source_pdf_name"):
            value = str(entry.get(key, ""))
            if value:
                legacy_paths.append(value)
        markdown_path = Path(str(entry.get("processed_markdown_path", "")))
        if markdown_path.name:
            from src.sectioning import stable_id

            legacy_doc_ids.append(stable_id("doc", markdown_path.stem))
    return sorted(set(legacy_paths)), sorted(set(legacy_doc_ids))


def _source_entries_for_hash(processed_dir: Path, source_hash: str) -> list[dict[str, Any]]:
    documents = load_source_map(processed_dir).get("documents", {})
    return [
        dict(entry)
        for entry in documents.values()
        if isinstance(entry, dict) and str(entry.get("source_hash") or "") == source_hash
    ]


def _delete_source_vectors(
    source_hash: str,
    *,
    document: dict[str, Any],
    source_entries: list[dict[str, Any]],
    db_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_db_dir = Path(db_dir or DB_DIR)
    legacy_paths, legacy_doc_ids = _legacy_delete_identifiers(source_hash, document, source_entries)
    with INDEX_LOCK:
        store = _index_store(resolved_db_dir)
        if not store.exists():
            return {"deleted": 0, "remaining": 0}
        result = store.delete_records_by_source_hash(
            source_hashes=[source_hash],
            legacy_file_paths=legacy_paths,
            legacy_doc_ids=legacy_doc_ids,
        )
        model, dim = store.metadata()
        write_index_manifest(
            resolved_db_dir,
            store.all_records(),
            embedding_model=model,
            embedding_dim=dim,
        )
        return result


def _delete_processed_markdown_files(
    source_entries: list[dict[str, Any]],
    *,
    processed_dir: Path,
    root_dir: Path = ROOT_DIR,
) -> int:
    deleted = 0
    processed_root = processed_dir.resolve()
    paths: set[Path] = set()
    for entry in source_entries:
        path = _resolve_workspace_path(str(entry.get("processed_markdown_path", "")), root_dir=root_dir)
        if path is not None:
            paths.add(path)
    for path in paths:
        try:
            resolved = path.resolve()
            resolved.relative_to(processed_root)
        except (OSError, ValueError):
            continue
        if not resolved.exists():
            continue
        try:
            resolved.unlink()
        except PermissionError:
            resolved.write_text("", encoding="utf-8")
        deleted += 1
    return deleted


def _remove_document_trust(source_hash: str, *, trust_path: Path | None = None) -> bool:
    with TRUST_LOCK:
        payload = _load_trust_registry(trust_path)
        documents = payload.setdefault("documents", {})
        removed = documents.pop(source_hash, None)
        if removed is not None:
            _write_trust_registry(payload, trust_path)
            return True
        return False


def _shared_pdf_paths(
    source_hash: str,
    documents: list[dict[str, Any]],
    *,
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> set[Path]:
    shared: set[Path] = set()
    for document in documents:
        if str(document.get("hash") or "") == source_hash:
            continue
        for raw_path in (document.get("upload_path"), document.get("source_pdf_path")):
            try:
                shared.add(_resolve_pdf_path(str(raw_path or ""), root_dir=root_dir, data_dir=data_dir).resolve())
            except (FileNotFoundError, PermissionError, OSError):
                continue
    return shared


def _delete_document_pdf_files(
    document: dict[str, Any],
    *,
    shared_paths: set[Path],
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> tuple[int, list[str]]:
    deleted = 0
    errors: list[str] = []
    candidates: set[Path] = set()
    for raw_path in (document.get("upload_path"), document.get("source_pdf_path")):
        try:
            candidates.add(_resolve_pdf_path(str(raw_path or ""), root_dir=root_dir, data_dir=data_dir).resolve())
        except (FileNotFoundError, PermissionError, OSError):
            continue
    for path in candidates:
        if path in shared_paths or not path.exists():
            continue
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        parent = path.parent
        try:
            if _path_is_relative_to(parent, data_dir) and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
    return deleted, errors


def delete_pdf_document(
    source_hash: str,
    *,
    registry_path: Path | None = None,
    processed_dir: Path | None = None,
    db_dir: Path | None = None,
    trust_path: Path | None = None,
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> dict[str, Any]:
    source_hash = str(source_hash or "").strip()
    if not source_hash:
        raise ValueError("source_hash is required.")
    registry_path = registry_path or PDF_REGISTRY_PATH
    processed_dir = processed_dir or PROCESSED_DIR
    trust_path = trust_path or DOCUMENT_TRUST_PATH
    document, documents = _pdf_document_by_hash(
        source_hash,
        registry_path=registry_path,
        processed_dir=processed_dir,
        root_dir=root_dir,
        data_dir=data_dir,
    )
    source_entries = _source_entries_for_hash(processed_dir, source_hash)
    if document.get("processed_markdown_path") and not source_entries:
        source_entries = [
            {
                "source_hash": source_hash,
                "processed_markdown_path": str(document.get("processed_markdown_path") or ""),
                "source_pdf_path": str(document.get("source_pdf_path") or document.get("upload_path") or ""),
                "source_pdf_name": str(document.get("filename") or ""),
            }
        ]
    vector_result = _delete_source_vectors(
        source_hash,
        document=document,
        source_entries=source_entries,
        db_dir=db_dir,
    )
    clear_overrides_for_sources(Path(db_dir or DB_DIR), {source_hash})
    markdown_deleted = _delete_processed_markdown_files(
        source_entries,
        processed_dir=processed_dir,
        root_dir=root_dir,
    )
    from src.asset_store import ImageAssetStore

    assets_deleted = ImageAssetStore(ASSET_DIR).remove_source_assets(source_hash)
    shared_paths = _shared_pdf_paths(source_hash, documents, root_dir=root_dir, data_dir=data_dir)
    pdfs_deleted, pdf_delete_errors = _delete_document_pdf_files(
        document,
        shared_paths=shared_paths,
        root_dir=root_dir,
        data_dir=data_dir,
    )
    removed_source_entries = remove_source_entries_by_hash(processed_dir, {source_hash})
    registry_entry = PdfRegistry(registry_path).delete_source(source_hash)
    trust_deleted = _remove_document_trust(source_hash, trust_path=trust_path)
    return {
        "source_hash": source_hash,
        "deleted": True,
        "vectors": vector_result,
        "markdown_deleted": markdown_deleted,
        "source_map_deleted": len(removed_source_entries),
        "assets_deleted": assets_deleted,
        "pdfs_deleted": pdfs_deleted,
        "pdf_delete_errors": pdf_delete_errors,
        "registry_deleted": registry_entry is not None,
        "trust_deleted": trust_deleted,
    }


def _reprocess_options_for_source(source_hash: str, *, registry_path: Path = PDF_REGISTRY_PATH) -> dict[str, Any]:
    payload = PdfRegistry(registry_path).load()
    entry = payload.get("pdfs", {}).get(source_hash, {})
    options = entry.get("options") if isinstance(entry, dict) else {}
    return _upload_options_from_form(options if isinstance(options, dict) else {})


def enqueue_source_reprocess(
    source_hash: str,
    *,
    registry_path: Path | None = None,
    processed_dir: Path | None = None,
    staging_root: Path | None = None,
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> QueueJob:
    registry_path = registry_path or PDF_REGISTRY_PATH
    processed_dir = processed_dir or PROCESSED_DIR
    staging_root = staging_root or STAGING_DIR
    path, filename = resolve_pdf_download_path(
        source_hash,
        registry_path=registry_path,
        processed_dir=processed_dir,
        root_dir=root_dir,
        data_dir=data_dir,
    )
    actual_hash = sha256_file(path)
    job_id = uuid.uuid4().hex
    staging_dir = staging_root / job_id
    staging_dir.mkdir(parents=True, exist_ok=False)
    staged_name = _safe_filename(filename or path.name)
    staged_path = staging_dir / staged_name
    try:
        shutil.copy2(path, staged_path)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    upload = {
        "filename": staged_name,
        "hash": actual_hash,
        "staging_path": str(staged_path),
    }
    options = _reprocess_options_for_source(source_hash, registry_path=registry_path)
    PdfRegistry(registry_path).register_queued(
        job_id=job_id,
        files=[upload],
        forced_hashes={actual_hash},
        options=options,
    )
    return job_queue.enqueue_upload(
        staging_dir=staging_dir,
        filenames=[staged_name],
        uploads=[upload],
        force_duplicate_hashes=[source_hash],
        job_id=job_id,
        options=options,
    )


def _known_source_hashes(
    *,
    registry_path: Path = PDF_REGISTRY_PATH,
    processed_dir: Path = PROCESSED_DIR,
) -> list[str]:
    """All distinct source hashes known from the upload registry and source map."""
    hashes: list[str] = []
    seen: set[str] = set()
    registry_payload = PdfRegistry(registry_path).load()
    for source_hash, entry in registry_payload.get("pdfs", {}).items():
        source_hash = str(source_hash or "")
        if source_hash and source_hash not in seen and isinstance(entry, dict):
            seen.add(source_hash)
            hashes.append(source_hash)
    for entry in load_source_map(processed_dir).get("documents", {}).values():
        if not isinstance(entry, dict):
            continue
        source_hash = str(entry.get("source_hash", ""))
        if source_hash and source_hash not in seen:
            seen.add(source_hash)
            hashes.append(source_hash)
    return hashes


def enqueue_full_reingest(
    *,
    registry_path: Path | None = None,
    processed_dir: Path | None = None,
    staging_root: Path | None = None,
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> QueueJob:
    """Queue a full PDF -> Markdown -> index re-run for every known source.

    Re-runs ingestion (PDF extraction) and indexing for all registered PDFs in a
    single upload job. Useful after changing OCR/parser/vision settings or when
    extracted Markdown looks wrong across the board. Existing per-source vectors,
    assets, overrides, and source-map entries for each hash are dropped by the
    upload job's forced-duplicate handling before the fresh extraction runs.
    """
    registry_path = registry_path or PDF_REGISTRY_PATH
    processed_dir = processed_dir or PROCESSED_DIR
    staging_root = staging_root or STAGING_DIR

    source_hashes = _known_source_hashes(
        registry_path=registry_path,
        processed_dir=processed_dir,
    )
    if not source_hashes:
        raise FileNotFoundError("No PDFs are registered to re-ingest.")

    job_id = uuid.uuid4().hex
    staging_dir = staging_root / job_id
    staging_dir.mkdir(parents=True, exist_ok=False)
    filenames: list[str] = []
    uploads: list[dict[str, Any]] = []
    used_names: set[str] = set()

    try:
        for source_hash in source_hashes:
            try:
                path, filename = resolve_pdf_download_path(
                    source_hash,
                    registry_path=registry_path,
                    processed_dir=processed_dir,
                    root_dir=root_dir,
                    data_dir=data_dir,
                )
            except FileNotFoundError:
                continue
            staged_name = _safe_filename(filename or path.name)
            base = staged_name
            counter = 1
            while staged_name.lower() in used_names:
                stem = Path(base).stem
                staged_name = f"{stem}-{counter}.pdf"
                counter += 1
            used_names.add(staged_name.lower())
            staged_path = staging_dir / staged_name
            shutil.copy2(path, staged_path)
            filenames.append(staged_name)
            uploads.append(
                {
                    "filename": staged_name,
                    "hash": source_hash,
                    "staging_path": str(staged_path),
                }
            )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    if not uploads:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise FileNotFoundError("No source PDF files could be located to re-ingest.")

    options = _upload_options_from_form({})
    file_uploads = [
        {
            "filename": str(upload["filename"]),
            "hash": str(upload["hash"]),
            "staging_path": str(upload["staging_path"]),
            "source_group": SOURCE_GROUP_UNGROUPED,
        }
        for upload in uploads
    ]
    PdfRegistry(registry_path).register_queued(
        job_id=job_id,
        files=file_uploads,
        forced_hashes=set(source_hashes),
        options=options,
    )
    return job_queue.enqueue_upload(
        staging_dir=staging_dir,
        filenames=filenames,
        uploads=file_uploads,
        force_duplicate_hashes=sorted(source_hashes),
        job_id=job_id,
        options=options,
    )


def _source_markdown_path(
    source_hash: str,
    *,
    processed_dir: Path | None = None,
    root_dir: Path = ROOT_DIR,
) -> Path | None:
    """Resolve the processed Markdown file backing a source hash, if any."""
    processed_dir = processed_dir or PROCESSED_DIR
    documents = load_source_map(processed_dir).get("documents", {})
    for entry in documents.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("source_hash", "")) != str(source_hash):
            continue
        path = _resolve_workspace_path(str(entry.get("processed_markdown_path", "")), root_dir=root_dir)
        if path is not None:
            return path
    return None


def enqueue_source_reindex(
    source_hash: str,
    *,
    registry_path: Path | None = None,
    processed_dir: Path | None = None,
    root_dir: Path = ROOT_DIR,
    data_dir: Path = DATA_DIR,
) -> QueueJob:
    """Queue an index-only rebuild for a source using its existing Markdown.

    Unlike ``enqueue_source_reprocess``, ingestion (PDF -> Markdown) is skipped;
    the source's processed Markdown must already exist. Existing vectors for the
    source are dropped by the job so the index is genuinely rebuilt.
    """
    registry_path = registry_path or PDF_REGISTRY_PATH
    processed_dir = processed_dir or PROCESSED_DIR
    markdown_path = _source_markdown_path(source_hash, processed_dir=processed_dir, root_dir=root_dir)
    if markdown_path is None or not markdown_path.exists():
        raise FileNotFoundError(f"No processed Markdown to re-index for source hash: {source_hash}")
    options = _reprocess_options_for_source(source_hash, registry_path=registry_path)
    return job_queue.enqueue_reindex_source(source_hashes=[source_hash], options=options)


QueueJob = import_split_class("src.web_app_classes.queue_job", "QueueJob")
QueueJob.__module__ = __name__


RagJobQueue = import_split_class("src.web_app_classes.rag_job_queue", "RagJobQueue")
RagJobQueue.__module__ = __name__


ChatRequest = import_split_class("src.web_app_classes.chat_request", "ChatRequest")
ChatRequest.__module__ = __name__


IndexUpdateRequest = import_split_class("src.web_app_classes.index_update_request", "IndexUpdateRequest")
IndexUpdateRequest.__module__ = __name__


IndexDeleteRequest = import_split_class("src.web_app_classes.index_delete_request", "IndexDeleteRequest")
IndexDeleteRequest.__module__ = __name__


IndexVectorSearchRequest = import_split_class("src.web_app_classes.index_vector_search_request", "IndexVectorSearchRequest")
IndexVectorSearchRequest.__module__ = __name__


ReindexRequest = import_split_class("src.web_app_classes.reindex_request", "ReindexRequest")
ReindexRequest.__module__ = __name__


RestoreBackupRequest = import_split_class("src.web_app_classes.restore_backup_request", "RestoreBackupRequest")
RestoreBackupRequest.__module__ = __name__


DocumentTrustRequest = import_split_class("src.web_app_classes.document_trust_request", "DocumentTrustRequest")
DocumentTrustRequest.__module__ = __name__


class BulkDocumentTrustRequest(BaseModel):
    source_hashes: list[str] = Field(default_factory=list)
    source_group: str
    reviewed_by: str | None = None


RenderRequest = import_split_class("src.web_app_classes.render_request", "RenderRequest")
RenderRequest.__module__ = __name__


def _model_dump(model: BaseModel) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper()
    return model.dict()


# Lazily constructed, module-level markdown renderer + LaTeX converter.
# ``/api/render`` is hit ~10x/second during chat streaming, so constructing a fresh
# ``MarkdownIt`` and re-resolving the converter import on every call is wasteful.
_MARKDOWN_RENDERER: Any = None
_LATEX_TO_MATHML: Any = None
_MATH_PATTERN = re.compile(
    r"(?s)\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|(?<!\\)\$(?!\s)(.+?)(?<!\s)(?<!\\)\$"
)


def _ensure_markdown_renderer() -> None:
    global _MARKDOWN_RENDERER, _LATEX_TO_MATHML
    if _MARKDOWN_RENDERER is None:
        from latex2mathml.converter import convert as latex_to_mathml
        from markdown_it import MarkdownIt

        _LATEX_TO_MATHML = latex_to_mathml
        _MARKDOWN_RENDERER = MarkdownIt("commonmark", {"html": False, "linkify": False})


def render_markdown_text(text: str) -> str:
    _ensure_markdown_renderer()

    math_blocks: list[str] = []

    def replace_math(match: re.Match[str]) -> str:
        latex = next(group for group in match.groups() if group is not None)
        display = "block" if match.group(1) is not None or match.group(2) is not None else "inline"
        placeholder = f"@@RAG_MATH_{len(math_blocks)}@@"
        try:
            math_blocks.append(_LATEX_TO_MATHML(latex.strip(), display=display))
        except Exception:
            math_blocks.append(html.escape(match.group(0)))
        return placeholder

    protected = _MATH_PATTERN.sub(replace_math, text)
    rendered = _MARKDOWN_RENDERER.render(protected)
    for index, math_html in enumerate(math_blocks):
        rendered = rendered.replace(f"@@RAG_MATH_{index}@@", math_html)
    return rendered


job_queue = RagJobQueue()


def recover_pending_upload_jobs_on_startup() -> dict[str, Any]:
    return job_queue.recover_pending_uploads()


@asynccontextmanager
async def lifespan(app: FastAPI):
    recover_pending_upload_jobs_on_startup()
    yield


app = FastAPI(title="Local FSAE RAG Pipeline", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def root():
    index_path = WEB_DIR / "index.html"
    try:
        text = index_path.read_text(encoding="utf-8")
    except OSError:
        return FileResponse(index_path)
    version = _file_signature(WEB_DIR / "styles.css", WEB_DIR / "app.js", WEB_DIR / "vendor" / "fflate.min.js")
    text = text.replace('/static/styles.css"', f'/static/styles.css?v={version}"')
    text = text.replace('/static/vendor/fflate.min.js"', f'/static/vendor/fflate.min.js?v={version}"')
    text = text.replace('/static/app.js"', f'/static/app.js?v={version}"')
    return Response(content=text, media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health(request: Request):
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
    payload = {
        "ok": True,
        "paths": {
            "data_dir": str(DATA_DIR),
            "upload_dir": str(UPLOAD_DIR),
            "processed_dir": str(PROCESSED_DIR),
            "db_dir": str(DB_DIR),
            "asset_dir": str(ASSET_DIR),
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
    seed = _payload_signature(payload, _file_signature(_index_store().table_version_hint_path(), DB_DIR / INDEX_MANIFEST_FILENAME))
    return _conditional_json(request, payload, seed)


@app.get("/api/update/status")
def update_status():
    return get_update_status(fetch=True)


@app.post("/api/update/apply")
def update_apply(request: Request, background_tasks: BackgroundTasks):
    _require_local_update_request(request)
    response = apply_available_update()
    background_tasks.add_task(_schedule_process_exit)
    return response


@app.post("/api/render")
def render_markdown(payload: RenderRequest):
    return {"html": render_markdown_text(payload.text)}


def _upload_options_from_form(form: Any) -> dict[str, Any]:
    return {
        "asset_dir": str(_resolve_root_path(form.get("asset_dir") or INGESTION_CONFIG["asset_dir"])),
        "parser_mode": str(form.get("parser_mode") or INGESTION_CONFIG["parser_mode"]),
        "accelerator": str(form.get("accelerator") or INGESTION_CONFIG["accelerator"]),
        "asset_triggers": str(form.get("asset_triggers") or INGESTION_CONFIG["asset_triggers"]),
        "code_enrichment": _bool_value(form.get("code_enrichment"), INGESTION_CONFIG["code_enrichment"]),
        "formula_enrichment": _bool_value(
            form.get("formula_enrichment"),
            INGESTION_CONFIG["formula_enrichment"],
        ),
        "vision_model": str(form.get("vision_model") or INGESTION_CONFIG["vision_model"]),
        "vision_enabled": _bool_value(form.get("vision_enabled"), INGESTION_CONFIG["vision_enabled"]),
        "ocr_backend": str(form.get("ocr_backend") or INGESTION_CONFIG["ocr_backend"]),
        "ocr_langs": _string_list(form.get("ocr_langs"), tuple(INGESTION_CONFIG["ocr_langs"])),
        "ocr_force_full_page": _bool_value(
            form.get("ocr_force_full_page"),
            INGESTION_CONFIG["ocr_force_full_page"],
        ),
        "ocr_bitmap_area_threshold": _positive_float(
            form.get("ocr_bitmap_area_threshold"),
            INGESTION_CONFIG["ocr_bitmap_area_threshold"],
        ),
        "rapidocr_backend": str(form.get("rapidocr_backend") or INGESTION_CONFIG["rapidocr_backend"]),
        "tesseract_cmd": str(form.get("tesseract_cmd") or INGESTION_CONFIG["tesseract_cmd"]),
        "tesseract_data_path": str(form.get("tesseract_data_path") or INGESTION_CONFIG["tesseract_data_path"]),
        "tesseract_psm": _optional_int(form.get("tesseract_psm"), INGESTION_CONFIG["tesseract_psm"]),
        "embedding_model": str(form.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
        "embedding_batch_size": _positive_int(form.get("embedding_batch_size"), DEFAULT_EMBEDDING_BATCH_SIZE),
        "embedding_timeout": _positive_float(form.get("embedding_timeout"), DEFAULT_EMBEDDING_TIMEOUT),
        "index_backend": str(form.get("index_backend") or DEFAULT_INDEX_BACKEND),
        "summary_mode": str(form.get("summary_mode") or DEFAULT_SUMMARY_MODE),
        "chunk_target_tokens": _positive_int(form.get("chunk_target_tokens"), DEFAULT_CHUNK_TARGET_TOKENS),
        "chunk_overlap_tokens": _nonnegative_int(form.get("chunk_overlap_tokens"), DEFAULT_CHUNK_OVERLAP_TOKENS),
        "progress_enabled": False,
    }


def _sha256_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_sha256_hash(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", _sha256_text(value)))


def _assign_upload_source_groups(
    uploads: list[dict[str, Any]],
    raw_groups: list[str],
    *,
    require_groups: bool,
) -> None:
    groups = [str(value or "").strip().lower() for value in raw_groups]
    if require_groups and len(groups) != len(uploads):
        raise HTTPException(status_code=400, detail="Each uploaded PDF must include a source group.")
    if not require_groups and groups and len(groups) != len(uploads):
        raise HTTPException(status_code=400, detail="source_groups count must match uploaded files when provided.")

    if not groups:
        for upload in uploads:
            upload["source_group"] = SOURCE_GROUP_UNGROUPED
        return

    valid_choices = ", ".join(sorted(TRUST_SOURCE_GROUPS))
    for upload, group in zip(uploads, groups):
        if group not in TRUST_SOURCE_GROUPS:
            raise HTTPException(status_code=400, detail=f"source_group must be one of: {valid_choices}")
        upload["source_group"] = group


def _apply_upload_source_groups(uploads: list[dict[str, Any]]) -> None:
    for upload in uploads:
        source_hash = str(upload.get("hash") or "")
        if not source_hash:
            continue
        group = normalize_source_group(upload.get("source_group"))
        if group == SOURCE_GROUP_UNGROUPED:
            continue
        update_document_trust(source_hash, {"source_group": group})


def _upload_hashes(files: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("hash", "")) for item in files if item.get("hash")})


def _duplicate_hashes(duplicates: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("hash", "")) for item in duplicates if item.get("hash")})


def _force_upload_token(
    *,
    upload_hashes: list[str],
    duplicate_hashes: list[str],
    now: datetime | None = None,
) -> str:
    issued_at = int((now or datetime.now(timezone.utc)).timestamp())
    payload = {
        "upload_hashes": sorted(upload_hashes),
        "duplicate_hashes": sorted(duplicate_hashes),
        "expires_at": issued_at + FORCE_UPLOAD_TOKEN_TTL_SECONDS,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(UPLOAD_FORCE_TOKEN_SECRET, payload_bytes, hashlib.sha256).digest()
    return ".".join(
        [
            base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("="),
            base64.urlsafe_b64encode(signature).decode("ascii").rstrip("="),
        ]
    )


def _decode_force_token_part(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _force_upload_token_valid(
    token: str,
    *,
    upload_hashes: list[str],
    duplicate_hashes: list[str],
    now: datetime | None = None,
) -> bool:
    try:
        encoded_payload, encoded_signature = str(token or "").split(".", 1)
        payload_bytes = _decode_force_token_part(encoded_payload)
        signature = _decode_force_token_part(encoded_signature)
        expected_signature = hmac.new(UPLOAD_FORCE_TOKEN_SECRET, payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected_signature):
            return False
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return False

    if sorted(payload.get("upload_hashes") or []) != sorted(upload_hashes):
        return False
    if sorted(payload.get("duplicate_hashes") or []) != sorted(duplicate_hashes):
        return False
    expires_at = int(payload.get("expires_at") or 0)
    current_time = int((now or datetime.now(timezone.utc)).timestamp())
    return expires_at >= current_time


def _duplicate_response_detail(
    *,
    message: str,
    duplicates: list[dict[str, Any]],
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "message": message,
        "can_force": True,
        "force_token": _force_upload_token(
            upload_hashes=_upload_hashes(files),
            duplicate_hashes=_duplicate_hashes(duplicates),
        ),
        "force_token_expires_seconds": FORCE_UPLOAD_TOKEN_TTL_SECONDS,
        "duplicates": duplicates,
    }


def _indexed_source_duplicate_entries(
    files: list[dict[str, Any]],
    *,
    known_hashes: set[str],
    processed_dir: Path | None = None,
) -> list[dict[str, Any]]:
    processed_dir = processed_dir or PROCESSED_DIR
    source_documents = load_source_map(processed_dir).get("documents", {})
    indexed_by_hash: dict[str, dict[str, Any]] = {}
    for entry in source_documents.values():
        if not isinstance(entry, dict):
            continue
        source_hash = str(entry.get("source_hash", ""))
        if source_hash and source_hash not in indexed_by_hash:
            indexed_by_hash[source_hash] = entry

    duplicates: list[dict[str, Any]] = []
    for item in files:
        file_hash = str(item.get("hash", ""))
        if not file_hash or file_hash in known_hashes:
            continue
        existing = indexed_by_hash.get(file_hash)
        if not existing:
            continue
        duplicates.append(
            {
                "filename": str(item.get("filename", "")),
                "hash": file_hash,
                "existing_filename": str(existing.get("source_pdf_name", "")),
                "status": "indexed",
                "job_id": "",
            }
        )
    return duplicates


def _vector_store_duplicate_entries(
    files: list[dict[str, Any]],
    *,
    known_hashes: set[str],
    db_dir: Path | None = None,
) -> list[dict[str, Any]]:
    store = _index_store(db_dir)
    if not store.exists():
        return []
    file_hashes = {str(item.get("hash", "")) for item in files if item.get("hash")}
    target_hashes = file_hashes - known_hashes
    if not target_hashes:
        return []

    try:
        rows = store.records_by_source_hash(sorted(target_hashes))
    except Exception:
        return []

    indexed_by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_hash = str(row.get("source_hash", ""))
        if source_hash in target_hashes and source_hash not in indexed_by_hash:
            indexed_by_hash[source_hash] = row

    duplicates: list[dict[str, Any]] = []
    for item in files:
        file_hash = str(item.get("hash", ""))
        if not file_hash or file_hash in known_hashes:
            continue
        existing = indexed_by_hash.get(file_hash)
        if not existing:
            continue
        existing_filename = (
            str(existing.get("source_pdf_name", ""))
            or Path(str(existing.get("source_pdf_path") or existing.get("file_path") or "")).name
        )
        duplicates.append(
            {
                "filename": str(item.get("filename", "")),
                "hash": file_hash,
                "existing_filename": existing_filename,
                "status": "indexed",
                "job_id": "",
                "record_id": str(existing.get("id", "")),
            }
        )
    return duplicates


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


_PDF_HASH_CACHE: dict[str, tuple[tuple[int, int], str]] = {}


def _cached_pdf_hash(path: Path) -> str:
    stat = path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    key = str(path.resolve())
    cached = _PDF_HASH_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = sha256_file(path)
    _PDF_HASH_CACHE[key] = (signature, digest)
    return digest


def _copy_upload_file_to_path(upload: StarletteUploadFile, destination: Path) -> str:
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    upload.file.seek(0)
    with destination.open("wb") as handle:
        for chunk in iter(lambda: upload.file.read(1024 * 1024), b""):
            digest.update(chunk)
            handle.write(chunk)
    upload.file.seek(0)
    return digest.hexdigest()


def _known_data_pdf_paths(
    *,
    data_dir: Path | None = None,
    registry_path: Path | None = None,
    processed_dir: Path | None = None,
) -> set[Path]:
    data_dir = data_dir or DATA_DIR
    registry_path = registry_path or PDF_REGISTRY_PATH
    processed_dir = processed_dir or PROCESSED_DIR
    known_paths: set[Path] = set()

    def add_path(raw_path: Any) -> None:
        text = str(raw_path or "")
        if not text:
            return
        try:
            known_paths.add(_resolve_pdf_path(text, data_dir=data_dir).resolve())
        except (OSError, PermissionError, FileNotFoundError):
            return

    registry_payload = PdfRegistry(registry_path).load()
    for source_hash, entry in registry_payload.get("pdfs", {}).items():
        if not source_hash or not isinstance(entry, dict):
            continue
        add_path(entry.get("upload_path"))
        add_path(entry.get("source_pdf_path"))

    source_map = load_source_map(processed_dir)
    for entry in source_map.get("documents", {}).values():
        if not isinstance(entry, dict) or not entry.get("source_hash"):
            continue
        add_path(entry.get("source_pdf_path"))

    return known_paths


def _data_pdf_duplicate_entries(
    files: list[dict[str, Any]],
    *,
    known_hashes: set[str],
    data_dir: Path | None = None,
    staging_dir: Path | None = None,
) -> list[dict[str, Any]]:
    data_dir = data_dir or DATA_DIR
    staging_dir = staging_dir or STAGING_DIR
    file_hashes = {str(item.get("hash", "")) for item in files if item.get("hash")}
    target_hashes = file_hashes - known_hashes
    if not target_hashes or not data_dir.exists():
        return []

    current_paths: set[Path] = set()
    for item in files:
        raw_path = str(item.get("staging_path", ""))
        if raw_path:
            try:
                current_paths.add(Path(raw_path).resolve())
            except OSError:
                pass

    known_pdf_paths = _known_data_pdf_paths(data_dir=data_dir)
    existing_by_hash: dict[str, Path] = {}
    for candidate in data_dir.rglob("*.pdf"):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in current_paths:
            continue
        if resolved in known_pdf_paths:
            continue
        if staging_dir.exists() and _path_is_relative_to(resolved, staging_dir):
            continue
        try:
            digest = _cached_pdf_hash(resolved)
        except OSError:
            continue
        if digest in target_hashes and digest not in existing_by_hash:
            existing_by_hash[digest] = resolved

    duplicates: list[dict[str, Any]] = []
    for item in files:
        file_hash = str(item.get("hash", ""))
        if not file_hash or file_hash in known_hashes:
            continue
        existing = existing_by_hash.get(file_hash)
        if not existing:
            continue
        duplicates.append(
            {
                "filename": str(item.get("filename", "")),
                "hash": file_hash,
                "existing_filename": existing.name,
                "status": "uploaded",
                "job_id": "",
            }
        )
    return duplicates


def _blocking_duplicate_entries(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry = PdfRegistry(PDF_REGISTRY_PATH)
    duplicates = registry.blocking_duplicates(files)
    known_hashes = {str(entry.get("hash", "")) for entry in duplicates}

    source_duplicates = _indexed_source_duplicate_entries(files, known_hashes=known_hashes)
    duplicates.extend(source_duplicates)
    known_hashes.update(str(entry.get("hash", "")) for entry in source_duplicates)

    vector_duplicates = _vector_store_duplicate_entries(files, known_hashes=known_hashes)
    duplicates.extend(vector_duplicates)
    known_hashes.update(str(entry.get("hash", "")) for entry in vector_duplicates)

    data_duplicates = _data_pdf_duplicate_entries(files, known_hashes=known_hashes)
    duplicates.extend(data_duplicates)
    return duplicates


def _duplicate_entries_for_hash(file_hash: str) -> list[dict[str, Any]]:
    if not _is_sha256_hash(file_hash):
        raise HTTPException(status_code=400, detail="hash must be a 64-character SHA-256 hex string.")
    return _blocking_duplicate_entries([{"filename": "", "hash": _sha256_text(file_hash)}])


async def _handle_upload_request(request: Request, *, require_source_groups: bool) -> dict[str, Any]:
    filenames: list[str] = []
    uploads: list[dict[str, Any]] = []
    used_names: set[str] = set()
    staging_dirs: dict[str, Path] = {}
    registered_jobs: list[tuple[str, list[dict[str, Any]]]] = []
    queued_job_ids: set[str] = set()

    try:
        form = await request.form()
        force_duplicates = str(form.get("force_duplicates") or "").lower() in {"1", "true", "yes", "on"}
        job_id = uuid.uuid4().hex
        staging_dir = STAGING_DIR / job_id
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_dirs[job_id] = staging_dir
        raw_source_groups = [
            str(value or "")
            for key, value in form.multi_items()
            if key == "source_groups" and not isinstance(value, StarletteUploadFile)
        ]
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
            try:
                digest = await run_in_threadpool(_copy_upload_file_to_path, value, destination)
            finally:
                await value.close()
            filenames.append(filename)
            uploads.append(
                {
                    "filename": filename,
                    "hash": digest,
                    "staging_path": str(destination),
                    "staging_dir": str(staging_dir),
                    "job_id": job_id,
                }
            )

        if not filenames:
            raise HTTPException(status_code=400, detail="No PDF files were uploaded.")

        _assign_upload_source_groups(
            uploads,
            raw_source_groups,
            require_groups=require_source_groups,
        )

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
        duplicate_entries = await run_in_threadpool(_blocking_duplicate_entries, uploads)
        force_token = str(form.get("force_token") or "")
        force_token_valid = (
            force_duplicates
            and bool(duplicate_entries)
            and _force_upload_token_valid(
                force_token,
                upload_hashes=_upload_hashes(uploads),
                duplicate_hashes=_duplicate_hashes(duplicate_entries),
            )
        )
        if duplicate_entries and not force_token_valid:
            raise HTTPException(
                status_code=409,
                detail=_duplicate_response_detail(
                    message="One or more PDFs have already been uploaded or queued.",
                    duplicates=duplicate_entries,
                    files=uploads,
                ),
            )

        forced_hashes = {entry["hash"] for entry in duplicate_entries} if force_token_valid else set()
        options = _upload_options_from_form(form)
        _apply_upload_source_groups(uploads)
        file_uploads = [
            {
                "filename": str(upload["filename"]),
                "hash": str(upload["hash"]),
                "staging_path": str(upload["staging_path"]),
                "source_group": normalize_source_group(upload.get("source_group")),
            }
            for upload in uploads
        ]
        registry.register_queued(
            job_id=job_id,
            files=file_uploads,
            forced_hashes=forced_hashes,
            options=options,
        )
        registered_jobs.append((job_id, file_uploads))
        job = job_queue.enqueue_upload(
            staging_dir=staging_dir,
            filenames=filenames,
            uploads=file_uploads,
            force_duplicate_hashes=sorted(forced_hashes),
            job_id=job_id,
            options=options,
        )
        queued_job_ids.add(job_id)

        job_payloads = [job.to_dict()]
        response = dict(job_payloads[0]) if job_payloads else {}
        response["jobs"] = job_payloads
        response["job_count"] = len(job_payloads)
        response["filenames"] = filenames
        return response
    except Exception:
        for registered_job_id, files in registered_jobs:
            if registered_job_id not in queued_job_ids:
                PdfRegistry(PDF_REGISTRY_PATH).mark_job_status(
                    job_id=registered_job_id,
                    files=files,
                    status="failed",
                    error="Upload was not queued.",
                )
        for staging_job_id, staging_dir in staging_dirs.items():
            if staging_job_id not in queued_job_ids:
                shutil.rmtree(staging_dir, ignore_errors=True)
        raise


@app.get("/api/uploads/check-hash")
def check_upload_hash(hash: str):
    duplicates = _duplicate_entries_for_hash(hash)
    return {
        "hash": _sha256_text(hash),
        "exists": bool(duplicates),
        "duplicates": duplicates,
    }


@app.post("/api/uploads")
async def upload_files(request: Request):
    return await _handle_upload_request(request, require_source_groups=True)


@app.post("/api/uploads/direct")
async def upload_files_direct(request: Request):
    return await _handle_upload_request(request, require_source_groups=False)


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


def _request_validation_error(exc: ValidationError) -> HTTPException:
    return HTTPException(status_code=422, detail=exc.errors())


@app.post("/api/reindex")
async def reindex(request: Request):
    try:
        payload = ReindexRequest(**await _optional_json(request))
    except ValidationError as exc:
        raise _request_validation_error(exc) from exc
    job = job_queue.enqueue_reindex(options=_model_dump(payload))
    return job.to_dict()


@app.post("/api/reingest")
def reingest():
    blocker = _index_mutation_blocker()
    if blocker:
        raise HTTPException(status_code=409, detail=blocker)
    try:
        job = enqueue_full_reingest(
            registry_path=PDF_REGISTRY_PATH,
            processed_dir=PROCESSED_DIR,
            staging_root=STAGING_DIR,
            root_dir=ROOT_DIR,
            data_dir=DATA_DIR,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    response = job.to_dict()
    response["reingest"] = True
    return response


@app.get("/api/index/backups")
def index_backups(request: Request):
    payload = {"backups": list_index_backups(DB_DIR), "keep": LANCEDB_BACKUP_KEEP}
    seed = _file_signature(index_backup_root(DB_DIR))
    return _conditional_json(request, payload, seed)


@app.post("/api/index/backup")
def backup_index():
    blocker = _index_mutation_blocker()
    if blocker:
        raise HTTPException(status_code=409, detail=blocker)
    job = job_queue.enqueue_backup()
    return job.to_dict()


@app.post("/api/index/rebuild")
def rebuild_index(request: Request):
    blocker = _index_mutation_blocker()
    if blocker:
        raise HTTPException(status_code=409, detail=blocker)
    job = job_queue.enqueue_rebuild()
    return job.to_dict()


@app.post("/api/index/restore")
async def restore_index(request: Request):
    try:
        payload = RestoreBackupRequest(**await _optional_json(request))
    except ValidationError as exc:
        raise _request_validation_error(exc) from exc
    blocker = _index_mutation_blocker()
    if blocker:
        raise HTTPException(status_code=409, detail=blocker)
    try:
        job = job_queue.enqueue_restore(backup_name=payload.backup_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs(request: Request, offset: int = 0, limit: int = 10, search: str = ""):
    payload = list_job_rows(offset=offset, limit=(None if limit <= 0 else limit), search=search)
    seed = _payload_signature(payload, "jobs", offset, limit, search)
    return _conditional_json(request, payload, seed)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return job_queue.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/pdfs")
def pdf_documents(request: Request, search: str = "", offset: int = 0, limit: int = 10):
    # The PDF list only changes when ingestion/indexing writes the registry,
    # source map, trust registry, or index manifest. Short-circuit unchanged
    # polls with a conditional 304 instead of rebuilding the whole list.
    seed = _scoped_signature(
        _file_signature(
            PDF_REGISTRY_PATH,
            source_map_path(PROCESSED_DIR),
            DOCUMENT_TRUST_PATH,
            DB_DIR / INDEX_MANIFEST_FILENAME,
        ),
        search=search,
        offset=offset,
        limit=limit,
    )
    if not_modified := _not_modified_or_etag(request, seed):
        return not_modified
    payload = list_pdf_documents(search=search, offset=offset, limit=(None if limit <= 0 else limit))
    return _etagged_json(request, payload, seed)


def _pdf_row_for_hash(source_hash: str) -> dict[str, Any] | None:
    documents = list_pdf_documents(search=source_hash, offset=0, limit=None)["pdfs"]
    return next((item for item in documents if item.get("hash") == source_hash), None)


@app.get("/api/assets/{asset_id}")
def image_asset(asset_id: str):
    from src.asset_store import ImageAssetStore

    store = ImageAssetStore(ASSET_DIR)
    entry = store.get_asset(asset_id)
    path = store.asset_path(asset_id)
    if entry is None or path is None:
        raise HTTPException(status_code=404, detail="Image asset not found.")
    return FileResponse(
        path,
        media_type=str(entry.get("mime_type") or "image/png"),
        filename=path.name,
        content_disposition_type="inline",
    )


@app.get("/api/pdfs/{source_hash}/download")
def download_pdf(source_hash: str):
    try:
        path, filename = resolve_pdf_download_path(source_hash)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="application/pdf", filename=filename)


@app.get("/api/pdfs/{source_hash}/view")
def view_pdf(source_hash: str):
    try:
        path, filename = resolve_pdf_download_path(source_hash)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=filename,
        content_disposition_type="inline",
    )


@app.delete("/api/pdfs/{source_hash}")
def delete_pdf(source_hash: str):
    blocker = _index_mutation_blocker()
    if blocker:
        raise HTTPException(status_code=409, detail=blocker)
    try:
        return delete_pdf_document(
            source_hash,
            registry_path=PDF_REGISTRY_PATH,
            processed_dir=PROCESSED_DIR,
            db_dir=DB_DIR,
            trust_path=DOCUMENT_TRUST_PATH,
            root_dir=ROOT_DIR,
            data_dir=DATA_DIR,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pdfs/{source_hash}/reprocess")
def reprocess_pdf(source_hash: str):
    try:
        job = enqueue_source_reprocess(
            source_hash,
            registry_path=PDF_REGISTRY_PATH,
            processed_dir=PROCESSED_DIR,
            staging_root=STAGING_DIR,
            root_dir=ROOT_DIR,
            data_dir=DATA_DIR,
        )
        return job.to_dict()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/pdfs/{source_hash}/reindex")
def reindex_pdf(source_hash: str):
    try:
        job = enqueue_source_reindex(
            source_hash,
            registry_path=PDF_REGISTRY_PATH,
            processed_dir=PROCESSED_DIR,
            root_dir=ROOT_DIR,
            data_dir=DATA_DIR,
        )
        return job.to_dict()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/pdfs/{source_hash}/trust")
def update_pdf_trust(source_hash: str, payload: DocumentTrustRequest):
    try:
        entry = update_document_trust(
            source_hash,
            {key: value for key, value in _model_dump(payload).items() if value is not None},
        )
        return {"trust": entry, "pdf": _pdf_row_for_hash(source_hash)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pdfs/trust/bulk")
def update_pdf_trust_bulk(payload: BulkDocumentTrustRequest):
    source_group = normalize_source_group(payload.source_group)
    if source_group not in TRUST_SOURCE_GROUPS or source_group == SOURCE_GROUP_UNGROUPED:
        choices = ", ".join(sorted(group for group in TRUST_SOURCE_GROUPS if group != SOURCE_GROUP_UNGROUPED))
        raise HTTPException(status_code=400, detail=f"source_group must be one of: {choices}")

    hashes = []
    seen = set()
    failed = []
    for raw_hash in payload.source_hashes:
        source_hash = str(raw_hash or "").strip()
        if not source_hash:
            failed.append({"source_hash": source_hash, "error": "source_hash cannot be empty"})
            continue
        if source_hash in seen:
            continue
        seen.add(source_hash)
        hashes.append(source_hash)
    if not hashes and not failed:
        raise HTTPException(status_code=400, detail="At least one source hash is required.")

    updated = []
    updates: dict[str, Any] = {"source_group": source_group}
    reviewed_by = str(payload.reviewed_by or "").strip()
    if reviewed_by:
        updates["reviewed_by"] = reviewed_by
    for source_hash in hashes:
        try:
            trust = update_document_trust(source_hash, updates)
            updated.append({"source_hash": source_hash, "trust": trust, "pdf": _pdf_row_for_hash(source_hash)})
        except ValueError as exc:
            failed.append({"source_hash": source_hash, "error": str(exc)})
    return {"updated": updated, "failed": failed}


@app.get("/api/index")
def index_rows(request: Request, offset: int = 0, limit: int = 50, search: str = ""):
    # Index rows only change when the LanceDB table is rewritten; key the ETag on
    # the table version-hint file so unchanged polls short-circuit to a 304.
    version_hint = _index_store().table_version_hint_path()
    seed = _scoped_signature(
        _file_signature(version_hint, DB_DIR / INDEX_MANIFEST_FILENAME, index_overrides_path(DB_DIR)),
        offset=offset,
        limit=limit,
        search=search,
    )
    if not_modified := _not_modified_or_etag(request, seed):
        return not_modified
    try:
        payload = list_index_rows(offset=offset, limit=limit, search=search)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _etagged_json(request, payload, seed)


@app.get("/api/index/summaries")
def index_summary_rows(request: Request, offset: int = 0, limit: int = 20, search: str = ""):
    seed = _scoped_signature(
        _file_signature(
            _index_store().table_version_hint_path(),
            DB_DIR / INDEX_MANIFEST_FILENAME,
            index_overrides_path(DB_DIR),
        ),
        offset=offset,
        limit=limit,
        search=search,
    )
    if not_modified := _not_modified_or_etag(request, seed):
        return not_modified
    try:
        payload = list_index_summary_rows(offset=offset, limit=limit, search=search)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _etagged_json(request, payload, seed)


@app.get("/api/index/children")
def index_child_rows(
    request: Request,
    parent_id: str,
    offset: int = 0,
    limit: int = INDEX_CHILD_DEFAULT_LIMIT,
    search: str = "",
):
    seed = _scoped_signature(
        _file_signature(
            _index_store().table_version_hint_path(),
            DB_DIR / INDEX_MANIFEST_FILENAME,
            index_overrides_path(DB_DIR),
        ),
        parent_id=parent_id,
        offset=offset,
        limit=limit,
        search=search,
    )
    if not_modified := _not_modified_or_etag(request, seed):
        return not_modified
    try:
        payload = list_index_child_rows(
            parent_id=parent_id,
            offset=offset,
            limit=limit,
            search=search,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Record not found: {exc}") from exc
    return _etagged_json(request, payload, seed)


@app.get("/api/index/stream")
def index_rows_stream(
    batch_size: int = INDEX_STREAM_DEFAULT_BATCH_SIZE,
    search: str = "",
):
    try:
        events = iter_index_row_events(batch_size=batch_size, search=search)
        first_event = next(events)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StopIteration:
        first_event = {
            "type": "done",
            "received": 0,
            "total": 0,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "embedding_dim": 768,
        }

    def generate():
        def encode_event(event: dict[str, Any]) -> str:
            return json.dumps(event, ensure_ascii=False) + "\n"

        yield encode_event(first_event)
        try:
            for event in events:
                yield encode_event(event)
        except Exception as exc:
            yield encode_event({"type": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="application/x-ndjson; charset=utf-8")


@app.post("/api/index/update")
def update_index(payload: IndexUpdateRequest):
    blocker = _index_mutation_blocker()
    if blocker:
        raise HTTPException(status_code=409, detail=blocker)
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
    blocker = _index_mutation_blocker()
    if blocker:
        raise HTTPException(status_code=409, detail=blocker)
    try:
        return delete_index_records(record_ids=payload.record_ids)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Record not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/index/vector-search")
def vector_search_index(payload: IndexVectorSearchRequest):
    try:
        return vector_search_index_rows(
            query=payload.query,
            relevance_floor=payload.relevance_floor,
            embedding_model=payload.embedding_model,
            embedding_batch_size=payload.embedding_batch_size,
            embedding_timeout=payload.embedding_timeout,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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
                asset_dir=str(ASSET_DIR),
                trust_path=str(DOCUMENT_TRUST_PATH),
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
                planner_model=payload.planner_model,
                planner_enabled=payload.planner_enabled,
                planner_max_queries=payload.planner_max_queries,
                progress_enabled=False,
            )
            if hasattr(engine, "ask_stream_events"):
                for event in engine.ask_stream_events(question):
                    if event.get("text") or event.get("sources") or event.get("result") or event.get("content"):
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


def run_server() -> None:
    import uvicorn

    uvicorn.run(
        "src.web_app:app",
        host=str(SERVER_CONFIG["host"]),
        port=int(SERVER_CONFIG["port"]),
    )


if __name__ == "__main__":
    run_server()
