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
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

from src.defaults import (
    DEFAULT_ASSET_TRIGGERS,
    DEFAULT_DOCLING_ACCELERATOR,
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
)
from src.pdf_registry import PdfRegistry, load_source_map, remove_source_entries_by_hash, sha256_file
from src.vector_store import default_store, lancedb_path, record_matches, record_row


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
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
DEFAULT_UPDATE_REMOTE = "origin"
DEFAULT_UPDATE_BRANCH = "main"
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


def _load_server_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or (ROOT_DIR / "config.toml")
    payload = _load_toml_config(config_path)
    server_config = payload.get("server", {}) if isinstance(payload.get("server"), dict) else {}

    return {
        "host": _nonempty_str(server_config.get("host"), DEFAULT_SERVER_HOST),
        "port": _positive_int(server_config.get("port"), DEFAULT_SERVER_PORT),
        "health_poll_interval_ms": _positive_int(
            server_config.get("health_poll_interval_ms"),
            DEFAULT_HEALTH_POLL_INTERVAL_MS,
        ),
        "jobs_poll_interval_ms": _positive_int(
            server_config.get("jobs_poll_interval_ms"),
            DEFAULT_JOBS_POLL_INTERVAL_MS,
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
    }


def _load_ingestion_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or (ROOT_DIR / "config.toml")
    payload = _load_toml_config(config_path)
    ingestion = payload.get("ingestion", {}) if isinstance(payload.get("ingestion"), dict) else {}
    models = payload.get("models", {}) if isinstance(payload.get("models"), dict) else {}

    return {
        "parser_mode": _nonempty_str(ingestion.get("parser_mode"), DEFAULT_PDF_PARSER_MODE),
        "accelerator": _nonempty_str(ingestion.get("accelerator"), DEFAULT_DOCLING_ACCELERATOR),
        "asset_triggers": _nonempty_str(ingestion.get("asset_triggers"), DEFAULT_ASSET_TRIGGERS),
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


def _index_store(db_dir: Path | None = None):
    return default_store(db_dir or DB_DIR)


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
        return _index_store(db_dir).list_records(offset=offset, limit=limit, search=search)


def _index_records_snapshot(db_dir: Path | None = None) -> tuple[list[dict[str, Any]], str, int]:
    store = _index_store(db_dir)
    if not store.exists():
        raise FileNotFoundError(f"LanceDB table not found at {lancedb_path(db_dir or DB_DIR) / 'chunks'}")
    count = store.count()
    if count <= 0:
        model, dim = store.metadata()
        return [], model, dim
    payload = store.list_records(offset=0, limit=count, search="")
    return (
        list(payload.get("rows") or []),
        str(payload.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
        int(payload.get("embedding_dim") or 768),
    )


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
        "rows": [
            _with_index_hierarchy_metadata(row, children=[], parent=parent)
            for row in page["rows"]
        ],
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
            "rows": rows,
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
        row["citation"] = str(item.get("citation") or "")
        row["source_id"] = str(item.get("source_id") or "")
        row["location"] = str(item.get("location") or "")
        row["vector_query"] = query
        rows.append(row)

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
    page = _page_slice(rows, offset=offset, limit=limit)
    return {
        "pdfs": page["rows"],
        "total": page["total"],
        "offset": page["offset"],
        "limit": page["limit"],
    }


def list_job_rows(*, offset: int = 0, limit: int | None = 10) -> dict[str, Any]:
    page = _page_slice(job_queue.list_jobs(), offset=offset, limit=limit)
    return {
        "jobs": page["rows"],
        "total": page["total"],
        "offset": page["offset"],
        "limit": page["limit"],
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
            parser_mode=options.get("parser_mode", INGESTION_CONFIG["parser_mode"]),
            accelerator=options.get("accelerator", INGESTION_CONFIG["accelerator"]),
            asset_triggers=options.get("asset_triggers", INGESTION_CONFIG["asset_triggers"]),
            vision_model=options.get("vision_model", INGESTION_CONFIG["vision_model"]),
            vision_enabled=options.get("vision_enabled", INGESTION_CONFIG["vision_enabled"]),
            ocr_backend=options.get("ocr_backend", INGESTION_CONFIG["ocr_backend"]),
            ocr_langs=options.get("ocr_langs", INGESTION_CONFIG["ocr_langs"]),
            ocr_force_full_page=options.get("ocr_force_full_page", INGESTION_CONFIG["ocr_force_full_page"]),
            ocr_bitmap_area_threshold=options.get(
                "ocr_bitmap_area_threshold",
                INGESTION_CONFIG["ocr_bitmap_area_threshold"],
            ),
            rapidocr_backend=options.get("rapidocr_backend", INGESTION_CONFIG["rapidocr_backend"]),
            tesseract_cmd=options.get("tesseract_cmd", INGESTION_CONFIG["tesseract_cmd"]),
            tesseract_data_path=options.get("tesseract_data_path", INGESTION_CONFIG["tesseract_data_path"]),
            tesseract_psm=options.get("tesseract_psm", INGESTION_CONFIG["tesseract_psm"]),
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


class IndexVectorSearchRequest(BaseModel):
    query: str
    relevance_floor: float = DEFAULT_INDEX_VECTOR_RELEVANCE_FLOOR
    embedding_model: str | None = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int | None = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_timeout: float | None = DEFAULT_EMBEDDING_TIMEOUT


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
        "parser_mode": str(form.get("parser_mode") or INGESTION_CONFIG["parser_mode"]),
        "accelerator": str(form.get("accelerator") or INGESTION_CONFIG["accelerator"]),
        "asset_triggers": str(form.get("asset_triggers") or INGESTION_CONFIG["asset_triggers"]),
        "vision_model": str(form.get("vision_model") or INGESTION_CONFIG["vision_model"]),
        "vision_enabled": _bool_value(form.get("vision_enabled"), INGESTION_CONFIG["vision_enabled"]),
        "ocr_backend": str(form.get("ocr_backend") or INGESTION_CONFIG["ocr_backend"]),
        "ocr_langs": _string_list(form.get("ocr_langs"), tuple(INGESTION_CONFIG["ocr_langs"])),
        "ocr_force_full_page": _bool_value(
            form.get("ocr_force_full_page"),
            INGESTION_CONFIG["ocr_force_full_page"],
        ),
        "ocr_bitmap_area_threshold": float(
            form.get("ocr_bitmap_area_threshold") or INGESTION_CONFIG["ocr_bitmap_area_threshold"]
        ),
        "rapidocr_backend": str(form.get("rapidocr_backend") or INGESTION_CONFIG["rapidocr_backend"]),
        "tesseract_cmd": str(form.get("tesseract_cmd") or INGESTION_CONFIG["tesseract_cmd"]),
        "tesseract_data_path": str(form.get("tesseract_data_path") or INGESTION_CONFIG["tesseract_data_path"]),
        "tesseract_psm": _optional_int(form.get("tesseract_psm"), INGESTION_CONFIG["tesseract_psm"]),
        "embedding_model": str(form.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
        "embedding_batch_size": int(form.get("embedding_batch_size") or DEFAULT_EMBEDDING_BATCH_SIZE),
        "embedding_timeout": float(form.get("embedding_timeout") or DEFAULT_EMBEDDING_TIMEOUT),
        "index_backend": str(form.get("index_backend") or DEFAULT_INDEX_BACKEND),
        "summary_mode": str(form.get("summary_mode") or DEFAULT_SUMMARY_MODE),
        "chunk_target_tokens": int(form.get("chunk_target_tokens") or DEFAULT_CHUNK_TARGET_TOKENS),
        "chunk_overlap_tokens": int(form.get("chunk_overlap_tokens") or DEFAULT_CHUNK_OVERLAP_TOKENS),
        "progress_enabled": False,
    }


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
        rows = store.list_records(offset=0, limit=store.count(), search="").get("rows") or []
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

    existing_by_hash: dict[str, Path] = {}
    for candidate in data_dir.rglob("*.pdf"):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in current_paths:
            continue
        if staging_dir.exists() and _path_is_relative_to(resolved, staging_dir):
            continue
        try:
            digest = sha256_file(resolved)
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


@app.post("/api/uploads")
async def upload_files(request: Request):
    filenames: list[str] = []
    uploads: list[dict[str, Any]] = []
    used_names: set[str] = set()
    staging_dirs: dict[str, Path] = {}
    registered_jobs: list[tuple[str, list[dict[str, Any]]]] = []
    queued_job_ids: set[str] = set()

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

            job_id = uuid.uuid4().hex
            staging_dir = STAGING_DIR / job_id
            staging_dir.mkdir(parents=True, exist_ok=True)
            staging_dirs[job_id] = staging_dir
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
                    "staging_dir": str(staging_dir),
                    "job_id": job_id,
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
        duplicate_entries = _blocking_duplicate_entries(uploads)
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
        jobs: list[QueueJob] = []
        for upload in uploads:
            job_id = str(upload["job_id"])
            file_upload = {
                "filename": str(upload["filename"]),
                "hash": str(upload["hash"]),
                "staging_path": str(upload["staging_path"]),
            }
            file_forced_hashes = {file_upload["hash"]} if file_upload["hash"] in forced_hashes else set()
            registry.register_queued(
                job_id=job_id,
                files=[file_upload],
                forced_hashes=file_forced_hashes,
            )
            registered_jobs.append((job_id, [file_upload]))
            job = job_queue.enqueue_upload(
                staging_dir=Path(str(upload["staging_dir"])),
                filenames=[file_upload["filename"]],
                uploads=[file_upload],
                force_duplicate_hashes=sorted(file_forced_hashes),
                job_id=job_id,
                options=options,
            )
            queued_job_ids.add(job_id)
            jobs.append(job)

        job_payloads = [job.to_dict() for job in jobs]
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
def list_jobs(offset: int = 0, limit: int = 10):
    return list_job_rows(offset=offset, limit=limit)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/api/pdfs")
def pdf_documents(search: str = "", offset: int = 0, limit: int = 10):
    return list_pdf_documents(search=search, offset=offset, limit=limit)


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


@app.get("/api/index/summaries")
def index_summary_rows(offset: int = 0, limit: int = 20, search: str = ""):
    try:
        return list_index_summary_rows(offset=offset, limit=limit, search=search)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/index/children")
def index_child_rows(
    parent_id: str,
    offset: int = 0,
    limit: int = INDEX_CHILD_DEFAULT_LIMIT,
    search: str = "",
):
    try:
        return list_index_child_rows(
            parent_id=parent_id,
            offset=offset,
            limit=limit,
            search=search,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Record not found: {exc}") from exc


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
