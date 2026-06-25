from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_FILENAME = ".pdf_upload_registry.json"
SOURCE_MAP_FILENAME = ".source_map.json"
REGISTRY_VERSION = 1
BLOCKING_STATUSES = {"queued", "saving_uploads", "ingesting", "ingested", "indexed"}

_LOCK = threading.RLock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return payload if isinstance(payload, dict) else default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(data, encoding="utf-8")


class PdfRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        with _LOCK:
            payload = _load_json(self.path, {"version": REGISTRY_VERSION, "pdfs": {}})
            payload.setdefault("version", REGISTRY_VERSION)
            payload.setdefault("pdfs", {})
            if not isinstance(payload["pdfs"], dict):
                payload["pdfs"] = {}
            return payload

    def blocking_duplicates(self, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = self.load()
        pdfs = payload.get("pdfs", {})
        duplicates: list[dict[str, Any]] = []
        for item in files:
            file_hash = str(item.get("hash", ""))
            existing = pdfs.get(file_hash)
            if not existing or existing.get("status") not in BLOCKING_STATUSES:
                continue
            duplicates.append(
                {
                    "filename": str(item.get("filename", "")),
                    "hash": file_hash,
                    "existing_filename": str(existing.get("filename", "")),
                    "status": str(existing.get("status", "")),
                    "job_id": str(existing.get("job_id", "")),
                }
            )
        return duplicates

    def register_queued(
        self,
        *,
        job_id: str,
        files: list[dict[str, Any]],
        forced_hashes: set[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        forced_hashes = forced_hashes or set()
        options = dict(options or {})
        with _LOCK:
            payload = self.load()
            pdfs = payload.setdefault("pdfs", {})
            now = utcnow()
            for item in files:
                file_hash = str(item["hash"])
                existing = pdfs.get(file_hash)
                entry = {
                    "hash": file_hash,
                    "filename": str(item["filename"]),
                    "status": "queued",
                    "job_id": job_id,
                    "staging_path": str(item.get("staging_path", "")),
                    "upload_path": str(item.get("upload_path", "")),
                    "processed_markdown_path": str(item.get("processed_markdown_path", "")),
                    "created_at": now,
                    "updated_at": now,
                }
                if options:
                    entry["options"] = dict(options)
                if isinstance(existing, dict) and file_hash in forced_hashes:
                    entry["previous_entry"] = existing
                pdfs[file_hash] = entry
            _write_json(self.path, payload)

    def mark_job_status(
        self,
        *,
        job_id: str,
        files: list[dict[str, Any]],
        status: str,
        error: str | None = None,
    ) -> None:
        with _LOCK:
            payload = self.load()
            pdfs = payload.setdefault("pdfs", {})
            now = utcnow()
            for item in files:
                file_hash = str(item.get("hash", ""))
                entry = pdfs.get(file_hash)
                if not isinstance(entry, dict) or entry.get("job_id") != job_id:
                    continue

                if status == "failed" and entry.get("status") in {"ingested", "indexed"}:
                    entry["last_error"] = error or ""
                    entry["last_failed_at"] = now
                    entry["updated_at"] = now
                    continue

                if status == "failed" and isinstance(entry.get("previous_entry"), dict):
                    restored = dict(entry["previous_entry"])
                    restored["last_failed_job_id"] = job_id
                    restored["last_error"] = error or ""
                    restored["updated_at"] = now
                    pdfs[file_hash] = restored
                    continue

                entry["status"] = status
                entry["updated_at"] = now
                for key in ("staging_path", "upload_path", "processed_markdown_path"):
                    if item.get(key):
                        entry[key] = str(item[key])
                if status in {"ingested", "indexed"}:
                    entry.pop("previous_entry", None)
                if error:
                    entry["last_error"] = error
            self._supersede_same_processed_paths(payload)
            _write_json(self.path, payload)

    @staticmethod
    def _supersede_same_processed_paths(payload: dict[str, Any]) -> None:
        pdfs = payload.get("pdfs", {})
        active_by_path: dict[str, str] = {}
        for file_hash, entry in pdfs.items():
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("processed_markdown_path", ""))
            if not path or entry.get("status") not in {"ingested", "indexed"}:
                continue
            active_by_path[path] = str(file_hash)

        for file_hash, entry in pdfs.items():
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("processed_markdown_path", ""))
            if path and active_by_path.get(path) not in {None, str(file_hash)}:
                entry["status"] = "superseded"
                entry["updated_at"] = utcnow()


def source_map_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / SOURCE_MAP_FILENAME


def load_source_map(processed_dir: str | Path) -> dict[str, Any]:
    payload = _load_json(source_map_path(processed_dir), {"version": REGISTRY_VERSION, "documents": {}})
    payload.setdefault("version", REGISTRY_VERSION)
    payload.setdefault("documents", {})
    if not isinstance(payload["documents"], dict):
        payload["documents"] = {}
    return payload


def write_source_entry(
    *,
    processed_dir: str | Path,
    markdown_path: str | Path,
    source_hash: str,
    source_pdf_name: str,
    source_pdf_path: str | Path,
) -> dict[str, Any]:
    markdown = Path(markdown_path)
    entry = {
        "source_hash": str(source_hash),
        "source_pdf_name": str(source_pdf_name),
        "source_pdf_path": str(source_pdf_path),
        "processed_markdown_path": str(markdown),
        "updated_at": utcnow(),
    }
    with _LOCK:
        payload = load_source_map(processed_dir)
        payload["documents"][markdown.name] = entry
        _write_json(source_map_path(processed_dir), payload)
    return entry


def source_entry_for_markdown(markdown_path: str | Path) -> dict[str, Any]:
    markdown = Path(markdown_path)
    payload = load_source_map(markdown.parent)
    entry = payload.get("documents", {}).get(markdown.name, {})
    return dict(entry) if isinstance(entry, dict) else {}


def remove_source_entries_by_hash(
    processed_dir: str | Path,
    source_hashes: set[str],
) -> list[dict[str, Any]]:
    hashes = {str(value) for value in source_hashes if value}
    if not hashes:
        return []
    with _LOCK:
        payload = load_source_map(processed_dir)
        documents = payload.get("documents", {})
        removed: list[dict[str, Any]] = []
        kept: dict[str, Any] = {}
        for markdown_name, entry in documents.items():
            if isinstance(entry, dict) and str(entry.get("source_hash", "")) in hashes:
                removed.append(dict(entry))
            else:
                kept[markdown_name] = entry
        if removed:
            payload["documents"] = kept
            _write_json(source_map_path(processed_dir), payload)
        return removed
