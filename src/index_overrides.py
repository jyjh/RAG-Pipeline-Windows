from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.atomic_io import write_json_atomic


INDEX_OVERRIDES_FILENAME = "index_overrides.json"
INDEX_OVERRIDES_VERSION = 1


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def index_overrides_path(db_dir: str | Path) -> Path:
    return Path(db_dir) / INDEX_OVERRIDES_FILENAME


def _empty_payload() -> dict[str, Any]:
    return {"version": INDEX_OVERRIDES_VERSION, "edits": {}, "deletions": {}}


def load_index_overrides(db_dir: str | Path) -> dict[str, Any]:
    path = index_overrides_path(db_dir)
    if not path.exists():
        return _empty_payload()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_payload()
    if not isinstance(payload, dict):
        return _empty_payload()
    payload.setdefault("version", INDEX_OVERRIDES_VERSION)
    payload.setdefault("edits", {})
    payload.setdefault("deletions", {})
    if not isinstance(payload["edits"], dict):
        payload["edits"] = {}
    if not isinstance(payload["deletions"], dict):
        payload["deletions"] = {}
    return payload


def write_index_overrides(db_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = index_overrides_path(db_dir)
    normalized = _empty_payload()
    normalized["edits"] = {
        str(record_id): dict(entry)
        for record_id, entry in (payload.get("edits") or {}).items()
        if record_id and isinstance(entry, dict)
    }
    normalized["deletions"] = {
        str(record_id): dict(entry)
        for record_id, entry in (payload.get("deletions") or {}).items()
        if record_id and isinstance(entry, dict)
    }
    write_json_atomic(path, normalized)
    return normalized


def persist_index_edit(db_dir: str | Path, record: dict[str, Any], content: str) -> dict[str, Any]:
    record_id = str(record.get("id") or "")
    if not record_id:
        raise ValueError("record id is required.")
    payload = load_index_overrides(db_dir)
    payload["deletions"].pop(record_id, None)
    payload["edits"][record_id] = {
        "record_id": record_id,
        "content": str(content),
        "source_hash": str(record.get("source_hash") or ""),
        "doc_id": str(record.get("doc_id") or ""),
        "file_path": str(record.get("file_path") or ""),
        "updated_at": utcnow(),
    }
    return write_index_overrides(db_dir, payload)


def persist_index_deletions(db_dir: str | Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    payload = load_index_overrides(db_dir)
    now = utcnow()
    for record in records:
        record_id = str(record.get("id") or "")
        if not record_id:
            continue
        payload["edits"].pop(record_id, None)
        payload["deletions"][record_id] = {
            "record_id": record_id,
            "source_hash": str(record.get("source_hash") or ""),
            "doc_id": str(record.get("doc_id") or ""),
            "file_path": str(record.get("file_path") or ""),
            "updated_at": now,
        }
    return write_index_overrides(db_dir, payload)


def apply_overrides_to_records(records: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply a loaded overrides payload to a batch of records.

    Split from :func:`apply_index_overrides` so the streaming indexer can load
    overrides once and apply them per-file without re-reading disk for every
    Markdown file.
    """
    edits = payload.get("edits") or {}
    deletions = set(str(record_id) for record_id in (payload.get("deletions") or {}).keys())
    if not edits and not deletions:
        return records

    applied: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id") or "")
        if record_id in deletions:
            continue
        edit = edits.get(record_id)
        if isinstance(edit, dict) and "content" in edit:
            item = dict(record)
            item["content"] = str(edit.get("content") or "")
            item["manual_edit"] = True
            applied.append(item)
        else:
            applied.append(record)
    return applied


def apply_index_overrides(records: list[dict[str, Any]], db_dir: str | Path) -> list[dict[str, Any]]:
    payload = load_index_overrides(db_dir)
    return apply_overrides_to_records(records, payload)


def clear_overrides_for_sources(db_dir: str | Path, source_hashes: set[str]) -> dict[str, Any]:
    hashes = {str(value) for value in source_hashes if value}
    payload = load_index_overrides(db_dir)
    if not hashes:
        return payload
    payload["edits"] = {
        record_id: entry
        for record_id, entry in payload.get("edits", {}).items()
        if str(entry.get("source_hash") or "") not in hashes
    }
    payload["deletions"] = {
        record_id: entry
        for record_id, entry in payload.get("deletions", {}).items()
        if str(entry.get("source_hash") or "") not in hashes
    }
    return write_index_overrides(db_dir, payload)


def edited_record_ids(db_dir: str | Path) -> set[str]:
    return set(str(record_id) for record_id in load_index_overrides(db_dir).get("edits", {}).keys())


def copy_index_overrides(source_db_dir: str | Path, target_db_dir: str | Path) -> None:
    source = index_overrides_path(source_db_dir)
    target = index_overrides_path(target_db_dir)
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
