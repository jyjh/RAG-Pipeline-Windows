"""PDF upload registry and ingestion source map.

Both stores previously lived in single JSON documents rewritten in full on
every mutation, which made bulk ingestion O(N^2) in parse/serialize work and
convoyed every ingestion worker through one file lock. They are now backed by
SQLite (one database per logical store), with a one-time import from the
legacy ``.json`` document when it exists.

Compatibility contract preserved for callers:

* :class:`PdfRegistry` and the module-level source-map functions keep their
  signatures and return shapes. The on-disk layout is an implementation
  detail: the database lives beside the legacy path as ``<path>.sqlite3``.
* A corrupt legacy JSON still raises :class:`json.JSONDecodeError` during the
  import rather than silently starting empty.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


REGISTRY_FILENAME = ".pdf_upload_registry.json"
SOURCE_MAP_FILENAME = ".source_map.json"
REGISTRY_VERSION = 1
BLOCKING_STATUSES = {"queued", "saving_uploads", "ingesting", "ingested", "indexed"}

_LOCK = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _db_path_for(path: str | Path) -> Path:
    """The SQLite store lives beside the legacy JSON path."""
    return Path(str(path) + ".sqlite3")


def _connect(db_path: Path) -> sqlite3.Connection:
    # Short-lived per-operation connections: cheap for a local file and safe
    # across threads, subprocesses, and tmp-path churn in tests. WAL lets
    # readers proceed while another process writes; busy_timeout absorbs
    # writer-writer contention that portalocker used to convoy.
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript(_SCHEMA)
    return conn


@contextlib.contextmanager
def _store(json_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = _db_path_for(json_path)
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        _import_legacy_if_needed(conn, json_path)
        yield conn
    finally:
        conn.close()


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _import_legacy_if_needed(conn: sqlite3.Connection, json_path: Path) -> None:
    """One-time import of the legacy JSON document.

    Errors propagate deliberately: corruption must surface, never silently
    become an empty registry that would lose upload/ingest state.
    """
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if int(count) > 0 or _meta_get(conn, "legacy_imported"):
        return
    if not json_path.exists():
        _meta_set(conn, "legacy_imported", utcnow())
        return
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows: list[tuple[str, str]] = []
    for container_key in ("pdfs", "documents"):
        container = payload.get(container_key) if isinstance(payload, dict) else None
        if isinstance(container, dict):
            for key, entry in container.items():
                rows.append((str(key), json.dumps(entry)))
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO documents(key, payload) VALUES(?, ?)", rows
        )
        _meta_set(conn, "legacy_imported", utcnow())
        _bump_version(conn)


def _bump_version(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('data_version', '1') "
        "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
    )


def _state_version(json_path: Path) -> str:
    """Monotonic change token for ETag seeding (0 until first write)."""
    try:
        with _store(json_path) as conn:
            value = _meta_get(conn, "data_version")
        return value or "0"
    except json.JSONDecodeError:
        raise
    except (sqlite3.Error, OSError):
        return "?"


# Payload documents are cached keyed by (db path, data_version) so hot read
# paths skip re-parsing every row between writes.
_PAYLOAD_CACHE_LOCK = threading.Lock()
_PAYLOAD_CACHE: dict[str, tuple[str, dict[str, Any]]] = {}
_PAYLOAD_CACHE_LIMIT = 64


def _cache_get(cache_key: str, version: str) -> dict[str, Any] | None:
    with _PAYLOAD_CACHE_LOCK:
        cached = _PAYLOAD_CACHE.get(cache_key)
        if cached is not None and cached[0] == version:
            return cached[1]
    return None


def _cache_put(cache_key: str, version: str, payload: dict[str, Any]) -> None:
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_CACHE[cache_key] = (version, payload)
        while len(_PAYLOAD_CACHE) > _PAYLOAD_CACHE_LIMIT:
            _PAYLOAD_CACHE.pop(next(iter(_PAYLOAD_CACHE)))


def _load_documents(conn: sqlite3.Connection, cache_key: str) -> dict[str, Any]:
    version = _meta_get(conn, "data_version") or "0"
    cached = _cache_get(cache_key, version)
    if cached is not None:
        return cached
    documents: dict[str, Any] = {}
    for key, raw in conn.execute("SELECT key, payload FROM documents"):
        try:
            entry = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(entry, dict):
            documents[str(key)] = entry
    result = {"version": REGISTRY_VERSION, "documents": documents}
    _cache_put(cache_key, version, result)
    return result


def _write_documents(
    conn: sqlite3.Connection,
    cache_key: str,
    updates: dict[str, Any],
    deletions: set[str] | None = None,
) -> None:
    rows = [(key, json.dumps(entry)) for key, entry in updates.items()]
    with conn:
        if rows:
            conn.executemany(
                "INSERT INTO documents(key, payload) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET payload = excluded.payload",
                rows,
            )
        for key in deletions or set():
            conn.execute("DELETE FROM documents WHERE key = ?", (key,))
        _bump_version(conn)
    # Drop any cached snapshot; the next read rescans rows once and repopulates
    # at the fresh version. Cross-process writes make trust-based seeding risky.
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_CACHE.pop(cache_key, None)


class PdfRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        with _LOCK, _store(self.path) as conn:
            payload = _load_documents(conn, f"reg:{self.path}")
            pdfs: dict[str, Any] = {}
            for key, entry in payload["documents"].items():
                if isinstance(entry, dict):
                    pdfs[key] = dict(entry)
            return {"version": REGISTRY_VERSION, "pdfs": pdfs}

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
        with _LOCK, _store(self.path) as conn:
            cache_key = f"reg:{self.path}"
            documents = dict(_load_documents(conn, cache_key)["documents"])
            before_snapshot = dict(documents)
            now = utcnow()
            for item in files:
                file_hash = str(item["hash"])
                previous = documents.get(file_hash)
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
                if isinstance(previous, dict) and file_hash in forced_hashes:
                    entry["previous_entry"] = previous
                documents[file_hash] = entry
            self._commit_changed(conn, cache_key, before=before_snapshot, state=documents)

    def mark_job_status(
        self,
        *,
        job_id: str,
        files: list[dict[str, Any]],
        status: str,
        error: str | None = None,
    ) -> None:
        with _LOCK, _store(self.path) as conn:
            documents = dict(_load_documents(conn, f"reg:{self.path}")["documents"])
            state = {
                key: dict(entry) if isinstance(entry, dict) else entry
                for key, entry in documents.items()
            }
            now = utcnow()
            for item in files:
                file_hash = str(item.get("hash", ""))
                entry = state.get(file_hash)
                if not isinstance(entry, dict) or entry.get("job_id") != job_id:
                    continue

                if status == "interrupted":
                    interrupted = {
                        "last_interrupted_job_id": job_id,
                        "last_interrupted_at": now,
                        "last_error": error or "Job interrupted.",
                        "updated_at": now,
                    }
                    if isinstance(entry.get("previous_entry"), dict):
                        restored = dict(entry["previous_entry"])
                        restored.update(interrupted)
                        state[file_hash] = restored
                        continue
                    entry = dict(entry)
                    entry["status"] = "interrupted"
                    entry.update(interrupted)
                    for key in ("staging_path", "upload_path", "processed_markdown_path"):
                        if item.get(key):
                            entry[key] = str(item[key])
                    state[file_hash] = entry
                    continue

                if status == "failed" and entry.get("status") in {"ingested", "indexed"}:
                    entry = dict(entry)
                    entry["last_error"] = error or ""
                    entry["last_failed_at"] = now
                    entry["updated_at"] = now
                    state[file_hash] = entry
                    continue

                if status == "failed" and isinstance(entry.get("previous_entry"), dict):
                    restored = dict(entry["previous_entry"])
                    restored["last_failed_job_id"] = job_id
                    restored["last_error"] = error or ""
                    restored["updated_at"] = now
                    state[file_hash] = restored
                    continue

                entry = dict(entry)
                entry["status"] = status
                entry["updated_at"] = now
                for key in ("staging_path", "upload_path", "processed_markdown_path"):
                    if item.get(key):
                        entry[key] = str(item[key])
                if status in {"ingested", "indexed"}:
                    entry.pop("previous_entry", None)
                if error:
                    entry["last_error"] = error
                state[file_hash] = entry

            self._supersede_same_processed_paths(state)
            self._commit_changed(conn, f"reg:{self.path}", before=documents, state=state)

    def mark_sources_interrupted(
        self,
        *,
        job_id: str,
        source_hashes: list[str],
        error: str | None = None,
    ) -> None:
        hashes = [str(value) for value in source_hashes if value]
        if not hashes:
            return
        with _LOCK, _store(self.path) as conn:
            documents = dict(_load_documents(conn, f"reg:{self.path}")["documents"])
            state = {
                key: dict(entry) if isinstance(entry, dict) else entry
                for key, entry in documents.items()
            }
            now = utcnow()
            for source_hash in hashes:
                entry = state.get(source_hash)
                if not isinstance(entry, dict):
                    entry = {
                        "hash": source_hash,
                        "filename": "",
                        "status": "",
                        "job_id": "",
                        "created_at": now,
                    }
                else:
                    entry = dict(entry)
                entry["last_interrupted_job_id"] = job_id
                entry["last_interrupted_at"] = now
                entry["last_error"] = error or "Job interrupted."
                entry["updated_at"] = now
                state[source_hash] = entry
            self._commit_changed(conn, f"reg:{self.path}", before=documents, state=state)

    def delete_source(self, source_hash: str) -> dict[str, Any] | None:
        source_hash = str(source_hash or "")
        if not source_hash:
            return None
        with _LOCK, _store(self.path) as conn:
            cache_key = f"reg:{self.path}"
            documents = dict(_load_documents(conn, cache_key)["documents"])
            victim = documents.pop(source_hash, None)
            # The "before" snapshot keeps the victim so _commit_changed sees a
            # deletion rather than no-difference.
            before = documents
            if isinstance(victim, dict):
                before = {**documents, source_hash: victim}
            self._commit_changed(conn, cache_key, before=before, state=documents)
            return dict(victim) if isinstance(victim, dict) else None

    @staticmethod
    def _supersede_same_processed_paths(state: dict[str, Any]) -> None:
        """Global recomputation, kept byte-for-byte faithful to the legacy rule:

        a document whose processed_markdown_path belongs to a different active
        (ingested/indexed) document is marked ``superseded`` -- regardless of
        its previous status. Every mutation re-runs this so the stored state
        converges exactly like the old whole-file implementation.
        """
        pdfs = state
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

    @staticmethod
    def _commit_changed(
        conn: sqlite3.Connection,
        cache_key: str,
        *,
        before: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        """Persist only the rows whose payload actually changed."""
        before = before or {}
        changed: dict[str, Any] = {}
        deleted: set[str] = set()
        for key, entry in state.items():
            if before.get(key) != entry:
                changed[key] = entry
        for key in before:
            if key not in state:
                deleted.add(key)
        if changed or deleted:
            _write_documents(conn, cache_key, changed, deletions=deleted)


def source_map_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / SOURCE_MAP_FILENAME


def registry_state_version(registry_path: str | Path) -> str:
    return _state_version(Path(registry_path))


def source_map_state_version(processed_dir: str | Path) -> str:
    return _state_version(source_map_path(processed_dir))


def load_source_map(processed_dir: str | Path) -> dict[str, Any]:
    json_path = source_map_path(processed_dir)
    with _LOCK, _store(json_path) as conn:
        payload = _load_documents(conn, f"srcmap:{json_path}")
        documents: dict[str, Any] = {}
        for key, entry in payload["documents"].items():
            if isinstance(entry, dict):
                documents[key] = dict(entry)
        return {"version": REGISTRY_VERSION, "documents": documents}


def write_source_entry(
    *,
    processed_dir: str | Path,
    markdown_path: str | Path,
    source_hash: str,
    source_pdf_name: str,
    source_pdf_path: str | Path,
    source_size: int | None = None,
    source_mtime_ns: int | None = None,
) -> dict[str, Any]:
    markdown = Path(markdown_path)
    entry = {
        "source_hash": str(source_hash),
        "source_pdf_name": str(source_pdf_name),
        "source_pdf_path": str(source_pdf_path),
        "processed_markdown_path": str(markdown),
        "updated_at": utcnow(),
    }
    if source_size is not None:
        entry["source_size"] = int(source_size)
    if source_mtime_ns is not None:
        entry["source_mtime_ns"] = int(source_mtime_ns)
    json_path = source_map_path(processed_dir)
    # Single-row upsert: this used to rewrite the whole source map per ingested
    # PDF, which made bulk ingestion O(N^2) in metadata churn.
    with _LOCK, _store(json_path) as conn:
        _write_documents(conn, f"srcmap:{json_path}", {markdown.name: entry})
    return entry


def source_entry_for(
    processed_dir: str | Path,
    markdown_name: str,
) -> dict[str, Any]:
    """Point lookup of one source-map entry by markdown filename."""
    json_path = source_map_path(processed_dir)
    try:
        with _LOCK, _store(json_path) as conn:
            row = conn.execute(
                "SELECT payload FROM documents WHERE key = ?", (str(markdown_name),)
            ).fetchone()
    except sqlite3.Error:
        return {}
    if not row:
        return {}
    try:
        entry = json.loads(row[0])
    except (TypeError, ValueError):
        return {}
    return dict(entry) if isinstance(entry, dict) else {}


def source_entry_for_markdown(markdown_path: str | Path) -> dict[str, Any]:
    markdown = Path(markdown_path)
    return source_entry_for(markdown.parent, markdown.name)


def remove_source_entries_by_hash(
    processed_dir: str | Path,
    source_hashes: set[str],
) -> list[dict[str, Any]]:
    hashes = {str(value) for value in source_hashes if value}
    if not hashes:
        return []
    json_path = source_map_path(processed_dir)
    with _LOCK, _store(json_path) as conn:
        documents = dict(_load_documents(conn, f"srcmap:{json_path}")["documents"])
        removed: list[dict[str, Any]] = []
        deletions: set[str] = set()
        for markdown_name, entry in documents.items():
            if isinstance(entry, dict) and str(entry.get("source_hash", "")) in hashes:
                removed.append(dict(entry))
                deletions.add(markdown_name)
        if removed:
            _write_documents(conn, f"srcmap:{json_path}", {}, deletions=deletions)
        return removed
