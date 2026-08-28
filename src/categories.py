"""Category registry: user-created document categories ("split databases").

A category partitions the corpus into an independent LanceDB working directory
so documents can be segmented (e.g. general textbooks vs. team design docs),
queried individually or together, and moved between without re-embedding.

Layout contract:

* ``General`` is implicit and always exists: it IS the configured ``db/``
  directory, and a source with no membership row belongs to it. This keeps
  every pre-category index working unchanged.
* Each custom category owns the working directory ``<db>/categories/<key>``,
  derived from the anchor ``db/`` directory rather than persisted, so the
  whole tree moves with a re-anchored ``[paths].db_dir`` and tests can
  isolate by pointing the anchor at a temp dir.
* Membership is per SOURCE (document), keyed by ``source_hash`` -- the same
  identity the trust registry and index manifest use -- never per chunk.
* The store itself is SQLite (``<data_dir>/.categories.json.sqlite3``, beside
  the conventional legacy-JSON path) following the same compatibility
  pattern as :mod:`src.pdf_registry`.
"""

from __future__ import annotations

import contextlib
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


GENERAL_CATEGORY_KEY = "general"
CATEGORIES_FILENAME = ".categories.json"
CATEGORIES_DIRNAME = "categories"

# Category keys become directory names under <db>/categories/ and appear in
# API paths, so they are restricted to a conservative slug alphabet. Labels
# are free-form.
_CATEGORY_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_LOCK = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    key TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memberships (
    source_hash TEXT PRIMARY KEY,
    category_key TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def categories_path(data_dir: str | Path) -> Path:
    """Legacy-style JSON path; the SQLite store lives beside it."""
    return Path(data_dir) / CATEGORIES_FILENAME


def normalize_category_key(value: Any) -> str:
    """Lowercase/strip a client-supplied key; raises ValueError when invalid."""
    key = str(value or "").strip().lower()
    if not _CATEGORY_KEY_RE.match(key):
        raise ValueError(
            "Category key must be 1-64 characters of a-z, 0-9, '-' or '_' "
            "and start with a letter or digit."
        )
    return key


def slugify_category_key(value: Any) -> str:
    """Best-effort slug from free-form user input ("Team A" -> "team-a").

    Used by the create endpoint so a typed name can be accepted as-is; strict
    :func:`normalize_category_key` still has the final word, so input that
    slugifies to nothing (e.g. "///") is rejected. Lookups/targets never
    slugify -- they must match an existing key exactly.
    """
    text = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", text)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    return normalize_category_key(slug)


def normalize_category_label(value: Any, *, fallback: str = "") -> str:
    label = str(value or "").strip()
    return label[:120] if label else (fallback or "")


def category_db_dir(anchor_db_dir: str | Path, key: str) -> Path:
    """Resolve a category key to its working directory under ``anchor_db_dir``.

    General IS the anchor directory; custom categories nest under it so the
    whole tree relocates with ``[paths].db_dir``.
    """
    key = normalize_category_key(key)
    anchor = Path(anchor_db_dir)
    if key == GENERAL_CATEGORY_KEY:
        return anchor
    return anchor / CATEGORIES_DIRNAME / key


class CategoryStore:
    """SQLite-backed category + membership store for one ``data_dir``."""

    def __init__(self, data_dir: str | Path):
        self.path = categories_path(data_dir)

    # -- low-level ----------------------------------------------------------

    def _db_path(self) -> Path:
        return Path(str(self.path) + ".sqlite3")

    def _connect(self) -> sqlite3.Connection:
        db_path = self._db_path()
        if not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.executescript(_SCHEMA)
        return conn

    @contextlib.contextmanager
    def _store(self) -> Iterator[sqlite3.Connection]:
        with _LOCK:
            conn = self._connect()
            try:
                yield conn
            finally:
                conn.close()

    @staticmethod
    def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else None

    @staticmethod
    def _bump_version(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('data_version', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
        )

    def state_version(self) -> str:
        """Monotonic change token for ETag seeding ('0' until first write)."""
        try:
            with self._store() as conn:
                return self._meta_get(conn, "data_version") or "0"
        except sqlite3.Error:
            return "?"

    # -- categories ---------------------------------------------------------

    def list_categories(self, *, anchor_db_dir: str | Path | None = None) -> list[dict[str, Any]]:
        """All categories, General first, then custom keys in creation order.

        Each entry carries its resolved working directory when
        ``anchor_db_dir`` is supplied, plus ``exists`` (whether that directory
        contains an index yet).
        """
        rows: list[dict[str, Any]] = [
            {
                "key": GENERAL_CATEGORY_KEY,
                "label": "General",
                "created_at": "",
                "updated_at": "",
                "custom": False,
            }
        ]
        try:
            with self._store() as conn:
                cursor = conn.execute(
                    "SELECT key, label, created_at, updated_at FROM categories ORDER BY created_at, key"
                )
                for key, label, created_at, updated_at in cursor:
                    rows.append(
                        {
                            "key": str(key),
                            "label": str(label),
                            "created_at": str(created_at or ""),
                            "updated_at": str(updated_at or ""),
                            "custom": True,
                        }
                    )
        except sqlite3.Error:
            return rows

        if anchor_db_dir is not None:
            for entry in rows:
                db_dir = category_db_dir(anchor_db_dir, entry["key"])
                entry["db_dir"] = str(db_dir)
                entry["exists"] = (db_dir / "lancedb").exists()
        return rows

    def get_category(self, key: str, *, anchor_db_dir: str | Path | None = None) -> dict[str, Any] | None:
        key = normalize_category_key(key)
        if key == GENERAL_CATEGORY_KEY:
            entry = {
                "key": GENERAL_CATEGORY_KEY,
                "label": "General",
                "created_at": "",
                "updated_at": "",
                "custom": False,
            }
        else:
            try:
                with self._store() as conn:
                    row = conn.execute(
                        "SELECT key, label, created_at, updated_at FROM categories WHERE key = ?",
                        (key,),
                    ).fetchone()
            except sqlite3.Error:
                return None
            if not row:
                return None
            entry = {
                "key": str(row[0]),
                "label": str(row[1]),
                "created_at": str(row[2] or ""),
                "updated_at": str(row[3] or ""),
                "custom": True,
            }
        if anchor_db_dir is not None:
            db_dir = category_db_dir(anchor_db_dir, entry["key"])
            entry["db_dir"] = str(db_dir)
            entry["exists"] = (db_dir / "lancedb").exists()
        return entry

    def create_category(self, key: Any, label: Any = "") -> dict[str, Any]:
        key = normalize_category_key(key)
        if key == GENERAL_CATEGORY_KEY:
            raise ValueError(f"'{GENERAL_CATEGORY_KEY}' is reserved for the default category.")
        label = normalize_category_label(label, fallback=key)
        now = utcnow()
        with self._store() as conn:
            existing = conn.execute("SELECT key FROM categories WHERE key = ?", (key,)).fetchone()
            if existing:
                raise ValueError(f"Category '{key}' already exists.")
            with conn:
                conn.execute(
                    "INSERT INTO categories(key, label, created_at, updated_at) VALUES(?, ?, ?, ?)",
                    (key, label, now, now),
                )
                self._bump_version(conn)
        return {"key": key, "label": label, "created_at": now, "updated_at": now, "custom": True}

    def rename_category(self, key: Any, label: Any) -> dict[str, Any]:
        """Update a category's display label. Keys (and thus directories) are
        immutable once created, so index state and API paths stay stable."""
        key = normalize_category_key(key)
        if key == GENERAL_CATEGORY_KEY:
            raise ValueError("The General category label cannot be changed.")
        label = normalize_category_label(label)
        if not label:
            raise ValueError("A non-empty label is required.")
        with self._store() as conn:
            existing = conn.execute("SELECT key FROM categories WHERE key = ?", (key,)).fetchone()
            if not existing:
                raise ValueError(f"Unknown category: {key}")
            with conn:
                conn.execute(
                    "UPDATE categories SET label = ?, updated_at = ? WHERE key = ?",
                    (label, utcnow(), key),
                )
                self._bump_version(conn)
        return {"key": key, "label": label, "updated_at": utcnow(), "custom": True}

    def delete_category(self, key: Any) -> dict[str, Any]:
        """Remove an empty custom category. Refuses when sources are still
        members -- move them back to General (or another category) first, so a
        delete can never orphan indexed rows inside the category directory."""
        key = normalize_category_key(key)
        if key == GENERAL_CATEGORY_KEY:
            raise ValueError("The General category cannot be deleted.")
        with self._store() as conn:
            existing = conn.execute("SELECT key FROM categories WHERE key = ?", (key,)).fetchone()
            if not existing:
                raise ValueError(f"Unknown category: {key}")
            members = conn.execute(
                "SELECT COUNT(*) FROM memberships WHERE category_key = ?", (key,)
            ).fetchone()[0]
            if int(members):
                raise ValueError(
                    f"Category '{key}' still has {int(members)} document(s); "
                    "move them to another category before deleting it."
                )
            with conn:
                conn.execute("DELETE FROM categories WHERE key = ?", (key,))
                self._bump_version(conn)
        return {"key": key, "deleted": True}

    # -- memberships --------------------------------------------------------

    def set_memberships(self, source_hashes: list[str], key: Any) -> int:
        """Assign sources to a category. General membership is the ABSENCE of
        a row, so moving to General deletes the membership rows. Returns the
        number of sources updated."""
        key = normalize_category_key(key)
        hashes = sorted({str(value or "").strip() for value in source_hashes if str(value or "").strip()})
        if not hashes:
            return 0
        if key != GENERAL_CATEGORY_KEY:
            with self._store() as conn:
                existing = conn.execute("SELECT key FROM categories WHERE key = ?", (key,)).fetchone()
                if not existing:
                    raise ValueError(f"Unknown category: {key}")
        now = utcnow()
        with self._store() as conn:
            with conn:
                if key == GENERAL_CATEGORY_KEY:
                    conn.executemany(
                        "DELETE FROM memberships WHERE source_hash = ?",
                        [(h,) for h in hashes],
                    )
                else:
                    conn.executemany(
                        "INSERT INTO memberships(source_hash, category_key, updated_at) VALUES(?, ?, ?) "
                        "ON CONFLICT(source_hash) DO UPDATE SET "
                        "category_key = excluded.category_key, updated_at = excluded.updated_at",
                        [(h, key, now) for h in hashes],
                    )
                self._bump_version(conn)
        return len(hashes)

    def memberships_for(self, source_hashes: list[str]) -> dict[str, str]:
        """``{source_hash: category_key}`` for the requested hashes; hashes
        with no row are absent (they belong to General)."""
        hashes = list({str(value or "") for value in source_hashes if str(value or "")})
        if not hashes:
            return {}
        # Chunked IN clauses: SQLite's host-parameter ceiling (999 on older
        # builds) must not cap how many memberships a corpus-wide lookup asks
        # for. Result order is irrelevant; callers key on the hash.
        found: dict[str, str] = {}
        try:
            with self._store() as conn:
                for start in range(0, len(hashes), 500):
                    chunk = hashes[start : start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT source_hash, category_key FROM memberships WHERE source_hash IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for row in rows:
                        found[str(row[0])] = str(row[1])
        except sqlite3.Error:
            return found
        return found

    def all_memberships(self) -> dict[str, str]:
        """``{source_hash: category_key}`` for every membership row."""
        try:
            with self._store() as conn:
                rows = conn.execute("SELECT source_hash, category_key FROM memberships").fetchall()
        except sqlite3.Error:
            return {}
        return {str(row[0]): str(row[1]) for row in rows}

    def membership_counts(self) -> dict[str, int]:
        """``{category_key: source_count}`` for custom categories only."""
        try:
            with self._store() as conn:
                rows = conn.execute(
                    "SELECT category_key, COUNT(*) FROM memberships GROUP BY category_key"
                ).fetchall()
        except sqlite3.Error:
            return {}
        return {str(row[0]): int(row[1]) for row in rows}
