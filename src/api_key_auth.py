"""Per-user API-key authentication, rate limiting, and usage tracking.

This module layers on top of the legacy single shared ``[server] api_token``
(which stays available as an admin/owner *master bypass*, counted under the
synthetic key id ``MASTER_KEY_ID``). Everyone else authenticates with a
per-user API key issued offline by ``scripts/manage_api_keys.py``.

Design follows the rest of the pipeline:

* State is a JSON sidecar under ``data/`` (``.api_keys.json``), written
  atomically via :func:`src.atomic_io.write_json_atomic` and guarded by an
  in-process ``threading.Lock`` plus a cross-process ``portalocker`` lock --
  the same pattern as ``pdf_registry.py``.
* Only **hashed** keys are stored (``sha256`` hex of the full secret). The
  plaintext is shown exactly once at creation time, GitHub-token style. A short
  human-readable ``prefix`` (``rag_…<last4>``) is stored so admins can identify
  keys in listings without the secret.
* Rate limiting is a per-key sliding 60s window kept in memory (single-worker
  uvicorn deployment). Usage counters likewise live in memory and are flushed
  to disk every N increments and on shutdown, so a request never blocks on disk
  for bookkeeping.

Permission sets: every key is assigned a named permission set that defines an
allowlist of document categories (or ``["*"]`` for all), a ``can_write`` flag
(upload/edit/delete inside the allowed categories), and an ``admin`` flag (key
and permission-set management, category management, maintenance). Sets live in
the same store under ``"permission_sets"``; two builtin sets (``admin`` and
``user``) are provisioned automatically and act as the migration target for the
legacy two-value ``role`` field, which is still persisted (derived from the
set's admin flag) so older tooling keeps working.

The server is always in an authentication posture: loopback clients are
auto-authenticated as the synthetic ``LOCAL_KEY_ID`` admin identity (built by
:func:`local_auth_result`), and every other client must present the master
token or a valid key. There is no open mode.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.atomic_io import write_json_atomic
from src.file_lock import acquire_registry_lock

# The synthetic identity the master token authenticates as. Never a real key id.
MASTER_KEY_ID = "__master__"
MASTER_PREFIX = "master"
# The synthetic identity auto-assigned to loopback (127.0.0.1/::1) clients.
LOCAL_KEY_ID = "__local__"
LOCAL_LABEL = "localhost"

# Key format: "<prefix>_<base62 secret>". The secret is opaque and unguessable.
DEFAULT_KEY_PREFIX = "rag_"
SECRET_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SECRET_LENGTH = 32

logger = logging.getLogger(__name__)

STORE_VERSION = 2
STORE_FILENAME = ".api_keys.json"
STATUS_ACTIVE = "active"
STATUS_DISABLED = "disabled"
ROLE_USER = "user"
ROLE_ADMIN = "admin"
VALID_ROLES = {ROLE_USER, ROLE_ADMIN}

# Permission sets ----------------------------------------------------------------
# A set's ``categories`` is either the wildcard ["*"] (every category, including
# ones created later) or an explicit list of category keys. ``can_write`` gates
# uploads/edits/deletes inside the allowed categories; ``admin`` gates key and
# permission-set management, category management, and index maintenance.
PERMSET_WILDCARD = "*"
PERMSET_ADMIN = "admin"
PERMSET_USER = "user"
BUILTIN_PERMSETS = (PERMSET_ADMIN, PERMSET_USER)
# Same slug rules as category keys (src/categories.normalize_category_key).
_PERMSET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CATEGORY_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
GENERAL_CATEGORY_KEY = "general"


def normalize_permission_set_name(raw: Any) -> str:
    """Validate + normalize a permission-set name (slug); raises ``ValueError``."""
    name = str(raw or "").strip().lower()
    if not _PERMSET_NAME_RE.match(name):
        raise ValueError(
            f"Invalid permission set name {raw!r}; expected a short slug "
            "(lowercase letters, digits, '-', '_')."
        )
    return name


def normalize_permission_categories(raw: Any) -> list[str]:
    """Coerce a category allowlist into a stored list.

    Accepts ``["*"]`` (all categories), a list of slugs, a comma-separated
    string, or ``None`` (treated as the wildcard). ``general`` is always
    allowed as an entry. Raises ``ValueError`` on malformed entries.
    """
    if raw is None or raw == "":
        return [PERMSET_WILDCARD]
    if isinstance(raw, str):
        items = [part for part in (chunk.strip().lower() for chunk in raw.split(",")) if part]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(part or "").strip().lower() for part in raw]
        items = [part for part in items if part]
    else:
        raise ValueError(f"Invalid category list: {raw!r}")
    if not items or PERMSET_WILDCARD in items:
        return [PERMSET_WILDCARD]
    normalized: list[str] = []
    for item in items:
        if item == GENERAL_CATEGORY_KEY:
            key = item
        elif not _CATEGORY_KEY_RE.match(item):
            raise ValueError(
                f"Invalid category key {item!r}; expected a short slug "
                "(lowercase letters, digits, '-', '_') or '*'."
            )
        else:
            key = item
        if key not in normalized:
            normalized.append(key)
    return normalized

# Window (seconds) for the sliding rate-limit bucket.
RATE_WINDOW_SECONDS = 60


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat(timespec="seconds")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # fromisoformat handles the "seconds" precision we write (no tz tricks).
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    # Hand-edited stores may hold tz-less timestamps; comparing those against
    # the aware _utcnow() raises TypeError, so anchor them to UTC first.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_expired(expires_at: str | None) -> bool:
    parsed = _parse_iso(expires_at)
    if parsed is None:
        return False
    return parsed <= _utcnow()


def generate_secret(*, length: int = SECRET_LENGTH) -> str:
    """Return a cryptographically random base62 secret (no prefix)."""
    return "".join(secrets.choice(SECRET_ALPHABET) for _ in range(length))


def generate_full_key(prefix: str = DEFAULT_KEY_PREFIX) -> str:
    """Return a full key string ``<prefix><secret>`` ready to hand to a user."""
    return f"{prefix}{generate_secret()}"


def hash_key(full_key: str) -> str:
    """Return the ``sha256`` hex digest of a full key string.

    Only the hash is ever persisted, so a leaked store file reveals nothing.
    """
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


def make_prefix(full_key: str, *, prefix: str = DEFAULT_KEY_PREFIX) -> str:
    """Return a display-only prefix ``<key-prefix-head>…<last4>``.

    Only the caller-known head (e.g. ``rag_``) and the last four secret
    characters are shown. The middle of the secret is NEVER stored, listed,
    or logged -- the store must not contain anything a reader could turn
    back into the full key.
    """
    secret = str(full_key or "")
    if len(secret) < 8:
        return secret
    head = str(prefix or "")
    if not head or not secret.startswith(head):
        head = secret[:4]
    return f"{head}…{secret[-4:]}"


def _parse_expires(raw: str | None) -> str | None:
    """Coerce a user-supplied expiry (e.g. ``2026-12-31``) into a stored value.

    Accepts ``YYYY-MM-DD`` (treated as end-of-that-day UTC) or a full ISO
    timestamp. Empty/None clears the expiry.
    """
    if not raw or not str(raw).strip():
        return None
    text = str(raw).strip()
    # Bare date -> end of day in UTC.
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:
            return datetime.fromisoformat(text).replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            ).isoformat(timespec="seconds")
        except ValueError:
            pass
    parsed = _parse_iso(text)
    if parsed is None:
        raise ValueError(f"Unrecognized expiry value: {raw!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat(timespec="seconds")


@dataclass(frozen=True)
class AuthResult:
    """Outcome of a successful authentication (always a valid identity)."""

    key_id: str
    role: str
    label: str
    # Effective per-minute rate limit for this identity (override or default).
    rate_limit_per_minute: int
    is_master: bool
    # Name of the permission set this identity resolves to. ``role`` is kept in
    # sync (admin iff the set has the admin flag) for legacy callers.
    permission_set: str = PERMSET_USER
    # True for the synthetic loopback identity (127.0.0.1/::1 auto-auth).
    is_local: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


def local_auth_result(*, default_rate_limit: int) -> AuthResult:
    """Full-admin identity for loopback clients (never rate-limited or tracked;
    the middleware returns it before ``authenticate`` is consulted)."""
    return AuthResult(
        key_id=LOCAL_KEY_ID,
        role=ROLE_ADMIN,
        label=LOCAL_LABEL,
        rate_limit_per_minute=max(1, int(default_rate_limit)),
        is_master=False,
        permission_set=PERMSET_ADMIN,
        is_local=True,
    )


@dataclass(frozen=True)
class Rejection:
    """Why a credential was rejected, with the HTTP status to return."""

    status_code: int
    detail: str
    retry_after: int | None = None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class KeyStore:
    """JSON-backed registry of API keys and permission sets, keyed by ``sha256`` hash.

    The on-disk shape is::

        {"version": 2,
         "permission_sets": {
            "admin": {"name": "admin", "label": "Administrator",
                      "categories": ["*"], "can_write": true, "admin": true,
                      "builtin": true, "created_at": "...", "updated_at": "..."},
            "user": {...}, ...},
         "keys": {
            "<sha256 hex>": {
                "label": "alice laptop",
                "prefix": "rag_…a1b2",
                "status": "active",
                "role": "user",
                "permission_set": "user",
                "created_at": "2026-07-30T...",
                "expires_at": null,
                "rate_limit_per_minute": null,
                "usage": {"requests": 0, "last_used_at": null, "last_used_ip": null}
            }, ...}}

    Reads are cached on the file's ``(mtime_ns, size)`` signature so the hot
    request path does not re-parse JSON on every call. Writes invalidate the
    cache. All mutation goes through ``_lock_for`` which holds an in-process
    ``threading.Lock`` and a best-effort cross-process ``portalocker`` lock.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache: tuple[tuple[int, int], dict[str, Any]] | None = None
        # True when the store FILE exists but could not be read/parsed. Auth
        # must fail CLOSED in that state: treating an unreadable store as
        # empty would silently disable authentication for every request.
        self._unreadable = False

    # -- locking -----------------------------------------------------------
    @contextlib.contextmanager
    def _lock_for(self) -> Iterator[None]:
        """Hold the in-process lock and a best-effort cross-process lock.

        Only the cross-process ACQUIRE may fail soft (falling back to the
        in-process lock alone). A failure raised by the wrapped body -- e.g.
        a disk-full OSError from ``_save_raw`` -- must propagate unchanged;
        swallowing it here would yield twice from this generator and surface
        as a contextlib RuntimeError instead of the real error.
        """
        with self._lock:
            with contextlib.ExitStack() as stack:
                try:
                    stack.enter_context(
                        acquire_registry_lock(self.path.parent, timeout=30.0)
                    )
                except (TimeoutError, OSError):
                    # Fall back to the in-process lock alone rather than
                    # failing the mutation (matches
                    # pdf_registry._registry_lock_for behavior).
                    pass
                yield

    # -- low-level io ------------------------------------------------------
    def _load_raw(self) -> dict[str, Any]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            # A missing store is a legitimate fresh deployment: empty and open.
            self._cache = None
            self._unreadable = False
            return self._empty()
        except OSError:
            self._cache = None
            self._unreadable = True
            return self._empty()
        signature = (stat.st_mtime_ns, stat.st_size)
        if self._cache is not None and self._cache[0] == signature:
            return self._cache[1]
        try:
            payload = json_loads_dict(self.path)
        except (OSError, ValueError):
            # A torn write should be impossible (writes are atomic), so a parse
            # error here means the file is temporarily unreadable (e.g. an
            # antivirus lock on Windows) or truly corrupt. Fail CLOSED: report
            # an empty payload but keep ``_unreadable`` set so ``has_any_key``
            # still reports keys exist and gating stays on, instead of silently
            # disabling authentication. The next successful write clears it.
            self._cache = None
            self._unreadable = True
            return self._empty()
        normalized = self._normalize(payload)
        self._cache = (signature, normalized)
        self._unreadable = False
        return normalized

    @staticmethod
    def _empty() -> dict[str, Any]:
        # A missing store (fresh deployment) still resolves the builtin
        # permission sets, so key creation works before the first write.
        return {
            "version": STORE_VERSION,
            "keys": {},
            "permission_sets": KeyStore._default_permission_sets(),
        }

    @staticmethod
    def _default_permission_sets() -> dict[str, dict[str, Any]]:
        """The two builtin sets provisioned into every (fresh or v1) store."""
        now = _utcnow_iso()
        return {
            PERMSET_ADMIN: {
                "name": PERMSET_ADMIN,
                "label": "Administrator",
                "categories": [PERMSET_WILDCARD],
                "can_write": True,
                "admin": True,
                "builtin": True,
                "created_at": now,
                "updated_at": now,
            },
            PERMSET_USER: {
                "name": PERMSET_USER,
                "label": "Standard user",
                "categories": [PERMSET_WILDCARD],
                "can_write": True,
                "admin": False,
                "builtin": True,
                "created_at": now,
                "updated_at": now,
            },
        }

    @staticmethod
    def _normalize_permission_set(name: str, record: Any) -> dict[str, Any] | None:
        """Coerce one stored set record into shape; ``None`` drops garbage rows."""
        if not isinstance(record, dict):
            return None
        try:
            name = normalize_permission_set_name(name)
        except ValueError:
            return None
        record = dict(record)
        record["name"] = name
        label = str(record.get("label") or "").strip()
        record["label"] = label or name
        try:
            categories = normalize_permission_categories(record.get("categories"))
        except ValueError:
            categories = [PERMSET_WILDCARD]
        record["categories"] = categories
        record["can_write"] = bool(record.get("can_write", True))
        record["admin"] = bool(record.get("admin", False))
        if name == PERMSET_ADMIN:
            # The builtin admin set can be renamed in label only; its powers
            # are part of the bootstrap story and must never be edited away.
            record["admin"] = True
            record["can_write"] = True
            record["categories"] = [PERMSET_WILDCARD]
        record["builtin"] = bool(record.get("builtin", False)) or name in BUILTIN_PERMSETS
        record.setdefault("created_at", _utcnow_iso())
        record["updated_at"] = str(record.get("updated_at") or record["created_at"])
        return record

    @staticmethod
    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        payload.setdefault("version", STORE_VERSION)
        keys = payload.get("keys")
        if not isinstance(keys, dict):
            payload["keys"] = {}
        # Permission sets: provision builtins into fresh/v1 stores, then coerce
        # every record (hand-edited or written by an older version).
        sets = payload.get("permission_sets")
        if not isinstance(sets, dict):
            sets = {}
            payload["permission_sets"] = sets
        defaults = KeyStore._default_permission_sets()
        for name, record in defaults.items():
            sets.setdefault(name, record)
        for name in list(sets):
            normalized = KeyStore._normalize_permission_set(name, sets[name])
            if normalized is None:
                del sets[name]
            else:
                sets[name] = normalized
        # Ensure each record has the expected fields. The legacy two-value
        # ``role`` migrates to a permission set once; ``role`` itself stays
        # stored (derived from the set's admin flag) for older tooling.
        for record in payload["keys"].values():
            if not isinstance(record, dict):
                continue
            record.setdefault("label", "")
            record.setdefault("prefix", "")
            record.setdefault("status", STATUS_ACTIVE)
            legacy_role = record.get("role", ROLE_USER)
            record.setdefault(
                "permission_set",
                PERMSET_ADMIN if legacy_role == ROLE_ADMIN else PERMSET_USER,
            )
            if record["permission_set"] not in sets:
                # Unknown set (hand-edited store): fail toward the builtin user
                # set rather than a 401 on every request from a typo.
                record["permission_set"] = (
                    PERMSET_ADMIN if legacy_role == ROLE_ADMIN else PERMSET_USER
                )
            record["role"] = ROLE_ADMIN if sets[record["permission_set"]].get("admin") else ROLE_USER
            record.setdefault("created_at", _utcnow_iso())
            record.setdefault("expires_at", None)
            record.setdefault("rate_limit_per_minute", None)
            usage = record.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            usage.setdefault("requests", 0)
            usage.setdefault("last_used_at", None)
            usage.setdefault("last_used_ip", None)
            record["usage"] = usage
        return payload

    def _save_raw(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.path, payload)
        # Invalidate; next read repopulates under the fresh signature.
        self._cache = None

    # -- public read API ---------------------------------------------------
    def load(self) -> dict[str, Any]:
        with self._lock:
            return self._load_raw()

    def has_any_key(self) -> bool:
        """True if the store contains at least one key record (any status).

        Also True when the store file exists but cannot currently be read
        (locked, corrupt): callers gate authentication on this, so an unreadable
        store must look "keys exist" (fail closed) rather than "no keys"
        (which would disable auth entirely).
        """
        if self._unreadable:
            return True
        return bool(self.load().get("keys"))

    def get_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        return self.load().get("keys", {}).get(key_hash)

    def list_keys(self) -> list[dict[str, Any]]:
        """Return all key records with their hash id under ``key_id``."""
        payload = self.load()
        records: list[dict[str, Any]] = []
        for key_hash, record in payload.get("keys", {}).items():
            out = dict(record)
            out["key_id"] = key_hash
            records.append(out)
        return records

    def find_by_prefix(self, prefix: str) -> dict[str, Any] | None:
        """Locate a record by its display ``prefix`` (e.g. ``rag_…a1b2``).

        Prefixes are unique by construction (secret is random 32 chars), so the
        first match wins. Returns ``None`` when not found.
        """
        for record in self.list_keys():
            if record.get("prefix") == prefix:
                return record
        return None

    # -- permission sets: read --------------------------------------------
    def list_permission_sets(self) -> list[dict[str, Any]]:
        """All permission set records, builtin sets first, then by name."""
        sets = self.load().get("permission_sets", {})
        ordered = sorted(
            sets.values(),
            key=lambda record: (0 if record.get("builtin") else 1, str(record.get("name"))),
        )
        return [dict(record) for record in ordered]

    def get_permission_set(self, name: str) -> dict[str, Any] | None:
        record = self.load().get("permission_sets", {}).get(str(name or "").strip().lower())
        return dict(record) if record else None

    def permission_set_key_counts(self) -> dict[str, int]:
        """Keys assigned per permission-set name (for delete guarding + UI)."""
        counts: dict[str, int] = {}
        for record in self.load().get("keys", {}).values():
            name = str(
                record.get("permission_set")
                or (PERMSET_ADMIN if record.get("role") == ROLE_ADMIN else PERMSET_USER)
            )
            counts[name] = counts.get(name, 0) + 1
        return counts

    # -- public mutation API ----------------------------------------------
    def add(self, record: dict[str, Any]) -> None:
        with self._lock_for():
            payload = self._load_raw()
            payload["keys"][record["key_hash"]] = record["stored"]
            self._save_raw(payload)

    def update(self, key_hash: str, mutate: dict[str, Any]) -> dict[str, Any] | None:
        """Merge ``mutate`` into the record at ``key_hash``; return the new record."""
        with self._lock_for():
            payload = self._load_raw()
            record = payload["keys"].get(key_hash)
            if record is None:
                return None
            record.update(mutate)
            self._save_raw(payload)
            return {**record, "key_id": key_hash}

    def delete(self, key_hash: str) -> bool:
        with self._lock_for():
            payload = self._load_raw()
            if key_hash not in payload["keys"]:
                return False
            del payload["keys"][key_hash]
            self._save_raw(payload)
            return True

    def replace_key(self, old_hash: str, new_hash: str, stored: dict[str, Any]) -> bool:
        """Atomically remove ``old_hash`` and insert ``new_hash`` -> ``stored``.

        Performed as a single load-modify-save under one lock so a crash between
        a separate ``add`` and ``delete`` cannot leave BOTH the old (possibly
        compromised) and new secrets active (fail-open). Returns True if
        ``old_hash`` was present and replaced, False if it was missing (in which
        case the store is left untouched -- the caller decides whether to treat
        that as an error).
        """
        with self._lock_for():
            payload = self._load_raw()
            keys = payload["keys"]
            if old_hash not in keys:
                return False
            del keys[old_hash]
            keys[new_hash] = stored
            self._save_raw(payload)
            return True

    def rotate_stored(self, key_hash: str, build_stored) -> str | None:
        """Replace ``key_hash`` in one locked write, deriving the new record
        inside the lock via ``build_stored(record) -> (new_hash, stored)``.

        Reading the existing record inside the same lock as the write means a
        concurrent ``update``/``set_role`` cannot be silently reverted by a
        stale read (the rotate always sees the latest record).
        Returns the new hash, or ``None`` if ``key_hash`` was not found.
        """
        with self._lock_for():
            payload = self._load_raw()
            keys = payload["keys"]
            record = keys.get(key_hash)
            if record is None:
                return None
            new_hash, stored = build_stored(record)
            del keys[key_hash]
            keys[new_hash] = stored
            self._save_raw(payload)
            return new_hash

    def merge_usage(self, deltas: dict[str, dict[str, Any]]) -> None:
        """Apply accumulated usage deltas (``requests``/``last_used_*``)."""
        if not deltas:
            return
        with self._lock_for():
            payload = self._load_raw()
            keys = payload["keys"]
            changed = False
            for key_hash, delta in deltas.items():
                record = keys.get(key_hash)
                if record is None:
                    continue
                usage = record.setdefault("usage", {})
                usage["requests"] = int(usage.get("requests", 0)) + int(delta.get("requests", 0))
                if delta.get("last_used_at"):
                    usage["last_used_at"] = delta["last_used_at"]
                if delta.get("last_used_ip"):
                    usage["last_used_ip"] = delta["last_used_ip"]
                changed = True
            if changed:
                self._save_raw(payload)

    # -- permission sets: mutation ----------------------------------------
    def add_permission_set(
        self,
        *,
        name: str,
        label: str = "",
        categories: Any = None,
        can_write: bool = True,
        admin: bool = False,
    ) -> dict[str, Any]:
        """Create a permission set. Raises ``ValueError`` on bad name/dup."""
        name = normalize_permission_set_name(name)
        if name in BUILTIN_PERMSETS:
            raise ValueError(f"Permission set {name!r} is built in; it always exists.")
        normalized_categories = normalize_permission_categories(categories)
        with self._lock_for():
            payload = self._load_raw()
            sets = payload["permission_sets"]
            if name in sets:
                raise ValueError(f"Permission set {name!r} already exists.")
            now = _utcnow_iso()
            record = {
                "name": name,
                "label": str(label or "").strip() or name,
                "categories": normalized_categories,
                "can_write": bool(can_write),
                "admin": bool(admin),
                "builtin": False,
                "created_at": now,
                "updated_at": now,
            }
            sets[name] = record
            self._save_raw(payload)
            return dict(record)

    def update_permission_set(self, name: str, mutate: dict[str, Any]) -> dict[str, Any] | None:
        """Merge ``mutate`` into a set. Builtins keep their structural flags."""
        name = normalize_permission_set_name(name)
        with self._lock_for():
            payload = self._load_raw()
            sets = payload["permission_sets"]
            record = sets.get(name)
            if record is None:
                return None
            merged = {**record, **(mutate or {})}
            if "categories" in mutate:
                merged["categories"] = normalize_permission_categories(mutate["categories"])
            if name == PERMSET_ADMIN:
                merged["admin"] = True
                merged["can_write"] = True
                merged["categories"] = [PERMSET_WILDCARD]
            merged["name"] = name
            merged["label"] = str(merged.get("label") or "").strip() or name
            merged["can_write"] = bool(merged.get("can_write", True))
            merged["admin"] = bool(merged.get("admin", False))
            merged["builtin"] = bool(merged.get("builtin", False)) or name in BUILTIN_PERMSETS
            merged["created_at"] = str(record.get("created_at") or _utcnow_iso())
            merged["updated_at"] = _utcnow_iso()
            sets[name] = merged
            self._sync_key_roles(payload, name)
            self._save_raw(payload)
            return dict(merged)

    def delete_permission_set(self, name: str) -> bool:
        """Delete a custom set. Raises ``ValueError`` for builtins and for sets
        still assigned to at least one key (deleting those would orphan the
        keys' authorization); returns False when the set does not exist."""
        name = normalize_permission_set_name(name)
        if name in BUILTIN_PERMSETS:
            raise ValueError(f"Permission set {name!r} is built in and cannot be deleted.")
        with self._lock_for():
            payload = self._load_raw()
            sets = payload["permission_sets"]
            if name not in sets:
                return False
            in_use = any(
                str(record.get("permission_set")) == name
                for record in payload["keys"].values()
            )
            if in_use:
                raise ValueError(
                    f"Permission set {name!r} is still assigned to at least one API key."
                )
            del sets[name]
            self._save_raw(payload)
            return True

    @staticmethod
    def _sync_key_roles(payload: dict[str, Any], set_name: str) -> None:
        """Keep the legacy ``role`` field in step with a set's admin flag."""
        admin = bool(payload["permission_sets"].get(set_name, {}).get("admin"))
        role = ROLE_ADMIN if admin else ROLE_USER
        for record in payload["keys"].values():
            if isinstance(record, dict) and record.get("permission_set") == set_name:
                record["role"] = role

    def repair_leaked_prefixes(self) -> int:
        """Rewrite legacy prefixes that embedded the whole secret.

        An earlier ``make_prefix`` stored ``<secret[:-4]>…<secret[-4:]>``,
        which contained the full key (the ellipsis hid nothing). Removing the
        ellipsis from such a value reproduces the key, and the record's
        sha256 id lets us VERIFY that reconstruction before rewriting -- a
        candidate that does not hash to the record's id is left untouched.
        Returns the number of records repaired.
        """
        repaired = 0
        with self._lock_for():
            payload = self._load_raw()
            keys = payload.get("keys", {})
            for key_hash, record in keys.items():
                if not isinstance(record, dict):
                    continue
                display = str(record.get("prefix") or "")
                if "…" not in display:
                    continue
                candidate = display.replace("…", "")
                # A genuine display prefix only ever held the ``rag_`` head;
                # anything longer than head+4 chars may have been the
                # leaked-secret form and is worth checking.
                if len(candidate) <= len(DEFAULT_KEY_PREFIX) + 4:
                    continue
                if hash_key(candidate) != key_hash:
                    continue
                record["prefix"] = make_prefix(candidate)
                repaired += 1
            if repaired:
                self._save_raw(payload)
        return repaired


def json_loads_dict(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window per-identity rate limiter (in memory).

    Each identity (key hash or :data:`MASTER_KEY_ID`) has a deque of request
    timestamps within the last ``window`` seconds. A request is allowed when
    the count after pruning is below ``limit``; otherwise rejected with a
    ``retry_after`` hint in seconds.

    Single-worker uvicorn assumption: state lives only in this process. For a
    multi-worker deployment this would need a shared store (Redis).
    """

    def __init__(self, *, window_seconds: float = RATE_WINDOW_SECONDS):
        self._window = window_seconds
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = {}

    def check(self, identity: str, limit: int) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``.

        ``retry_after`` is 0 when allowed, otherwise an estimate of how long
        until enough timestamps age out of the window.
        """
        if limit <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = deque()
                self._buckets[identity] = bucket
            # Drop timestamps outside the window.
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) < limit:
                bucket.append(now)
                return True, 0
            # Over limit: earliest timestamp ages out at cutoff + window.
            retry_after = max(1, int(bucket[0] + self._window - now) + 1)
            return False, retry_after


# ---------------------------------------------------------------------------
# Usage tracker
# ---------------------------------------------------------------------------


class UsageTracker:
    """In-memory usage accumulator that flushes to the store on a throttle.

    The hot request path only mutates an in-process counter. Every
    ``persist_interval`` increments (per identity) the deltas are merged into
    the store. :meth:`flush` forces a write (called on shutdown).
    """

    def __init__(self, store: KeyStore, *, persist_interval: int):
        self._store = store
        self._persist_interval = max(1, int(persist_interval))
        self._lock = threading.Lock()
        # identity -> {"requests": int, "last_used_at": str|None, "last_used_ip": str|None}
        self._pending: dict[str, dict[str, Any]] = {}

    def record(self, identity: str, *, last_used_at: str, last_used_ip: str | None) -> None:
        with self._lock:
            entry = self._pending.setdefault(
                identity, {"requests": 0, "last_used_at": None, "last_used_ip": None}
            )
            entry["requests"] += 1
            entry["last_used_at"] = last_used_at
            if last_used_ip:
                entry["last_used_ip"] = last_used_ip
            should_flush = entry["requests"] >= self._persist_interval
        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Persist and clear all pending deltas. Safe to call from any thread.

        If the store write fails (disk full, transient lock timeout), the
        deltas are merged back into ``_pending`` so the usage they represent is
        retried on the next flush instead of being silently lost.
        """
        with self._lock:
            if not self._pending:
                return
            deltas = self._pending
            self._pending = {}
        try:
            self._store.merge_usage(deltas)
        except OSError:
            with self._lock:
                for identity, entry in deltas.items():
                    pending = self._pending.setdefault(
                        identity,
                        {"requests": 0, "last_used_at": None, "last_used_ip": None},
                    )
                    pending["requests"] += int(entry.get("requests", 0))
                    if entry.get("last_used_at"):
                        pending["last_used_at"] = entry["last_used_at"]
                    if entry.get("last_used_ip"):
                        pending["last_used_ip"] = entry["last_used_ip"]


# ---------------------------------------------------------------------------
# Authenticator
# ---------------------------------------------------------------------------


class ApiKeyAuthenticator:
    """Resolves a supplied credential into an :class:`AuthResult` or rejection.

    The authenticator holds a :class:`KeyStore`, :class:`RateLimiter`, and
    :class:`UsageTracker`. The middleware constructs one instance at startup
    and calls :meth:`authenticate` per request. The admin script uses the same
    store + CRUD helpers directly.
    """

    def __init__(
        self,
        store: KeyStore,
        *,
        default_rate_limit: int,
        persist_interval: int,
        window_seconds: float = RATE_WINDOW_SECONDS,
    ):
        self.store = store
        self.default_rate_limit = max(1, int(default_rate_limit))
        self.rate_limiter = RateLimiter(window_seconds=window_seconds)
        self.usage = UsageTracker(store, persist_interval=persist_interval)

    # -- core --------------------------------------------------------------
    def authenticate(
        self,
        supplied: str | None,
        *,
        master_token: str,
        client_ip: str | None = None,
        track: bool = True,
    ) -> tuple[AuthResult | None, Rejection | None]:
        """Resolve ``supplied`` into an identity or a rejection.

        * If ``master_token`` is set and matches (constant-time) -> admin.
        * Else look up ``sha256(supplied)`` in the store; must be active,
          unexpired, and its permission set must still exist (fail closed).
          The resulting ``role`` mirrors the set's ``admin`` flag.

        Authentication has no open mode: with no master token and an empty
        store every supplied credential is simply rejected. Loopback clients
        are handled before this method (see :func:`local_auth_result`).

        On success, the rate limit is enforced and usage is recorded (unless
        ``track`` is False). Returns ``(AuthResult, None)`` on success or
        ``(None, Rejection)`` on rejection.
        """
        supplied_str = (supplied or "").strip()
        if not supplied_str:
            return None, Rejection(401, "An API key is required. Provide it via the X-API-Token header.")

        # Master bypass. compare_digest on str inputs requires ASCII-only text
        # and raises TypeError otherwise (a non-ASCII credential must yield a
        # clean rejection, not a 500), so compare encoded bytes instead.
        if (
            master_token
            and hmac.compare_digest(supplied_str.encode("utf-8"), master_token.encode("utf-8"))
        ):
            result = AuthResult(
                key_id=MASTER_KEY_ID,
                role=ROLE_ADMIN,
                label="master",
                rate_limit_per_minute=self.default_rate_limit,
                is_master=True,
                permission_set=PERMSET_ADMIN,
            )
            return self._enforce_limit_and_track(result, client_ip, track)

        key_hash = hash_key(supplied_str)
        record = self.store.get_by_hash(key_hash)
        if record is None:
            return None, Rejection(401, "Invalid or missing API key.")

        status = record.get("status", STATUS_ACTIVE)
        if status == STATUS_DISABLED:
            return None, Rejection(401, "This API key has been disabled.")

        if _is_expired(record.get("expires_at")):
            return None, Rejection(401, "This API key has expired.")

        permission_set = self.resolve_permission_set(record)
        if permission_set is None:
            # Should be unreachable (normalize reassigns unknown sets), but a
            # hand-edited store must fail closed, never open.
            return None, Rejection(401, "This API key's permission set no longer exists.")

        per_key_limit = record.get("rate_limit_per_minute")
        try:
            limit = int(per_key_limit) if per_key_limit else self.default_rate_limit
        except (TypeError, ValueError):
            limit = self.default_rate_limit

        result = AuthResult(
            key_id=key_hash,
            role=ROLE_ADMIN if permission_set.get("admin") else ROLE_USER,
            label=str(record.get("label", "")),
            rate_limit_per_minute=limit,
            is_master=False,
            permission_set=str(permission_set.get("name") or PERMSET_USER),
        )
        return self._enforce_limit_and_track(result, client_ip, track)

    def resolve_permission_set(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """The permission-set record a key record resolves to, or ``None``."""
        name = str(
            record.get("permission_set")
            or (PERMSET_ADMIN if record.get("role") == ROLE_ADMIN else PERMSET_USER)
        )
        found = self.store.get_permission_set(name)
        if found is None:
            return None
        return found

    def _enforce_limit_and_track(
        self, result: AuthResult, client_ip: str | None, track: bool
    ) -> tuple[AuthResult, None]:
        allowed, retry_after = self.rate_limiter.check(result.key_id, result.rate_limit_per_minute)
        if not allowed:
            # Rate-limited identities are still authenticated; surface a 429.
            raise RateLimitExceeded(retry_after)
        if track:
            self.usage.record(result.key_id, last_used_at=_utcnow_iso(), last_used_ip=client_ip)
        return result, None

    # -- CRUD used by both middleware state and the admin script ------------
    def create_key(
        self,
        *,
        label: str,
        role: str = ROLE_USER,
        expires_at: str | None = None,
        rate_limit_per_minute: int | None = None,
        prefix: str = DEFAULT_KEY_PREFIX,
        permission_set: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate + store a new key. Returns ``(full_key_plaintext, record)``.

        The plaintext is returned here and nowhere else -- the store keeps only
        the hash. Callers (the admin script) print it once. ``permission_set``
        wins over the legacy ``role`` argument; when both are empty the role
        maps onto the builtin sets (admin -> ``admin``, else ``user``).
        """
        if permission_set:
            set_record = self.store.get_permission_set(permission_set)
            if set_record is None:
                raise ValueError(f"Unknown permission set {permission_set!r}")
        else:
            if role not in VALID_ROLES:
                raise ValueError(f"Invalid role {role!r}; expected one of {sorted(VALID_ROLES)}")
            set_record = self.store.get_permission_set(
                PERMSET_ADMIN if role == ROLE_ADMIN else PERMSET_USER
            )
            if set_record is None:  # builtins are provisioned on load; belt+braces
                raise ValueError("Built-in permission sets are missing from the store.")
        full_key = generate_full_key(prefix=prefix)
        key_hash = hash_key(full_key)
        stored = {
            "label": str(label or "").strip(),
            "prefix": make_prefix(full_key, prefix=prefix),
            "status": STATUS_ACTIVE,
            "role": ROLE_ADMIN if set_record.get("admin") else ROLE_USER,
            "permission_set": str(set_record["name"]),
            "created_at": _utcnow_iso(),
            "expires_at": _parse_expires(expires_at),
            "rate_limit_per_minute": _coerce_optional_int(rate_limit_per_minute),
            "usage": {"requests": 0, "last_used_at": None, "last_used_ip": None},
        }
        self.store.add({"key_hash": key_hash, "stored": stored})
        record = {"key_id": key_hash, **stored}
        return full_key, record

    def rotate_key(self, key_hash: str, *, prefix: str = DEFAULT_KEY_PREFIX) -> tuple[str, dict[str, Any]] | None:
        """Atomically issue a fresh secret for an existing key and drop the old one.

        Preserves label/permission set/expiry/rate-limit; resets usage and bumps
        ``created_at`` + ``prefix``. The old secret stops working in the SAME
        single locked write (``KeyStore.rotate_stored``), and the record the
        new secret is derived from is read inside that same lock, so a
        concurrent ``update``/``set_role`` cannot be silently reverted and a
        crash mid-rotate cannot leave the old (possibly compromised) secret
        active. Returns ``(full_key_plaintext, record)`` or ``None`` if
        ``key_hash`` is not found.
        """
        generated: list[tuple[str, dict[str, Any]]] = []

        # Resolve the permission set BEFORE the locked write: _build runs inside
        # rotate_stored's lock, and any KeyStore read there (load() takes the
        # same non-reentrant threading.Lock) would deadlock the caller.
        existing = self.store.get_by_hash(key_hash)
        set_name = str(
            (existing or {}).get("permission_set")
            or (PERMSET_ADMIN if (existing or {}).get("role") == ROLE_ADMIN else PERMSET_USER)
        )
        set_record = self.store.get_permission_set(set_name)
        role = ROLE_ADMIN if (set_record or {}).get("admin") else ROLE_USER

        def _build(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            full_key = generate_full_key(prefix=prefix)
            new_hash = hash_key(full_key)
            stored = {
                "label": str(record.get("label", "")),
                "prefix": make_prefix(full_key, prefix=prefix),
                "status": STATUS_ACTIVE,
                "role": role,
                "permission_set": set_name,
                "created_at": _utcnow_iso(),
                "expires_at": record.get("expires_at"),
                "rate_limit_per_minute": _coerce_optional_int(record.get("rate_limit_per_minute")),
                "usage": {"requests": 0, "last_used_at": None, "last_used_ip": None},
            }
            generated.append((full_key, new_hash, stored))
            return new_hash, stored

        if self.store.rotate_stored(key_hash, _build) is None or not generated:
            # Not found (or lost a race with a concurrent delete); surface it.
            return None
        full_key, new_hash, stored = generated[0]
        new_record = {"key_id": new_hash, **stored}
        return full_key, new_record

    def set_status(self, key_hash: str, status: str) -> dict[str, Any] | None:
        if status not in {STATUS_ACTIVE, STATUS_DISABLED}:
            raise ValueError(f"Invalid status {status!r}")
        return self.store.update(key_hash, {"status": status})

    def set_role(self, key_hash: str, role: str) -> dict[str, Any] | None:
        """Legacy two-value role assignment; maps onto the builtin sets."""
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role {role!r}")
        return self.set_permission_set(
            key_hash, PERMSET_ADMIN if role == ROLE_ADMIN else PERMSET_USER
        )

    def set_permission_set(self, key_hash: str, permission_set: str) -> dict[str, Any] | None:
        """Assign a key to a permission set; raises ``ValueError`` if unknown."""
        set_record = self.store.get_permission_set(permission_set)
        if set_record is None:
            raise ValueError(f"Unknown permission set {permission_set!r}")
        updated = self.store.update(
            key_hash,
            {
                "permission_set": str(set_record["name"]),
                "role": ROLE_ADMIN if set_record.get("admin") else ROLE_USER,
            },
        )
        return updated

    # -- permission set management (thin, validated wrappers over the store) -
    def create_permission_set(
        self,
        *,
        name: str,
        label: str = "",
        categories: Any = None,
        can_write: bool = True,
        admin: bool = False,
    ) -> dict[str, Any]:
        return self.store.add_permission_set(
            name=name,
            label=label,
            categories=categories,
            can_write=can_write,
            admin=admin,
        )

    def update_permission_set(self, name: str, mutate: dict[str, Any]) -> dict[str, Any] | None:
        if "categories" in (mutate or {}):
            # Validate before taking the store lock so a bad list raises the
            # same ValueError the create path raises.
            normalize_permission_categories(mutate["categories"])
        return self.store.update_permission_set(name, mutate)

    def delete_permission_set(self, name: str) -> bool:
        return self.store.delete_permission_set(name)

    def list_permission_sets(self) -> list[dict[str, Any]]:
        return self.store.list_permission_sets()

    def get_permission_set(self, name: str) -> dict[str, Any] | None:
        return self.store.get_permission_set(name)

    def permission_set_key_counts(self) -> dict[str, int]:
        return self.store.permission_set_key_counts()

    def delete_key(self, key_hash: str) -> bool:
        return self.store.delete(key_hash)

    def reset_usage(self, key_hash: str) -> dict[str, Any] | None:
        return self.store.update(
            key_hash,
            {"usage": {"requests": 0, "last_used_at": None, "last_used_ip": None}},
        )


class RateLimitExceeded(Exception):
    """Raised internally when an authenticated identity exceeds its limit."""

    def __init__(self, retry_after: int):
        super().__init__(f"Rate limit exceeded; retry after {retry_after}s")
        self.retry_after = max(1, int(retry_after))


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = [
    "AuthResult",
    "ApiKeyAuthenticator",
    "KeyStore",
    "RateLimiter",
    "Rejection",
    "UsageTracker",
    "RateLimitExceeded",
    "DEFAULT_KEY_PREFIX",
    "MASTER_KEY_ID",
    "MASTER_PREFIX",
    "LOCAL_KEY_ID",
    "PERMSET_ADMIN",
    "PERMSET_USER",
    "PERMSET_WILDCARD",
    "BUILTIN_PERMSETS",
    "ROLE_ADMIN",
    "ROLE_USER",
    "STATUS_ACTIVE",
    "STATUS_DISABLED",
    "STORE_FILENAME",
    "STORE_VERSION",
    "VALID_ROLES",
    "create_default_authenticator",
    "generate_full_key",
    "generate_secret",
    "hash_key",
    "local_auth_result",
    "make_prefix",
    "normalize_permission_categories",
    "normalize_permission_set_name",
]


def create_default_authenticator(
    data_dir: str | Path,
    *,
    default_rate_limit: int,
    persist_interval: int,
    window_seconds: float = RATE_WINDOW_SECONDS,
) -> ApiKeyAuthenticator:
    """Build an :class:`ApiKeyAuthenticator` rooted at ``data_dir/.api_keys.json``."""
    store = KeyStore(Path(data_dir) / STORE_FILENAME)
    try:
        # One-time repair of legacy records whose display prefix embedded the
        # whole secret (see repair_leaked_prefixes). Best-effort: a read-only
        # or corrupt store must not block startup.
        store.repair_leaked_prefixes()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("API key prefix repair skipped: %s", exc)
    return ApiKeyAuthenticator(
        store,
        default_rate_limit=default_rate_limit,
        persist_interval=persist_interval,
        window_seconds=window_seconds,
    )
