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

Backward compatibility: if the master token is empty **and** the key store has
no active keys, :func:`authenticate` returns ``None`` for any supplied value
and the middleware becomes a no-op -- a fresh deployment stays fully open.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
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

# Key format: "<prefix>_<base62 secret>". The secret is opaque and unguessable.
DEFAULT_KEY_PREFIX = "rag_"
SECRET_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SECRET_LENGTH = 32

STORE_VERSION = 1
STORE_FILENAME = ".api_keys.json"
STATUS_ACTIVE = "active"
STATUS_DISABLED = "disabled"
ROLE_USER = "user"
ROLE_ADMIN = "admin"
VALID_ROLES = {ROLE_USER, ROLE_ADMIN}

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
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


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
    """JSON-backed registry of API keys, keyed by ``sha256`` hash.

    The on-disk shape is::

        {"version": 1, "keys": {
            "<sha256 hex>": {
                "label": "alice laptop",
                "prefix": "rag_…a1b2",
                "status": "active",
                "role": "user",
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
        return {"version": STORE_VERSION, "keys": {}}

    @staticmethod
    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        payload.setdefault("version", STORE_VERSION)
        keys = payload.get("keys")
        if not isinstance(keys, dict):
            payload["keys"] = {}
        # Ensure each record has the expected fields.
        for record in payload["keys"].values():
            if not isinstance(record, dict):
                continue
            record.setdefault("label", "")
            record.setdefault("prefix", "")
            record.setdefault("status", STATUS_ACTIVE)
            record.setdefault("role", ROLE_USER)
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
          unexpired, and (optionally) role-tagged.
        * If neither path applies AND the store has no keys AND no master token
          is set, the whole system is disabled: returns ``(None, None)`` so the
          middleware becomes a no-op (fresh-deploy backward compatibility).

        On success, the rate limit is enforced and usage is recorded (unless
        ``track`` is False). Returns ``(AuthResult, None)`` on success,
        ``(None, Rejection)`` on rejection, or ``(None, None)`` when auth is
        fully disabled.
        """
        master_active = bool(master_token)
        store_has_keys = self.store.has_any_key()
        # Fresh-deploy no-op: nothing to check against.
        if not master_active and not store_has_keys:
            return None, None

        supplied_str = (supplied or "").strip()

        # Master bypass. compare_digest on str inputs requires ASCII-only text
        # and raises TypeError otherwise (a non-ASCII credential must yield a
        # clean rejection, not a 500), so compare encoded bytes instead.
        if (
            master_active
            and supplied_str
            and hmac.compare_digest(supplied_str.encode("utf-8"), master_token.encode("utf-8"))
        ):
            result = AuthResult(
                key_id=MASTER_KEY_ID,
                role=ROLE_ADMIN,
                label="master",
                rate_limit_per_minute=self.default_rate_limit,
                is_master=True,
            )
            return self._enforce_limit_and_track(result, client_ip, track)

        if not supplied_str:
            return None, Rejection(401, "An API key is required. Provide it via the X-API-Token header.")

        key_hash = hash_key(supplied_str)
        record = self.store.get_by_hash(key_hash)
        if record is None:
            return None, Rejection(401, "Invalid or missing API key.")

        status = record.get("status", STATUS_ACTIVE)
        if status == STATUS_DISABLED:
            return None, Rejection(401, "This API key has been disabled.")

        if _is_expired(record.get("expires_at")):
            return None, Rejection(401, "This API key has expired.")

        per_key_limit = record.get("rate_limit_per_minute")
        try:
            limit = int(per_key_limit) if per_key_limit else self.default_rate_limit
        except (TypeError, ValueError):
            limit = self.default_rate_limit

        result = AuthResult(
            key_id=key_hash,
            role=record.get("role", ROLE_USER),
            label=str(record.get("label", "")),
            rate_limit_per_minute=limit,
            is_master=False,
        )
        return self._enforce_limit_and_track(result, client_ip, track)

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
    ) -> tuple[str, dict[str, Any]]:
        """Generate + store a new key. Returns ``(full_key_plaintext, record)``.

        The plaintext is returned here and nowhere else -- the store keeps only
        the hash. Callers (the admin script) print it once.
        """
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role {role!r}; expected one of {sorted(VALID_ROLES)}")
        full_key = generate_full_key(prefix=prefix)
        key_hash = hash_key(full_key)
        stored = {
            "label": str(label or "").strip(),
            "prefix": make_prefix(full_key, prefix=prefix),
            "status": STATUS_ACTIVE,
            "role": role,
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

        Preserves label/role/expiry/rate-limit; resets usage and bumps
        ``created_at`` + ``prefix``. The old secret stops working in the SAME
        single locked write (``KeyStore.rotate_stored``), and the record the
        new secret is derived from is read inside that same lock, so a
        concurrent ``update``/``set_role`` cannot be silently reverted and a
        crash mid-rotate cannot leave the old (possibly compromised) secret
        active. Returns ``(full_key_plaintext, record)`` or ``None`` if
        ``key_hash`` is not found.
        """
        generated: list[tuple[str, dict[str, Any]]] = []

        def _build(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            full_key = generate_full_key(prefix=prefix)
            new_hash = hash_key(full_key)
            stored = {
                "label": str(record.get("label", "")),
                "prefix": make_prefix(full_key, prefix=prefix),
                "status": STATUS_ACTIVE,
                "role": record.get("role", ROLE_USER),
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
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role {role!r}")
        return self.store.update(key_hash, {"role": role})

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
    "make_prefix",
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
    except Exception:
        pass
    return ApiKeyAuthenticator(
        store,
        default_rate_limit=default_rate_limit,
        persist_interval=persist_interval,
        window_seconds=window_seconds,
    )
