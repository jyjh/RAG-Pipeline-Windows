"""Bounded LRU cache used by module-level memoization dicts.

Several modules (``web_app``, ``job_ledger``, ``pdf_registry``) keep a
process-wide cache of parsed JSON / computed hashes keyed by file path. Over a
multi-day 100GB-scale ingestion, these caches grow without bound: every PDF
hashed, every markdown quality-scored, every JSON file loaded adds an entry
that is never evicted. The signature-staleness idiom (compare ``(mtime_ns,
size)`` on read and overwrite the entry when it changes) means a path is only
ever re-stored, never removed -- so distinct paths accumulate indefinitely.

``BoundedLRU`` is a dict-compatible bounded LRU: same access pattern the
callers already use (``.get()``, ``cache[key] = value``, ``.pop(key, None)``)
plus the usual dict methods for forward safety. It mirrors the eviction
strategy already used by ``EmbeddingEngine._cache`` (an ``OrderedDict`` with
``move_to_end`` on hit + ``popitem(last=False)`` on overflow).

Thread-safety: callers hold their own coarse locks (e.g. ``job_ledger._LOCK``)
around read-modify-write sequences. This class is NOT internally locked -- the
existing per-module locks cover the compound operations. A single
``__setitem__`` or ``pop`` is not atomic on its own, but every call site that
depends on atomicity already serializes via the module lock, so preserving
that contract (no internal lock) avoids a double-lock and keeps the hot path
lock-free for the read-only ``.get()`` path.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable, Iterator


class BoundedLRU:
    """A dict-compatible bounded LRU with ``maxsize`` entries.

    Implements the subset of ``dict`` the module-level caches use
    (``__getitem__``, ``__setitem__``, ``__contains__``, ``get``, ``pop``,
    ``__delitem__``, ``clear``, ``__len__``, ``__iter__``) so it is a drop-in
    for ``dict``-typed cache declarations. Eviction is true LRU: every read
    (``__getitem__``/``get`` hit) marks the entry most-recently-used; inserts
    and updates evict the least-recently-used entry when over capacity.
    """

    __slots__ = ("_data", "_maxsize")

    def __init__(self, maxsize: int = 1024) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = int(maxsize)
        self._data: OrderedDict[Hashable, Any] = OrderedDict()

    def _bump(self, key: Hashable) -> None:
        # Mark ``key`` as most-recently-used. O(1).
        self._data.move_to_end(key)

    def _evict_if_needed(self) -> None:
        # Evict the least-recently-used entry (FIFO end) until within bounds.
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    # --- dict-compatible API ------------------------------------------------

    def __getitem__(self, key: Hashable) -> Any:
        value = self._data[key]  # raises KeyError on miss (matches dict)
        self._bump(key)
        return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        # An update to an existing key must also bump its recency.
        if key in self._data:
            self._data[key] = value
            self._bump(key)
        else:
            self._data[key] = value
            self._evict_if_needed()

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __delitem__(self, key: Hashable) -> None:
        del self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BoundedLRU(maxsize={self._maxsize}, len={len(self._data)})"

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Return the cached value (bumping recency) or ``default`` on miss.

        Matches ``dict.get`` semantics exactly, including the recency bump on
        hit so frequently-accessed paths survive eviction.
        """
        if key in self._data:
            value = self._data[key]
            self._bump(key)
            return value
        return default

    def pop(self, key: Hashable, default: Any = None) -> Any:
        """Remove and return ``key``'s value, or ``default`` if absent.

        Matches ``dict.pop`` semantics used by the write-through invalidation
        paths (``cache.pop(str(path), None)`` after an atomic JSON write).
        """
        try:
            return self._data.pop(key)
        except KeyError:
            return default

    def clear(self) -> None:
        self._data.clear()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def setdefault(self, key: Hashable, default: Any = None) -> Any:
        if key in self._data:
            self._bump(key)
            return self._data[key]
        self[key] = default
        return default
