"""Tests for the bounded LRU used by module-level memoization caches."""

from __future__ import annotations

import pytest

from src.caches import BoundedLRU


def test_set_get_round_trip():
    cache = BoundedLRU(maxsize=4)
    cache["a"] = 1
    assert cache["a"] == 1
    assert cache.get("a") == 1
    assert cache.get("missing") is None
    assert cache.get("missing", "fallback") == "fallback"


def test_eviction_is_lru_not_fifo():
    cache = BoundedLRU(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    # Touch "a" so it becomes most-recently-used; "b" is now LRU.
    _ = cache["a"]
    # Inserting "c" should evict "b" (the LRU), not "a".
    cache["c"] = 3
    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache


def test_get_bumps_recency():
    cache = BoundedLRU(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    # A .get() hit must bump recency so the entry survives eviction.
    assert cache.get("a") == 1
    cache["c"] = 3
    assert "a" in cache
    assert "b" not in cache


def test_update_existing_key_bumps_recency_and_preserves_size():
    cache = BoundedLRU(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    # Updating "a" should bump it, evicting "b" on next insert, not "a".
    cache["a"] = 10
    assert cache["a"] == 10
    assert len(cache) == 2
    cache["c"] = 3
    assert "a" in cache
    assert "b" not in cache


def test_pop_matches_dict_semantics():
    cache = BoundedLRU(maxsize=4)
    cache["a"] = 1
    assert cache.pop("a") == 1
    assert "a" not in cache
    # pop on missing key returns the default, never raises (matches the
    # write-through invalidation call sites: cache.pop(str(path), None)).
    assert cache.pop("missing") is None
    assert cache.pop("missing", "fallback") == "fallback"


def test_delete_and_contains():
    cache = BoundedLRU(maxsize=4)
    cache["a"] = 1
    del cache["a"]
    assert "a" not in cache
    with pytest.raises(KeyError):
        del cache["a"]


def test_clear_and_len():
    cache = BoundedLRU(maxsize=4)
    cache["a"] = 1
    cache["b"] = 2
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


def test_getitem_miss_raises_keyerror():
    cache = BoundedLRU(maxsize=4)
    with pytest.raises(KeyError):
        cache["missing"]


def test_maxsize_one_evicts_on_every_insert():
    cache = BoundedLRU(maxsize=1)
    cache["a"] = 1
    cache["b"] = 2
    assert "a" not in cache
    assert cache["b"] == 2


def test_invalid_maxsize_rejected():
    with pytest.raises(ValueError):
        BoundedLRU(maxsize=0)


def test_iteration_and_keys():
    cache = BoundedLRU(maxsize=4)
    cache["a"] = 1
    cache["b"] = 2
    assert set(cache.keys()) == {"a", "b"}
    assert set(iter(cache)) == {"a", "b"}


def test_setdefault():
    cache = BoundedLRU(maxsize=2)
    assert cache.setdefault("a", 1) == 1
    assert cache.setdefault("a", 999) == 1  # already present, not overwritten
    assert cache["a"] == 1


def test_large_capacity_does_not_leak():
    cache = BoundedLRU(maxsize=100)
    for i in range(1000):
        cache[f"key-{i}"] = i
    # Capacity is honored exactly regardless of insert volume.
    assert len(cache) == 100
    # The most-recently-inserted keys survived; the oldest were evicted.
    assert "key-999" in cache
    assert "key-0" not in cache
