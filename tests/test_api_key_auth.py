"""Tests for the per-user API-key auth, rate limiting, and usage tracking.

These cover the unit-level behavior of ``src/api_key_auth`` (store CRUD,
hashing/prefix, expiry, rate limiting, usage persistence) and the integration
of that module into the FastAPI middleware + admin endpoint in
``src.web_app``. The middleware tests follow the established pattern in
``tests/test_web_app.py`` (TestClient + monkeypatch of module globals).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api_key_auth as aka
import src.web_app as web_app

# This module tests credential handling itself, so every test here runs in the
# strict remote posture (see the ``_local_operator`` fixture in conftest.py):
# TestClient requests are NOT treated as loopback. The localhost auto-auth
# bypass stays enabled -- the tests that exercise it use an explicit
# 127.0.0.1 client address.
pytestmark = pytest.mark.remote_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def authenticator(safe_tmp_path):
    """A fresh authenticator rooted at a temp data dir per test."""
    store = aka.KeyStore(safe_tmp_path / aka.STORE_FILENAME)
    return aka.ApiKeyAuthenticator(
        store, default_rate_limit=60, persist_interval=5, window_seconds=60
    )


def _make_key(authenticator, **kwargs):
    """Create a key and return its full plaintext secret."""
    full_key, record = authenticator.create_key(**kwargs)
    return full_key, record


# ===========================================================================
# Key generation / hashing primitives
# ===========================================================================


def test_generate_secret_is_random_and_correct_length():
    a = aka.generate_secret()
    b = aka.generate_secret()
    assert len(a) == aka.SECRET_LENGTH
    assert a != b  # overwhelmingly likely; guards against a constant fallback
    assert all(c in aka.SECRET_ALPHABET for c in a)


def test_generate_full_key_has_prefix():
    key = aka.generate_full_key(prefix="rag_")
    assert key.startswith("rag_")
    assert len(key) == len("rag_") + aka.SECRET_LENGTH


def test_hash_key_is_stable_and_hex():
    key = "rag_testsecret"
    h = aka.hash_key(key)
    assert h == aka.hash_key(key)  # deterministic
    assert len(h) == 64  # sha256 hex
    assert all(c in "0123456789abcdef" for c in h)


def test_make_prefix_redacts_middle():
    prefix = aka.make_prefix("rag_abcdefgh")
    assert prefix.endswith("efgh")
    assert "…" in prefix
    # Only the key-prefix head and the last 4 chars may appear: the middle of
    # the secret must not be recoverable from the display value.
    assert prefix == "rag_…efgh"
    assert "abcd" not in prefix


def test_make_prefix_custom_prefix_head():
    prefix = aka.make_prefix("team_A1b2C3d4", prefix="team_")
    assert prefix == "team_…C3d4"
    assert "A1b2" not in prefix
    # A key that does not start with the given prefix still never shows more
    # than a short head.
    fallback = aka.make_prefix("rag_abcdefgh", prefix="zzz_")
    assert fallback == "rag_…efgh"


def test_make_prefix_handles_short_input():
    assert aka.make_prefix("ab") == "ab"


# ===========================================================================
# KeyStore CRUD
# ===========================================================================


def test_store_starts_empty(authenticator):
    assert authenticator.store.has_any_key() is False
    assert authenticator.store.list_keys() == []


def test_create_key_returns_plaintext_and_stores_hash(authenticator):
    full_key, record = _make_key(authenticator, label="alice")
    assert full_key.startswith(aka.DEFAULT_KEY_PREFIX)
    # The stored prefix is the redacted form, never the full secret.
    assert record["prefix"] == aka.make_prefix(full_key)
    assert record["label"] == "alice"
    assert record["role"] == "user"
    assert record["status"] == "active"
    assert record["expires_at"] is None
    # The hash of the secret must resolve to the stored record.
    stored = authenticator.store.get_by_hash(aka.hash_key(full_key))
    assert stored is not None
    assert stored["label"] == "alice"


def test_repair_leaked_prefixes_rewrites_verifiable_records(authenticator, safe_tmp_path):
    """Legacy stores recorded <secret[:-4]>…<secret[-4:]> as the display
    prefix, which contained the entire secret. The repair must rewrite those
    (verifiable via the record's hash id) to the redacted form -- and leave
    anything it cannot verify alone."""
    import json

    full_key, record = _make_key(authenticator, label="legacy")
    key_hash = record["key_id"]
    # Simulate the legacy leaked form.
    store_path = safe_tmp_path / aka.STORE_FILENAME
    payload = json.loads(store_path.read_text())
    leaked = full_key[:-4] + "…" + full_key[-4:]
    assert leaked.replace("…", "") == full_key  # the leak really is the key
    payload["keys"][key_hash]["prefix"] = leaked
    # An unverifiable entry with the same shape must be left untouched.
    payload["keys"]["0" * 64] = {
        "label": "foreign",
        "prefix": "rag_deadbeef…beef",
        "status": "active",
        "role": "user",
        "created_at": "2026-01-01T00:00:00+00:00",
        "expires_at": None,
        "rate_limit_per_minute": None,
        "usage": {"requests": 0, "last_used_at": None, "last_used_ip": None},
    }
    store_path.write_text(json.dumps(payload))

    repaired = authenticator.store.repair_leaked_prefixes()

    assert repaired == 1
    on_disk = json.loads(store_path.read_text())
    fixed = on_disk["keys"][key_hash]["prefix"]
    assert fixed == aka.make_prefix(full_key)
    assert full_key not in json.dumps(on_disk)
    # The foreign record could not be verified, so it survives verbatim.
    assert on_disk["keys"]["0" * 64]["prefix"] == "rag_deadbeef…beef"
    # The key itself still authenticates after the rewrite.
    result, rejection = authenticator.authenticate(full_key, master_token="", track=False)
    assert result is not None and rejection is None


def test_plaintext_secret_is_never_stored(authenticator, safe_tmp_path):
    import json

    full_key, _ = _make_key(authenticator, label="alice")
    raw = json.loads((safe_tmp_path / aka.STORE_FILENAME).read_text())
    blob = json.dumps(raw)
    # The full secret must not appear anywhere in the persisted store.
    assert full_key not in blob
    assert aka.hash_key(full_key) in blob  # only the hash is stored


def test_find_by_prefix(authenticator):
    full_key, record = _make_key(authenticator, label="bob")
    found = authenticator.store.find_by_prefix(record["prefix"])
    assert found is not None
    assert found["key_id"] == aka.hash_key(full_key)
    assert authenticator.store.find_by_prefix("rag_…nope") is None


def test_set_status_disable_then_enable(authenticator):
    _, record = _make_key(authenticator, label="x")
    key_hash = record["key_id"]
    disabled = authenticator.set_status(key_hash, aka.STATUS_DISABLED)
    assert disabled["status"] == "disabled"
    re_enabled = authenticator.set_status(key_hash, aka.STATUS_ACTIVE)
    assert re_enabled["status"] == "active"


def test_set_status_rejects_invalid_value(authenticator):
    _, record = _make_key(authenticator, label="x")
    with pytest.raises(ValueError):
        authenticator.set_status(record["key_id"], "paused")


def test_delete_key(authenticator):
    _, record = _make_key(authenticator, label="x")
    key_hash = record["key_id"]
    assert authenticator.delete_key(key_hash) is True
    assert authenticator.store.get_by_hash(key_hash) is None
    # Deleting again is a no-op (returns False).
    assert authenticator.delete_key(key_hash) is False


# ===========================================================================
# Atomic rotation (replace_key / rotate_key)
# ===========================================================================


def test_replace_key_is_atomic_single_write(authenticator, monkeypatch):
    """replace_key must perform old-del + new-add in ONE store write, so a crash
    cannot leave both the old and new secrets active (fail-open)."""
    old_full, old_record = _make_key(authenticator, label="old")
    old_hash = old_record["key_id"]

    # Count store persists. add/delete each call _save_raw once; a non-atomic
    # rotate (create_key + delete_key) would call it twice. replace_key calls once.
    save_calls: list[int] = []
    orig_save = aka.KeyStore._save_raw
    monkeypatch.setattr(
        authenticator.store, "_save_raw",
        lambda payload: save_calls.append(1) or orig_save(authenticator.store, payload),
    )

    new_full, new_record = authenticator.rotate_key(old_hash)

    assert len(save_calls) == 1, f"rotate must be a single write; got {len(save_calls)}"
    assert new_record["key_id"] != old_hash
    # Old secret is gone, new one is present.
    assert authenticator.store.get_by_hash(old_hash) is None
    assert authenticator.store.get_by_hash(new_record["key_id"]) is not None


def test_rotate_key_drops_old_secret_and_preserves_attributes(authenticator):
    """After rotation only the NEW secret authenticates; label/role/expiry carry over."""
    full_key, record = authenticator.create_key(
        label="alice", role=aka.ROLE_ADMIN, rate_limit_per_minute=99,
    )
    old_hash = record["key_id"]

    new_full, new_record = authenticator.rotate_key(old_hash)

    assert new_full != full_key
    # Preserved attributes.
    assert new_record["label"] == "alice"
    assert new_record["role"] == aka.ROLE_ADMIN
    assert new_record["rate_limit_per_minute"] == 99
    # Usage is reset on the new record.
    assert new_record["usage"]["requests"] == 0
    # The OLD secret no longer authenticates; the NEW one does.
    result_old, rej_old = authenticator.authenticate(full_key, master_token="")
    assert result_old is None and rej_old is not None and rej_old.status_code == 401
    result_new, rej_new = authenticator.authenticate(new_full, master_token="")
    assert rej_new is None and result_new is not None
    assert result_new.key_id == new_record["key_id"]


def test_rotate_missing_key_returns_none(authenticator):
    """Rotating a hash that isn't there returns None and leaves the store intact."""
    assert authenticator.rotate_key("0" * 64) is None


def test_set_role(authenticator):
    _, record = _make_key(authenticator, label="x")
    updated = authenticator.set_role(record["key_id"], aka.ROLE_ADMIN)
    assert updated["role"] == "admin"
    with pytest.raises(ValueError):
        authenticator.set_role(record["key_id"], "superuser")


def test_create_rejects_invalid_role(authenticator):
    with pytest.raises(ValueError):
        authenticator.create_key(label="x", role="superuser")


# ===========================================================================
# Permission sets
# ===========================================================================


def test_builtin_sets_provisioned_on_fresh_store(authenticator):
    sets = {s["name"]: s for s in authenticator.list_permission_sets()}
    assert sets[aka.PERMSET_ADMIN]["admin"] is True
    assert sets[aka.PERMSET_ADMIN]["builtin"] is True
    assert sets[aka.PERMSET_ADMIN]["categories"] == [aka.PERMSET_WILDCARD]
    assert sets[aka.PERMSET_USER]["admin"] is False
    assert sets[aka.PERMSET_USER]["builtin"] is True


def test_create_key_defaults_map_roles_to_builtin_sets(authenticator):
    _, user_record = _make_key(authenticator, label="u")
    assert user_record["permission_set"] == aka.PERMSET_USER
    _, admin_record = _make_key(authenticator, label="a", role=aka.ROLE_ADMIN)
    assert admin_record["permission_set"] == aka.PERMSET_ADMIN


def test_create_key_with_permission_set_overrides_role(authenticator):
    authenticator.create_permission_set(
        name="team-a", label="Team A", categories=["general", "team-a"]
    )
    _, record = _make_key(authenticator, label="k", role=aka.ROLE_ADMIN, permission_set="team-a")
    assert record["permission_set"] == "team-a"
    # The legacy role mirrors the set's admin flag (not the role argument).
    assert record["role"] == aka.ROLE_USER


def test_create_key_rejects_unknown_permission_set(authenticator):
    with pytest.raises(ValueError):
        _make_key(authenticator, label="k", permission_set="nope")


def test_key_authenticates_through_its_set(authenticator):
    authenticator.create_permission_set(
        name="ops", categories=["*"], admin=True, can_write=True
    )
    full_key, _ = _make_key(authenticator, label="k", permission_set="ops")
    result, rejection = authenticator.authenticate(full_key, master_token="")
    assert rejection is None
    assert result.role == aka.ROLE_ADMIN
    assert result.permission_set == "ops"


def test_permission_set_roundtrip_and_counts(authenticator):
    created = authenticator.create_permission_set(name="viewers", categories=["general"], can_write=False)
    assert created["name"] == "viewers"
    got = authenticator.get_permission_set("viewers")
    assert got["can_write"] is False
    assert got["categories"] == ["general"]
    _make_key(authenticator, label="k", permission_set="viewers")
    assert authenticator.permission_set_key_counts().get("viewers") == 1


def test_permission_set_rejects_bad_name_and_duplicate(authenticator):
    with pytest.raises(ValueError):
        authenticator.create_permission_set(name="Bad Name!")
    authenticator.create_permission_set(name="dup")
    with pytest.raises(ValueError):
        authenticator.create_permission_set(name="dup")


def test_permission_set_category_validation(authenticator):
    with pytest.raises(ValueError):
        authenticator.create_permission_set(name="bad-cats", categories=["Not A Slug"])
    # Wildcard collapses whatever else was listed.
    record = authenticator.create_permission_set(name="wild", categories=["*", "general"])
    assert record["categories"] == [aka.PERMSET_WILDCARD]
    # Comma-separated strings are accepted for CLI convenience.
    record = authenticator.create_permission_set(name="csv", categories="general, team-a")
    assert record["categories"] == ["general", "team-a"]


def test_delete_builtin_set_refused(authenticator):
    with pytest.raises(ValueError):
        authenticator.delete_permission_set(aka.PERMSET_ADMIN)
    with pytest.raises(ValueError):
        authenticator.delete_permission_set(aka.PERMSET_USER)


def test_delete_set_in_use_refused(authenticator):
    authenticator.create_permission_set(name="used")
    full_key, _ = _make_key(authenticator, label="k", permission_set="used")
    with pytest.raises(ValueError):
        authenticator.delete_permission_set("used")
    # Still functional after the refused delete.
    result, rejection = authenticator.authenticate(full_key, master_token="")
    assert rejection is None


def test_delete_unused_set_succeeds(authenticator):
    authenticator.create_permission_set(name="unused")
    assert authenticator.delete_permission_set("unused") is True
    assert authenticator.get_permission_set("unused") is None


def test_update_set_revalidates_and_syncs_key_roles(authenticator):
    authenticator.create_permission_set(name="promote")
    full_key, _ = _make_key(authenticator, label="k", permission_set="promote")
    updated = authenticator.update_permission_set("promote", {"admin": True, "categories": ["team-x"]})
    assert updated["admin"] is True
    assert updated["categories"] == ["team-x"]
    # The member key's legacy role follows the set.
    fresh = authenticator.store.get_by_hash(aka.hash_key(full_key))
    assert fresh["role"] == aka.ROLE_ADMIN
    # The builtin admin set cannot be demoted.
    demoted = authenticator.update_permission_set(aka.PERMSET_ADMIN, {"admin": False})
    assert demoted["admin"] is True


def test_legacy_role_store_migrates_to_permission_sets(authenticator, safe_tmp_path):
    import json

    full_key = aka.generate_full_key()
    key_hash = aka.hash_key(full_key)
    legacy_payload = {
        "version": 1,
        "keys": {
            key_hash: {
                "label": "legacy",
                "prefix": aka.make_prefix(full_key),
                "status": aka.STATUS_ACTIVE,
                "role": aka.ROLE_ADMIN,
                "created_at": "2026-01-01T00:00:00+00:00",
                "expires_at": None,
                "rate_limit_per_minute": None,
                "usage": {"requests": 0, "last_used_at": None, "last_used_ip": None},
            }
        },
    }
    store_path = safe_tmp_path / aka.STORE_FILENAME
    store_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    result, rejection = authenticator.authenticate(full_key, master_token="")
    assert rejection is None
    assert result.permission_set == aka.PERMSET_ADMIN
    assert result.role == aka.ROLE_ADMIN
    record = authenticator.store.get_by_hash(key_hash)
    assert record["permission_set"] == aka.PERMSET_ADMIN


def test_store_with_unknown_set_reference_fails_toward_builtin(authenticator, safe_tmp_path):
    import json

    full_key = aka.generate_full_key()
    key_hash = aka.hash_key(full_key)
    payload = {
        "version": 2,
        "permission_sets": {
            aka.PERMSET_ADMIN: {"name": aka.PERMSET_ADMIN, "admin": True, "builtin": True},
            aka.PERMSET_USER: {"name": aka.PERMSET_USER, "admin": False, "builtin": True},
        },
        "keys": {
            key_hash: {
                "label": "orphan",
                "prefix": aka.make_prefix(full_key),
                "status": aka.STATUS_ACTIVE,
                "role": aka.ROLE_USER,
                "permission_set": "vanished",
                "created_at": "2026-01-01T00:00:00+00:00",
                "expires_at": None,
                "rate_limit_per_minute": None,
                "usage": {"requests": 0, "last_used_at": None, "last_used_ip": None},
            }
        },
    }
    store_path = safe_tmp_path / aka.STORE_FILENAME
    store_path.write_text(json.dumps(payload), encoding="utf-8")
    record = authenticator.store.get_by_hash(key_hash)
    # Reassigned to the builtin user set rather than a permanent 401.
    assert record["permission_set"] == aka.PERMSET_USER


# ===========================================================================
# Expiry
# ===========================================================================


def test_parse_expires_bare_date_is_end_of_day_utc():
    parsed = aka._parse_expires("2026-12-31")
    assert parsed is not None
    assert parsed.endswith("23:59:59+00:00")


def test_parse_expires_full_iso_passthrough():
    iso = "2026-12-31T10:00:00+00:00"
    assert aka._parse_expires(iso) == iso


def test_parse_expires_empty_clears():
    assert aka._parse_expires("") is None
    assert aka._parse_expires(None) is None


def test_parse_expires_invalid_raises():
    with pytest.raises(ValueError):
        aka._parse_expires("not-a-date")


def test_expired_key_is_rejected(authenticator):
    full_key, _ = _make_key(authenticator, label="expired", expires_at="2020-01-01")
    result, rejection = authenticator.authenticate(full_key, master_token="")
    assert result is None
    assert rejection is not None
    assert rejection.status_code == 401
    assert "expired" in rejection.detail.lower()


def test_unexpired_future_key_succeeds(authenticator):
    full_key, _ = _make_key(authenticator, label="future", expires_at="2999-12-31")
    result, rejection = authenticator.authenticate(full_key, master_token="")
    assert rejection is None
    assert result is not None
    assert result.role == "user"


# ===========================================================================
# authenticate(): identity resolution, master bypass, rejections
# ===========================================================================


def test_authenticate_disabled_key_rejected(authenticator):
    full_key, record = _make_key(authenticator, label="x")
    authenticator.set_status(record["key_id"], aka.STATUS_DISABLED)
    result, rejection = authenticator.authenticate(full_key, master_token="")
    assert result is None
    assert rejection.status_code == 401
    assert "disabled" in rejection.detail.lower()


def test_authenticate_wrong_key_rejected(authenticator):
    _make_key(authenticator, label="x")  # store non-empty so gating engages
    result, rejection = authenticator.authenticate("rag_wrongsecret", master_token="")
    assert result is None
    assert rejection.status_code == 401


def test_authenticate_valid_key_succeeds(authenticator):
    full_key, _ = _make_key(authenticator, label="alice")
    result, rejection = authenticator.authenticate(full_key, master_token="")
    assert rejection is None
    assert result is not None
    assert result.is_master is False
    assert result.role == "user"
    assert result.key_id == aka.hash_key(full_key)


def test_master_token_bypass(authenticator):
    # Store non-empty so the system is "on", but authenticate with master.
    _make_key(authenticator, label="some key")
    result, rejection = authenticator.authenticate("super-secret", master_token="super-secret")
    assert rejection is None
    assert result is not None
    assert result.is_master is True
    assert result.role == "admin"


def test_master_token_constant_time_wrong_rejected(authenticator):
    _make_key(authenticator, label="some key")
    result, rejection = authenticator.authenticate("wrong", master_token="super-secret")
    assert result is None
    assert rejection.status_code == 401


def test_rejects_when_no_master_and_no_keys(authenticator):
    # Auth has no open mode: empty store + no master -> any credential fails
    # closed with a clean 401 (loopback auto-auth is handled by the middleware,
    # not here).
    result, rejection = authenticator.authenticate("anything", master_token="")
    assert result is None
    assert rejection is not None
    assert rejection.status_code == 401


def test_master_engages_gating_even_with_empty_store(safe_tmp_path):
    # If a master token is set but no keys exist, an empty/missing credential
    # must still be rejected (the owner has chosen to lock down).
    store = aka.KeyStore(safe_tmp_path / aka.STORE_FILENAME)
    auth = aka.ApiKeyAuthenticator(store, default_rate_limit=60, persist_interval=5)
    result, rejection = auth.authenticate("", master_token="owner-token")
    assert result is None
    assert rejection is not None
    assert rejection.status_code == 401
    # And the master still works.
    result, rejection = auth.authenticate("owner-token", master_token="owner-token")
    assert result is not None and result.is_master


def test_admin_role_key_resolves_as_admin(authenticator):
    full_key, _ = _make_key(authenticator, label="ops", role="admin")
    result, _ = authenticator.authenticate(full_key, master_token="")
    assert result.role == "admin"


def test_per_key_rate_limit_override(authenticator):
    full_key, _ = _make_key(authenticator, label="ci", rate_limit_per_minute=3)
    result, _ = authenticator.authenticate(full_key, master_token="")
    assert result.rate_limit_per_minute == 3


# ===========================================================================
# Rate limiting
# ===========================================================================


def test_rate_limiter_allows_under_limit():
    limiter = aka.RateLimiter(window_seconds=60)
    for _ in range(5):
        allowed, retry = limiter.check("k1", limit=5)
        assert allowed is True
        assert retry == 0


def test_rate_limiter_blocks_over_limit():
    limiter = aka.RateLimiter(window_seconds=60)
    for _ in range(3):
        assert limiter.check("k1", limit=3)[0] is True
    allowed, retry = limiter.check("k1", limit=3)
    assert allowed is False
    assert retry >= 1


def test_rate_limiter_isolated_per_identity():
    limiter = aka.RateLimiter(window_seconds=60)
    for _ in range(3):
        limiter.check("k1", limit=3)
    # A different identity is unaffected.
    allowed, _ = limiter.check("k2", limit=3)
    assert allowed is True


def test_rate_limiter_window_recovery(monkeypatch):
    real_monotonic = time.monotonic
    limiter = aka.RateLimiter(window_seconds=60)
    # Exhaust the limit.
    for _ in range(2):
        limiter.check("k1", limit=2)
    assert limiter.check("k1", limit=2)[0] is False
    # Simulate the window rolling forward. Capture the real function first so
    # the lambda doesn't recurse into the patched one.
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 61)
    allowed, _ = limiter.check("k1", limit=2)
    assert allowed is True


def test_authenticate_returns_429_on_rate_limit(authenticator, safe_tmp_path):
    # persist_interval high so usage flushes don't interfere with the test.
    store = aka.KeyStore(safe_tmp_path / aka.STORE_FILENAME)
    auth = aka.ApiKeyAuthenticator(store, default_rate_limit=2, persist_interval=999)
    full_key, _ = auth.create_key(label="spammy")
    # Two requests allowed.
    assert auth.authenticate(full_key, master_token="")[0] is not None
    assert auth.authenticate(full_key, master_token="")[0] is not None
    # Third is rate-limited.
    with pytest.raises(aka.RateLimitExceeded) as exc_info:
        auth.authenticate(full_key, master_token="")
    assert exc_info.value.retry_after >= 1


# ===========================================================================
# Usage tracking
# ===========================================================================


def test_usage_tracker_flushes_on_interval(authenticator):
    authenticator._persist_interval = None  # ensure default path
    full_key, _ = _make_key(authenticator, label="x")
    key_hash = aka.hash_key(full_key)
    for _ in range(5):  # persist_interval=5 (see fixture) triggers a flush
        authenticator.authenticate(full_key, master_token="")
    record = authenticator.store.get_by_hash(key_hash)
    assert record["usage"]["requests"] >= 5


def test_usage_tracker_pending_flushed_on_demand(authenticator):
    # persist_interval high so nothing auto-flushes.
    authenticator.usage._persist_interval = 999
    full_key, _ = _make_key(authenticator, label="x")
    key_hash = aka.hash_key(full_key)
    authenticator.authenticate(full_key, master_token="")
    authenticator.authenticate(full_key, master_token="")
    # Not yet persisted (below interval).
    assert authenticator.store.get_by_hash(key_hash)["usage"]["requests"] == 0
    authenticator.usage.flush()
    assert authenticator.store.get_by_hash(key_hash)["usage"]["requests"] == 2


def test_usage_records_last_used_ip(authenticator):
    full_key, _ = _make_key(authenticator, label="x")
    key_hash = aka.hash_key(full_key)
    authenticator.authenticate(full_key, master_token="", client_ip="10.0.0.5")
    authenticator.usage.flush()
    assert authenticator.store.get_by_hash(key_hash)["usage"]["last_used_ip"] == "10.0.0.5"


# ===========================================================================
# Store persistence / caching
# ===========================================================================


def test_store_cache_invalidated_on_write(authenticator):
    full_key, record = _make_key(authenticator, label="a")
    # First read populates the cache.
    authenticator.store.load()
    # Mutate via a second store instance pointing at the same file -- the
    # first store's cache must not serve a stale copy after its own write.
    authenticator.set_status(record["key_id"], aka.STATUS_DISABLED)
    fresh = authenticator.store.get_by_hash(aka.hash_key(full_key))
    assert fresh["status"] == "disabled"


def test_store_reload_picks_up_external_writes(authenticator, safe_tmp_path):
    full_key, record = _make_key(authenticator, label="a")
    # A second store instance (simulating the admin script) disables the key.
    other = aka.KeyStore(safe_tmp_path / aka.STORE_FILENAME)
    other.update(record["key_id"], {"status": aka.STATUS_DISABLED})
    # The original store's mtime-cached read now reflects the change.
    fresh = authenticator.store.get_by_hash(aka.hash_key(full_key))
    assert fresh["status"] == "disabled"


# ===========================================================================
# Middleware integration (src.web_app)
# ===========================================================================


@pytest.fixture
def patched_authenticator(monkeypatch, safe_tmp_path):
    """Replace the module-level authenticator with one rooted in a temp dir."""
    store = aka.KeyStore(safe_tmp_path / aka.STORE_FILENAME)
    auth = aka.ApiKeyAuthenticator(store, default_rate_limit=60, persist_interval=999)
    monkeypatch.setattr(web_app, "api_authenticator", auth)
    monkeypatch.setattr(web_app, "_API_TOKEN", "")
    return auth


def _stub_queue(monkeypatch):
    """Stub the job queue so /api/reindex reaches a real handler state."""

    class StubQueue:
        def __init__(self):
            self.enqueued = None

        def enqueue_reindex(self, **kwargs):
            class Job:
                id = "job-reindex-1"

                def to_dict(self_inner):
                    return {"id": self_inner.id, "kind": "reindex", "status": "queued"}

            self.enqueued = kwargs
            return Job()

        def summary(self):
            return {
                "indexing_job_ids": [],
                "running_job_ids": [],
                "queued_count": 0,
                "active_query_count": 0,
            }

        def state_version(self):
            return "stub-1"

        def list_jobs(self, **kwargs):
            return []

        def begin_query(self):
            pass

        def finish_query(self):
            pass

    monkeypatch.setattr(web_app, "job_queue", StubQueue())


@pytest.fixture(autouse=True)
def _no_real_job_queue(monkeypatch):
    """Route every test in this module away from the app-level job queue.

    These tests authenticate against the real app object, and several
    exercise POST endpoints (e.g. /api/reindex) whose handlers enqueue real
    jobs once auth passes. Routed to the real queue, a passing test records
    a reindex in the live durable ledger (data/.job_ledger.json); the next
    server boot recovers that entry and re-indexes the whole corpus. The
    stub keeps middleware assertions meaningful without touching real job
    state; tests that assert on enqueue kwargs read ``enqueued`` off it.
    """
    _stub_queue(monkeypatch)


def test_middleware_gates_even_when_store_empty(patched_authenticator):
    # Auth is always the basis: even with no keys and no master token, a
    # remote (non-loopback) client without a credential gets 401. The
    # TestClient's host ("testclient") is not loopback, so no auto-auth.
    response = TestClient(web_app.app).post("/api/reindex")
    assert response.status_code == 401


def test_middleware_requires_key_when_store_nonempty(patched_authenticator):
    patched_authenticator.create_key(label="alice")  # store now non-empty
    response = TestClient(web_app.app).post("/api/reindex")
    assert response.status_code == 401
    assert "api key" in response.json()["detail"].lower()


def test_middleware_accepts_valid_key(patched_authenticator):
    full_key, _ = patched_authenticator.create_key(label="alice")
    response = TestClient(web_app.app).post(
        "/api/reindex", headers={"X-API-Token": full_key}
    )
    assert response.status_code != 401  # auth passed


def test_middleware_accepts_key_via_query_param(patched_authenticator):
    """The ?token= transport exists for GET consumers that cannot set headers
    (EventSource/media), so gated GETs accept it -- but mutating requests must
    present the header: a credential in a URL leaks into access logs and
    browser history, so it is never accepted for state changes."""
    full_key, _ = patched_authenticator.create_key(label="alice")
    client = TestClient(web_app.app)
    # Gated GET: query param authenticates.
    get_response = client.get(f"/api/jobs?token={full_key}")
    assert get_response.status_code == 200
    # Mutating request: query param alone is rejected; the header works.
    post_response = client.post(f"/api/reindex?token={full_key}")
    assert post_response.status_code == 401
    header_response = client.post("/api/reindex", headers={"X-API-Token": full_key})
    assert header_response.status_code != 401


def test_middleware_rejects_disabled_key(patched_authenticator):
    full_key, record = patched_authenticator.create_key(label="alice")
    patched_authenticator.set_status(record["key_id"], aka.STATUS_DISABLED)
    response = TestClient(web_app.app).post(
        "/api/reindex", headers={"X-API-Token": full_key}
    )
    assert response.status_code == 401
    assert "disabled" in response.json()["detail"].lower()


def test_middleware_rejects_expired_key(patched_authenticator):
    full_key, _ = patched_authenticator.create_key(label="alice", expires_at="2020-01-01")
    response = TestClient(web_app.app).post(
        "/api/reindex", headers={"X-API-Token": full_key}
    )
    assert response.status_code == 401


def test_middleware_gets_stay_open(patched_authenticator):
    # Even with keys present, GET reads are not gated.
    patched_authenticator.create_key(label="alice")
    response = TestClient(web_app.app).get("/api/health")
    assert response.status_code == 200


def test_sensitive_gets_require_key_when_store_nonempty(patched_authenticator):
    # /api/metrics and /api/index/stream exfiltrate bulk data / server internals,
    # so they are gated when auth is active (unlike ordinary GETs).
    patched_authenticator.create_key(label="alice")
    r_metrics = TestClient(web_app.app).get("/api/metrics")
    assert r_metrics.status_code == 401
    r_stream = TestClient(web_app.app).get("/api/index/stream")
    assert r_stream.status_code == 401


def test_sensitive_gets_accept_valid_key(patched_authenticator):
    full_key, _ = patched_authenticator.create_key(label="alice")
    # Auth passes (not 401); downstream status depends on the handler.
    r_metrics = TestClient(web_app.app).get(
        "/api/metrics", headers={"X-API-Token": full_key}
    )
    assert r_metrics.status_code != 401


def test_sensitive_gets_gated_when_no_auth_configured(patched_authenticator):
    # No open mode: even with an empty store and no master token, sensitive
    # GETs require a credential from remote clients.
    # (patched_authenticator fixture starts with an empty store.)
    r_metrics = TestClient(web_app.app).get("/api/metrics")
    assert r_metrics.status_code == 401


def test_metrics_history_gated_when_auth_configured(patched_authenticator):
    # /api/metrics/history exposes the same operational data as /api/metrics;
    # it must be gated exactly like /api/metrics (it was missed by the
    # exact-match sensitive-GET list once).
    patched_authenticator.create_key(label="alice")
    response = TestClient(web_app.app).get("/api/metrics/history")
    assert response.status_code == 401


def test_middleware_chat_stream_is_gated(patched_authenticator):
    # /api/chat/stream is the most expensive endpoint (LLM + GPU + corpus
    # answers) and the browser fetch carries X-API-Token fine, so it is gated
    # like every other POST when auth is configured.
    full_key, _record = patched_authenticator.create_key(label="alice")
    client = TestClient(web_app.app)
    anonymous = client.post(
        "/api/chat/stream",
        json={"question": "hi"},
        headers={"Content-Type": "application/json"},
    )
    assert anonymous.status_code == 401
    # A valid credential still gets past auth (the endpoint then fails
    # downstream with no model -- but not with 401).
    response = client.post(
        "/api/chat/stream",
        json={"question": "hi"},
        headers={"Content-Type": "application/json", "X-API-Token": full_key},
    )
    assert response.status_code != 401


def test_middleware_master_token_bypass(monkeypatch, safe_tmp_path):
    # Keys exist, but the master token still bypasses.
    store = aka.KeyStore(safe_tmp_path / aka.STORE_FILENAME)
    auth = aka.ApiKeyAuthenticator(store, default_rate_limit=60, persist_interval=999)
    auth.create_key(label="alice")
    monkeypatch.setattr(web_app, "api_authenticator", auth)
    monkeypatch.setattr(web_app, "_API_TOKEN", "owner-token")
    response = TestClient(web_app.app).post(
        "/api/reindex", headers={"X-API-Token": "owner-token"}
    )
    assert response.status_code != 401


def test_middleware_legacy_token_only_when_keys_disabled(monkeypatch):
    # [api_keys] disabled (api_authenticator=None) but master token set:
    # legacy single-token path must still work and reject wrong tokens.
    monkeypatch.setattr(web_app, "api_authenticator", None)
    monkeypatch.setattr(web_app, "_API_TOKEN", "owner-token")
    ok = TestClient(web_app.app).post(
        "/api/reindex", headers={"X-API-Token": "owner-token"}
    )
    bad = TestClient(web_app.app).post(
        "/api/reindex", headers={"X-API-Token": "wrong"}
    )
    assert ok.status_code != 401
    assert bad.status_code == 401


def test_middleware_rate_limit_429(monkeypatch, safe_tmp_path):
    store = aka.KeyStore(safe_tmp_path / aka.STORE_FILENAME)
    auth = aka.ApiKeyAuthenticator(store, default_rate_limit=2, persist_interval=999)
    monkeypatch.setattr(web_app, "api_authenticator", auth)
    monkeypatch.setattr(web_app, "_API_TOKEN", "")
    full_key, _ = auth.create_key(label="spammy")
    headers = {"X-API-Token": full_key}
    r1 = TestClient(web_app.app).post("/api/reindex", headers=headers)
    r2 = TestClient(web_app.app).post("/api/reindex", headers=headers)
    r3 = TestClient(web_app.app).post("/api/reindex", headers=headers)
    assert r1.status_code != 429
    assert r2.status_code != 429
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


# ===========================================================================
# Localhost auto-auth + permission-set category scoping (end-to-end)
# ===========================================================================


@pytest.fixture
def scoped_world(patched_authenticator, monkeypatch, safe_tmp_path):
    """Two categories, one two-doc corpus, and a set factory.

    The TestClient host ("testclient") is NOT loopback, so requests act as
    remote clients; the localhost case is exercised with an explicit
    127.0.0.1 client address.
    """
    entries = [
        {"key": "general", "label": "General", "db_dir": str(safe_tmp_path / "db"), "exists": True},
        {"key": "team-a", "label": "Team A", "db_dir": str(safe_tmp_path / "db" / "team-a"), "exists": True},
    ]
    memberships = {"hash-a": "general", "hash-b": "team-a"}

    class StubCategoryStore:
        def list_categories(self, *, anchor_db_dir=None):
            return [dict(entry) for entry in entries]

        def memberships_for(self, hashes):
            return {h: memberships.get(h, "general") for h in hashes}

        def state_version(self):
            return "stub-1"

    monkeypatch.setattr(web_app, "_category_store", lambda *a, **k: StubCategoryStore())

    def make_set(name, *, categories, can_write=True, admin=False):
        return patched_authenticator.create_permission_set(
            name=name, categories=categories, can_write=can_write, admin=admin,
        )

    return {"make_set": make_set, "auth": patched_authenticator}


def test_localhost_requests_are_auto_authenticated(scoped_world):
    # No credential at all, but the socket peer is 127.0.0.1: the middleware
    # stashes the full-admin localhost identity and the request passes.
    local_client = TestClient(web_app.app, client=("127.0.0.1", 51000))
    assert local_client.post("/api/reindex").status_code != 401
    # The TestClient's default host ("testclient") is remote: a bare request,
    # even with a spoofed X-Forwarded-For (only trusted from loopback peers),
    # is rejected.
    remote_client = TestClient(web_app.app)
    assert remote_client.post("/api/reindex").status_code == 401


def test_category_listing_filtered_by_permission_set(patched_authenticator, monkeypatch, scoped_world):
    monkeypatch.setattr(
        web_app,
        "_category_listing",
        lambda: {"categories": [
            {"key": "general", "label": "General"},
            {"key": "team-a", "label": "Team A"},
        ], "total_sources": 2},
    )
    scoped_world["make_set"]("team-a-only", categories=["team-a"])
    scoped_key, _ = patched_authenticator.create_key(label="a", permission_set="team-a-only")
    open_key, _ = patched_authenticator.create_key(label="b")
    client = TestClient(web_app.app)
    scoped = client.get("/api/categories", headers={"X-API-Token": scoped_key}).json()
    assert [entry["key"] for entry in scoped["categories"]] == ["team-a"]
    open_view = client.get("/api/categories", headers={"X-API-Token": open_key}).json()
    assert [entry["key"] for entry in open_view["categories"]] == ["general", "team-a"]


def test_pdfs_endpoint_passes_scope_into_listing(patched_authenticator, monkeypatch, scoped_world):
    captured = {}

    def fake_list_pdf_documents(**kwargs):
        captured["allowed"] = kwargs.get("allowed_categories")
        return {"pdfs": [], "total": 0, "offset": 0, "limit": 10, "facets": {}, "categories": []}

    monkeypatch.setattr(web_app, "list_pdf_documents", fake_list_pdf_documents)
    scoped_world["make_set"]("general-only", categories=["general"])
    full_key, _ = patched_authenticator.create_key(label="c", permission_set="general-only")
    response = TestClient(web_app.app).get("/api/pdfs", headers={"X-API-Token": full_key})
    assert response.status_code == 200
    assert captured["allowed"] == {"general"}


def test_pdfs_endpoint_403_for_out_of_scope_category_param(patched_authenticator, scoped_world):
    scoped_world["make_set"]("general-only", categories=["general"])
    full_key, _ = patched_authenticator.create_key(label="d", permission_set="general-only")
    response = TestClient(web_app.app).get(
        "/api/pdfs", params={"category": "team-a"}, headers={"X-API-Token": full_key}
    )
    assert response.status_code == 403


def test_index_endpoint_403_for_out_of_scope_category(patched_authenticator, monkeypatch, scoped_world, safe_tmp_path):
    # Stub the dir resolution so the scope check is what rejects (the real one
    # would 400 on the unknown key first).
    monkeypatch.setattr(web_app, "_resolve_category_db_dir", lambda key, **kwargs: Path(safe_tmp_path))
    scoped_world["make_set"]("general-only", categories=["general"])
    full_key, _ = patched_authenticator.create_key(label="e", permission_set="general-only")
    response = TestClient(web_app.app).get(
        "/api/index", params={"category": "team-a"}, headers={"X-API-Token": full_key}
    )
    assert response.status_code == 403


def test_chat_stream_403_for_out_of_scope_category(patched_authenticator, scoped_world):
    scoped_world["make_set"]("general-only", categories=["general"])
    full_key, _ = patched_authenticator.create_key(label="f", permission_set="general-only")
    response = TestClient(web_app.app).post(
        "/api/chat/stream",
        json={"question": "hi", "categories": ["team-a"]},
        headers={"X-API-Token": full_key, "Content-Type": "application/json"},
    )
    assert response.status_code == 403


def test_upload_403_for_out_of_scope_category(patched_authenticator, scoped_world):
    scoped_world["make_set"]("team-a-only", categories=["team-a"])
    full_key, _ = patched_authenticator.create_key(label="g", permission_set="team-a-only")
    response = TestClient(web_app.app).post(
        "/api/uploads",
        files={"files": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"category": "general"},
        headers={"X-API-Token": full_key},
    )
    assert response.status_code == 403


def test_read_only_set_cannot_upload(patched_authenticator, scoped_world):
    scoped_world["make_set"]("viewers", categories=["*"], can_write=False)
    full_key, _ = patched_authenticator.create_key(label="h", permission_set="viewers")
    response = TestClient(web_app.app).post(
        "/api/uploads",
        files={"files": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
        headers={"X-API-Token": full_key},
    )
    assert response.status_code == 403
    assert "read-only" in response.json()["detail"].lower()


def test_permission_set_admin_api_lifecycle(patched_authenticator, monkeypatch, scoped_world):
    auth = scoped_world["auth"]
    monkeypatch.setattr(web_app, "_API_TOKEN", "master-token")
    headers = {"X-API-Token": "master-token"}
    client = TestClient(web_app.app)

    created = client.post(
        "/api/admin/permission-sets",
        headers=headers,
        json={"name": "reviewers", "label": "Reviewers", "categories": ["general"], "can_write": False},
    )
    assert created.status_code == 200, created.text
    record = created.json()["permission_set"]
    assert record["name"] == "reviewers"
    assert record["categories"] == ["general"]

    listed = client.get("/api/admin/permission-sets", headers=headers).json()
    names = [s["name"] for s in listed["permission_sets"]]
    assert {"admin", "user", "reviewers"}.issubset(set(names))

    # Assign a key to the set via the dedicated endpoint.
    full_key, key_record = auth.create_key(label="k")
    assign = client.post(
        f"/api/admin/api-keys/{key_record['prefix']}/permission-set",
        headers=headers,
        json={"permission_set": "reviewers"},
    )
    assert assign.status_code == 200, assign.text
    assert assign.json()["record"]["permission_set"] == "reviewers"

    # A set still referenced by a key cannot be deleted; builtins never delete.
    refused = client.delete("/api/admin/permission-sets/reviewers", headers=headers)
    assert refused.status_code == 409
    assert client.delete("/api/admin/permission-sets/admin", headers=headers).status_code == 409


# ===========================================================================
# Admin endpoint GET /api/admin/api-keys
# ===========================================================================


def test_admin_endpoint_gated_even_when_no_auth_configured(monkeypatch):
    # No open mode: a remote (non-loopback) client without a credential gets
    # 401 even when the store is empty and no master token is set. The local
    # operator reaches the same endpoint via the localhost auto-auth bypass.
    monkeypatch.setattr(web_app, "_API_TOKEN", "")
    store = aka.KeyStore(web_app.DATA_DIR / aka.STORE_FILENAME)
    # Use the real module authenticator but ensure its store is empty for the
    # test by pointing at a temp file via a fresh authenticator.
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / aka.STORE_FILENAME
    auth = aka.ApiKeyAuthenticator(aka.KeyStore(tmp), default_rate_limit=60, persist_interval=999)
    monkeypatch.setattr(web_app, "api_authenticator", auth)
    response = TestClient(web_app.app).get("/api/admin/api-keys")
    assert response.status_code == 401


def test_admin_endpoint_requires_master_or_admin_key(patched_authenticator):
    patched_authenticator.create_key(label="alice")  # store non-empty -> gated
    # No credential.
    assert TestClient(web_app.app).get("/api/admin/api-keys").status_code == 401
    # A normal user key is forbidden (403), not just unauthorized.
    user_key, _ = patched_authenticator.create_key(label="user")
    r = TestClient(web_app.app).get(
        "/api/admin/api-keys", headers={"X-API-Token": user_key}
    )
    assert r.status_code == 403


def test_admin_endpoint_admin_key_can_list(patched_authenticator):
    admin_key, _ = patched_authenticator.create_key(label="ops", role="admin")
    patched_authenticator.create_key(label="alice", role="user")
    response = TestClient(web_app.app).get(
        "/api/admin/api-keys", headers={"X-API-Token": admin_key}
    )
    assert response.status_code == 200
    keys = response.json()["keys"]
    assert len(keys) == 2
    # No secret material leaks; prefixes are the redacted form.
    for k in keys:
        assert "key_id" not in k  # internal hash is not exposed
        assert "…" in k["prefix"]
    labels = {k["label"] for k in keys}
    assert labels == {"ops", "alice"}


def test_admin_endpoint_master_can_list(patched_authenticator, monkeypatch):
    patched_authenticator.create_key(label="alice")
    monkeypatch.setattr(web_app, "_API_TOKEN", "owner-token")
    response = TestClient(web_app.app).get(
        "/api/admin/api-keys", headers={"X-API-Token": "owner-token"}
    )
    assert response.status_code == 200
    assert response.json()["master_configured"] is True
