from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.atomic_io import write_bytes_atomic, write_json_atomic
from src.file_lock import acquire_asset_lock


ASSET_MANIFEST_FILENAME = "assets_manifest.json"
IMAGE_ASSET_MARKER_RE = re.compile(r"\[Image Asset:\s*([A-Za-z0-9_.-]+)\]")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ASSET_MANIFEST_VERSION = 1


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def image_asset_marker(asset_id: str) -> str:
    return f"[Image Asset: {asset_id}]"


def image_asset_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for match in IMAGE_ASSET_MARKER_RE.finditer(text or ""):
        asset_id = match.group(1)
        if asset_id in seen:
            continue
        seen.add(asset_id)
        ids.append(asset_id)
    return ids


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    # A JSON decode failure now indicates real corruption (writes are atomic),
    # so surface it rather than silently returning an empty manifest.
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else default


# The asset manifest scales with every extracted image in the corpus, and the
# query/citation path re-reads it once per retrieved chunk. Cache the parsed
# document keyed on its (mtime_ns, size) signature -- the same trick the PDF
# registry uses. Two access flavors:
#   _shared_manifest_payload -- returns the CACHED dict itself; callers must
#     never mutate it. Used by read-only paths (get_asset, assets_for_text,
#     pre-copy bases for writes) so a per-chunk citation lookup costs one dict
#     lookup instead of a whole-manifest deepcopy.
#   _cached_manifest_payload -- returns a deep copy; safe to mutate. Kept for
#     callers that treat the returned manifest as theirs (load_manifest).
_MANIFEST_CACHE_LOCK = threading.Lock()
_MANIFEST_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def _shared_manifest_payload(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    """Return the signature-cached manifest. READ-ONLY: never mutate the result."""
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        with _MANIFEST_CACHE_LOCK:
            _MANIFEST_CACHE.pop(key, None)
        return {"version": ASSET_MANIFEST_VERSION, "assets": {}}
    signature = (stat.st_mtime_ns, stat.st_size)
    with _MANIFEST_CACHE_LOCK:
        cached = _MANIFEST_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
    payload = _load_json(path, json.loads(json.dumps(default)))
    normalized = {
        "version": ASSET_MANIFEST_VERSION,
        "assets": {},
    }
    normalized.update(payload)
    normalized.setdefault("assets", {})
    if not isinstance(normalized["assets"], dict):
        normalized["assets"] = {}
    with _MANIFEST_CACHE_LOCK:
        _MANIFEST_CACHE[key] = (signature, normalized)
    return normalized


def _cached_manifest_payload(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(_shared_manifest_payload(path, default))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def _safe_component(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip(".-")
    return text or fallback


class ImageAssetStore:
    def __init__(self, asset_dir: str | Path):
        self.asset_dir = Path(asset_dir)
        self.manifest_path = self.asset_dir / ASSET_MANIFEST_FILENAME
        # Batch mode: while active, ``save_image`` mutates a single in-memory
        # manifest and defers the (expensive, O(manifest size)) rewrite to
        # ``commit_batch``. Without this, ingesting a document with K images does
        # K full read+rewrite passes (O(K**2) in manifest size). The depth counter
        # makes ``begin_batch``/``commit_batch`` safe to nest.
        self._batch_manifest: dict[str, Any] | None = None
        self._batch_depth = 0
        # Snapshot of the manifest at ``begin_batch`` time (asset_id -> entry).
        # ``commit_batch`` three-way merges batch vs base vs the fresh on-disk
        # manifest so concurrent writers' entries survive our flush instead of
        # being clobbered by a stale full rewrite.
        self._batch_base: dict[str, dict[str, Any]] | None = None

    def load_manifest(self) -> dict[str, Any]:
        if self._batch_manifest is not None:
            return self._batch_manifest
        return _cached_manifest_payload(
            self.manifest_path,
            {"version": ASSET_MANIFEST_VERSION, "assets": {}},
        )

    def begin_batch(self) -> None:
        """Load the manifest once so subsequent ``save_image`` calls mutate it
        in memory until ``commit_batch`` flushes a single rewrite. Reentrant."""
        self._batch_depth += 1
        if self._batch_manifest is None:
            self._batch_manifest = self.load_manifest()
            self._batch_base = {
                asset_id: dict(entry)
                for asset_id, entry in (self._batch_manifest.get("assets") or {}).items()
                if isinstance(entry, dict)
            }

    def commit_batch(self) -> None:
        """Write the pending batched manifest (if any) to disk in a single pass.
        Reentrant: only flushes when the outermost batch commits.

        The flush merges this batch's changes (new/updated/removed entries,
        detected against the ``begin_batch`` snapshot) into the CURRENT on-disk
        manifest under the cross-process asset lock. A full stale rewrite here
        would silently drop entries another ingestion worker committed while
        this batch was open.
        """
        if self._batch_depth <= 0:
            return
        self._batch_depth -= 1
        if self._batch_depth != 0 or self._batch_manifest is None:
            return
        batch = self._batch_manifest
        base = self._batch_base or {}
        self._batch_manifest = None
        self._batch_base = None

        batch_assets = batch.get("assets") or {}
        removed = {asset_id for asset_id in base if asset_id not in batch_assets}
        changed = {
            asset_id: entry
            for asset_id, entry in batch_assets.items()
            if asset_id not in base or base[asset_id] != entry
        }
        try:
            with acquire_asset_lock(self.asset_dir):
                fresh = _shared_manifest_payload(
                    self.manifest_path,
                    {"version": ASSET_MANIFEST_VERSION, "assets": {}},
                )
                merged_assets = {
                    asset_id: entry
                    for asset_id, entry in (fresh.get("assets") or {}).items()
                    if asset_id not in removed
                }
                merged_assets.update(changed)
                payload = dict(fresh)
                payload["assets"] = merged_assets
                _write_json(self.manifest_path, payload)
        except (TimeoutError, OSError):
            # The image FILES for this batch are already on disk; losing their
            # manifest entries would strand them unretrievable. Re-arm the
            # batch so the next begin/commit on this store retries the flush.
            self._batch_manifest = batch
            self._batch_base = base
            raise

    def abort_batch(self) -> None:
        """Discard any pending batched manifest without writing."""
        self._batch_manifest = None
        self._batch_base = None
        self._batch_depth = 0

    def save_image(
        self,
        *,
        image_data: bytes,
        source_hash: str,
        source_pdf_name: str,
        page_no: int | None,
        description: str,
    ) -> dict[str, Any]:
        source_hash = str(source_hash or "").strip()
        image_sha = hashlib.sha256(image_data).hexdigest()
        source_key = _safe_component(source_hash, "unknown-source")
        page_key = str(int(page_no or 0))
        asset_id = f"img_{source_key[:12]}_p{page_key}_{image_sha[:16]}"
        relative_path = Path(source_key) / f"{asset_id}.png"
        image_path = self.asset_dir / relative_path
        write_bytes_atomic(image_path, image_data)

        manifest = self.load_manifest()
        assets = manifest.setdefault("assets", {})
        existing = assets.get(asset_id) if isinstance(assets, dict) else {}
        created_at = str(existing.get("created_at") or utcnow()) if isinstance(existing, dict) else utcnow()
        entry = {
            "asset_id": asset_id,
            "source_hash": source_hash,
            "source_pdf_name": str(source_pdf_name or ""),
            "page_no": int(page_no or 0),
            "description": str(description or "").strip(),
            "mime_type": "image/png",
            "relative_path": relative_path.as_posix(),
            "image_sha": image_sha,
            "created_at": created_at,
        }
        assets[asset_id] = entry
        if self._batch_manifest is not None:
            # Batched: defer to commit_batch's merged flush.
            self._batch_manifest = manifest
        else:
            self._save_image_now(entry)
        return dict(entry)

    def _save_image_now(self, entry: dict[str, Any]) -> None:
        """Non-batch persist of one entry: locked read-modify-write.

        Merges into the current on-disk manifest (not a stale copy) so a
        concurrent worker's entries are preserved; the cross-process lock makes
        the read-modify-write atomic.
        """
        with acquire_asset_lock(self.asset_dir):
            fresh = _shared_manifest_payload(
                self.manifest_path,
                {"version": ASSET_MANIFEST_VERSION, "assets": {}},
            )
            payload = dict(fresh)
            payload["assets"] = {**(fresh.get("assets") or {}), str(entry["asset_id"]): entry}
            _write_json(self.manifest_path, payload)

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        if not SAFE_ID_RE.fullmatch(str(asset_id or "")):
            return None
        entry = _shared_manifest_payload(
            self.manifest_path,
            {"version": ASSET_MANIFEST_VERSION, "assets": {}},
        ).get("assets", {}).get(asset_id)
        return dict(entry) if isinstance(entry, dict) else None

    def asset_path(self, asset_id: str) -> Path | None:
        entry = self.get_asset(asset_id)
        if not entry:
            return None
        relative_path = str(entry.get("relative_path") or "")
        if not relative_path:
            return None
        root = self.asset_dir.resolve()
        candidate = (self.asset_dir / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.exists() and candidate.is_file() else None

    def public_metadata(
        self,
        entry: dict[str, Any],
        *,
        url_for: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        asset_id = str(entry.get("asset_id") or "")
        payload = {
            "asset_id": asset_id,
            "source_hash": str(entry.get("source_hash") or ""),
            "source_pdf_name": str(entry.get("source_pdf_name") or ""),
            "page_no": int(entry.get("page_no") or 0),
            "description": str(entry.get("description") or ""),
            "mime_type": str(entry.get("mime_type") or "image/png"),
            "image_sha": str(entry.get("image_sha") or ""),
        }
        if url_for is not None and asset_id:
            payload["url"] = url_for(asset_id)
        return payload

    def assets_for_text(
        self,
        text: str,
        *,
        url_for: Callable[[str], str] | None = None,
    ) -> list[dict[str, Any]]:
        # Load the manifest once rather than re-reading it twice per asset id
        # (once via get_asset, once via asset_path -> get_asset). Uses the
        # shared cached view -- this is the per-retrieved-chunk hot path on
        # every query, so a whole-manifest deepcopy here would make query cost
        # scale with the corpus's total image count instead of with k.
        manifest_assets = _shared_manifest_payload(
            self.manifest_path,
            {"version": ASSET_MANIFEST_VERSION, "assets": {}},
        ).get("assets", {})
        if not isinstance(manifest_assets, dict):
            manifest_assets = {}
        root = self.asset_dir.resolve()
        assets: list[dict[str, Any]] = []
        for asset_id in image_asset_ids(text):
            entry = manifest_assets.get(asset_id)
            if not isinstance(entry, dict):
                continue
            relative_path = str(entry.get("relative_path") or "")
            if not relative_path:
                continue
            candidate = (self.asset_dir / relative_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if not (candidate.exists() and candidate.is_file()):
                continue
            assets.append(self.public_metadata(entry, url_for=url_for))
        return assets

    def remove_source_assets(self, source_hash: str) -> int:
        source_hash = str(source_hash or "").strip()
        if not source_hash:
            return 0
        if self._batch_manifest is not None:
            # Batched: drop entries from the in-memory manifest; commit_batch's
            # three-way merge replays the removal against the fresh on-disk
            # manifest without resurrecting foreign entries.
            manifest = self._batch_manifest
            assets = manifest.get("assets", {})
            removed_paths: list[Path] = []
            if isinstance(assets, dict):
                kept: dict[str, Any] = {}
                for asset_id, entry in assets.items():
                    if isinstance(entry, dict) and str(entry.get("source_hash") or "") == source_hash:
                        removed_paths.extend(self._asset_file_paths(entry))
                    else:
                        kept[str(asset_id)] = entry
                manifest["assets"] = kept
        else:
            # Locked read-modify-write against the shared cached view: without
            # the lock, two ingestion workers rewriting the full manifest can
            # drop each other's entries (last writer wins). Skip the rewrite
            # entirely when nothing matches -- a per-PDF pre-clean on an
            # already-clean manifest must not cost an O(assets) JSON rewrite.
            removed_paths = []
            with acquire_asset_lock(self.asset_dir):
                shared = _shared_manifest_payload(
                    self.manifest_path,
                    {"version": ASSET_MANIFEST_VERSION, "assets": {}},
                )
                assets = shared.get("assets", {})
                kept: dict[str, Any] = {}
                for asset_id, entry in assets.items():
                    if isinstance(entry, dict) and str(entry.get("source_hash") or "") == source_hash:
                        removed_paths.extend(self._asset_file_paths(entry))
                    else:
                        kept[str(asset_id)] = entry
                if removed_paths:
                    payload = dict(shared)
                    payload["assets"] = kept
                    _write_json(self.manifest_path, payload)
        # File deletions happen outside the lock: idempotent, and paths were
        # captured from the pre-removal manifest so cache staleness can't
        # strand a file.
        for path in removed_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        source_dir = self.asset_dir / _safe_component(source_hash, "unknown-source")
        if source_dir.exists():
            shutil.rmtree(source_dir, ignore_errors=True)
        return len(removed_paths)

    def _asset_file_paths(self, entry: dict[str, Any]) -> list[Path]:
        """Container-escaped on-disk path(s) for a manifest entry (may not exist)."""
        relative_path = str(entry.get("relative_path") or "")
        if not relative_path:
            return []
        try:
            root = self.asset_dir.resolve()
            candidate = (self.asset_dir / relative_path).resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            return []
        return [candidate]
