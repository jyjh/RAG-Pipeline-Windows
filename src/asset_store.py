from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return payload if isinstance(payload, dict) else default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_component(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip(".-")
    return text or fallback


class ImageAssetStore:
    def __init__(self, asset_dir: str | Path):
        self.asset_dir = Path(asset_dir)
        self.manifest_path = self.asset_dir / ASSET_MANIFEST_FILENAME

    def load_manifest(self) -> dict[str, Any]:
        payload = _load_json(
            self.manifest_path,
            {"version": ASSET_MANIFEST_VERSION, "assets": {}},
        )
        payload.setdefault("version", ASSET_MANIFEST_VERSION)
        payload.setdefault("assets", {})
        if not isinstance(payload["assets"], dict):
            payload["assets"] = {}
        return payload

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
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_data)

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
        _write_json(self.manifest_path, manifest)
        return dict(entry)

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        if not SAFE_ID_RE.fullmatch(str(asset_id or "")):
            return None
        entry = self.load_manifest().get("assets", {}).get(asset_id)
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
        assets: list[dict[str, Any]] = []
        for asset_id in image_asset_ids(text):
            entry = self.get_asset(asset_id)
            if not entry or self.asset_path(asset_id) is None:
                continue
            assets.append(self.public_metadata(entry, url_for=url_for))
        return assets

    def remove_source_assets(self, source_hash: str) -> int:
        source_hash = str(source_hash or "").strip()
        if not source_hash:
            return 0
        manifest = self.load_manifest()
        assets = manifest.get("assets", {})
        if not isinstance(assets, dict):
            return 0
        removed = 0
        kept: dict[str, Any] = {}
        for asset_id, entry in assets.items():
            if isinstance(entry, dict) and str(entry.get("source_hash") or "") == source_hash:
                path = self.asset_path(str(asset_id))
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                removed += 1
            else:
                kept[str(asset_id)] = entry
        manifest["assets"] = kept
        _write_json(self.manifest_path, manifest)

        source_key = _safe_component(source_hash, "unknown-source")
        source_dir = self.asset_dir / source_key
        if source_dir.exists():
            shutil.rmtree(source_dir, ignore_errors=True)
        return removed
