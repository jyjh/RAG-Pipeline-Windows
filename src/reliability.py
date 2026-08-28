from __future__ import annotations

import json
import logging
import threading

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any


SOURCE_GROUP_UNGROUPED = "ungrouped"
SOURCE_GROUP_OFFICIAL = "official"
SOURCE_GROUP_STUDENT_RESEARCH = "student_research"
SOURCE_GROUP_UNOFFICIAL = "unofficial"

SOURCE_GROUP_METADATA: dict[str, dict[str, Any]] = {
    SOURCE_GROUP_OFFICIAL: {
        "key": SOURCE_GROUP_OFFICIAL,
        "label": "Official",
        "weight": 1.0,
        "assignable": True,
    },
    SOURCE_GROUP_STUDENT_RESEARCH: {
        "key": SOURCE_GROUP_STUDENT_RESEARCH,
        "label": "Student Research",
        "weight": 0.9,
        "assignable": True,
    },
    SOURCE_GROUP_UNOFFICIAL: {
        "key": SOURCE_GROUP_UNOFFICIAL,
        "label": "Unofficial",
        "weight": 0.8,
        "assignable": True,
    },
    SOURCE_GROUP_UNGROUPED: {
        "key": SOURCE_GROUP_UNGROUPED,
        "label": "Ungrouped",
        "weight": 0.1,
        "assignable": False,
    },
}


def valid_assignable_source_groups() -> tuple[str, ...]:
    return tuple(
        key
        for key, meta in SOURCE_GROUP_METADATA.items()
        if bool(meta.get("assignable"))
    )


def normalize_source_group(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in SOURCE_GROUP_METADATA else SOURCE_GROUP_UNGROUPED


def source_group_details(value: Any) -> dict[str, Any]:
    return dict(SOURCE_GROUP_METADATA[normalize_source_group(value)])


def source_group_label(value: Any) -> str:
    return str(source_group_details(value)["label"])


def source_group_weight(value: Any) -> float:
    return float(source_group_details(value)["weight"])


def source_group_is_assignable(value: Any) -> bool:
    return bool(source_group_details(value).get("assignable"))


# The query engine is constructed per chat request and reads the trust map
# each time, so the O(corpdocs) parse runs on the hot retrieval path. Cache
# the normalized map keyed on the file's (mtime_ns, size) signature -- the
# same trick the PDF registry and asset manifest use -- so unchanged files
# cost one stat() per query instead of a full read+parse.
_SOURCE_GROUP_CACHE_LOCK = threading.Lock()
_SOURCE_GROUP_CACHE: dict[str, tuple[tuple[int, int], dict[str, dict[str, Any]]]] = {}
_SOURCE_GROUP_CACHE_MAX = 8


def load_source_group_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    trust_path = Path(path)
    key = str(trust_path)
    try:
        stat = trust_path.stat()
    except OSError:
        with _SOURCE_GROUP_CACHE_LOCK:
            _SOURCE_GROUP_CACHE.pop(key, None)
        return {}
    signature = (stat.st_mtime_ns, stat.st_size)
    with _SOURCE_GROUP_CACHE_LOCK:
        cached = _SOURCE_GROUP_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return dict(cached[1])
    try:
        payload = json.loads(trust_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse trust map %s: %s", trust_path, exc)
        return {}
    except OSError:
        return {}
    documents = payload.get("documents", {}) if isinstance(payload, dict) else {}
    if not isinstance(documents, dict):
        return {}

    groups: dict[str, dict[str, Any]] = {}
    for source_hash, entry in documents.items():
        if not isinstance(entry, dict):
            continue
        details = source_group_details(entry.get("source_group"))
        groups[str(source_hash)] = details
    with _SOURCE_GROUP_CACHE_LOCK:
        # Do not cache against a corrupt parse: a later repair must be
        # observed (an empty parse is a legitimate result, so it may cache).
        if groups or not documents:
            _SOURCE_GROUP_CACHE[key] = (signature, groups)
            while len(_SOURCE_GROUP_CACHE) > _SOURCE_GROUP_CACHE_MAX:
                _SOURCE_GROUP_CACHE.pop(next(iter(_SOURCE_GROUP_CACHE)))
    return dict(groups)
