from __future__ import annotations

import json
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


def load_source_group_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    trust_path = Path(path)
    if not trust_path.exists():
        return {}
    try:
        payload = json.loads(trust_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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
    return groups
