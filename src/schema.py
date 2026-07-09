from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return {} if default is None else default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {} if default is None else default


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    source_path: str
    source_sha256: str = ""
    created_at: str = ""
    status: str = "ready"
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageRecord:
    doc_id: str
    page_num: int
    page_type: str = "unknown"
    width: float | None = None
    height: float | None = None
    text_chars: int = 0
    image_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssetRecord:
    asset_id: str
    doc_id: str
    block_id: str
    page_num: int
    asset_type: str
    path: str
    mime_type: str = "image/png"
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlockRecord:
    block_id: str
    doc_id: str
    page_num: int
    modality: str
    reading_order: int
    text: str = ""
    markdown: str = ""
    latex: str = ""
    table_json: str = ""
    bbox: list[float] | None = None
    confidence: float | None = None
    asset_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_for_index(self) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for value in (self.markdown, self.latex, self.table_json, self.text):
            normalized = normalize_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                parts.append(normalized)
        return normalize_text("\n\n".join(parts))

    def snippet(self, limit: int = 400) -> str:
        return self.content_for_index()[:limit].rstrip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    block_ids: list[str]
    token_count: int
    section_path: str = ""
    duplicate_group_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    doc_id: str
    document_title: str
    page: int
    block_id: str
    modality: str
    score: float
    snippet: str
    asset_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedBlock:
    block_id: str
    chunk_id: str
    doc_id: str
    document_title: str
    page: int
    modality: str
    score: float
    text: str
    asset_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryResponse:
    answer: str
    citations: list[Citation]
    retrieved_blocks: list[RetrievedBlock]
    timings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_blocks": [b.to_dict() for b in self.retrieved_blocks],
            "timings": self.timings,
        }

