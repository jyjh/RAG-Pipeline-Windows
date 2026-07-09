from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from src.config import ChunkingConfig
from src.schema import BlockRecord, ChunkRecord, normalize_text, stable_id


@dataclass
class ChunkBuildResult:
    chunks: list[ChunkRecord]
    duplicate_groups: dict[str, list[str]]


def count_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text or "")))


def build_chunks_for_blocks(doc_id: str, blocks: list[BlockRecord], config: ChunkingConfig) -> ChunkBuildResult:
    chunks: list[ChunkRecord] = []
    groups: dict[str, list[str]] = {}
    current: list[BlockRecord] = []
    tokens = 0
    section = ""
    for block in blocks:
        content = block.content_for_index()
        if not content:
            continue
        if block.modality in {"title", "section_header", "heading"}:
            section = content[:200]
        block_tokens = count_tokens(content)
        if current and tokens + block_tokens > config.max_tokens:
            chunk = _make_chunk(doc_id, current, section)
            chunks.append(chunk)
            groups.setdefault(chunk.duplicate_group_id, []).append(chunk.chunk_id)
            current, tokens = _overlap(current, config.overlap_tokens)
        current.append(block)
        tokens += block_tokens
    if current:
        chunk = _make_chunk(doc_id, current, section)
        chunks.append(chunk)
        groups.setdefault(chunk.duplicate_group_id, []).append(chunk.chunk_id)
    return ChunkBuildResult(chunks, groups)


def _make_chunk(doc_id: str, blocks: list[BlockRecord], section: str) -> ChunkRecord:
    parts = [f"Section: {section}"] if section else []
    for block in blocks:
        parts.append(f"[page {block.page_num} | {block.modality} | {block.block_id}]\n{block.content_for_index()}")
    text = normalize_text("\n\n".join(parts))
    raw = normalize_text("\n\n".join(block.content_for_index() for block in blocks))
    dup = "dup_" + hashlib.sha1(re.sub(r"\W+", " ", raw.lower()).encode("utf-8")).hexdigest()[:16]
    ids = [b.block_id for b in blocks]
    return ChunkRecord(stable_id("chunk", doc_id, ids[0], ids[-1], dup), doc_id, text, ids, count_tokens(text), section, dup,
                       {"page_start": blocks[0].page_num, "page_end": blocks[-1].page_num,
                        "modalities": sorted({b.modality for b in blocks})})


def _overlap(blocks: list[BlockRecord], overlap_tokens: int) -> tuple[list[BlockRecord], int]:
    if overlap_tokens <= 0:
        return [], 0
    kept: list[BlockRecord] = []
    total = 0
    for block in reversed(blocks):
        t = count_tokens(block.content_for_index())
        if kept and total + t > overlap_tokens:
            break
        kept.append(block)
        total += t
    kept.reverse()
    return kept, total
