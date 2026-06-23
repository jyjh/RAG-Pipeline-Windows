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


HEADING_MODALITIES = {"title", "section_header", "heading"}


def build_chunks_for_blocks(
    doc_id: str,
    blocks: list[BlockRecord],
    config: ChunkingConfig,
) -> ChunkBuildResult:
    chunks: list[ChunkRecord] = []
    duplicate_groups: dict[str, list[str]] = {}
    current_blocks: list[BlockRecord] = []
    current_tokens = 0
    section_path = ""

    for block in blocks:
        content = block.content_for_index()
        if not content:
            continue
        if block.modality in HEADING_MODALITIES:
            section_path = content[:200]

        block_tokens = count_tokens(content)
        if current_blocks and current_tokens + block_tokens > config.max_tokens:
            chunk = _make_chunk(doc_id, current_blocks, section_path)
            chunks.append(chunk)
            duplicate_groups.setdefault(chunk.duplicate_group_id, []).append(chunk.chunk_id)
            current_blocks, current_tokens = _overlap_blocks(current_blocks, config.overlap_tokens)

        current_blocks.append(block)
        current_tokens += block_tokens

    if current_blocks:
        chunk = _make_chunk(doc_id, current_blocks, section_path)
        chunks.append(chunk)
        duplicate_groups.setdefault(chunk.duplicate_group_id, []).append(chunk.chunk_id)

    return ChunkBuildResult(chunks=chunks, duplicate_groups=duplicate_groups)


def count_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text or "")))


def _make_chunk(doc_id: str, blocks: list[BlockRecord], section_path: str) -> ChunkRecord:
    block_ids = [block.block_id for block in blocks]
    parts = []
    if section_path:
        parts.append(f"Section: {section_path}")
    for block in blocks:
        label = f"[page {block.page_num} | {block.modality} | {block.block_id}]"
        parts.append(f"{label}\n{block.content_for_index()}")
    text = normalize_text("\n\n".join(parts))
    raw_content = normalize_text("\n\n".join(block.content_for_index() for block in blocks))
    duplicate_group_id = duplicate_group(raw_content)
    return ChunkRecord(
        chunk_id=stable_id("chunk", doc_id, block_ids[0], block_ids[-1], duplicate_group_id),
        doc_id=doc_id,
        text=text,
        block_ids=block_ids,
        token_count=count_tokens(text),
        section_path=section_path,
        duplicate_group_id=duplicate_group_id,
        metadata={
            "page_start": blocks[0].page_num,
            "page_end": blocks[-1].page_num,
            "modalities": sorted({block.modality for block in blocks}),
        },
    )


def duplicate_group(text: str) -> str:
    normalized = re.sub(r"\W+", " ", (text or "").lower()).strip()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"dup_{digest}"


def _overlap_blocks(blocks: list[BlockRecord], overlap_tokens: int) -> tuple[list[BlockRecord], int]:
    if overlap_tokens <= 0:
        return [], 0
    kept: list[BlockRecord] = []
    total = 0
    for block in reversed(blocks):
        block_tokens = count_tokens(block.content_for_index())
        if kept and total + block_tokens > overlap_tokens:
            break
        kept.append(block)
        total += block_tokens
    kept.reverse()
    return kept, total
