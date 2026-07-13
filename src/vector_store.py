from __future__ import annotations

import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Protocol

from src._class_module_support import import_split_class

_CLASS_MODULE_PROXY_FUNCTIONS = (
    "default_store",
    "lancedb_path",
    "record_row",
    "record_matches",
    "_record_batch_rows",
    "path_variants",
    "record_matches_source",
    "sql_match_clauses",
    "sql_string",
    "sql_like_escape",
)


LANCEDB_DIRNAME = "lancedb"
TABLE_NAME = "chunks"
REQUIRED_LANCEDB_COLUMNS = {
    "id",
    "doc_id",
    "parent_id",
    "node_type",
    "file_path",
    "chunk_index",
    "content",
    "title",
    "section_path",
    "page_start",
    "page_end",
    "summary",
    "tags",
    "source_hash",
    "source_pdf_name",
    "source_pdf_path",
    "embedding_model",
    "embedding_dim",
    "vector",
}
LIST_RECORD_COLUMNS = [
    "id",
    "doc_id",
    "parent_id",
    "node_type",
    "file_path",
    "chunk_index",
    "content",
    "title",
    "section_path",
    "page_start",
    "page_end",
    "summary",
    "tags",
    "source_hash",
    "source_pdf_name",
    "source_pdf_path",
    "embedding_model",
    "embedding_dim",
]

# Approximate-nearest-neighbour (ANN) vector index settings. Below ANN_MIN_ROWS
# LanceDB's flat (brute-force) scan is faster, so we skip building an index for
# small corpora. Above it, an IVF_PQ index turns every query from an O(N) scan
# into a sublinear lookup. Vectors are L2-normalized at embed time, so L2
# distance is monotonic with cosine and the existing score formula stays valid.
ANN_INDEX_TYPE = "IVF_PQ"
ANN_NUM_PARTITIONS = 0  # 0 = auto sqrt(N), capped at ANN_MAX_PARTITIONS
ANN_MAX_PARTITIONS = 1024
ANN_NUM_SUB_VECTORS = 16
ANN_NUM_BITS = 8
ANN_NPROBES = 50
ANN_REFINE_FACTOR = 20
ANN_MIN_ROWS = 50_000


VectorStore = import_split_class("src.vector_store_classes.vector_store", "VectorStore")
VectorStore.__module__ = __name__


def default_store(working_dir: str | Path, *, prefer_lancedb: bool = True) -> VectorStore:
    return LanceDBVectorStore(working_dir)


def lancedb_path(db_dir: str | Path) -> Path:
    return Path(db_dir) / LANCEDB_DIRNAME


def record_row(record: dict[str, Any]) -> dict[str, Any]:
    content = str(record.get("content", ""))
    return {
        "id": str(record.get("id", "")),
        "doc_id": str(record.get("doc_id", "")),
        "parent_id": str(record.get("parent_id", "")),
        "node_type": str(record.get("node_type", "chunk")),
        "file_path": str(record.get("file_path", "")),
        "chunk_index": record.get("chunk_index"),
        "content": content,
        "title": str(record.get("title", "")),
        "section_path": str(record.get("section_path", "")),
        "page_start": record.get("page_start"),
        "page_end": record.get("page_end"),
        "summary": str(record.get("summary", "")),
        "tags": list(record.get("tags") or []),
        "source_hash": str(record.get("source_hash", "")),
        "source_pdf_name": str(record.get("source_pdf_name", "")),
        "source_pdf_path": str(record.get("source_pdf_path", "")),
        "char_count": len(content),
    }


def record_matches(record: dict[str, Any], search: str) -> bool:
    if not search:
        return True
    haystack = "\n".join(
        [
            str(record.get("id", "")),
            str(record.get("doc_id", "")),
            str(record.get("node_type", "")),
            str(record.get("file_path", "")),
            str(record.get("title", "")),
            str(record.get("section_path", "")),
            str(record.get("source_hash", "")),
            str(record.get("source_pdf_name", "")),
            str(record.get("source_pdf_path", "")),
            str(record.get("content", "")),
            " ".join(str(tag) for tag in record.get("tags") or []),
        ]
    ).lower()
    return search.lower() in haystack


def _record_batch_rows(raw_rows: list[dict[str, Any]], search: str) -> list[dict[str, Any]]:
    return [record_row(row) for row in raw_rows if record_matches(row, search)]


LanceDBVectorStore = import_split_class("src.vector_store_classes.lance_db_vector_store", "LanceDBVectorStore")
LanceDBVectorStore.__module__ = __name__


def path_variants(values: list[str]) -> set[str]:
    variants: set[str] = set()
    for value in values:
        raw = str(value)
        if not raw:
            continue
        variants.add(raw)
        variants.add(raw.replace("\\", "/"))
        variants.add(raw.replace("/", "\\"))
        try:
            path = Path(raw)
            variants.add(str(path))
            variants.add(path.as_posix())
        except (OSError, ValueError):
            pass
    return variants


def record_matches_source(
    record: dict[str, Any],
    *,
    source_hashes: set[str],
    legacy_file_paths: set[str],
    legacy_doc_ids: set[str],
) -> bool:
    if str(record.get("source_hash", "")) in source_hashes:
        return True
    if str(record.get("file_path", "")) in legacy_file_paths:
        return True
    return str(record.get("doc_id", "")) in legacy_doc_ids


def sql_match_clauses(
    *,
    source_hashes: list[str],
    legacy_file_paths: list[str] | None = None,
    legacy_doc_ids: list[str] | None = None,
) -> list[str]:
    clauses: list[str] = []
    for source_hash in sorted({str(value) for value in source_hashes if value}):
        clauses.append(f"source_hash = {sql_string(source_hash)}")
    for file_path in sorted(path_variants(legacy_file_paths or [])):
        clauses.append(f"file_path = {sql_string(file_path)}")
    for doc_id in sorted({str(value) for value in legacy_doc_ids or [] if value}):
        clauses.append(f"doc_id = {sql_string(doc_id)}")
    return clauses


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_like_escape(value: str) -> str:
    """Escape a literal for use inside a SQL ``LIKE`` predicate.

    Escapes single quotes (like :func:`sql_string`) and the LIKE wildcard
    characters ``%`` and ``_`` so user-supplied search text is matched literally
    rather than interpreted as a pattern. Callers wrap the result in a
    ``'%...%'`` pattern and prefix with ``ESCAPE '\\'`` on the predicate.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "''")
    escaped = escaped.replace("%", "\\%").replace("_", "\\_")
    return escaped
