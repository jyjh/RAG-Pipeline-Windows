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
    "apply_indexing_config",
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
# During incremental reindex, only retrain the IVF_PQ index when the changed
# row fraction reaches this threshold. Below it, stale centroids are retained
# (recall degrades gracefully, not incorrectly). Tunable via [indexing].
ANN_RETRAIN_THRESHOLD = 0.01


def apply_indexing_config(config: dict[str, Any] | None) -> None:
    """Apply the ``[indexing]`` config values to the module-level ANN constants.

    Called once at startup so :meth:`create_vector_index` defaults and
    ``_apply_ann_search_params`` both honor ``config.toml``. Unknown, missing,
    or non-integer keys leave the hardcoded defaults in place, so this is safe
    to call with an empty/partial config (behavior-preserving). Idempotent.
    """
    if not config:
        return
    global ANN_MIN_ROWS, ANN_NPROBES, ANN_REFINE_FACTOR, ANN_RETRAIN_THRESHOLD
    try:
        ANN_MIN_ROWS = int(config.get("ann_min_rows", ANN_MIN_ROWS))
    except (TypeError, ValueError):
        pass
    try:
        ANN_NPROBES = int(config.get("ann_nprobes", ANN_NPROBES))
    except (TypeError, ValueError):
        pass
    try:
        ANN_REFINE_FACTOR = int(config.get("ann_refine_factor", ANN_REFINE_FACTOR))
    except (TypeError, ValueError):
        pass
    try:
        ANN_RETRAIN_THRESHOLD = float(config.get("ann_retrain_threshold", ANN_RETRAIN_THRESHOLD))
    except (TypeError, ValueError):
        pass


VectorStore = import_split_class("src.vector_store_classes.vector_store", "VectorStore")
VectorStore.__module__ = __name__


class MultiVectorStore:
    """Query-facing union of several per-category ``LanceDBVectorStore`` dirs.

    Implements the subset of the :class:`VectorStore` protocol the query
    engine uses (``exists/count/metadata/search/child_chunks``). ``search``
    fans one ANN query out per store and merges by score -- sound across
    categories because every store must share the configured embedding
    model/dimension (guarded at query time). Matched records are annotated
    with ``category`` (the store's label) so results, sources, and child-chunk
    routing can name the category they came from.

    A single selected category bypasses this wrapper entirely (the engine
    constructs the plain store), so the union only exists when it is needed.
    """

    def __init__(
        self,
        stores: list[Any],
        *,
        labels: list[str] | None = None,
        keys: list[str] | None = None,
        weights: list[float] | None = None,
    ):
        if not stores:
            raise ValueError("MultiVectorStore requires at least one store.")
        self.stores = list(stores)
        self.labels = list(labels) if labels is not None else [f"store{i}" for i in range(len(stores))]
        self.keys = list(keys) if keys is not None else [str(index) for index in range(len(stores))]
        self.weights = list(weights) if weights is not None else [1.0] * len(stores)
        if (
            len(self.labels) != len(self.stores)
            or len(self.keys) != len(self.stores)
            or len(self.weights) != len(self.stores)
        ):
            raise ValueError("labels/keys/weights must align with stores.")

    def _store_for(self, record: dict[str, Any]):
        """Route a previously-returned record back to the store it came from."""
        category = str(record.get("category") or "")
        if category:
            for label, store in zip(self.labels, self.stores):
                if label == category:
                    return store
        return self.stores[0]

    def exists(self) -> bool:
        return any(store.exists() for store in self.stores)

    def count(self) -> int:
        return sum(store.count() for store in self.stores if store.exists())

    def metadata(self) -> tuple[str, int]:
        for store in self.stores:
            if store.exists():
                return store.metadata()
        return self.stores[0].metadata()

    def search(self, vector: list[float], *, top_k: int) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for label, weight, store in zip(self.labels, self.weights, self.stores):
            for row in store.search(vector, top_k=top_k):
                record_id = str(row.get("id") or "")
                if not record_id:
                    continue
                # Record ids are content-derived and identical across
                # categories, so a duplicate here means the same source is
                # (transiently, mid-transfer) indexed in two stores. Keep the
                # higher-scoring copy so ranking and the score cutoff are not
                # dragged down by a stale lower-scoring index.
                if record_id in merged:
                    if float(row.get("score") or 0.0) <= float(merged[record_id].get("score") or 0.0):
                        continue
                    row = dict(row)
                    row["category"] = label
                    row["category_weight"] = float(weight)
                    merged[record_id] = row
                    continue
                row = dict(row)
                row["category"] = label
                row["category_weight"] = float(weight)
                merged[record_id] = row
        return sorted(
            merged.values(),
            key=lambda row: float(row.get("score") or 0.0),
            reverse=True,
        )

    def child_chunks(self, parent: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        store = self._store_for(parent)
        rows = store.child_chunks(parent, limit=limit)
        index = self.stores.index(store) if store in self.stores else 0
        return [dict(row, category=self.labels[index], category_weight=float(self.weights[index])) for row in rows]


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
