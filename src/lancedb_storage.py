"""
LanceDB vector storage adapter for LightRAG-HKU.

LightRAG-HKU 1.4.10 does not ship a LanceDB backend. This module implements
BaseVectorStorage using lancedb 0.29.x, providing:
  - Disk-backed columnar storage (Lance format)
  - ANN search via cosine distance
  - Memory-mapped reads for near-zero-latency retrieval

Registered in LightRAG via create_lightrag_instance() in src/utils.py by
monkey-patching lightrag.kg.STORAGES before the LightRAG constructor resolves
the storage class name.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, final

import lancedb
import numpy as np
import pyarrow as pa

from lightrag.base import BaseVectorStorage
from lightrag.kg.shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)
from lightrag.utils import compute_mdhash_id

logger = logging.getLogger(__name__)


@final
@dataclass
class LanceDBStorage(BaseVectorStorage):
    """LanceDB-backed vector storage for LightRAG.

    Each LightRAG namespace (entities_vdb, relationships_vdb, chunks_vdb)
    becomes a separate LanceDB table inside the working directory.
    """

    _db: Any = field(default=None, init=False, repr=False)
    _table: Any = field(default=None, init=False, repr=False)
    _table_name: str = field(default="", init=False, repr=False)
    _storage_lock: Any = field(default=None, init=False, repr=False)
    storage_updated: Any = field(default=None, init=False, repr=False)
    _max_batch_size: int = field(default=32, init=False, repr=False)

    def __post_init__(self):
        self._validate_embedding_func()

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        working_dir = self.global_config["working_dir"]
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
            self._table_name = f"{self.workspace}_{self.namespace}"
        else:
            self._table_name = self.namespace
            self.workspace = ""
            workspace_dir = working_dir

        os.makedirs(workspace_dir, exist_ok=True)

        lance_dir = os.path.join(workspace_dir, "lancedb")
        self._db = lancedb.connect(lance_dir)
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

        # Open existing table or defer creation to first upsert
        if self._table_name in self._db.table_names():
            self._table = self._db.open_table(self._table_name)
        else:
            self._table = None

    async def initialize(self):
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )

    # ------------------------------------------------------------------
    #  Schema helpers
    # ------------------------------------------------------------------
    def _build_schema(self, dim: int) -> pa.Schema:
        """Build the Arrow schema for a LanceDB table."""
        fields = [
            pa.field("id", pa.utf8()),
            pa.field("created_at", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
        for mf in sorted(self.meta_fields):
            fields.append(pa.field(mf, pa.utf8()))
        return pa.schema(fields)

    def _ensure_table(self, dim: int):
        """Create the table on first write using the now-known embedding dim."""
        if self._table is not None:
            return
        schema = self._build_schema(dim)
        self._table = self._db.create_table(
            self._table_name, schema=schema, mode="overwrite"
        )

    # ------------------------------------------------------------------
    #  Core CRUD
    # ------------------------------------------------------------------
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        current_time = int(time.time())
        contents = [v["content"] for v in data.values()]
        ids = list(data.keys())

        # Embed in batches (outside lock for concurrency)
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        if len(embeddings) != len(ids):
            logger.error(
                f"[{self.workspace}] embedding count mismatch: "
                f"{len(embeddings)} != {len(ids)}"
            )
            return

        dim = embeddings.shape[1]
        self._ensure_table(dim)

        rows = []
        for i, doc_id in enumerate(ids):
            row = {
                "id": doc_id,
                "created_at": current_time,
                "vector": embeddings[i].tolist(),
            }
            for mf in self.meta_fields:
                row[mf] = data[doc_id].get(mf, "")
            rows.append(row)

        # Upsert via merge_insert (update existing, insert new)
        (
            self._table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if self._table is None:
            return []

        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding = await self.embedding_func([query], _priority=5)
            embedding = embedding[0]

        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        try:
            results = (
                self._table.search(embedding)
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] LanceDB search error: {e}")
            return []

        out = []
        for row in results:
            # LanceDB returns cosine distance in _distance; convert to similarity
            distance = row.get("_distance", 1.0)
            similarity = 1.0 - distance
            if similarity < self.cosine_better_than_threshold:
                continue
            entry = {
                "id": row["id"],
                "distance": similarity,
                "created_at": row.get("created_at"),
            }
            for mf in self.meta_fields:
                if mf in row:
                    entry[mf] = row[mf]
            out.append(entry)
        return out

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        if self._table is None:
            return None
        try:
            rows = self._table.search().where(f"id = '{id}'").limit(1).to_list()
            if rows:
                row = rows[0]
                return {
                    "id": row["id"],
                    "created_at": row.get("created_at"),
                    **{mf: row.get(mf, "") for mf in self.meta_fields},
                }
        except Exception as e:
            logger.error(f"[{self.workspace}] get_by_id error: {e}")
        return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids or self._table is None:
            return [None] * len(ids)

        id_set = ", ".join(f"'{i}'" for i in ids)
        try:
            rows = (
                self._table.search()
                .where(f"id IN ({id_set})")
                .limit(len(ids))
                .to_list()
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] get_by_ids error: {e}")
            return [None] * len(ids)

        row_map = {r["id"]: r for r in rows}
        result = []
        for doc_id in ids:
            row = row_map.get(doc_id)
            if row:
                result.append({
                    "id": row["id"],
                    "created_at": row.get("created_at"),
                    **{mf: row.get(mf, "") for mf in self.meta_fields},
                })
            else:
                result.append(None)
        return result

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids or self._table is None:
            return {}

        id_set = ", ".join(f"'{i}'" for i in ids)
        try:
            rows = (
                self._table.search()
                .where(f"id IN ({id_set})")
                .limit(len(ids))
                .to_list()
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] get_vectors_by_ids error: {e}")
            return {}

        return {
            r["id"]: (
                r["vector"].tolist()
                if isinstance(r["vector"], np.ndarray)
                else r["vector"]
            )
            for r in rows
            if "vector" in r
        }

    async def delete(self, ids: list[str]):
        if not ids or self._table is None:
            return
        id_set = ", ".join(f"'{i}'" for i in ids)
        try:
            self._table.delete(f"id IN ({id_set})")
        except Exception as e:
            logger.error(f"[{self.workspace}] delete error: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        if self._table is None:
            return
        try:
            self._table.delete(
                f"src_id = '{entity_name}' OR tgt_id = '{entity_name}'"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] delete_entity_relation error: {e}"
            )

    async def index_done_callback(self) -> bool:
        if self._storage_lock is None:
            return True
        async with self._storage_lock:
            try:
                # LanceDB auto-persists writes to disk; just notify peers
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                if self.storage_updated is not None:
                    self.storage_updated.value = False
                return True
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] index_done_callback error: {e}"
                )
                return False

    async def drop(self) -> dict[str, str]:
        try:
            if self._table_name in self._db.table_names():
                self._db.drop_table(self._table_name)
            self._table = None
            if self._storage_lock is not None:
                async with self._storage_lock:
                    await set_all_update_flags(
                        self.namespace, workspace=self.workspace
                    )
                    if self.storage_updated is not None:
                        self.storage_updated.value = False
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] drop error: {e}")
            return {"status": "error", "message": str(e)}
