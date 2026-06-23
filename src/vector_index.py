from __future__ import annotations

import logging
from pathlib import Path

from src.store import SQLiteBlockStore

logger = logging.getLogger(__name__)


class VectorIndex:
    """Chunk vector index using LanceDB when present, SQLite otherwise."""

    def __init__(self, db_dir: str | Path, store: SQLiteBlockStore):
        self.db_dir = Path(db_dir)
        self.store = store
        self.backend = "sqlite"
        self._db = None
        self._table = None
        self._table_name = "chunks"
        self._connect_lancedb()

    def _connect_lancedb(self) -> None:
        try:
            import lancedb

            lance_dir = self.db_dir / "lancedb"
            lance_dir.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(lance_dir))
            if self._table_name in self._db.table_names():
                self._table = self._db.open_table(self._table_name)
            self.backend = "lancedb"
        except Exception as exc:
            logger.warning("Using SQLite vector fallback: %s", exc)
            self.backend = "sqlite"
            self._db = None
            self._table = None

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        if self.backend == "lancedb":
            self._upsert_lancedb(rows)
            return
        self.store.upsert_vectors({row["chunk_id"]: row["vector"] for row in rows})

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        if self.backend == "lancedb" and self._table is not None:
            try:
                rows = (
                    self._table.search(vector)
                    .metric("cosine")
                    .limit(top_k)
                    .to_list()
                )
                return [
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "text": row.get("text", ""),
                        "score": 1.0 - float(row.get("_distance", 1.0)),
                    }
                    for row in rows
                ]
            except Exception as exc:
                logger.warning("LanceDB search failed, using SQLite fallback: %s", exc)
        return self.store.vector_search(vector, top_k)

    def _upsert_lancedb(self, rows: list[dict]) -> None:
        if self._table is None:
            self._table = self._db.create_table(self._table_name, rows, mode="overwrite")
            return
        ids = [row["chunk_id"] for row in rows]
        for chunk_id in ids:
            try:
                self._table.delete(f"chunk_id = '{chunk_id}'")
            except Exception:
                pass
        self._table.add(rows)

