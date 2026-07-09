from __future__ import annotations

import logging
from pathlib import Path

from src.store import SQLiteBlockStore

logger = logging.getLogger(__name__)


class VectorIndex:
    def __init__(self, db_dir: str | Path, store: SQLiteBlockStore):
        self.db_dir = Path(db_dir)
        self.store = store
        self.backend = "sqlite"
        self._db = None
        self._table = None
        self._connect_lancedb()

    def _connect_lancedb(self) -> None:
        try:
            import lancedb

            path = self.db_dir / "lancedb"
            path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(path))
            self._table = self._db.open_table("chunks") if "chunks" in self._db.table_names() else None
            self.backend = "lancedb"
        except Exception as exc:
            logger.warning("Using SQLite vector fallback: %s", exc)

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        if self.backend == "lancedb":
            if self._table is None:
                self._table = self._db.create_table("chunks", rows, mode="overwrite")
            else:
                for row in rows:
                    try:
                        self._table.delete(f"chunk_id = '{row['chunk_id']}'")
                    except Exception:
                        pass
                self._table.add(rows)
        else:
            self.store.upsert_vectors({row["chunk_id"]: row["vector"] for row in rows})

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        if self.backend == "lancedb" and self._table is not None:
            rows = self._table.search(vector).metric("cosine").limit(top_k).to_list()
            return [{"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "text": r.get("text", ""),
                     "score": 1.0 - float(r.get("_distance", 1.0))} for r in rows]
        return self.store.vector_search(vector, top_k)

