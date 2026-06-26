from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.vector_store as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class LanceDBVectorStore:
    def __init__(self, working_dir: str | Path):
        self.working_dir = Path(working_dir)
        self.db_path = lancedb_path(self.working_dir)

    def exists(self) -> bool:
        if not self.db_path.exists():
            return False
        try:
            if TABLE_NAME in self._table_names():
                return self.is_compatible()
            return False
        except Exception:
            return False

    def is_compatible(self) -> bool:
        try:
            schema_names = set(self._table().schema.names)
        except Exception:
            return False
        return REQUIRED_LANCEDB_COLUMNS.issubset(schema_names)

    def count(self) -> int:
        if not self.exists():
            return 0
        return int(self._table().count_rows())

    def write_records(
        self,
        records: list[dict[str, Any]],
        *,
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        rows = [
            self._row_for_lance(record, embedding_model=embedding_model, embedding_dim=embedding_dim)
            for record in records
        ]
        db = self._db()
        if TABLE_NAME in self._table_names(db):
            db.drop_table(TABLE_NAME)
        if rows:
            db.create_table(TABLE_NAME, data=rows, mode="overwrite")
        else:
            db.create_table(TABLE_NAME, schema=self._schema(embedding_dim), mode="overwrite")

    def list_records(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        search: str = "",
    ) -> dict[str, Any]:
        if not self.exists():
            raise FileNotFoundError(f"LanceDB table not found at {self.db_path / TABLE_NAME}")
        if search:
            rows = [row for row in self._scan_rows(columns=LIST_RECORD_COLUMNS) if record_matches(row, search)]
            total = len(rows)
            page = rows[offset : offset + limit]
        else:
            total = self.count()
            page = self._scan_rows(
                offset=offset,
                limit=limit,
                columns=LIST_RECORD_COLUMNS,
            )
        model, dim = self.metadata()
        return {
            "offset": offset,
            "limit": limit,
            "total": total,
            "rows": [record_row(row) for row in page],
            "embedding_model": model,
            "embedding_dim": dim,
        }

    def iter_record_batches(
        self,
        *,
        batch_size: int = 250,
        search: str = "",
        workers: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        if not self.exists():
            raise FileNotFoundError(f"LanceDB table not found at {self.db_path / TABLE_NAME}")
        batch_size = max(1, int(batch_size))
        worker_count = max(1, int(workers or min(8, max(2, os.cpu_count() or 2))))
        pending: deque[Future[list[dict[str, Any]]]] = deque()
        executor = ThreadPoolExecutor(max_workers=worker_count)

        try:
            raw_batches = self._raw_record_batches(batch_size=batch_size)

            def submit_next() -> bool:
                try:
                    raw_rows = next(raw_batches)
                except StopIteration:
                    return False
                pending.append(executor.submit(_record_batch_rows, raw_rows, search))
                return True

            for _ in range(worker_count * 2):
                if not submit_next():
                    break

            while pending:
                future = pending.popleft()
                rows = future.result()
                submit_next()
                if rows:
                    yield rows
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    def metadata(self) -> tuple[str, int]:
        rows = self._scan_rows(limit=1, columns=["embedding_model", "embedding_dim"])
        return self._metadata(rows)

    def all_records(self) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        return self._scan_rows()

    def get_record(self, record_id: str) -> dict[str, Any]:
        matches = self._where(f"id = {sql_string(record_id)}", limit=1)
        if not matches:
            raise KeyError(record_id)
        return dict(matches[0])

    def update_record(
        self,
        *,
        record_id: str,
        content: str,
        vector: list[float],
        embedding_model: str,
        embedding_dim: int,
    ) -> dict[str, Any]:
        table = self._table()
        matches = self._where(f"id = {sql_string(record_id)}", limit=1)
        if not matches:
            raise KeyError(record_id)
        record = dict(matches[0])
        record["content"] = content
        record["vector"] = list(vector)
        record["embedding_model"] = embedding_model
        record["embedding_dim"] = embedding_dim
        table.update(
            where=f"id = {sql_string(record_id)}",
            values={
                "content": content,
                "vector": list(vector),
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
            },
        )
        return record_row(record)

    def delete_records(self, *, record_ids: list[str]) -> dict[str, Any]:
        ids = [record_id for record_id in record_ids if record_id]
        if not ids:
            raise ValueError("At least one record ID is required.")
        before = self.count()
        existing = {
            row.get("id")
            for row in self._where(" OR ".join(f"id = {sql_string(record_id)}" for record_id in ids))
        }
        if not existing:
            raise KeyError(", ".join(sorted(ids)))
        self._table().delete(" OR ".join(f"id = {sql_string(record_id)}" for record_id in ids))
        after = self.count()
        return {"deleted": before - after, "remaining": after}

    def delete_records_by_source_hash(
        self,
        *,
        source_hashes: list[str],
        legacy_file_paths: list[str] | None = None,
        legacy_doc_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.exists():
            return {"deleted": 0, "remaining": 0}
        clauses = sql_match_clauses(
            source_hashes=source_hashes,
            legacy_file_paths=legacy_file_paths,
            legacy_doc_ids=legacy_doc_ids,
        )
        if not clauses:
            return {"deleted": 0, "remaining": self.count()}
        where = " OR ".join(clauses)
        before = self.count()
        if not self._where(where):
            return {"deleted": 0, "remaining": before}
        self._table().delete(where)
        after = self.count()
        return {"deleted": before - after, "remaining": after}

    def search(self, vector: list[float], *, top_k: int) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        rows = self._table().search(vector).limit(max(1, top_k)).to_list()
        return [self._normalize_result(row) for row in rows]

    def child_chunks(self, parent: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        node_type = str(parent.get("node_type", ""))
        if node_type == "document_summary":
            where = f"doc_id = {sql_string(str(parent.get('doc_id', '')))} AND node_type = 'chunk'"
            rows = self._where(where, limit=max(1, limit))
        else:
            where = f"parent_id = {sql_string(str(parent.get('id', '')))} AND node_type = 'chunk'"
            rows = self._where(where, limit=max(1, limit))
            if not rows and parent.get("section_path"):
                prefix = str(parent.get("section_path", ""))
                rows = [
                    row
                    for row in self._scan_rows()
                    if row.get("doc_id") == parent.get("doc_id")
                    and row.get("node_type") == "chunk"
                    and str(row.get("section_path", "")).startswith(prefix)
                ]
        rows.sort(key=lambda row: int(row.get("chunk_index") or 0))
        return [self._normalize_result(row) for row in rows[:limit]]

    def _db(self):
        import lancedb

        return lancedb.connect(str(self.db_path))

    def _table(self):
        return self._db().open_table(TABLE_NAME)

    def _table_names(self, db=None) -> list[str]:
        db = db or self._db()
        names = db.list_tables()
        if isinstance(names, list):
            return [str(name) for name in names]
        if hasattr(names, "tables"):
            return [str(name) for name in names.tables]
        return [str(name) for name in names]

    def _where(self, where: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        query = self._table().search().where(where)
        if limit is not None:
            query = query.limit(limit)
        return query.to_list()

    def _scan_rows(
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        count = self.count()
        if count <= 0:
            return []
        offset = max(0, int(offset))
        if offset >= count:
            return []
        row_limit = count - offset if limit is None else max(0, min(int(limit), count - offset))
        if row_limit <= 0:
            return []
        query = self._table().search()
        if columns:
            query = query.select(columns)
        if offset:
            query = query.offset(offset)
        return query.limit(row_limit).to_list()

    def _raw_record_batches(self, *, batch_size: int) -> Iterator[list[dict[str, Any]]]:
        count = self.count()
        if count <= 0:
            return
        reader = (
            self._table()
            .search()
            .select(LIST_RECORD_COLUMNS)
            .limit(count)
            .to_batches(batch_size=max(1, int(batch_size)))
        )
        for batch in reader:
            rows = batch.to_pylist()
            if rows:
                yield rows

    def _normalize_result(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        if "_distance" in normalized:
            distance = float(normalized.pop("_distance") or 0.0)
            normalized["score"] = 1.0 / (1.0 + distance)
        normalized["tags"] = list(normalized.get("tags") or [])
        return normalized

    @staticmethod
    def _metadata(rows: list[dict[str, Any]]) -> tuple[str, int]:
        if not rows:
            return "nomic-embed-text", 768
        first = rows[0]
        return str(first.get("embedding_model") or "nomic-embed-text"), int(first.get("embedding_dim") or 768)

    @staticmethod
    def _row_for_lance(
        record: dict[str, Any],
        *,
        embedding_model: str,
        embedding_dim: int,
    ) -> dict[str, Any]:
        vector = list(record.get("vector") or [])
        if len(vector) < embedding_dim:
            vector.extend([0.0] * (embedding_dim - len(vector)))
        elif len(vector) > embedding_dim:
            vector = vector[:embedding_dim]
        return {
            "id": str(record.get("id", "")),
            "doc_id": str(record.get("doc_id", "")),
            "parent_id": str(record.get("parent_id", "")),
            "node_type": str(record.get("node_type", "chunk")),
            "file_path": str(record.get("file_path", "")),
            "chunk_index": int(record.get("chunk_index") if record.get("chunk_index") is not None else -1),
            "content": str(record.get("content", "")),
            "title": str(record.get("title", "")),
            "section_path": str(record.get("section_path", "")),
            "page_start": int(record.get("page_start") or 0),
            "page_end": int(record.get("page_end") or 0),
            "summary": str(record.get("summary", "")),
            "tags": [str(tag) for tag in (record.get("tags") or [])],
            "source_hash": str(record.get("source_hash", "")),
            "source_pdf_name": str(record.get("source_pdf_name", "")),
            "source_pdf_path": str(record.get("source_pdf_path", "")),
            "embedding_model": embedding_model,
            "embedding_dim": int(embedding_dim),
            "vector": [float(value) for value in vector],
        }

    @staticmethod
    def _schema(embedding_dim: int):
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("parent_id", pa.string()),
                pa.field("node_type", pa.string()),
                pa.field("file_path", pa.string()),
                pa.field("chunk_index", pa.int64()),
                pa.field("content", pa.string()),
                pa.field("title", pa.string()),
                pa.field("section_path", pa.string()),
                pa.field("page_start", pa.int64()),
                pa.field("page_end", pa.int64()),
                pa.field("summary", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("source_hash", pa.string()),
                pa.field("source_pdf_name", pa.string()),
                pa.field("source_pdf_path", pa.string()),
                pa.field("embedding_model", pa.string()),
                pa.field("embedding_dim", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
            ]
        )

LanceDBVectorStore.__module__ = _source_module.__name__
finalize_split_class(_source_module, LanceDBVectorStore)

