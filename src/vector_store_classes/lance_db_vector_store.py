from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.vector_store as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


# Upper bound on how many server-side-filtered rows a list_records(search=...)
# call will scan before applying the Python-side post-filter. The previous
# implementation loaded EVERY row of the table into Python and filtered there,
# which degraded badly at millions of chunks. The pushdown narrows server-side
# via a content LIKE first; this cap protects against a wildly broad search
# (e.g. a single common letter) returning an unbounded intermediate set.
LIST_RECORDS_SEARCH_SCAN_CAP = 50_000


class LanceDBVectorStore:
    def __init__(self, working_dir: str | Path):
        self.working_dir = Path(working_dir)
        self.db_path = lancedb_path(self.working_dir)
        # Cached handles. LanceDB indexing runs in a subprocess, so the table can
        # change on disk between requests. We detect that cheaply via the mtime of
        # the version-hint file and drop the cached table/existence state when it
        # changes. The DB connection itself is path-based and stateless, so it is
        # reused for the lifetime of the instance.
        self._db_conn = None
        self._table_obj = None
        self._table_signature: tuple[int, int] | None = None
        self._exists_cache: tuple[tuple[int, int] | None, bool] | None = None

    def _version_hint_path(self) -> Path:
        return self.db_path / TABLE_NAME / "_versions" / "latest_version_hint.json"

    def _read_table_signature(self):
        try:
            stat = self._version_hint_path().stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _invalidate_table(self) -> None:
        self._table_obj = None
        self._table_signature = None
        self._exists_cache = None

    def table_version_hint_path(self) -> Path:
        """Path to the LanceDB version-hint file; a cheap freshness signal for ETags."""
        return self._version_hint_path()

    def exists(self) -> bool:
        signature = self._read_table_signature()
        if self._exists_cache is not None and self._exists_cache[0] == signature:
            return self._exists_cache[1]
        result = self._exists_fresh()
        self._exists_cache = (signature, result)
        return result

    def _exists_fresh(self) -> bool:
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
        self._invalidate_table()

    def append_records(
        self,
        records: list[dict[str, Any]],
        *,
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        """Add ``records`` without touching existing rows.

        Used by the streaming indexer to write one file's records at a time so
        peak RAM is ~one file instead of the whole corpus. On the first call
        (table missing) the table is created with ``mode="overwrite"`` -- this
        is a brand-new staged directory, so there is nothing to preserve. On
        subsequent calls the rows are appended via ``table.add``.
        """
        self.db_path.mkdir(parents=True, exist_ok=True)
        rows = [
            self._row_for_lance(record, embedding_model=embedding_model, embedding_dim=embedding_dim)
            for record in records
        ]
        db = self._db()
        if TABLE_NAME not in self._table_names(db):
            # First write into a fresh (staged) directory: create the table.
            if rows:
                db.create_table(TABLE_NAME, data=rows, mode="overwrite")
            else:
                db.create_table(TABLE_NAME, schema=self._schema(embedding_dim), mode="overwrite")
        elif rows:
            self._table().add(rows)
        self._invalidate_table()

    def replace_records_by_source_hash(
        self,
        *,
        source_hash: str,
        records: list[dict[str, Any]],
        embedding_model: str,
        embedding_dim: int,
    ) -> dict[str, Any]:
        """Atomically replace one source's rows with ``records``.

        A delete-then-add of a single source within one logical operation, so
        re-indexing one file is idempotent and never touches other files' rows.
        Uses the existing :meth:`delete_records_by_source_hash`, which is a
        no-op when the table is absent or no rows match. Returns the delete
        summary for diagnostics.
        """
        source_hash = str(source_hash or "")
        deleted = self.delete_records_by_source_hash(source_hashes=[source_hash] if source_hash else [])
        if records:
            self.append_records(
                records,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
            )
        return deleted

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
            # Push the dominant filter (content substring) into LanceDB so we
            # don't load every row of a multi-million-chunk table into Python.
            # The full record_matches() post-filter still runs on the narrowed
            # candidate set to catch matches in title/tags/section_path/etc.
            rows = self._search_records_pushdown(search)
            total = len(rows)
            page = rows[offset : offset + limit]
        else:
            total = self.count()
            page = self._scan_rows(
                offset=offset,
                limit=limit,
                columns=LIST_RECORD_COLUMNS,
                total=total,
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
        self._invalidate_table()
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
        self._invalidate_table()
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
        self._invalidate_table()
        after = self.count()
        return {"deleted": before - after, "remaining": after}

    def records_by_source_hash(self, source_hashes: list[str]) -> list[dict[str, Any]]:
        hashes = sorted({str(value) for value in source_hashes if value})
        if not hashes or not self.exists():
            return []
        rows = self._where(" OR ".join(f"source_hash = {sql_string(source_hash)}" for source_hash in hashes))
        return [record_row(row) for row in rows]

    def records_by_file_path(self, file_path: str) -> list[dict[str, Any]]:
        """Return rows for one legacy file path using a server-side filter."""
        value = str(file_path or "")
        if not value or not self.exists():
            return []
        return [record_row(row) for row in self._where(f"file_path = {sql_string(value)}")]

    def vectors_by_source_hash(self, source_hashes: list[str]) -> list[dict[str, Any]]:
        """Raw rows (including ``vector``) for the given sources.

        :meth:`records_by_source_hash` strips the vector column (via
        ``record_row``) for API/listing use. Vector reuse needs the vector, so
        this returns the raw LanceDB rows -- one read per source, keeping the
        streaming indexer's memory bounded to ~one file.
        """
        hashes = sorted({str(value) for value in source_hashes if value})
        if not hashes or not self.exists():
            return []
        return self._where(" OR ".join(f"source_hash = {sql_string(source_hash)}" for source_hash in hashes))

    def search(self, vector: list[float], *, top_k: int) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        query = self._table().search(vector).limit(max(1, top_k))
        query = _apply_ann_search_params(query)
        rows = query.to_list()
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
                # Fallback for summary nodes whose children share a section_path
                # prefix but don't carry this node's parent_id. Previously this
                # loaded the entire index via _scan_rows() and filtered in Python
                # (OOM at scale). Instead, narrow server-side to the one document
                # first, then apply the prefix filter in Python on just that
                # document's rows.
                doc_id = sql_string(str(parent.get("doc_id", "")))
                prefix = str(parent.get("section_path", ""))
                candidates = self._where(
                    f"doc_id = {doc_id} AND node_type = 'chunk'",
                    limit=max(1, limit) * 4,
                )
                rows = [
                    row
                    for row in candidates
                    if str(row.get("section_path", "")).startswith(prefix)
                ]
        rows.sort(key=lambda row: int(row.get("chunk_index") or 0))
        return [self._normalize_result(row) for row in rows[:limit]]

    def _db(self):
        import lancedb

        if self._db_conn is None:
            self._db_conn = lancedb.connect(str(self.db_path))
        return self._db_conn

    def has_vector_index(self) -> bool:
        """True if an ANN index exists on the vector column.

        Below :data:`ANN_MIN_ROWS` the table is intentionally left un-indexed
        (flat scan is faster for small tables), so callers should check this
        before reporting whether queries use the ANN path.
        """
        if not self.exists():
            return False
        try:
            stats = self._table().index_stats("vector_idx")
            return stats is not None
        except Exception:
            return False

    def create_vector_index(
        self,
        *,
        index_type: str | None = None,
        num_partitions: int | None = None,
        max_partitions: int | None = None,
        num_sub_vectors: int | None = None,
        num_bits: int | None = None,
        min_rows: int | None = None,
        metric: str = "L2",
    ) -> dict[str, Any]:
        """Build (or rebuild) an ANN index on the vector column.

        Idempotent and safe to call on a freshly-built staged table before
        publish, so the published index is already query-accelerated. Returns a
        small diagnostics dict describing what was done. No-op (returns
        ``{"built": False, ...}``) when the table is below ``min_rows`` -- the
        flat scan is faster there and the build cost is wasted.

        Vectors are L2-normalized at embed time, so the metric is L2 (monotonic
        with cosine) to keep the existing score formula valid.
        """
        # Resolve defaults from the proxied module constants at call time (they
        # are not available as parameter defaults because the class body is
        # evaluated before the proxy binding populates them).
        index_type = str(index_type or ANN_INDEX_TYPE)
        max_partitions = int(max_partitions or ANN_MAX_PARTITIONS)
        num_sub_vectors = int(num_sub_vectors or ANN_NUM_SUB_VECTORS)
        num_bits = int(num_bits or ANN_NUM_BITS)
        min_rows = int(min_rows or ANN_MIN_ROWS)
        if num_partitions is None or num_partitions <= 0:
            num_partitions = 0  # auto

        if not self.exists():
            return {"built": False, "reason": "table_missing"}
        count = self.count()
        if count < max(1, int(min_rows)):
            return {"built": False, "reason": "below_min_rows", "rows": count, "min_rows": min_rows}

        # Resolve num_partitions. 0 means auto: sqrt(N) capped at max_partitions.
        import math

        if not num_partitions or num_partitions <= 0:
            num_partitions = min(int(max_partitions), max(1, int(math.isqrt(count))))

        metric_normalized = str(metric or "L2").strip().lower()
        if metric_normalized not in {"l2", "cosine", "dot", "hamming"}:
            metric_normalized = "l2"

        # PQ-family indices need >= 2^num_bits training rows per sub-vector to
        # build the codebook. If the corpus is too small for PQ, transparently
        # fall back to IVF_FLAT (no quantization) so the index still builds and
        # search is still accelerated. This only triggers near the min_rows
        # boundary; at true scale (millions of chunks) PQ trains easily.
        pq_types = {"IVF_PQ", "IVF_HNSW_PQ", "IVF_RQ"}
        pq_min_rows = 2 ** int(num_bits)
        if index_type.upper() in pq_types and count < pq_min_rows:
            index_type = "IVF_FLAT"

        # Clamp sub-vectors for short vectors (768/16 = 48 dims each is fine).
        num_sub_vectors = max(1, min(int(num_sub_vectors), 768 // 8))

        metric_normalized = str(metric or "L2").strip().lower()
        if metric_normalized not in {"l2", "cosine", "dot", "hamming"}:
            metric_normalized = "l2"

        table = self._table()
        kwargs: dict[str, Any] = {
            "metric": metric_normalized,
            "num_partitions": int(num_partitions),
            "vector_column_name": "vector",
            "index_type": str(index_type or "IVF_PQ").upper(),
            "replace": True,
        }
        # num_sub_vectors applies to PQ-family index types.
        if str(index_type or "IVF_PQ").upper() in {
            "IVF_PQ",
            "IVF_HNSW_PQ",
            "IVF_RQ",
        }:
            kwargs["num_sub_vectors"] = int(num_sub_vectors)
            kwargs["num_bits"] = int(num_bits)
        table.create_index(**kwargs)
        self._invalidate_table()
        try:
            stats = table.index_stats("vector_idx")
        except Exception:
            stats = None
        return {
            "built": True,
            "index_type": kwargs["index_type"],
            "num_partitions": int(num_partitions),
            "num_sub_vectors": kwargs.get("num_sub_vectors"),
            "metric": metric_normalized,
            "rows": count,
            "stats": str(stats) if stats is not None else None,
        }

    def drop_vector_index(self) -> dict[str, Any]:
        """Drop the ANN index if present (used before a rebuild)."""
        if not self.exists():
            return {"dropped": False, "reason": "table_missing"}
        if not self.has_vector_index():
            return {"dropped": False, "reason": "no_index"}
        try:
            self._table().drop_index("vector_idx")
            self._invalidate_table()
            return {"dropped": True}
        except Exception as exc:
            return {"dropped": False, "reason": str(exc)}

    def compact(self, *, cleanup_older_than_seconds: float | None = None) -> dict[str, Any]:
        """Compact the Lance table, reclaiming space from delete+add cycles.

        Re-indexing uses logical delete (deletion vectors) then add, so over
        many reindex rounds the ``data/*.lance`` fragments accumulate dead rows
        and scan cost rises. LanceDB's ``optimize()`` merges fragments and drops
        deletion tombstones. Safe to call any time -- it is the LanceDB-recommended
        maintenance op. Returns before/after on-disk size for logging.
        """
        if not self.exists():
            return {"compacted": False, "reason": "table_missing"}
        before = self._on_disk_bytes()
        table = self._table()
        cleanup = None
        if cleanup_older_than_seconds is not None and cleanup_older_than_seconds > 0:
            from datetime import timedelta

            cleanup = timedelta(seconds=float(cleanup_older_than_seconds))
        try:
            table.optimize(cleanup_older_than=cleanup)
        except TypeError:
            # Older LanceDB signatures may not accept kwargs; fall back to plain.
            table.optimize()
        self._invalidate_table()
        after = self._on_disk_bytes()
        return {
            "compacted": True,
            "bytes_before": before,
            "bytes_after": after,
            "bytes_reclaimed": max(0, before - after),
        }

    def _on_disk_bytes(self) -> int:
        """Total bytes used by the Lance table directory on disk."""
        try:
            table_dir = self.db_path / f"{TABLE_NAME}.lance"
            if not table_dir.exists():
                return 0
            return sum(f.stat().st_size for f in table_dir.rglob("*") if f.is_file())
        except OSError:
            return 0

    def _table(self):
        signature = self._read_table_signature()
        if self._table_obj is not None and self._table_signature == signature:
            return self._table_obj
        table = self._db().open_table(TABLE_NAME)
        self._table_obj = table
        self._table_signature = signature
        return table

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

    def _search_records_pushdown(self, search: str) -> list[dict[str, Any]]:
        """Server-side-filtered + Python-post-filtered record search.

        The previous list_records(search=...) loaded every row of the table into
        Python and filtered with record_matches -- O(N) in memory and time. This
        pushes a multi-column ``lower(col) LIKE '%search%' OR ...`` predicate
        into LanceDB to narrow the candidate set server-side, then applies the
        full record_matches() post-filter (which also checks tags and the
        remaining columns) on the narrowed set. The server-side predicate covers
        the same text columns as record_matches (minus ``tags``, which is a list
        and isn't LIKE-able -- the post-filter catches those) so the visible
        result set is unchanged. The intermediate server-side result is capped
        to LIST_RECORDS_SEARCH_SCAN_CAP rows so a very broad search can't
        exhaust memory.
        """
        search = (search or "").strip()
        if not search:
            return []
        escaped = sql_like_escape(search).lower()
        # Push down every text column record_matches checks. tags is a list and
        # isn't LIKE-able in LanceDB, so it's covered by the post-filter only.
        # id/source_hash are excluded (rarely searched; including them would add
        # clauses with no meaningful benefit).
        columns = [
            "content",
            "doc_id",
            "node_type",
            "file_path",
            "title",
            "section_path",
            "source_pdf_name",
            "source_pdf_path",
        ]
        clause = f"lower({columns[0]}) LIKE '%{escaped}%' ESCAPE '\\'"
        for col in columns[1:]:
            clause += f" OR lower({col}) LIKE '%{escaped}%' ESCAPE '\\'"
        try:
            raw_rows = (
                self._table()
                .search()
                .where(clause)
                .select(LIST_RECORD_COLUMNS)
                .limit(LIST_RECORDS_SEARCH_SCAN_CAP)
                .to_list()
            )
        except Exception:
            # If the LIKE/ESCAPE dialect is unsupported in a future LanceDB
            # version, fall back to the original full-scan behavior rather than
            # breaking the admin UI. Correctness over speed.
            raw_rows = self._scan_rows(columns=LIST_RECORD_COLUMNS)
        return [row for row in raw_rows if record_matches(row, search)]

    def _scan_rows(
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        columns: list[str] | None = None,
        total: int | None = None,
    ) -> list[dict[str, Any]]:
        count = total if total is not None else self.count()
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

def _apply_ann_search_params(query):
    """Chain nprobes/refine_factor onto a LanceDB vector query when applicable.

    These only affect the ANN index path; on a flat-scan table they are silently
    ignored by LanceDB, so it's safe to always apply them. Wrapped so a future
    LanceDB API change degrades to the plain query rather than crashing search.
    """
    try:
        nprobes = _source_module.ANN_NPROBES
        if nprobes and hasattr(query, "nprobes"):
            query = query.nprobes(int(nprobes))
    except Exception:
        pass
    try:
        refine = _source_module.ANN_REFINE_FACTOR
        if refine and hasattr(query, "refine_factor"):
            query = query.refine_factor(int(refine))
    except Exception:
        pass
    return query


LanceDBVectorStore.__module__ = _source_module.__name__
finalize_split_class(_source_module, LanceDBVectorStore)

