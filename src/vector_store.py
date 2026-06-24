from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Protocol


INDEX_FILENAME = "local_vector_index.json"
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


class VectorStore(Protocol):
    def exists(self) -> bool: ...

    def count(self) -> int: ...

    def write_records(
        self,
        records: list[dict[str, Any]],
        *,
        embedding_model: str,
        embedding_dim: int,
    ) -> None: ...

    def list_records(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        search: str = "",
    ) -> dict[str, Any]: ...

    def get_record(self, record_id: str) -> dict[str, Any]: ...

    def update_record(
        self,
        *,
        record_id: str,
        content: str,
        vector: list[float],
        embedding_model: str,
        embedding_dim: int,
    ) -> dict[str, Any]: ...

    def delete_records(self, *, record_ids: list[str]) -> dict[str, Any]: ...

    def delete_records_by_source_hash(
        self,
        *,
        source_hashes: list[str],
        legacy_file_paths: list[str] | None = None,
        legacy_doc_ids: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def search(self, vector: list[float], *, top_k: int) -> list[dict[str, Any]]: ...

    def child_chunks(self, parent: dict[str, Any], *, limit: int) -> list[dict[str, Any]]: ...


def default_store(working_dir: str | Path, *, prefer_lancedb: bool = True) -> VectorStore:
    working_path = Path(working_dir)
    lancedb_store = LanceDBVectorStore(working_path)
    if prefer_lancedb and lancedb_store.exists():
        return lancedb_store
    json_store = JsonVectorStore(working_path)
    if json_store.exists():
        return json_store
    return lancedb_store if prefer_lancedb else json_store


def vector_index_path(db_dir: str | Path) -> Path:
    return Path(db_dir) / INDEX_FILENAME


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


class LanceDBVectorStore:
    def __init__(self, working_dir: str | Path):
        self.working_dir = Path(working_dir)
        self.db_path = lancedb_path(self.working_dir)

    def exists(self) -> bool:
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
        rows = self._scan_rows()
        rows = [row for row in rows if record_matches(row, search)]
        total = len(rows)
        page = rows[offset : offset + limit]
        model, dim = self._metadata(rows)
        return {
            "offset": offset,
            "limit": limit,
            "total": total,
            "rows": [record_row(row) for row in page],
            "embedding_model": model,
            "embedding_dim": dim,
        }

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

    def _scan_rows(self) -> list[dict[str, Any]]:
        count = self.count()
        if count <= 0:
            return []
        return self._table().search().limit(count).to_list()

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


class JsonVectorStore:
    def __init__(self, working_dir: str | Path):
        self.working_dir = Path(working_dir)
        self.path = vector_index_path(self.working_dir)

    def exists(self) -> bool:
        return self.path.exists()

    def count(self) -> int:
        if not self.exists():
            return 0
        return len(self._load().get("records", []))

    def write_records(
        self,
        records: list[dict[str, Any]],
        *,
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        payload = {
            "backend": "local_vector_json",
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "records": records,
        }
        self._write(payload)

    def list_records(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        search: str = "",
    ) -> dict[str, Any]:
        payload = self._load()
        records = [record for record in payload.get("records", []) if record_matches(record, search)]
        return {
            "offset": offset,
            "limit": limit,
            "total": len(records),
            "rows": [record_row(record) for record in records[offset : offset + limit]],
            "embedding_model": payload.get("embedding_model", "nomic-embed-text"),
            "embedding_dim": int(payload.get("embedding_dim") or 768),
        }

    def get_record(self, record_id: str) -> dict[str, Any]:
        payload = self._load()
        record = next((item for item in payload.get("records", []) if item.get("id") == record_id), None)
        if record is None:
            raise KeyError(record_id)
        result = dict(record)
        result["embedding_model"] = payload.get("embedding_model", "nomic-embed-text")
        result["embedding_dim"] = int(payload.get("embedding_dim") or len(result.get("vector", [])) or 768)
        return result

    def update_record(
        self,
        *,
        record_id: str,
        content: str,
        vector: list[float],
        embedding_model: str,
        embedding_dim: int,
    ) -> dict[str, Any]:
        payload = self._load()
        record = next((item for item in payload.get("records", []) if item.get("id") == record_id), None)
        if record is None:
            raise KeyError(record_id)
        record["content"] = content
        record["vector"] = vector
        payload["embedding_model"] = embedding_model
        payload["embedding_dim"] = embedding_dim
        self._write(payload)
        return record_row(record)

    def delete_records(self, *, record_ids: list[str]) -> dict[str, Any]:
        ids = {record_id for record_id in record_ids if record_id}
        if not ids:
            raise ValueError("At least one record ID is required.")
        payload = self._load()
        original = payload.get("records", [])
        kept = [record for record in original if record.get("id") not in ids]
        deleted = len(original) - len(kept)
        if deleted == 0:
            raise KeyError(", ".join(sorted(ids)))
        payload["records"] = kept
        self._write(payload)
        return {"deleted": deleted, "remaining": len(kept)}

    def delete_records_by_source_hash(
        self,
        *,
        source_hashes: list[str],
        legacy_file_paths: list[str] | None = None,
        legacy_doc_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.exists():
            return {"deleted": 0, "remaining": 0}
        hashes = {str(value) for value in source_hashes if value}
        file_paths = path_variants(legacy_file_paths or [])
        doc_ids = {str(value) for value in legacy_doc_ids or [] if value}
        if not hashes and not file_paths and not doc_ids:
            return {"deleted": 0, "remaining": self.count()}

        payload = self._load()
        original = payload.get("records", [])
        kept = [
            record
            for record in original
            if not record_matches_source(
                record,
                source_hashes=hashes,
                legacy_file_paths=file_paths,
                legacy_doc_ids=doc_ids,
            )
        ]
        deleted = len(original) - len(kept)
        if deleted:
            payload["records"] = kept
            self._write(payload)
        return {"deleted": deleted, "remaining": len(kept)}

    def search(self, vector: list[float], *, top_k: int) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        import numpy as np

        payload = self._load()
        records = payload.get("records", [])
        if not records:
            return []
        vectors = np.asarray([record["vector"] for record in records], dtype=np.float32)
        scores = vectors @ np.asarray(vector, dtype=np.float32)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for index in top_indices:
            record = dict(records[int(index)])
            record["score"] = float(scores[int(index)])
            results.append(record)
        return results

    def child_chunks(self, parent: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        payload = self._load()
        rows = payload.get("records", [])
        if parent.get("node_type") == "document_summary":
            matches = [
                row
                for row in rows
                if row.get("doc_id") == parent.get("doc_id") and row.get("node_type", "chunk") == "chunk"
            ]
        else:
            matches = [
                row
                for row in rows
                if row.get("parent_id") == parent.get("id") and row.get("node_type", "chunk") == "chunk"
            ]
            if not matches and parent.get("section_path"):
                prefix = str(parent.get("section_path", ""))
                matches = [
                    row
                    for row in rows
                    if row.get("doc_id") == parent.get("doc_id")
                    and row.get("node_type", "chunk") == "chunk"
                    and str(row.get("section_path", "")).startswith(prefix)
                ]
        matches.sort(key=lambda row: int(row.get("chunk_index") or 0))
        return [dict(row) for row in matches[:limit]]

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Index file not found at {self.path}")
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        data = json.dumps(payload, indent=2)
        try:
            tmp_path.write_text(data, encoding="utf-8")
            try:
                os.replace(tmp_path, self.path)
            except PermissionError:
                try:
                    self.path.unlink(missing_ok=True)
                    os.replace(tmp_path, self.path)
                except PermissionError:
                    self.path.write_text(data, encoding="utf-8")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except PermissionError:
                pass

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
