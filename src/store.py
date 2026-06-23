from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from src.schema import (
    AssetRecord,
    BlockRecord,
    ChunkRecord,
    DocumentRecord,
    PageRecord,
    json_dumps,
    json_loads,
)


def _now_sql() -> str:
    return "datetime('now')"


class SQLiteBlockStore:
    """Structured metadata and citation store backed by local SQLite."""

    def __init__(self, db_dir: str | Path):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.db_dir / "rag.sqlite"
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._fts_enabled = False
        self.initialize()

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_sha256 TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                page_count INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS pages (
                doc_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                page_type TEXT NOT NULL,
                width REAL,
                height REAL,
                text_chars INTEGER DEFAULT 0,
                image_count INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                PRIMARY KEY (doc_id, page_num),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                block_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                asset_type TEXT NOT NULL,
                path TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                bbox_json TEXT DEFAULT 'null',
                metadata_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS blocks (
                block_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                modality TEXT NOT NULL,
                reading_order INTEGER NOT NULL,
                bbox_json TEXT DEFAULT 'null',
                text TEXT DEFAULT '',
                markdown TEXT DEFAULT '',
                latex TEXT DEFAULT '',
                table_json TEXT DEFAULT '',
                confidence REAL,
                asset_id TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}',
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_blocks_doc_order
                ON blocks(doc_id, page_num, reading_order);
            CREATE INDEX IF NOT EXISTS idx_blocks_modality
                ON blocks(modality);

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                section_path TEXT DEFAULT '',
                duplicate_group_id TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunk_blocks (
                chunk_id TEXT NOT NULL,
                block_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (chunk_id, block_id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                FOREIGN KEY (block_id) REFERENCES blocks(block_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
            );
            """
        )
        try:
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(chunk_id UNINDEXED, doc_id UNINDEXED, text, tokenize='unicode61')"
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False
        self.conn.commit()

    def reset(self) -> None:
        cur = self.conn.cursor()
        for table in ("chunk_vectors", "chunk_blocks", "chunks", "assets", "blocks", "pages", "documents"):
            cur.execute(f"DELETE FROM {table}")
        if self._fts_enabled:
            cur.execute("DELETE FROM chunks_fts")
        self.conn.commit()

    def clear_index(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunk_vectors")
        cur.execute("DELETE FROM chunk_blocks")
        cur.execute("DELETE FROM chunks")
        if self._fts_enabled:
            cur.execute("DELETE FROM chunks_fts")
        self.conn.commit()

    def upsert_document(self, record: DocumentRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO documents
                (doc_id, title, source_path, source_sha256, status, page_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                title=excluded.title,
                source_path=excluded.source_path,
                source_sha256=excluded.source_sha256,
                status=excluded.status,
                page_count=excluded.page_count,
                metadata_json=excluded.metadata_json
            """,
            (
                record.doc_id,
                record.title,
                record.source_path,
                record.source_sha256,
                record.status,
                record.page_count,
                json_dumps(record.metadata),
            ),
        )
        self.conn.commit()

    def upsert_page(self, record: PageRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO pages
                (doc_id, page_num, page_type, width, height, text_chars, image_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, page_num) DO UPDATE SET
                page_type=excluded.page_type,
                width=excluded.width,
                height=excluded.height,
                text_chars=excluded.text_chars,
                image_count=excluded.image_count,
                metadata_json=excluded.metadata_json
            """,
            (
                record.doc_id,
                record.page_num,
                record.page_type,
                record.width,
                record.height,
                record.text_chars,
                record.image_count,
                json_dumps(record.metadata),
            ),
        )
        self.conn.commit()

    def upsert_asset(self, record: AssetRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO assets
                (asset_id, doc_id, block_id, page_num, asset_type, path, mime_type, bbox_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                block_id=excluded.block_id,
                page_num=excluded.page_num,
                asset_type=excluded.asset_type,
                path=excluded.path,
                mime_type=excluded.mime_type,
                bbox_json=excluded.bbox_json,
                metadata_json=excluded.metadata_json
            """,
            (
                record.asset_id,
                record.doc_id,
                record.block_id,
                record.page_num,
                record.asset_type,
                record.path,
                record.mime_type,
                json_dumps(record.bbox),
                json_dumps(record.metadata),
            ),
        )
        self.conn.commit()

    def set_asset_block(self, asset_id: str, block_id: str) -> None:
        self.conn.execute(
            "UPDATE assets SET block_id = ? WHERE asset_id = ?",
            (block_id, asset_id),
        )
        self.conn.commit()

    def upsert_block(self, record: BlockRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO blocks
                (block_id, doc_id, page_num, modality, reading_order, bbox_json,
                 text, markdown, latex, table_json, confidence, asset_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(block_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                page_num=excluded.page_num,
                modality=excluded.modality,
                reading_order=excluded.reading_order,
                bbox_json=excluded.bbox_json,
                text=excluded.text,
                markdown=excluded.markdown,
                latex=excluded.latex,
                table_json=excluded.table_json,
                confidence=excluded.confidence,
                asset_id=excluded.asset_id,
                metadata_json=excluded.metadata_json
            """,
            (
                record.block_id,
                record.doc_id,
                record.page_num,
                record.modality,
                record.reading_order,
                json_dumps(record.bbox),
                record.text,
                record.markdown,
                record.latex,
                record.table_json,
                record.confidence,
                record.asset_id,
                json_dumps(record.metadata),
            ),
        )
        self.conn.commit()

    def upsert_chunk(self, record: ChunkRecord) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks
                (chunk_id, doc_id, text, token_count, section_path, duplicate_group_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                text=excluded.text,
                token_count=excluded.token_count,
                section_path=excluded.section_path,
                duplicate_group_id=excluded.duplicate_group_id,
                metadata_json=excluded.metadata_json
            """,
            (
                record.chunk_id,
                record.doc_id,
                record.text,
                record.token_count,
                record.section_path,
                record.duplicate_group_id,
                json_dumps(record.metadata),
            ),
        )
        cur.execute("DELETE FROM chunk_blocks WHERE chunk_id = ?", (record.chunk_id,))
        cur.executemany(
            "INSERT INTO chunk_blocks (chunk_id, block_id, ordinal) VALUES (?, ?, ?)",
            [(record.chunk_id, block_id, i) for i, block_id in enumerate(record.block_ids)],
        )
        if self._fts_enabled:
            cur.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (record.chunk_id,))
            cur.execute(
                "INSERT INTO chunks_fts (chunk_id, doc_id, text) VALUES (?, ?, ?)",
                (record.chunk_id, record.doc_id, record.text),
            )
        self.conn.commit()

    def upsert_vectors(self, vectors: dict[str, list[float]]) -> None:
        rows = [
            (chunk_id, len(vector), json.dumps(vector),)
            for chunk_id, vector in vectors.items()
        ]
        self.conn.executemany(
            """
            INSERT INTO chunk_vectors (chunk_id, dim, vector_json)
            VALUES (?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                dim=excluded.dim,
                vector_json=excluded.vector_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        self.conn.commit()

    def vector_search(self, query_vector: list[float], top_k: int) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT cv.chunk_id, cv.vector_json, c.doc_id, c.text
            FROM chunk_vectors cv
            JOIN chunks c ON c.chunk_id = cv.chunk_id
            """
        ).fetchall()
        scored = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            score = _cosine(query_vector, vector)
            scored.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "score": score,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def list_documents(self) -> list[DocumentRecord]:
        rows = self.conn.execute("SELECT * FROM documents ORDER BY title").fetchall()
        return [self._document_from_row(row) for row in rows]

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self.conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._document_from_row(row) if row else None

    def list_blocks(self, doc_id: str | None = None) -> list[BlockRecord]:
        if doc_id:
            rows = self.conn.execute(
                "SELECT * FROM blocks WHERE doc_id = ? ORDER BY page_num, reading_order",
                (doc_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM blocks ORDER BY doc_id, page_num, reading_order"
            ).fetchall()
        return [self._block_from_row(row) for row in rows]

    def get_block(self, block_id: str) -> BlockRecord | None:
        row = self.conn.execute("SELECT * FROM blocks WHERE block_id = ?", (block_id,)).fetchone()
        return self._block_from_row(row) if row else None

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        row = self.conn.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
        return self._asset_from_row(row) if row else None

    def list_chunks(self) -> list[ChunkRecord]:
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY doc_id, chunk_id").fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: Iterable[str]) -> dict[str, ChunkRecord]:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row["chunk_id"]: self._chunk_from_row(row) for row in rows}

    def get_blocks_for_chunk(self, chunk_id: str) -> list[BlockRecord]:
        rows = self.conn.execute(
            """
            SELECT b.*
            FROM chunk_blocks cb
            JOIN blocks b ON b.block_id = cb.block_id
            WHERE cb.chunk_id = ?
            ORDER BY cb.ordinal
            """,
            (chunk_id,),
        ).fetchall()
        return [self._block_from_row(row) for row in rows]

    def get_adjacent_blocks(self, block: BlockRecord, window: int) -> list[BlockRecord]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM blocks
            WHERE doc_id = ?
              AND page_num = ?
              AND reading_order BETWEEN ? AND ?
            ORDER BY reading_order
            """,
            (
                block.doc_id,
                block.page_num,
                block.reading_order - window,
                block.reading_order + window,
            ),
        ).fetchall()
        return [self._block_from_row(row) for row in rows]

    def search_chunks_fts(self, query: str, top_k: int) -> list[dict]:
        if not self._fts_enabled:
            return self._fallback_text_search(query, top_k)
        match = _fts_query(query)
        if not match:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT chunk_id, doc_id, text, bm25(chunks_fts) AS rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return self._fallback_text_search(query, top_k)
        return [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "text": row["text"],
                "score": 1.0 / (1.0 + max(0.0, float(row["rank"]))),
            }
            for row in rows
        ]

    def _fallback_text_search(self, query: str, top_k: int) -> list[dict]:
        terms = set(_terms(query))
        if not terms:
            return []
        scored = []
        for chunk in self.list_chunks():
            words = set(_terms(chunk.text))
            overlap = len(terms & words)
            if overlap:
                scored.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "score": overlap / max(1, len(terms)),
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _document_from_row(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            title=row["title"],
            source_path=row["source_path"],
            source_sha256=row["source_sha256"] or "",
            created_at=row["created_at"] or "",
            status=row["status"],
            page_count=row["page_count"] or 0,
            metadata=json_loads(row["metadata_json"]),
        )

    def _block_from_row(self, row: sqlite3.Row) -> BlockRecord:
        return BlockRecord(
            block_id=row["block_id"],
            doc_id=row["doc_id"],
            page_num=row["page_num"],
            modality=row["modality"],
            reading_order=row["reading_order"],
            bbox=json_loads(row["bbox_json"], None),
            text=row["text"] or "",
            markdown=row["markdown"] or "",
            latex=row["latex"] or "",
            table_json=row["table_json"] or "",
            confidence=row["confidence"],
            asset_id=row["asset_id"] or "",
            metadata=json_loads(row["metadata_json"]),
        )

    def _asset_from_row(self, row: sqlite3.Row) -> AssetRecord:
        return AssetRecord(
            asset_id=row["asset_id"],
            doc_id=row["doc_id"],
            block_id=row["block_id"],
            page_num=row["page_num"],
            asset_type=row["asset_type"],
            path=row["path"],
            mime_type=row["mime_type"],
            bbox=json_loads(row["bbox_json"], None),
            metadata=json_loads(row["metadata_json"]),
        )

    def _chunk_from_row(self, row: sqlite3.Row) -> ChunkRecord:
        block_rows = self.conn.execute(
            "SELECT block_id FROM chunk_blocks WHERE chunk_id = ? ORDER BY ordinal",
            (row["chunk_id"],),
        ).fetchall()
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            block_ids=[r["block_id"] for r in block_rows],
            token_count=row["token_count"],
            section_path=row["section_path"] or "",
            duplicate_group_id=row["duplicate_group_id"] or "",
            metadata=json_loads(row["metadata_json"]),
            created_at=row["created_at"] or "",
        )


def _terms(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]{2,}", text or "")]


def _fts_query(query: str) -> str:
    terms = _terms(query)
    return " OR ".join(f'"{term}"' for term in terms[:16])


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
