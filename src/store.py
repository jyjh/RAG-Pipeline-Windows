from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from src.schema import AssetRecord, BlockRecord, ChunkRecord, DocumentRecord, PageRecord, json_dumps, json_loads


class SQLiteBlockStore:
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
                doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, source_path TEXT NOT NULL,
                source_sha256 TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, status TEXT NOT NULL,
                page_count INTEGER DEFAULT 0, metadata_json TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS pages (
                doc_id TEXT NOT NULL, page_num INTEGER NOT NULL, page_type TEXT NOT NULL,
                width REAL, height REAL, text_chars INTEGER DEFAULT 0, image_count INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}', PRIMARY KEY (doc_id, page_num)
            );
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, block_id TEXT NOT NULL,
                page_num INTEGER NOT NULL, asset_type TEXT NOT NULL, path TEXT NOT NULL,
                mime_type TEXT NOT NULL, bbox_json TEXT DEFAULT 'null', metadata_json TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS blocks (
                block_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, page_num INTEGER NOT NULL,
                modality TEXT NOT NULL, reading_order INTEGER NOT NULL, bbox_json TEXT DEFAULT 'null',
                text TEXT DEFAULT '', markdown TEXT DEFAULT '', latex TEXT DEFAULT '', table_json TEXT DEFAULT '',
                confidence REAL, asset_id TEXT DEFAULT '', metadata_json TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_blocks_doc_order ON blocks(doc_id, page_num, reading_order);
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, text TEXT NOT NULL,
                token_count INTEGER NOT NULL, section_path TEXT DEFAULT '', duplicate_group_id TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}', created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunk_blocks (
                chunk_id TEXT NOT NULL, block_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                PRIMARY KEY (chunk_id, block_id)
            );
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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
        for table in ("chunk_vectors", "chunk_blocks", "chunks", "assets", "blocks", "pages", "documents"):
            self.conn.execute(f"DELETE FROM {table}")
        if self._fts_enabled:
            self.conn.execute("DELETE FROM chunks_fts")
        self.conn.commit()

    def clear_index(self) -> None:
        for table in ("chunk_vectors", "chunk_blocks", "chunks"):
            self.conn.execute(f"DELETE FROM {table}")
        if self._fts_enabled:
            self.conn.execute("DELETE FROM chunks_fts")
        self.conn.commit()

    def upsert_document(self, r: DocumentRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO documents (doc_id,title,source_path,source_sha256,status,page_count,metadata_json)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(doc_id) DO UPDATE SET title=excluded.title, source_path=excluded.source_path,
            source_sha256=excluded.source_sha256, status=excluded.status, page_count=excluded.page_count,
            metadata_json=excluded.metadata_json
            """,
            (r.doc_id, r.title, r.source_path, r.source_sha256, r.status, r.page_count, json_dumps(r.metadata)),
        )
        self.conn.commit()

    def upsert_page(self, r: PageRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO pages (doc_id,page_num,page_type,width,height,text_chars,image_count,metadata_json)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(doc_id,page_num) DO UPDATE SET page_type=excluded.page_type, width=excluded.width,
            height=excluded.height, text_chars=excluded.text_chars, image_count=excluded.image_count,
            metadata_json=excluded.metadata_json
            """,
            (r.doc_id, r.page_num, r.page_type, r.width, r.height, r.text_chars, r.image_count, json_dumps(r.metadata)),
        )
        self.conn.commit()

    def upsert_asset(self, r: AssetRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO assets (asset_id,doc_id,block_id,page_num,asset_type,path,mime_type,bbox_json,metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(asset_id) DO UPDATE SET block_id=excluded.block_id, path=excluded.path,
            metadata_json=excluded.metadata_json
            """,
            (r.asset_id, r.doc_id, r.block_id, r.page_num, r.asset_type, r.path, r.mime_type, json_dumps(r.bbox), json_dumps(r.metadata)),
        )
        self.conn.commit()

    def set_asset_block(self, asset_id: str, block_id: str) -> None:
        self.conn.execute("UPDATE assets SET block_id=? WHERE asset_id=?", (block_id, asset_id))
        self.conn.commit()

    def upsert_block(self, r: BlockRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO blocks (block_id,doc_id,page_num,modality,reading_order,bbox_json,text,markdown,latex,
            table_json,confidence,asset_id,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(block_id) DO UPDATE SET markdown=excluded.markdown, text=excluded.text,
            metadata_json=excluded.metadata_json
            """,
            (r.block_id, r.doc_id, r.page_num, r.modality, r.reading_order, json_dumps(r.bbox), r.text, r.markdown,
             r.latex, r.table_json, r.confidence, r.asset_id, json_dumps(r.metadata)),
        )
        self.conn.commit()

    def upsert_chunk(self, r: ChunkRecord) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks (chunk_id,doc_id,text,token_count,section_path,duplicate_group_id,metadata_json)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(chunk_id) DO UPDATE SET text=excluded.text, token_count=excluded.token_count,
            section_path=excluded.section_path, duplicate_group_id=excluded.duplicate_group_id,
            metadata_json=excluded.metadata_json
            """,
            (r.chunk_id, r.doc_id, r.text, r.token_count, r.section_path, r.duplicate_group_id, json_dumps(r.metadata)),
        )
        cur.execute("DELETE FROM chunk_blocks WHERE chunk_id=?", (r.chunk_id,))
        cur.executemany("INSERT INTO chunk_blocks (chunk_id,block_id,ordinal) VALUES (?,?,?)",
                        [(r.chunk_id, bid, i) for i, bid in enumerate(r.block_ids)])
        if self._fts_enabled:
            cur.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (r.chunk_id,))
            cur.execute("INSERT INTO chunks_fts (chunk_id,doc_id,text) VALUES (?,?,?)", (r.chunk_id, r.doc_id, r.text))
        self.conn.commit()

    def upsert_vectors(self, vectors: dict[str, list[float]]) -> None:
        self.conn.executemany(
            """
            INSERT INTO chunk_vectors (chunk_id,dim,vector_json) VALUES (?,?,?)
            ON CONFLICT(chunk_id) DO UPDATE SET dim=excluded.dim, vector_json=excluded.vector_json,
            updated_at=CURRENT_TIMESTAMP
            """,
            [(cid, len(vec), json.dumps(vec)) for cid, vec in vectors.items()],
        )
        self.conn.commit()

    def vector_search(self, query_vector: list[float], top_k: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT cv.chunk_id,cv.vector_json,c.doc_id,c.text FROM chunk_vectors cv JOIN chunks c ON c.chunk_id=cv.chunk_id"
        ).fetchall()
        scored = []
        for row in rows:
            scored.append({"chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "text": row["text"],
                           "score": _cosine(query_vector, json.loads(row["vector_json"]))})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def list_documents(self) -> list[DocumentRecord]:
        return [self._doc(r) for r in self.conn.execute("SELECT * FROM documents ORDER BY title").fetchall()]

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self.conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def list_blocks(self, doc_id: str | None = None) -> list[BlockRecord]:
        if doc_id:
            rows = self.conn.execute("SELECT * FROM blocks WHERE doc_id=? ORDER BY page_num,reading_order", (doc_id,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM blocks ORDER BY doc_id,page_num,reading_order").fetchall()
        return [self._block(r) for r in rows]

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        row = self.conn.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        return self._asset(row) if row else None

    def list_chunks(self) -> list[ChunkRecord]:
        return [self._chunk(r) for r in self.conn.execute("SELECT * FROM chunks ORDER BY doc_id,chunk_id").fetchall()]

    def get_chunks_by_ids(self, chunk_ids: Iterable[str]) -> dict[str, ChunkRecord]:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return {}
        rows = self.conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' for _ in ids)})", ids).fetchall()
        return {r["chunk_id"]: self._chunk(r) for r in rows}

    def get_blocks_for_chunk(self, chunk_id: str) -> list[BlockRecord]:
        rows = self.conn.execute(
            """
            SELECT b.* FROM chunk_blocks cb JOIN blocks b ON b.block_id=cb.block_id
            WHERE cb.chunk_id=? ORDER BY cb.ordinal
            """,
            (chunk_id,),
        ).fetchall()
        return [self._block(r) for r in rows]

    def get_adjacent_blocks(self, block: BlockRecord, window: int) -> list[BlockRecord]:
        rows = self.conn.execute(
            """
            SELECT * FROM blocks WHERE doc_id=? AND page_num=? AND reading_order BETWEEN ? AND ?
            ORDER BY reading_order
            """,
            (block.doc_id, block.page_num, block.reading_order - window, block.reading_order + window),
        ).fetchall()
        return [self._block(r) for r in rows]

    def search_chunks_fts(self, query: str, top_k: int) -> list[dict]:
        match = " OR ".join(f'"{t}"' for t in _terms(query)[:16])
        if not match:
            return []
        if self._fts_enabled:
            try:
                rows = self.conn.execute(
                    "SELECT chunk_id,doc_id,text,bm25(chunks_fts) AS rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, top_k),
                ).fetchall()
                return [{"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "text": r["text"],
                         "score": 1.0 / (1.0 + max(0.0, float(r["rank"])))} for r in rows]
            except sqlite3.OperationalError:
                pass
        terms = set(_terms(query))
        scored = []
        for chunk in self.list_chunks():
            overlap = len(terms & set(_terms(chunk.text)))
            if overlap:
                scored.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "text": chunk.text,
                               "score": overlap / max(1, len(terms))})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _doc(self, r: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(r["doc_id"], r["title"], r["source_path"], r["source_sha256"] or "",
                              r["created_at"] or "", r["status"], r["page_count"] or 0, json_loads(r["metadata_json"]))

    def _block(self, r: sqlite3.Row) -> BlockRecord:
        return BlockRecord(r["block_id"], r["doc_id"], r["page_num"], r["modality"], r["reading_order"],
                           r["text"] or "", r["markdown"] or "", r["latex"] or "", r["table_json"] or "",
                           json_loads(r["bbox_json"], None), r["confidence"], r["asset_id"] or "", json_loads(r["metadata_json"]))

    def _asset(self, r: sqlite3.Row) -> AssetRecord:
        return AssetRecord(r["asset_id"], r["doc_id"], r["block_id"], r["page_num"], r["asset_type"], r["path"],
                           r["mime_type"], json_loads(r["bbox_json"], None), json_loads(r["metadata_json"]))

    def _chunk(self, r: sqlite3.Row) -> ChunkRecord:
        rows = self.conn.execute("SELECT block_id FROM chunk_blocks WHERE chunk_id=? ORDER BY ordinal", (r["chunk_id"],)).fetchall()
        return ChunkRecord(r["chunk_id"], r["doc_id"], r["text"], [x["block_id"] for x in rows], r["token_count"],
                           r["section_path"] or "", r["duplicate_group_id"] or "", json_loads(r["metadata_json"]), r["created_at"] or "")


def _terms(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]{2,}", text or "")]


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

