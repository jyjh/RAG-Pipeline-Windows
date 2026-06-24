import shutil
import tempfile
import uuid
from pathlib import Path

from src.vector_store import JsonVectorStore, LanceDBVectorStore, default_store


def test_lancedb_store_create_query_update_delete():
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_vector_store_{uuid.uuid4().hex}"
    try:
        store = LanceDBVectorStore(tmp_path)
        records = [
            {
                "id": "doc-summary",
                "doc_id": "doc",
                "parent_id": "",
                "node_type": "document_summary",
                "file_path": "doc.pdf",
                "chunk_index": -1,
                "content": "document about alpha",
                "title": "Doc",
                "section_path": "Doc",
                "page_start": 1,
                "page_end": 2,
                "summary": "alpha summary",
                "tags": ["alpha"],
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "chunk-1",
                "doc_id": "doc",
                "parent_id": "leaf-summary",
                "node_type": "chunk",
                "file_path": "doc.pdf",
                "chunk_index": 0,
                "content": "alpha chunk",
                "title": "Alpha",
                "section_path": "Doc > Top > Alpha",
                "page_start": 2,
                "page_end": 2,
                "summary": "alpha summary",
                "tags": ["alpha"],
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "top-summary",
                "doc_id": "doc",
                "parent_id": "doc-summary",
                "node_type": "section_summary",
                "file_path": "doc.pdf",
                "chunk_index": -1,
                "content": "top alpha summary",
                "title": "Top",
                "section_path": "Doc > Top",
                "page_start": 2,
                "page_end": 2,
                "summary": "top summary",
                "tags": ["alpha"],
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "leaf-summary",
                "doc_id": "doc",
                "parent_id": "top-summary",
                "node_type": "section_summary",
                "file_path": "doc.pdf",
                "chunk_index": -1,
                "content": "leaf alpha summary",
                "title": "Alpha",
                "section_path": "Doc > Top > Alpha",
                "page_start": 2,
                "page_end": 2,
                "summary": "leaf summary",
                "tags": ["alpha"],
                "vector": [1.0, 0.0, 0.0],
            },
        ]

        store.write_records(records, embedding_model="fake-embed", embedding_dim=3)

        assert store.exists()
        assert store.count() == 4
        assert store.list_records(search="Alpha")["total"] == 4
        assert store.search([1.0, 0.0, 0.0], top_k=1)[0]["id"] in {
            "doc-summary",
            "top-summary",
            "leaf-summary",
            "chunk-1",
        }
        assert store.child_chunks({"id": "doc-summary", "doc_id": "doc", "node_type": "document_summary"}, limit=3)[
            0
        ]["id"] == "chunk-1"
        assert store.child_chunks(
            {
                "id": "top-summary",
                "doc_id": "doc",
                "node_type": "section_summary",
                "section_path": "Doc > Top",
            },
            limit=3,
        )[0]["id"] == "chunk-1"

        row = store.update_record(
            record_id="chunk-1",
            content="edited alpha chunk",
            vector=[0.0, 1.0, 0.0],
            embedding_model="fake-embed",
            embedding_dim=3,
        )
        assert row["content"] == "edited alpha chunk"

        assert store.delete_records(record_ids=["chunk-1"]) == {"deleted": 1, "remaining": 3}
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_default_store_ignores_incompatible_legacy_lancedb_schema():
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_vector_store_{uuid.uuid4().hex}"
    try:
        import lancedb

        legacy_db = lancedb.connect(str(tmp_path / "lancedb"))
        legacy_db.create_table(
            "chunks",
            data=[
                {
                    "chunk_id": "old-chunk",
                    "doc_id": "old-doc",
                    "text": "legacy text",
                    "duplicate_group_id": "",
                    "vector": [1.0, 0.0, 0.0],
                }
            ],
            mode="overwrite",
        )
        JsonVectorStore(tmp_path).write_records(
            [
                {
                    "id": "json-chunk",
                    "doc_id": "doc",
                    "parent_id": "",
                    "node_type": "chunk",
                    "file_path": "doc.md",
                    "chunk_index": 0,
                    "content": "json context",
                    "vector": [1.0, 0.0, 0.0],
                }
            ],
            embedding_model="fake-embed",
            embedding_dim=3,
        )

        assert not LanceDBVectorStore(tmp_path).exists()
        assert isinstance(default_store(tmp_path), JsonVectorStore)
        assert default_store(tmp_path).list_records()["rows"][0]["id"] == "json-chunk"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
