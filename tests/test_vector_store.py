import shutil
import tempfile
import uuid
from pathlib import Path

from src.vector_store import LIST_RECORD_COLUMNS, LanceDBVectorStore, default_store


def _source_records():
    return [
        {
            "id": "hash-row",
            "doc_id": "hash-doc",
            "parent_id": "",
            "node_type": "chunk",
            "file_path": "doc-a.pdf",
            "chunk_index": 0,
            "content": "hash content",
            "title": "Doc A",
            "section_path": "Doc A",
            "page_start": 1,
            "page_end": 1,
            "summary": "summary",
            "tags": ["a"],
            "source_hash": "hash-a",
            "source_pdf_name": "doc-a.pdf",
            "source_pdf_path": "uploads/doc-a.pdf",
            "vector": [1.0, 0.0, 0.0],
        },
        {
            "id": "legacy-row",
            "doc_id": "legacy-doc",
            "parent_id": "",
            "node_type": "chunk",
            "file_path": "processed_docs\\legacy.md",
            "chunk_index": 1,
            "content": "legacy content",
            "title": "Legacy",
            "section_path": "Legacy",
            "page_start": 1,
            "page_end": 1,
            "summary": "summary",
            "tags": ["legacy"],
            "vector": [0.0, 1.0, 0.0],
        },
        {
            "id": "kept-row",
            "doc_id": "kept-doc",
            "parent_id": "",
            "node_type": "chunk",
            "file_path": "kept.pdf",
            "chunk_index": 2,
            "content": "kept content",
            "title": "Kept",
            "section_path": "Kept",
            "page_start": 1,
            "page_end": 1,
            "summary": "summary",
            "tags": ["kept"],
            "source_hash": "hash-b",
            "source_pdf_name": "kept.pdf",
            "source_pdf_path": "uploads/kept.pdf",
            "vector": [0.0, 0.0, 1.0],
        },
    ]


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


def test_lancedb_store_preserves_source_metadata_and_deletes_by_hash_or_legacy_path():
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_vector_store_{uuid.uuid4().hex}"
    try:
        store = LanceDBVectorStore(tmp_path)
        store.write_records(_source_records(), embedding_model="fake-embed", embedding_dim=3)

        rows = store.list_records()["rows"]
        assert rows[0]["source_hash"] == "hash-a"
        assert rows[0]["source_pdf_path"] == "uploads/doc-a.pdf"

        assert store.delete_records_by_source_hash(source_hashes=["hash-a"]) == {
            "deleted": 1,
            "remaining": 2,
        }
        assert store.delete_records_by_source_hash(
            source_hashes=["missing"],
            legacy_file_paths=["processed_docs/legacy.md"],
        ) == {"deleted": 1, "remaining": 1}
        assert store.get_record("kept-row")["source_hash"] == "hash-b"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_default_store_never_falls_back_to_legacy_json():
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
        tmp_path.joinpath("local_vector_index.json").write_text('{"records": []}', encoding="utf-8")

        assert not LanceDBVectorStore(tmp_path).exists()
        assert isinstance(default_store(tmp_path), LanceDBVectorStore)
        try:
            default_store(tmp_path).list_records()
        except FileNotFoundError as exc:
            assert "LanceDB table not found" in str(exc)
        else:
            raise AssertionError("Expected incompatible LanceDB to remain unavailable")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def _make_store_with_records(tmp_path, records):
    """Helper: create a store, write records, return it."""
    store = LanceDBVectorStore(tmp_path)
    store.write_records(records, embedding_model="fake-embed", embedding_dim=3)
    return store


def _base_record(record_id, content, *, title="Title", tags=None):
    return {
        "id": record_id,
        "doc_id": "doc",
        "parent_id": "",
        "node_type": "chunk",
        "file_path": "doc.pdf",
        "chunk_index": 0,
        "content": content,
        "title": title,
        "section_path": "Doc",
        "page_start": 1,
        "page_end": 1,
        "summary": "",
        "tags": tags or [],
        "vector": [1.0, 0.0, 0.0],
    }


def test_list_records_search_pushdown_matches_title_only_via_postfilter():
    """A search term present only in title/tags (not content) still matches.

    The server-side pushdown filters on content LIKE, but the post-filter
    (record_matches) still checks title/tags so matches in other columns aren't
    lost. This test pins that two-stage behavior.
    """
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_search_pushdown_{uuid.uuid4().hex}"
    try:
        records = [
            _base_record("r1", "general engineering content", title="Dynamics"),
            _base_record("r2", "completely unrelated text", title="Other"),
            _base_record("r3", "dynamics appears in content too", title="Whatever"),
        ]
        store = _make_store_with_records(tmp_path, records)

        # "dynamics" matches r3 in content (pushdown hit) AND r1 in title only
        # (post-filter catch). r2 matches nowhere.
        result = store.list_records(search="dynamics", limit=50)
        ids = {row["id"] for row in result["rows"]}
        assert ids == {"r1", "r3"}, f"expected r1 (title) + r3 (content), got {ids}"
        assert result["total"] == 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_list_records_search_escapes_like_wildcards():
    """Literal % and _ in search text are matched literally, not as wildcards."""
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_like_escape_{uuid.uuid4().hex}"
    try:
        records = [
            _base_record("r1", "variable_name_5 end"),
            _base_record("r2", "variableXnameX5 end"),  # would match if _ were wildcard
            _base_record("r3", "100% complete"),
        ]
        store = _make_store_with_records(tmp_path, records)

        # Searching for the literal underscore must NOT match r2.
        result = store.list_records(search="name_5", limit=50)
        ids = {row["id"] for row in result["rows"]}
        assert ids == {"r1"}, f"underscore should be literal, got {ids}"

        # Searching for the literal percent must match r3 only.
        result = store.list_records(search="100%", limit=50)
        ids = {row["id"] for row in result["rows"]}
        assert ids == {"r3"}, f"percent should be literal, got {ids}"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_list_records_search_pushdown_narrows_scan(monkeypatch):
    """The pushdown must NOT load every row of the table into Python.

    We spy on the unfiltered _scan_rows path: a search must not trigger a full
    table scan (which is the old behavior). Instead the server-side LIKE narrows
    first. We approximate this by confirming a search over many rows where only
    a few match content returns quickly without materializing all rows.
    """
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_pushdown_perf_{uuid.uuid4().hex}"
    try:
        # 500 rows, only 3 contain the rare search term in content.
        records = [
            _base_record(f"r{i}", f"filler content number {i}")
            for i in range(500)
        ]
        for i, needle in enumerate(["UNIQUENEEDLE", "uniqueneedle", "UniqueNeedle"]):
            records[i * 100]["content"] = f"this row has {needle} in it"
        store = _make_store_with_records(tmp_path, records)

        # Spy on _scan_rows (the old full-scan path). The pushdown should use
        # _where/_table().search().where(...) instead, so _scan_rows must NOT
        # be called for the search branch.
        scan_calls = []
        original_scan = store._scan_rows

        def spy_scan(*args, **kwargs):
            scan_calls.append(kwargs)
            return original_scan(*args, **kwargs)

        monkeypatch.setattr(store, "_scan_rows", spy_scan)

        result = store.list_records(search="uniqueneedle", limit=50)

        assert result["total"] == 3
        # The search must NOT trigger a full-table scan. The old path called
        # _scan_rows(columns=LIST_RECORD_COLUMNS) over every row; the pushdown
        # uses server-side .where() instead. The only _scan_rows call permitted
        # is the tiny single-row metadata() read.
        full_scans = [
            call for call in scan_calls if call.get("columns") == LIST_RECORD_COLUMNS
        ]
        assert full_scans == [], (
            f"search should use server-side pushdown, not a full LIST_RECORD_COLUMNS scan: {full_scans}"
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_compact_runs_without_error_and_returns_diagnostics():
    """store.compact() runs LanceDB optimize and reports before/after sizes."""
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_compact_{uuid.uuid4().hex}"
    try:
        records = [_base_record(f"r{i}", f"content {i}") for i in range(20)]
        store = _make_store_with_records(tmp_path, records)

        # A delete+add cycle leaves a tombstone that compaction can reclaim.
        store.delete_records(record_ids=["r0", "r1"])
        before = store._on_disk_bytes()
        assert before > 0

        result = store.compact()
        assert result["compacted"] is True
        assert "bytes_before" in result
        assert "bytes_after" in result
        assert "bytes_reclaimed" in result
        # Record count is unchanged by compaction (only dead fragments are merged).
        assert store.count() == 18
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_compact_on_missing_table_returns_not_compacted():
    """compact() on a non-existent table is a safe no-op."""
    tmp_path = Path(tempfile.gettempdir()) / f"rag_test_compact_missing_{uuid.uuid4().hex}"
    try:
        store = LanceDBVectorStore(tmp_path)
        result = store.compact()
        assert result["compacted"] is False
        assert result["reason"] == "table_missing"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_apply_indexing_config_overrides_ann_constants():
    """apply_indexing_config writes [indexing] values onto the module constants.

    create_vector_index() defaults and _apply_ann_search_params both read these
    module globals, so this is the wiring that makes config.toml's [indexing]
    section actually take effect. Save/restore so the test is order-independent.
    """
    import src.vector_store as vs

    original = (vs.ANN_MIN_ROWS, vs.ANN_NPROBES, vs.ANN_REFINE_FACTOR, vs.ANN_RETRAIN_THRESHOLD)
    try:
        vs.apply_indexing_config(
            {
                "ann_min_rows": 12345,
                "ann_nprobes": 7,
                "ann_refine_factor": 11,
                "ann_retrain_threshold": 0.05,
            }
        )
        assert vs.ANN_MIN_ROWS == 12345
        assert vs.ANN_NPROBES == 7
        assert vs.ANN_REFINE_FACTOR == 11
        assert vs.ANN_RETRAIN_THRESHOLD == 0.05
    finally:
        vs.ANN_MIN_ROWS, vs.ANN_NPROBES, vs.ANN_REFINE_FACTOR, vs.ANN_RETRAIN_THRESHOLD = original


def test_apply_indexing_config_is_safe_noop_on_empty_or_invalid():
    """Empty/None/partial/invalid configs leave the hardcoded defaults intact."""
    import src.vector_store as vs

    original = (vs.ANN_MIN_ROWS, vs.ANN_NPROBES, vs.ANN_REFINE_FACTOR, vs.ANN_RETRAIN_THRESHOLD)
    try:
        # None and empty dict must be no-ops.
        vs.apply_indexing_config(None)
        assert (vs.ANN_MIN_ROWS, vs.ANN_NPROBES, vs.ANN_REFINE_FACTOR, vs.ANN_RETRAIN_THRESHOLD) == original
        vs.apply_indexing_config({})
        assert (vs.ANN_MIN_ROWS, vs.ANN_NPROBES, vs.ANN_REFINE_FACTOR, vs.ANN_RETRAIN_THRESHOLD) == original

        # Invalid types must not raise and must not mutate.
        vs.apply_indexing_config({"ann_nprobes": "not-a-number", "ann_min_rows": None})
        assert (vs.ANN_MIN_ROWS, vs.ANN_NPROBES, vs.ANN_REFINE_FACTOR, vs.ANN_RETRAIN_THRESHOLD) == original

        # Partial config applies only the valid keys.
        vs.apply_indexing_config({"ann_nprobes": 99, "ann_min_rows": "bad"})
        assert vs.ANN_NPROBES == 99
        assert vs.ANN_MIN_ROWS == original[0]  # unchanged by the bad value
        assert vs.ANN_REFINE_FACTOR == original[2]
        assert vs.ANN_RETRAIN_THRESHOLD == original[3]
    finally:
        vs.ANN_MIN_ROWS, vs.ANN_NPROBES, vs.ANN_REFINE_FACTOR, vs.ANN_RETRAIN_THRESHOLD = original
