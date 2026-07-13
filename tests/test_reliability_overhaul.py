"""Tests for the reliability & scaling overhaul layers.

Covers:
- LanceDBVectorStore.append_records / replace_records_by_source_hash (Layer 1)
- write_json_atomic atomicity (Layer 2)
- per-file ingestion isolation (Layer 3)
- embedding retry with backoff (Layer 3)
- ingestion resume after interruption (Layer 4)
- disk-space guard (Layer 5)
- corrupt-file loader now propagates (Layer 2)
"""

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from src.atomic_io import write_json_atomic
from src.disk_space import DiskSpaceError, check_disk_space, estimate_dir_bytes
from src.embeddings import EmbeddingEngine
from src.vector_store import LanceDBVectorStore


# --------------------------------------------------------------------------- #
# Layer 1: append_records / replace_records_by_source_hash                    #
# --------------------------------------------------------------------------- #

def _record(record_id, source_hash, content="content", vector=None):
    return {
        "id": record_id,
        "doc_id": "doc",
        "parent_id": "",
        "node_type": "chunk",
        "file_path": f"{source_hash}.md",
        "chunk_index": 0,
        "content": content,
        "title": record_id,
        "section_path": record_id,
        "page_start": 1,
        "page_end": 1,
        "summary": "",
        "tags": [],
        "source_hash": source_hash,
        "source_pdf_name": f"{source_hash}.pdf",
        "source_pdf_path": f"uploads/{source_hash}.pdf",
        "vector": vector if vector is not None else [1.0, 0.0, 0.0],
    }


def test_append_records_empty_creates_compatible_table():
    tmp = Path(tempfile.gettempdir()) / f"rag_overhaul_{uuid.uuid4().hex}"
    try:
        store = LanceDBVectorStore(tmp)
        store.append_records([], embedding_model="m", embedding_dim=3)
        assert store.exists(), "empty append should create a compatible table"
        assert store.count() == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_append_records_adds_rows_without_dropping():
    tmp = Path(tempfile.gettempdir()) / f"rag_overhaul_{uuid.uuid4().hex}"
    try:
        store = LanceDBVectorStore(tmp)
        store.append_records([_record("a1", "hash-a")], embedding_model="m", embedding_dim=3)
        store.append_records([_record("b1", "hash-b", vector=[0.0, 1.0, 0.0])], embedding_model="m", embedding_dim=3)
        assert store.count() == 2
        assert {r["id"] for r in store.all_records()} == {"a1", "b1"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_replace_records_by_source_hash_leaves_other_sources_untouched():
    tmp = Path(tempfile.gettempdir()) / f"rag_overhaul_{uuid.uuid4().hex}"
    try:
        store = LanceDBVectorStore(tmp)
        store.append_records(
            [_record("a1", "hash-a"), _record("b1", "hash-b", vector=[0.0, 1.0, 0.0])],
            embedding_model="m",
            embedding_dim=3,
        )
        # Replace hash-a with two new rows; hash-b must be untouched.
        store.replace_records_by_source_hash(
            source_hash="hash-a",
            records=[
                _record("a1", "hash-a", content="updated", vector=[1.0, 1.0, 0.0]),
                _record("a2", "hash-a", content="extra", vector=[1.0, 0.0, 1.0]),
            ],
            embedding_model="m",
            embedding_dim=3,
        )
        assert store.count() == 3
        a_rows = store.records_by_source_hash(["hash-a"])
        assert {r["id"] for r in a_rows} == {"a1", "a2"}
        assert store.get_record("b1")["content"] == "content"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_replace_records_by_source_hash_on_fresh_table():
    tmp = Path(tempfile.gettempdir()) / f"rag_overhaul_{uuid.uuid4().hex}"
    try:
        store = LanceDBVectorStore(tmp)
        # No table yet: replace should behave like an append (delete is a no-op).
        store.replace_records_by_source_hash(
            source_hash="hash-a",
            records=[_record("a1", "hash-a")],
            embedding_model="m",
            embedding_dim=3,
        )
        assert store.count() == 1
        assert store.get_record("a1")["source_hash"] == "hash-a"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Layer 2: write_json_atomic atomicity                                        #
# --------------------------------------------------------------------------- #

def test_write_json_atomic_produces_valid_file_and_no_tmp_leftbehind():
    tmp = Path(tempfile.gettempdir()) / f"rag_overhaul_{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        target = tmp / "data.json"
        write_json_atomic(target, {"a": 1, "b": [1, 2, 3]})
        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}
        # No temp files left in the directory.
        leftovers = [p.name for p in tmp.iterdir()]
        assert leftovers == ["data.json"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_json_atomic_replaces_corrupt_existing_file():
    tmp = Path(tempfile.gettempdir()) / f"rag_overhaul_{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        target = tmp / "data.json"
        target.write_text("{this is not valid json", encoding="utf-8")
        write_json_atomic(target, {"clean": True})
        assert json.loads(target.read_text(encoding="utf-8")) == {"clean": True}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_json_atomic_concurrent_writes_each_valid():
    """Overlapping atomic writes must each leave a fully-valid JSON file."""
    import threading

    tmp = Path(tempfile.gettempdir()) / f"rag_overhaul_{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    write_json_atomic(tmp / f"f{n}.json", {"writer": n, "i": i})
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        for n in range(5):
            payload = json.loads((tmp / f"f{n}.json").read_text(encoding="utf-8"))
            assert payload["writer"] == n
            assert payload["i"] == 19
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corrupt_registry_propagates_after_atomic_writes():
    """A truncated registry file must surface, not silently become empty."""
    from src.pdf_registry import PdfRegistry

    tmp = Path(tempfile.gettempdir()) / f"rag_overhaul_{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        registry_path = tmp / ".pdf_upload_registry.json"
        # Write a truncated/corrupt file directly (simulating an old torn write).
        registry_path.write_text('{"version": 1, "pdfs": {', encoding="utf-8")
        registry = PdfRegistry(registry_path)
        with pytest.raises(json.JSONDecodeError):
            registry.load()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Layer 3: per-file ingestion isolation                                       #
# --------------------------------------------------------------------------- #

def test_run_ingestion_isolates_per_pdf_failures(monkeypatch, tmp_path):
    """One failing PDF must not abort the batch; only all-fail raises."""
    import src.ingestion as ingestion

    # Two PDFs in the input dir.
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "good.pdf").write_bytes(b"%PDF-1.4 good")
    (input_path_bad := input_dir / "bad.pdf").write_bytes(b"%PDF-1.4 bad")

    # Stub the heavy parts: DocumentProcessor and asset cleanup.
    call_log = []

    class FakeProcessor:
        def __init__(self, **kwargs):
            call_log.append("init")

        def set_source_context(self, **kwargs):
            pass

        def process_pdf(self, path):
            if Path(path).name == "bad.pdf":
                raise RuntimeError("boom on bad.pdf")
            return "# Good markdown\n\ncontent"

    monkeypatch.setattr(ingestion, "DocumentProcessor", FakeProcessor)
    monkeypatch.setattr("src.pdf_registry.sha256_file", lambda p: f"hash-{Path(p).stem}")

    class FakeAssetStore:
        def __init__(self, _dir):
            pass

        def remove_source_assets(self, _hash):
            pass

    monkeypatch.setattr("src.asset_store.ImageAssetStore", FakeAssetStore)

    ingestion.run_ingestion(str(input_dir), str(output_dir), progress_enabled=False)

    # The good PDF was processed; the bad one was recorded as failed, not raised.
    assert (output_dir / "good.md").exists()
    result = json.loads((output_dir / ".ingest_result.json").read_text(encoding="utf-8"))
    assert len(result["processed"]) == 1
    assert result["processed"][0]["file"] == "good.pdf"
    assert len(result["failed"]) == 1
    assert result["failed"][0]["file"] == "bad.pdf"


def test_run_ingestion_raises_when_all_pdfs_fail(monkeypatch, tmp_path):
    import src.ingestion as ingestion

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "only.pdf").write_bytes(b"%PDF-1.4")

    class FakeProcessor:
        def __init__(self, **kwargs):
            pass

        def set_source_context(self, **kwargs):
            pass

        def process_pdf(self, _path):
            raise RuntimeError("always fails")

    monkeypatch.setattr(ingestion, "DocumentProcessor", FakeProcessor)
    monkeypatch.setattr("src.pdf_registry.sha256_file", lambda p: "hash-x")

    class FakeAssetStore:
        def __init__(self, _dir):
            pass

        def remove_source_assets(self, _hash):
            pass

    monkeypatch.setattr("src.asset_store.ImageAssetStore", FakeAssetStore)

    with pytest.raises(RuntimeError, match="failed for all"):
        ingestion.run_ingestion(str(input_dir), str(output_dir), progress_enabled=False)


# --------------------------------------------------------------------------- #
# Layer 3: embedding retry with backoff                                       #
# --------------------------------------------------------------------------- #

def test_embedding_retry_recovers_after_transient_failure(monkeypatch):
    """_ollama_api_with_retry should ride out N-1 failures then succeed."""
    import src.embeddings as embeddings_mod

    # Zero out backoff sleep so the test is fast.
    fake_time = type("FakeTime", (), {"sleep": staticmethod(lambda *_a, **_k: None)})()
    monkeypatch.setattr(embeddings_mod, "time", fake_time)

    engine = EmbeddingEngine(model_name="nomic-embed-text", ollama_retries=3)
    calls = {"n": 0}

    def flaky_api(_path, _payload, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient timeout")
        return {"embeddings": [[1.0, 0.0, 0.0]]}

    monkeypatch.setattr(engine, "_ollama_api", flaky_api)

    response = engine._ollama_api_with_retry("/api/embed", {"model": "x", "input": "y"})
    assert calls["n"] == 3
    assert response["embeddings"] == [[1.0, 0.0, 0.0]]


def test_embedding_retry_exhausts_and_raises(monkeypatch):
    engine = EmbeddingEngine(model_name="nomic-embed-text", ollama_retries=2)
    import src.embeddings as embeddings_mod

    monkeypatch.setattr(
        embeddings_mod, "time", type("FakeTime", (), {"sleep": staticmethod(lambda *_a, **_k: None)})()
    )
    calls = {"n": 0}

    def always_fails(_path, _payload, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("persistent failure")

    monkeypatch.setattr(engine, "_ollama_api", always_fails)

    with pytest.raises(RuntimeError, match="persistent failure"):
        engine._ollama_api_with_retry("/api/embed", {"model": "x", "input": "y"})
    assert calls["n"] == 2  # exactly ollama_retries attempts


# --------------------------------------------------------------------------- #
# Layer 4: ingestion resume                                                   #
# --------------------------------------------------------------------------- #

def test_run_ingestion_skips_already_processed_pdf(monkeypatch, tmp_path):
    """A PDF whose hash already maps to its processed Markdown is skipped."""
    import src.ingestion as ingestion
    from src.pdf_registry import write_source_entry

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4")

    # Pre-seed: doc.pdf already produced doc.md with its hash in the source map.
    md_path = output_dir / "doc.md"
    md_path.write_text("# existing", encoding="utf-8")
    write_source_entry(
        processed_dir=output_dir,
        markdown_path=md_path,
        source_hash="hash-doc",
        source_pdf_name="doc.pdf",
        source_pdf_path=input_dir / "doc.pdf",
    )

    processed_calls = []

    class FakeProcessor:
        def __init__(self, **kwargs):
            pass

        def set_source_context(self, **kwargs):
            pass

        def process_pdf(self, path):
            processed_calls.append(Path(path).name)
            return "# should not be called"

    monkeypatch.setattr(ingestion, "DocumentProcessor", FakeProcessor)
    monkeypatch.setattr("src.pdf_registry.sha256_file", lambda p: "hash-doc")

    class FakeAssetStore:
        def __init__(self, _dir):
            pass

        def remove_source_assets(self, _hash):
            pass

    monkeypatch.setattr("src.asset_store.ImageAssetStore", FakeAssetStore)

    ingestion.run_ingestion(str(input_dir), str(output_dir), progress_enabled=False)

    # The expensive parse was skipped.
    assert processed_calls == []
    result = json.loads((output_dir / ".ingest_result.json").read_text(encoding="utf-8"))
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["file"] == "doc.pdf"
    # Existing markdown untouched.
    assert md_path.read_text(encoding="utf-8") == "# existing"


def test_run_ingestion_reprocesses_pdf_when_fingerprint_changes(monkeypatch, tmp_path):
    """A stale Markdown entry must not hide a changed source PDF."""
    import src.ingestion as ingestion
    from src.pdf_registry import write_source_entry

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    input_pdf = input_dir / "doc.pdf"
    input_pdf.write_bytes(b"%PDF-1.4 changed")
    md_path = output_dir / "doc.md"
    md_path.write_text("# stale", encoding="utf-8")
    write_source_entry(
        processed_dir=output_dir,
        markdown_path=md_path,
        source_hash="old-hash",
        source_pdf_name="doc.pdf",
        source_pdf_path=input_pdf,
        source_size=1,
        source_mtime_ns=1,
    )

    processed_calls = []

    class FakeProcessor:
        def __init__(self, **kwargs):
            pass

        def set_source_context(self, **kwargs):
            pass

        def process_pdf(self, path):
            processed_calls.append(Path(path).name)
            return "# refreshed"

    monkeypatch.setattr(ingestion, "DocumentProcessor", FakeProcessor)

    class FakeAssetStore:
        def __init__(self, _dir):
            pass

        def remove_source_assets(self, _hash):
            pass

    monkeypatch.setattr("src.asset_store.ImageAssetStore", FakeAssetStore)

    ingestion.run_ingestion(str(input_dir), str(output_dir), progress_enabled=False)

    assert processed_calls == ["doc.pdf"]
    assert md_path.read_text(encoding="utf-8") == "# refreshed"


def test_pdf_discovery_walks_nested_corpus_directories(tmp_path):
    from src.ingestion import _iter_pdf_paths

    nested = tmp_path / "year" / "project"
    nested.mkdir(parents=True)
    first = nested / "first.pdf"
    second = tmp_path / "second.PDF"
    first.write_bytes(b"pdf")
    second.write_bytes(b"pdf")

    assert _iter_pdf_paths(str(tmp_path)) == sorted([first, second])


def test_nested_pdfs_with_duplicate_stems_get_distinct_markdown_names(tmp_path):
    from src.ingestion import _markdown_name_for_pdf

    root_pdf = tmp_path / "manual.pdf"
    nested_pdf = tmp_path / "2026" / "manual.pdf"
    duplicate_stems = {"manual"}

    assert _markdown_name_for_pdf(
        root_pdf,
        input_root=tmp_path,
        duplicate_stems=duplicate_stems,
    ) == "manual.md"
    nested_name = _markdown_name_for_pdf(
        nested_pdf,
        input_root=tmp_path,
        duplicate_stems=duplicate_stems,
    )
    assert nested_name.startswith("manual__")
    assert nested_name.endswith(".md")
    assert nested_name != "manual.md"


# --------------------------------------------------------------------------- #
# Layer 5: disk-space guard                                                   #
# --------------------------------------------------------------------------- #

def test_check_disk_space_passes_when_enough_free(tmp_path):
    # Require a tiny amount; the temp dir's filesystem always has this free.
    check_disk_space(tmp_path, 1)  # should not raise


def test_check_disk_space_raises_when_insufficient(tmp_path, monkeypatch):
    import src.disk_space as disk_space_mod

    class FakeUsage:
        free = 100
        total = 1000
        used = 900

    monkeypatch.setattr(disk_space_mod.shutil, "disk_usage", lambda _p: FakeUsage())
    with pytest.raises(DiskSpaceError) as exc_info:
        check_disk_space(tmp_path, 10_000)
    assert exc_info.value.required_bytes == 10_000
    assert exc_info.value.free_bytes == 100


def test_estimate_dir_bytes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 50)
    assert estimate_dir_bytes(tmp_path) == 150
    assert estimate_dir_bytes(tmp_path / "missing") == 0
