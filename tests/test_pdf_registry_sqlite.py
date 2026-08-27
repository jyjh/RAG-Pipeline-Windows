"""SQLite-backed PDF registry / source map migration and behaviour tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

from src.pdf_registry import (
    PdfRegistry,
    load_source_map,
    registry_state_version,
    remove_source_entries_by_hash,
    source_entry_for,
    source_entry_for_markdown,
    source_map_path,
    source_map_state_version,
    write_source_entry,
)


def _tmp_root() -> Path:
    root = Path(tempfile.gettempdir()) / f"rag_sqlite_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


def test_legacy_json_registry_is_imported_once():
    tmp = _tmp_root()
    try:
        legacy = tmp / ".pdf_upload_registry.json"
        legacy.write_text(
            json.dumps(
                {
                    "version": 1,
                    "pdfs": {
                        "hash-a": {
                            "hash": "hash-a",
                            "filename": "a.pdf",
                            "status": "indexed",
                        },
                        "hash-b": {
                            "hash": "hash-b",
                            "filename": "b.pdf",
                            "status": "superseded",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        registry = PdfRegistry(legacy)
        payload = registry.load()
        assert set(payload["pdfs"]) == {"hash-a", "hash-b"}
        assert payload["pdfs"]["hash-a"]["filename"] == "a.pdf"
        # The original document is untouched (rollback-friendly) and a second
        # open must not duplicate rows.
        assert legacy.exists()
        assert len(registry.load()["pdfs"]) == 2

        # Mutations land in SQLite from then on.
        registry.register_queued(
            job_id="job-1",
            files=[{"hash": "hash-c", "filename": "c.pdf"}],
        )
        refreshed = registry.load()
        assert set(refreshed["pdfs"]) == {"hash-a", "hash-b", "hash-c"}
        assert (_db := Path(str(legacy) + ".sqlite3")).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corrupt_legacy_json_raises_during_import():
    tmp = _tmp_root()
    try:
        legacy = tmp / ".pdf_upload_registry.json"
        legacy.write_text('{"version": 1, "pdfs": {', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            PdfRegistry(legacy).load()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fresh_store_creates_sqlite_beside_legacy_path():
    tmp = _tmp_root()
    try:
        legacy = tmp / ".pdf_upload_registry.json"
        registry = PdfRegistry(legacy)
        assert registry.load() == {"version": 1, "pdfs": {}}
        # No legacy file is fabricated for a fresh install; the DB appears.
        assert not legacy.exists()
        assert Path(str(legacy) + ".sqlite3").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mark_job_status_transition_and_supersede():
    tmp = _tmp_root()
    try:
        legacy = tmp / ".pdf_upload_registry.json"
        registry = PdfRegistry(legacy)
        files_a = [
            {
                "hash": "hash-a",
                "filename": "same-content.pdf",
                "staging_path": str(tmp / "a.pdf"),
                "upload_path": str(tmp / "a.pdf"),
                "processed_markdown_path": str(tmp / "out.md"),
            }
        ]
        registry.register_queued(job_id="job-a", files=files_a)
        registry.mark_job_status(job_id="job-a", files=files_a, status="indexed")

        # hash-b re-ingests the same processed markdown -> hash-a is superseded.
        files_b = [
            {
                "hash": "hash-b",
                "filename": "same-content-reupload.pdf",
                "upload_path": str(tmp / "b.pdf"),
                "processed_markdown_path": str(tmp / "out.md"),
            }
        ]
        registry.register_queued(
            job_id="job-b",
            files=files_b,
            forced_hashes={"hash-b"},
        )
        registry.mark_job_status(job_id="job-b", files=files_b, status="ingested")
        payload = registry.load()
        assert payload["pdfs"]["hash-b"]["status"] == "ingested"
        assert payload["pdfs"]["hash-a"]["status"] == "superseded"

        # Interruption restores the previous entry snapshot. The snapshot is
        # captured on the forced-duplicate re-registration and consumed by an
        # interrupt BEFORE any successful transition (successes clear it --
        # legacy behaviour).
        files_b_again = [dict(files_b[0], staging_path=str(tmp / "b1.pdf"))]
        registry.register_queued(
            job_id="job-b2",
            files=files_b_again,
            forced_hashes={"hash-b"},
        )
        registry.mark_job_status(job_id="job-b2", files=files_b_again, status="interrupted")
        entry = registry.load()["pdfs"]["hash-b"]
        assert entry["status"] == "ingested"  # restored from the forced-upload snapshot
        assert entry["last_interrupted_job_id"] == "job-b2"

        registry.delete_source("hash-b")
        assert set(registry.load()["pdfs"]) == {"hash-a"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_source_map_roundtrip_point_lookup_and_removal():
    tmp = _tmp_root()
    try:
        markdown = tmp / "doc.md"
        markdown.write_text("# Doc", encoding="utf-8")
        write_source_entry(
            processed_dir=tmp,
            markdown_path=markdown,
            source_hash="hash-1",
            source_pdf_name="doc.pdf",
            source_pdf_path=tmp / "doc.pdf",
            source_size=123,
        )
        write_source_entry(
            processed_dir=tmp,
            markdown_path=tmp / "other.md",
            source_hash="hash-2",
            source_pdf_name="other.pdf",
            source_pdf_path=tmp / "other.pdf",
        )

        loaded = load_source_map(tmp)
        assert set(loaded["documents"]) == {"doc.md", "other.md"}

        point = source_entry_for(tmp, "doc.md")
        assert point["source_hash"] == "hash-1"
        assert point["source_size"] == 123
        assert source_entry_for(tmp, "missing.md") == {}
        assert source_entry_for_markdown(markdown)["source_hash"] == "hash-1"

        removed = remove_source_entries_by_hash(tmp, {"hash-1"})
        assert [entry["source_hash"] for entry in removed] == ["hash-1"]
        remaining = load_source_map(tmp)["documents"]
        assert set(remaining) == {"other.md"}

        # Legacy source-map file is imported when present.
        legacy_map = tmp / "legacy_out"
        legacy_map.mkdir()
        (legacy_map / ".source_map.json").write_text(
            json.dumps({"version": 1, "documents": {"old.md": {"source_hash": "h0"}}}),
            encoding="utf-8",
        )
        assert load_source_map(legacy_map)["documents"]["old.md"]["source_hash"] == "h0"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_state_versions_track_writes():
    tmp = _tmp_root()
    try:
        legacy = tmp / ".pdf_upload_registry.json"
        before = registry_state_version(legacy)
        PdfRegistry(legacy).register_queued(
            job_id="j", files=[{"hash": "h", "filename": "f.pdf"}]
        )
        after = registry_state_version(legacy)
        assert int(after) > int(before)

        # source_map_state_version takes the processed dir (not the map path).
        map_before = source_map_state_version(tmp)
        write_source_entry(
            processed_dir=tmp,
            markdown_path=tmp / "m.md",
            source_hash="h",
            source_pdf_name="f.pdf",
            source_pdf_path=tmp / "f.pdf",
        )
        assert int(source_map_state_version(tmp)) > int(map_before)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_concurrent_processes_do_not_lose_updates():
    """Two external processes writing disjoint keys must both persist.

    This exercises the real deployment pattern: ingestion workers are separate
    processes appending to the same source map (previously serialized through a
    portalocker convoy; now via SQLite WAL + busy_timeout).
    """
    tmp = _tmp_root()
    worker_script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from src.pdf_registry import write_source_entry\n"
        "tag, count, target = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])\n"
        "for i in range(count):\n"
        "    name = f'{tag}_{i}.md'\n"
        "    write_source_entry(\n"
        "        processed_dir=target,\n"
        f"        markdown_path=target / name,\n"
        "        source_hash=f'{tag}-{i}',\n"
        "        source_pdf_name=name,\n"
        "        source_pdf_path=target / name,\n"
        "    )\n"
        "    time.sleep(0.001)\n"
    )
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", worker_script, tag, "15", str(tmp)],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        for tag in ("w1", "w2")
    ]
    codes = [worker.wait() for worker in workers]
    assert codes == [0, 0]
    documents = load_source_map(tmp)["documents"]
    expected = {f"w{w}_{i}.md" for w in (1, 2) for i in range(15)}
    assert set(documents) == expected
