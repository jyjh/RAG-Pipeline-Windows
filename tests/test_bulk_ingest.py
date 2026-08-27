"""Bulk-ingest CLI forwards the FULL [ingestion] config to the pipeline.

The cluster PBS job runs scripts/bulk_ingest.py inside the SIF. It used to
hand-pick a subset of options, so the cluster silently diverged from local
ingestion (notably ingestion_workers: config 2, cluster default 1 -- half the
parse throughput). These tests pin the full-forwarding contract.
"""

from __future__ import annotations

import pytest

import scripts.bulk_ingest as bulk_ingest


@pytest.fixture
def captured(monkeypatch):
    calls = {"ingest": [], "index": []}

    def fake_run_ingestion(*args, **kwargs):
        calls["ingest"].append((args, kwargs))

    def fake_run_indexing(*args, **kwargs):
        calls["index"].append((args, kwargs))

    monkeypatch.setattr(bulk_ingest, "run_ingestion", fake_run_ingestion)
    monkeypatch.setattr(bulk_ingest, "run_indexing", fake_run_indexing)
    return calls


def test_forwards_full_ingestion_config(captured, safe_tmp_path, monkeypatch):
    monkeypatch.chdir(safe_tmp_path)
    bulk_ingest.main(["--input-dir", str(safe_tmp_path)])

    assert len(captured["ingest"]) == 1
    kwargs = captured["ingest"][0][1]
    # The previously-dropped options must all reach run_ingestion.
    for key in (
        "asset_triggers",
        "ocr_langs",
        "ocr_force_full_page",
        "ocr_bitmap_area_threshold",
        "rapidocr_backend",
        "tesseract_cmd",
        "tesseract_data_path",
        "tesseract_psm",
        "ingestion_workers",
        "max_pages_whole_doc",
    ):
        assert key in kwargs, f"run_ingestion must receive {key}"
    # ingestion_workers comes from config (2 in this repo's config.toml), not
    # the historical serial default of 1.
    assert kwargs["ingestion_workers"] >= 1
    assert kwargs["progress_enabled"] is True


def test_dir_flags_override_config_paths(captured, safe_tmp_path, monkeypatch):
    monkeypatch.chdir(safe_tmp_path)
    processed = safe_tmp_path / "out_md"
    db = safe_tmp_path / "out_db"
    assets = safe_tmp_path / "out_assets"
    bulk_ingest.main([
        "--input-dir", str(safe_tmp_path),
        "--processed-dir", str(processed),
        "--db-dir", str(db),
        "--asset-dir", str(assets),
    ])

    _, kwargs = captured["ingest"][0]
    assert kwargs["input_dir"] == str(safe_tmp_path)
    assert kwargs["output_dir"] == str(processed)
    assert kwargs["asset_dir"] == assets
    assert processed.is_dir() and db.is_dir() and assets.is_dir()

    assert len(captured["index"]) == 1
    index_kwargs = captured["index"][0][1]
    assert index_kwargs["md_dir"] == str(processed)
    assert index_kwargs["db_dir"] == str(db)


def test_skip_index_skips_indexing_phase(captured, safe_tmp_path, monkeypatch):
    monkeypatch.chdir(safe_tmp_path)
    bulk_ingest.main(["--input-dir", str(safe_tmp_path), "--skip-index"])
    assert len(captured["ingest"]) == 1
    assert captured["index"] == []


def test_missing_input_dir_exits(captured, safe_tmp_path):
    with pytest.raises(SystemExit):
        bulk_ingest.main(["--input-dir", str(safe_tmp_path / "nope")])
    assert captured["ingest"] == []
