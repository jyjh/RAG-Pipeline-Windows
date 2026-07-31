"""Tests for the embedding-throughput estimator.

These cover the pure projection math (no network, no models) and the chunk-only
CLI path (uses the real chunker, still no network). The actual rate measurement
(``--native``/``--ollama``) requires a model download / a running Ollama server
and is intentionally NOT exercised here -- it's an interactive cluster tool.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT_DIR / "scripts" / "estimate_embed_throughput.py"

# Load the script as a module (it lives under scripts/, not src/, so a plain
# import wouldn't work). The functions under test are pure and side-effect-free.
_spec = importlib.util.spec_from_file_location("estimate_embed_throughput", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
est = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(est)


# --- project_wallclock -------------------------------------------------------


def test_project_wallclock_linear_scaling():
    """Halving throughput or doubling nodes should each halve the wall-clock;
    the two are interchangeable (embarrassingly-parallel chunks)."""
    assert est.project_wallclock(10000, 100.0, 1) == pytest.approx(100.0)
    assert est.project_wallclock(10000, 100.0, 2) == pytest.approx(50.0)
    assert est.project_wallclock(10000, 50.0, 1) == pytest.approx(200.0)
    # 4x the work at 4x the parallelism is the same wall-clock.
    assert est.project_wallclock(40000, 100.0, 4) == pytest.approx(100.0)


def test_project_wallclock_edge_cases():
    # Zero corpus -> 0, not inf (so callers don't special-case).
    assert est.project_wallclock(0, 100.0, 8) == 0.0
    # Zero/negative throughput or nodes -> infeasible (inf), never a crash.
    assert est.project_wallclock(1000, 0.0, 8) == float("inf")
    assert est.project_wallclock(1000, -1.0, 8) == float("inf")
    assert est.project_wallclock(1000, 100.0, 0) == float("inf")


# --- format_duration ---------------------------------------------------------


def test_format_duration_units():
    assert est.format_duration(45) == "45s"
    assert est.format_duration(120) == "2m0s"
    assert est.format_duration(3600) == "1h0m"
    assert est.format_duration(5400) == "1h30m"
    assert est.format_duration(90000) == "1d1h"  # 25h
    assert est.format_duration(0) == "0s"
    assert est.format_duration(float("inf")).startswith("n/a")


# --- parse_node_list ---------------------------------------------------------


def test_parse_node_list_variants():
    assert est.parse_node_list("8") == [8]
    assert est.parse_node_list("1,4,8,16") == [1, 4, 8, 16]
    # Dedup + sort, whitespace-tolerant.
    assert est.parse_node_list(" 4 , 1 ,4, 16") == [1, 4, 16]
    with pytest.raises(ValueError):
        est.parse_node_list("0")
    with pytest.raises(ValueError):
        est.parse_node_list("")
    with pytest.raises(ValueError):
        est.parse_node_list("-2")


# --- CLI: --count path (real chunker, no network/models) ---------------------


def test_count_mode_chunks_real_sample_without_embedding(tmp_path, capsys):
    """--count must chunk a .md sample via the production chunker and report a
    count, without touching any model or the network."""
    # A markdown doc long enough to produce multiple chunks.
    body = "# Report\n\n## Intro\n\n" + ("Torque curve brake power thermal. " * 400)
    (tmp_path / "doc.md").write_text(body, encoding="utf-8")

    rc = est.main(["--count", "--sample-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Chunked into" in out
    assert "chunks" in out
    assert "--count only: skipping embedding" in out


def test_count_mode_reports_zero_chunks_cleanly(tmp_path, capsys):
    # A directory with NO .md files at all -> FileNotFoundError -> error, not crash.
    (tmp_path / "notmarkdown.txt").write_text("some text", encoding="utf-8")
    rc = est.main(["--count", "--sample-dir", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no .md files found" in err


def test_cli_rejects_count_with_backend(tmp_path, capsys):
    rc = est.main(["--count", "--native", "--sample-dir", str(tmp_path)])
    assert rc == 2
    assert "drop --native" in capsys.readouterr().err


def test_cli_requires_a_source(capsys):
    rc = est.main(["--native"])
    assert rc == 2
    assert "provide --sample-dir or --raw-text" in capsys.readouterr().err


def test_cli_missing_sample_dir_errors(capsys):
    rc = est.main(["--native", "--sample-dir", "/does/not/exist/xyz"])
    assert rc == 2
    assert "is not a directory" in capsys.readouterr().err


# --- CLI: --help (argparse wiring smoke test) --------------------------------


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        est.main(["--help"])
    assert exc.value.code == 0
    assert "estimate" in capsys.readouterr().out.lower()


# --- synthetic chunk generator -----------------------------------------------


def test_synthetic_chunks_realistic_length():
    chunks = est.synthetic_chunks(5, chunk_target_tokens=900)
    assert len(chunks) == 5
    # ~4 chars/token * 900 tokens target; allow slack in the word boundary.
    avg_len = sum(len(c) for c in chunks) / 5
    assert 3000 <= avg_len <= 3800
    # Deterministic across runs (seeded RNG).
    again = est.synthetic_chunks(5, chunk_target_tokens=900)
    assert chunks == again
