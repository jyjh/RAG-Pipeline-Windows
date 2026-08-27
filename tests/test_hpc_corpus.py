"""Tests for the one-command initial corpus flow (zip -> HPC parse -> index).

All cluster interactions go through a fake HpcBackend and the index build is
an intercepted subprocess call, so nothing here touches the network.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hpc_corpus import (
    CorpusError,
    check_local_ollama,
    check_soclaas_vision_key,
    extract_corpus_zip,
    run_initial_corpus,
)
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _write_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return path


# --- Zip extraction ----------------------------------------------------------


def test_extract_preserves_nested_structure(safe_tmp_path):
    """Nested dirs are kept; discovery + collision-safe naming happen later.

    This is the core fix for the ``unzip -o -j`` data loss: flattening let
    A/report.pdf silently overwrite B/report.pdf.
    """
    zip_path = _write_zip(safe_tmp_path / "corpus.zip", {
        "manuals/v1/report.pdf": b"%PDF-v1",
        "manuals/v2/report.pdf": b"%PDF-v2",
        "datasheets/chip A.pdf": b"%PDF-chip",
        "readme.txt": b"not a pdf",
        "__MACOSX/._report.pdf": b"junk",
        ".DS_Store": b"junk",
    })
    summary = extract_corpus_zip(zip_path, safe_tmp_path / "out", log=lambda *_: None)
    names = sorted(str(p) for p in summary["pdfs"])
    assert names == [
        "datasheets/chip A.pdf",
        "manuals/v1/report.pdf",
        "manuals/v2/report.pdf",
    ]
    assert (safe_tmp_path / "out" / "manuals" / "v1" / "report.pdf").read_bytes() == b"%PDF-v1"
    assert (safe_tmp_path / "out" / "manuals" / "v2" / "report.pdf").read_bytes() == b"%PDF-v2"
    assert len(summary["skipped_non_pdf"]) == 1
    assert len(summary["skipped_unsafe"]) == 2


def test_extract_rejects_traversal_entries(safe_tmp_path):
    zip_path = _write_zip(safe_tmp_path / "evil.zip", {
        "../escape.pdf": b"%PDF-evil",
        "/absolute.pdf": b"%PDF-abs",
        "ok/fine.pdf": b"%PDF-ok",
    })
    summary = extract_corpus_zip(zip_path, safe_tmp_path / "out", log=lambda *_: None)
    assert [str(p) for p in summary["pdfs"]] == ["ok/fine.pdf"]
    assert not (safe_tmp_path / "escape.pdf").exists()


def test_extract_caps_entry_count(safe_tmp_path):
    entries = {f"dir/file{i}.pdf": b"x" for i in range(5)}
    zip_path = _write_zip(safe_tmp_path / "bomb.zip", entries)
    with pytest.raises(CorpusError, match="exceeding the cap"):
        extract_corpus_zip(
            zip_path, safe_tmp_path / "out", max_entries=3, log=lambda *_: None
        )


def test_extract_dedupes_case_collisions(safe_tmp_path):
    """Case-insensitive filesystems would otherwise overwrite A.pdf with a.pdf."""
    zip_path = _write_zip(safe_tmp_path / "case.zip", {
        "A/report.pdf": b"%PDF-upper",
        "a/report.pdf": b"%PDF-lower",
    })
    summary = extract_corpus_zip(zip_path, safe_tmp_path / "out", log=lambda *_: None)
    assert len(summary["pdfs"]) == 2
    contents = sorted(
        p.read_bytes() for p in (safe_tmp_path / "out").rglob("*.pdf")
    )
    assert contents == [b"%PDF-lower", b"%PDF-upper"]


def test_extract_bad_zip_raises(safe_tmp_path):
    bad = safe_tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip at all")
    with pytest.raises(CorpusError, match="valid zip"):
        extract_corpus_zip(bad, safe_tmp_path / "out", log=lambda *_: None)


# --- Preflights ---------------------------------------------------------------


def _config_with(tmp_path: Path, *, vision: bool, api_key: str = ""):
    (tmp_path / "config.toml").write_text(
        "[models]\nembedding_model = \"nomic-embed-text\"\n\n"
        f"[ingestion]\nvision_enabled = {str(vision).lower()}\n\n"
        f"[llm_api]\napi_key = \"{api_key}\"\n",
        encoding="utf-8",
    )
    return load_config(tmp_path / "config.toml")


class _Backend:
    def __init__(self, *, remote_key: bool = False):
        self.remote_key = remote_key

    def remote_file_nonempty(self, path):
        return self.remote_key


def test_vision_key_check_accepts_env_var(monkeypatch, safe_tmp_path):
    config = _config_with(safe_tmp_path, vision=True)
    monkeypatch.setenv("SOCLAAS_API_KEY", "sk-test")
    check_soclaas_vision_key(config, _Backend())


def test_vision_key_check_accepts_config_or_remote_key(safe_tmp_path):
    config = _config_with(safe_tmp_path, vision=True, api_key="sk-in-config")
    check_soclaas_vision_key(config, _Backend())

    config_no_key = _config_with(safe_tmp_path, vision=True)
    check_soclaas_vision_key(config_no_key, _Backend(remote_key=True))


def test_vision_key_check_fails_closed_without_any_key(monkeypatch, safe_tmp_path):
    monkeypatch.delenv("SOCLAAS_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = _config_with(safe_tmp_path, vision=True)
    with pytest.raises(CorpusError, match="no SoCLAaS API key is reachable"):
        check_soclaas_vision_key(config, _Backend())


def test_vision_key_check_skipped_when_vision_disabled(safe_tmp_path):
    config = _config_with(safe_tmp_path, vision=False)
    check_soclaas_vision_key(config, _Backend())


def test_ollama_check_requires_reachable_server(monkeypatch, safe_tmp_path):
    import scripts.hpc_corpus as hpc_corpus

    config = _config_with(safe_tmp_path, vision=False)
    monkeypatch.setattr(
        hpc_corpus, "_http_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("refused")),
    )
    with pytest.raises(CorpusError, match="Ollama is not reachable"):
        check_local_ollama(config)


def test_ollama_check_requires_embedding_model(monkeypatch, safe_tmp_path):
    import scripts.hpc_corpus as hpc_corpus

    config = _config_with(safe_tmp_path, vision=False)

    def fake_http_json(url, timeout=5.0):
        if url.endswith("/api/version"):
            return {"version": "0.1"}
        return {"models": [{"name": "llama3:latest"}]}

    monkeypatch.setattr(hpc_corpus, "_http_json", fake_http_json)
    with pytest.raises(CorpusError, match="ollama pull nomic-embed-text"):
        check_local_ollama(config)


# --- End-to-end flow (fake backend + intercepted index build) -----------------


HPC_CONFIG = """\
[paths]
data_dir = "data"
processed_dir = "processed_docs"
db_dir = "db"

[models]
embedding_model = "nomic-embed-text"

[ingestion]
vision_enabled = false

[hpc]
enabled = true
remote_data_dir = "/hpctmp/student/rag-corpus/data"
remote_processed_dir = "/hpctmp/student/rag-corpus/processed_docs"

[hpc.cpu]
ssh_host = "cpu-login"
remote_repo_dir = "rag-cpu"
container_sif = "rag_pipeline_cpu.sif"
storage_root = "/hpctmp/student"
"""


class FakeBackend:
    instances: list["FakeBackend"] = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls: list[tuple] = []
        FakeBackend.instances.append(self)

    def remote_file_nonempty(self, path):
        return False

    def push_corpus_dir(self, local_dir, remote_dir=None):
        self.calls.append(("push", str(local_dir), remote_dir))
        return remote_dir or self.cfg.remote_data_dir

    def submit_ingest_index(self, input_dir_on_hpc=None, *, skip_index=False,
                            progress_callback=None, log_callback=None,
                            cancel_event=None):
        self.calls.append(("submit", input_dir_on_hpc, skip_index))

    def fetch_processed_docs(self, remote_processed_dir=None, local_dir="processed_docs"):
        self.calls.append(("fetch", str(local_dir)))
        return Path(local_dir)


@pytest.fixture
def flow_env(monkeypatch, safe_tmp_path):
    """Isolated ROOT (config/data/processed) + faked cluster + index build."""
    import scripts.hpc_corpus as hpc_corpus

    fake_root = safe_tmp_path
    (fake_root / "config.toml").write_text(HPC_CONFIG, encoding="utf-8")
    monkeypatch.setattr(hpc_corpus, "ROOT", fake_root)
    FakeBackend.instances = []
    monkeypatch.setattr(hpc_corpus, "HpcBackend", FakeBackend)
    monkeypatch.setattr(hpc_corpus, "check_local_ollama", lambda config: None)
    monkeypatch.setattr(hpc_corpus, "_web_server_running", lambda config: False)

    index_commands: list[list[str]] = []
    monkeypatch.setattr(
        hpc_corpus.subprocess, "run",
        lambda cmd, **k: index_commands.append(cmd) or SimpleNamespace(returncode=0),
    )
    return fake_root, index_commands


def test_run_initial_corpus_happy_path(flow_env, safe_tmp_path, monkeypatch):
    fake_root, index_commands = flow_env
    zip_path = _write_zip(safe_tmp_path / "corpus.zip", {
        "a/one.pdf": b"%PDF-1",
        "b/two.pdf": b"%PDF-2",
    })
    # Pretend the cluster produced Markdown + a result summary (fetch is faked).
    processed = fake_root / "processed_docs"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "one.md").write_text("# one", encoding="utf-8")
    (processed / ".ingest_result.json").write_text(json.dumps({
        "processed": [{"file": "a/one.pdf"}, {"file": "b/two.pdf"}],
        "skipped": [],
        "failed": [],
    }), encoding="utf-8")

    monkeypatch.setattr("scripts.hpc_corpus.ROOT", fake_root)
    rc = run_initial_corpus(zip_path, config_path=fake_root / "config.toml", log=lambda *_: None)
    assert rc == 0

    backend = FakeBackend.instances[-1]
    push, submit, fetch = backend.calls
    assert push[0] == "push"
    assert push[1].endswith("corpus")  # data/corpus/<zip-stem>
    assert submit == ("submit", "/hpctmp/student/rag-corpus/data", True)
    assert fetch[0] == "fetch"

    # The zip was extracted locally with structure preserved.
    assert (fake_root / "data" / "corpus" / "corpus" / "a" / "one.pdf").exists()

    # The local index build ran via main.py --mode index.
    assert len(index_commands) == 1
    command = index_commands[0]
    assert "--mode" in command and command[command.index("--mode") + 1] == "index"


def test_run_initial_corpus_requires_hpc_mode(flow_env, safe_tmp_path, monkeypatch):
    import scripts.hpc_corpus as hpc_corpus

    fake_root, _ = flow_env
    (fake_root / "config.toml").write_text(
        HPC_CONFIG.replace("enabled = true", "enabled = false"), encoding="utf-8"
    )
    zip_path = _write_zip(safe_tmp_path / "corpus.zip", {"a.pdf": b"%PDF"})
    with pytest.raises(CorpusError, match="hpc"):
        run_initial_corpus(zip_path, config_path=fake_root / "config.toml", log=lambda *_: None)


def test_run_initial_corpus_refuses_while_server_runs(flow_env, safe_tmp_path, monkeypatch):
    import scripts.hpc_corpus as hpc_corpus

    fake_root, index_commands = flow_env
    monkeypatch.setattr(hpc_corpus, "_web_server_running", lambda config: True)
    zip_path = _write_zip(safe_tmp_path / "corpus.zip", {"a.pdf": b"%PDF"})
    with pytest.raises(CorpusError, match="web server is running"):
        run_initial_corpus(zip_path, config_path=fake_root / "config.toml", log=lambda *_: None)
    assert index_commands == []
    assert all(not backend.calls for backend in FakeBackend.instances)


def test_run_initial_corpus_skips_index_when_requested(flow_env, safe_tmp_path):
    fake_root, index_commands = flow_env
    zip_path = _write_zip(safe_tmp_path / "corpus.zip", {"a.pdf": b"%PDF"})
    processed = fake_root / "processed_docs"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "a.md").write_text("# a", encoding="utf-8")
    (processed / ".ingest_result.json").write_text(json.dumps({
        "processed": [{"file": "a.pdf"}], "skipped": [], "failed": [],
    }), encoding="utf-8")
    rc = run_initial_corpus(
        zip_path,
        config_path=fake_root / "config.toml",
        skip_index_build=True,
        log=lambda *_: None,
    )
    assert rc == 0
    assert index_commands == []


def test_run_initial_corpus_fails_when_nothing_parsed(flow_env, safe_tmp_path):
    fake_root, index_commands = flow_env
    zip_path = _write_zip(safe_tmp_path / "corpus.zip", {"a.pdf": b"%PDF"})
    (fake_root / "processed_docs").mkdir(parents=True, exist_ok=True)
    (fake_root / "processed_docs" / ".ingest_result.json").write_text(json.dumps({
        "processed": [], "skipped": [], "failed": [{"file": "a.pdf", "error": "boom"}],
    }), encoding="utf-8")
    with pytest.raises(CorpusError, match="no output"):
        run_initial_corpus(zip_path, config_path=fake_root / "config.toml", log=lambda *_: None)
    assert index_commands == []
