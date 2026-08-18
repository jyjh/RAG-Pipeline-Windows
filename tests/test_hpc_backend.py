"""Unit tests for the HPC build-delegation backend.

All cluster interactions are mocked -- these exercise the orchestration logic
(qsub/qstat parsing, progress relay, cancellation, rsync args) with no network.
The pure parsing helpers are tested directly; the SSH/rsync seams are tested by
monkeypatching ``HpcBackend._run_ssh`` / ``_run_rsync``.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import HpcClusterConfig, HpcConfig
from src.hpc_backend import (
    HpcBackend,
    HpcError,
    PbsJobResult,
    _parse_qsub_output,
    parse_job_exit_code,
    parse_qstat_job_state,
)
from src.progress_protocol import PROGRESS_PREFIX

BASE_CFG = HpcConfig(
    enabled=True,
    cpu=HpcClusterConfig(
        ssh_host="nus_hpc_cpu",            # free CPU cluster login node
        remote_repo_dir="rag-cpu",
        container_sif="rag_pipeline_cpu.sif",
        storage_root="/hpctmp/me",
        pbs_overrides={"ngpus": 0, "queue": "cpu"},
    ),
    remote_data_dir="data",
    remote_db_dir="db",
    poll_interval_seconds=0.01,  # tight loop in tests
)


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# --- pure parsing helpers ----------------------------------------------------


def test_parse_qsub_output_extracts_jobid():
    assert _parse_qsub_output("12345[].aspsus01\n") == "12345[].aspsus01"
    assert _parse_qsub_output("\n\n  67890\n") == "67890"
    assert _parse_qsub_output("99999.aspsus01") == "99999.aspsus01"


def test_parse_qsub_output_raises_on_empty():
    with pytest.raises(HpcError):
        _parse_qsub_output("")
    with pytest.raises(HpcError):
        _parse_qsub_output("\n\n  \n")


def test_parse_qstat_job_state():
    qstat = """
Job Id: 12345.aspsus01
    Job_Name = rag_ingest_index
    job_state = R
    Exit_status = 0
"""
    assert parse_qstat_job_state(qstat, "12345") == "R"
    assert parse_job_exit_code(qstat) == 0
    # empty / unknown job
    assert parse_qstat_job_state("", "x") is None
    assert parse_job_exit_code("") is None


# --- construction guards (validation is lazy, per-method) --------------------


def test_backend_constructs_without_validating():
    """Construction never fails -- a deployment may use only one cluster, so
    validation happens per-method via _require_cluster."""
    HpcBackend(HpcConfig())  # both clusters unconfigured: OK until a method is called


def test_submit_ingest_index_requires_cpu_cluster():
    b = HpcBackend(HpcConfig())  # no cpu config
    with pytest.raises(HpcError, match="hpc.cpu.ssh_host"):
        b.submit_ingest_index()


def test_cluster_repo_path_must_be_login_relative():
    absolute = replace(
        BASE_CFG,
        cpu=replace(BASE_CFG.cpu, remote_repo_dir="/home/me/rag"),
    )
    with pytest.raises(HpcError, match="relative to the SSH login"):
        HpcBackend(absolute).submit_ingest_index()

    traversal = replace(
        BASE_CFG,
        cpu=replace(BASE_CFG.cpu, remote_repo_dir="../other/rag"),
    )
    with pytest.raises(HpcError, match="relative to the SSH login"):
        HpcBackend(traversal).submit_ingest_index()


def test_connection_check_accepts_post_quantum_ssh_warnings(monkeypatch):
    warning = (
        '** WARNING: connection is not using a post-quantum key exchange algorithm.\n'
        '** This session may be vulnerable to "store now, decrypt later" attacks.\n'
        '** The server may need to be upgraded. See https://openssh.com/pq.html\n'
    )
    markers = "RAG_SSH_OK\nRAG_REPO_OK\nRAG_QSUB_OK\nRAG_SIF_OK\n"
    backend = HpcBackend(BASE_CFG)
    monkeypatch.setattr(
        backend,
        "_run_ssh",
        lambda *args, **kwargs: _completed(markers, warning, 0),
    )

    checks = backend.check_connections()
    assert checks["cpu"] == {"ok": True, "configured": True, "detail": "ready"}


def test_connection_check_reports_missing_prerequisite_not_pq_warning(monkeypatch):
    warning = (
        "** WARNING: connection is not using a post-quantum key exchange algorithm.\n"
        "** The server may need to be upgraded. See https://openssh.com/pq.html\n"
    )
    backend = HpcBackend(BASE_CFG)
    monkeypatch.setattr(
        backend,
        "_run_ssh",
        lambda *args, **kwargs: _completed(
            "RAG_SSH_OK\nRAG_REPO_OK\nRAG_QSUB_OK\n",
            warning,
            0,
        ),
    )

    checks = backend.check_connections()
    assert checks["cpu"]["ok"] is False
    assert "container image" in checks["cpu"]["detail"]
    assert "post-quantum" not in checks["cpu"]["detail"]


# --- PBS script generation via the backend -----------------------------------


def test_build_pbs_script_is_cpu_shaped_from_overrides():
    b = HpcBackend(BASE_CFG)
    script = b._build_pbs_script("data")
    assert "#PBS -q cpu" in script
    assert "rag_pipeline_cpu.sif" in script
    assert 'STORAGE_ROOT="/hpctmp/me"' in script
    # CPU bundle: no :ngpus= clause, no --nv.
    assert ":ngpus=" not in script
    assert "--nv " not in script
    assert "python3 scripts/bulk_ingest.py --input-dir \"data\"" in script


def test_build_pbs_script_input_dir_override():
    b = HpcBackend(BASE_CFG)
    script = b._build_pbs_script("/hpctmp/me/pdfs")
    assert 'python3 scripts/bulk_ingest.py --input-dir "/hpctmp/me/pdfs"' in script


# --- submit_ingest_index routes to the CPU cluster ---------------------------


def test_submit_ingest_index_happy_path(monkeypatch):
    """qsub -> poll (R) -> poll (F, exit 0) -> return result with remote db path.
    Also verifies a __RAG_PROGRESS__ line is relayed to progress_callback AND
    that every SSH call went to the CPU cluster host (not the GPU one)."""
    b = HpcBackend(BASE_CFG)

    # Record the (host, command) pairs issued so we can assert routing.
    calls: list[tuple[str, str]] = []
    qstat_calls = {"n": 0}
    progress_payloads: list[dict] = []

    # The simulated PBS stdout file content grows across polls.
    # Poll 1 (R): one progress line. Poll 2 (F): no new content.
    progress_line = PROGRESS_PREFIX + '{"phase":"index","done":5,"total":10,"unit":"chunks"}'

    def fake_run_ssh(host, remote_command, *, capture=True, check=True, timeout=None):
        calls.append((host, remote_command))
        if "cat > " in remote_command:
            return _completed()  # write PBS file
        if "qsub" in remote_command:
            return _completed("42.aspsus01\n")
        if "qstat -f" in remote_command:
            qstat_calls["n"] += 1
            if qstat_calls["n"] == 1:
                state = "R"
            else:
                state = "F"
            return _completed(f"    job_state = {state}\n    Exit_status = 0\n")
        if "tail -c" in remote_command:
            # First tail returns the progress line; subsequent tails return empty.
            if qstat_calls["n"] == 1:
                return _completed(progress_line + "\n")
            return _completed("")
        if "rm -f " in remote_command:
            return _completed()
        return _completed()

    monkeypatch.setattr(b, "_run_ssh", fake_run_ssh)

    def on_progress(payload):
        progress_payloads.append(payload)

    result = b.submit_ingest_index(progress_callback=on_progress)
    assert isinstance(result, PbsJobResult)
    assert result.job_id == "42.aspsus01"
    assert result.remote_db_dir == "rag-cpu/db"
    assert result.exit_code == 0
    # The progress line was parsed and relayed.
    assert len(progress_payloads) == 1
    assert progress_payloads[0]["phase"] == "index"
    assert progress_payloads[0]["done"] == 5
    # qsub ran from the repo dir.
    assert any("cd rag-cpu && qsub" in cmd for _, cmd in calls)
    assert not any("qsub rag-cpu/" in cmd for _, cmd in calls)
    # CRITICAL: every ingest call went to the CPU cluster host, never the GPU one.
    assert all(host == "nus_hpc_cpu" for host, _ in calls), \
        f"build work must route to the CPU cluster; got hosts {[h for h,_ in calls]}"


def test_submit_ingest_index_fails_on_nonzero_exit(monkeypatch):
    b = HpcBackend(BASE_CFG)
    poll = {"n": 0}

    def fake_run_ssh(host, remote_command, *, capture=True, check=True, timeout=None):
        if "cat > " in remote_command or "rm -f " in remote_command:
            return _completed()
        if "qsub" in remote_command:
            return _completed("7.aspsus01\n")
        if "qstat -f" in remote_command:
            poll["n"] += 1
            if poll["n"] == 1:
                return _completed("    job_state = R\n")
            return _completed("    job_state = F\n    Exit_status = 1\n")
        return _completed()  # tail: empty

    monkeypatch.setattr(b, "_run_ssh", fake_run_ssh)
    with pytest.raises(HpcError, match="exit code 1"):
        b.submit_ingest_index()


def test_submit_ingest_index_qdel_on_cancel(monkeypatch):
    b = HpcBackend(BASE_CFG)
    cancel = threading.Event()
    polled = {"n": 0}
    qdeleted = {"done": False}

    def fake_run_ssh(host, remote_command, *, capture=True, check=True, timeout=None):
        if "cat > " in remote_command or "rm -f " in remote_command:
            return _completed()
        if "qsub" in remote_command:
            return _completed("9.aspsus01\n")
        if "qstat -f" in remote_command:
            polled["n"] += 1
            # After the first poll returns R, set cancel so the loop qdels next.
            if polled["n"] == 1:
                cancel.set()
            return _completed("    job_state = R\n")
        if "qdel" in remote_command:
            qdeleted["done"] = True
            return _completed()
        return _completed()

    monkeypatch.setattr(b, "_run_ssh", fake_run_ssh)
    with pytest.raises(HpcError, match="cancelled"):
        b.submit_ingest_index(cancel_event=cancel)
    assert qdeleted["done"], "cancel_event must trigger qdel"


def test_await_job_uses_history_after_qstat_purges_job(monkeypatch):
    b = HpcBackend(BASE_CFG)
    calls: list[str] = []

    def fake_run_ssh(host, remote_command, *, capture=True, check=True, timeout=None):
        calls.append(remote_command)
        if "qstat -xf" in remote_command:
            return _completed("    job_state = F\n    Exit_status = 0\n")
        if "qstat -f" in remote_command:
            return _completed("", "Unknown Job Id", returncode=153)
        return _completed()

    monkeypatch.setattr(b, "_run_ssh", fake_run_ssh)
    b._await_job(
        "42.server",
        BASE_CFG.cpu,
        progress_callback=None,
        log_callback=None,
        cancel_event=None,
    )
    assert any("qstat -xf" in command for command in calls)


def test_progress_log_uses_numeric_pbs_sequence(monkeypatch):
    b = HpcBackend(BASE_CFG)
    calls: list[str] = []

    def fake_run_ssh(host, remote_command, *, capture=True, check=True, timeout=None):
        calls.append(remote_command)
        return _completed("")

    monkeypatch.setattr(b, "_run_ssh", fake_run_ssh)
    b._relay_new_progress(
        "12345[].scheduler",
        BASE_CFG.cpu,
        0,
        progress_callback=lambda payload: None,
        log_callback=None,
    )
    assert any("rag_ingest_index.o12345" in command for command in calls)
    assert not any("o12345[]" in command for command in calls)


def test_submit_ingest_index_cancels_before_qsub(monkeypatch):
    b = HpcBackend(BASE_CFG)
    cancel = threading.Event()
    cancel.set()
    wrote = {"v": False}

    def fake_run_ssh(host, remote_command, *, capture=True, check=True, timeout=None):
        if "cat > " in remote_command:
            wrote["v"] = True
            return _completed()
        if "rm -f " in remote_command:
            return _completed()
        return _completed()

    monkeypatch.setattr(b, "_run_ssh", fake_run_ssh)
    with pytest.raises(HpcError, match="cancelled before qsub"):
        b.submit_ingest_index(cancel_event=cancel)


# --- fetch_index (rsync, from the CPU cluster) -------------------------------


def test_fetch_index_invokes_rsync_with_trailing_slash(tmp_path, monkeypatch):
    b = HpcBackend(BASE_CFG)
    captured: list[tuple[str, str]] = []

    def fake_rsync(source, dest, *, check=True, extra_args=None):
        captured.append((source, dest))
        return _completed()

    monkeypatch.setattr(b, "_run_rsync", fake_rsync)
    local = tmp_path / "db"
    out = b.fetch_index(local_db_dir=local)
    assert out == local
    assert local.exists()
    # source is cpu-host:/abs/db/ (trailing slash = contents), dest is local path/.
    assert captured == [(f"{BASE_CFG.cpu.ssh_host}:rag-cpu/db/", str(local) + "/")]


def test_fetch_index_uses_explicit_remote_db(tmp_path, monkeypatch):
    b = HpcBackend(BASE_CFG)
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(b, "_run_rsync", lambda s, d, **k: captured.append((s, d)) or _completed())
    b.fetch_index(remote_db_dir="/abs/other/db", local_db_dir=tmp_path / "db")
    assert captured[0][0] == "nus_hpc_cpu:/abs/other/db/"


# --- config wiring: cfg.hpc exists, defaults off, cpu cluster ---------------


def test_pipeline_config_has_hpc_section_defaulting_disabled():
    from src.config import load_config

    cfg = load_config()  # no RAG_PIPELINE_CONFIG in tests -> all defaults
    assert hasattr(cfg, "hpc")
    assert cfg.hpc.enabled is False
    # The CPU cluster sub-config exists with sensible defaults.
    assert cfg.hpc.cpu.ssh_host == ""
    assert cfg.hpc.cpu.container_sif == "rag_pipeline_cpu.sif"
    assert cfg.hpc.cpu.storage_root == "/hpctmp/${USER}"
    assert cfg.hpc.cpu.pbs_overrides.get("ngpus") == 0
    assert cfg.hpc.cpu.pbs_overrides.get("queue") == "cpu"


def test_hpc_config_loads_cpu_cluster_from_toml(tmp_path, monkeypatch):
    """[hpc.cpu] merges into cfg.hpc.cpu."""
    toml = """
[paths]
db_dir = "db"

[hpc]
enabled = true
remote_data_dir = "data"
poll_interval_seconds = 30.0

[hpc.cpu]
ssh_host = "cpu_login"
remote_repo_dir = "rag_cpu"
storage_root = "/hpctmp/u"
"""
    p = tmp_path / "c.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("RAG_PIPELINE_CONFIG", str(p))
    from src.config import load_config

    cfg = load_config()
    assert cfg.hpc.enabled is True
    assert cfg.hpc.poll_interval_seconds == 30.0
    assert cfg.hpc.cpu.ssh_host == "cpu_login"
    assert cfg.hpc.cpu.remote_repo_dir == "rag_cpu"
    assert cfg.hpc.cpu.storage_root == "/hpctmp/u"


# --- SSH/rsync hardening: timeouts + BatchMode/ConnectTimeout ----------------
#
# A hung SSH (host-key prompt, MFA challenge, dead route) must not block the
# single job worker forever. _run_ssh passes BatchMode=yes + ConnectTimeout so
# failures are loud, and a default subprocess timeout converts a hang into a
# loud HpcError instead of a permanently blocked worker thread.


def test_run_ssh_includes_batchmode_and_connect_timeout(monkeypatch):
    """The ssh argv must carry BatchMode=yes and ConnectTimeout so an
    interactive prompt or dead route fails fast rather than hanging."""
    from src import hpc_backend

    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["timeout"] = kwargs.get("timeout")
        return _completed()

    monkeypatch.setattr(hpc_backend.subprocess, "run", fake_run)
    b = hpc_backend.HpcBackend(BASE_CFG)
    b._run_ssh("nus_hpc_cpu", "echo hi")

    argv = captured["argv"]
    assert argv[0] == "ssh", f"unexpected argv: {argv}"
    # BatchMode + ConnectTimeout must be present (order-independent assertion).
    joined = " ".join(argv)
    assert "BatchMode=yes" in joined, "missing BatchMode=yes (would hang on a prompt)"
    assert "ConnectTimeout=15" in joined, "missing ConnectTimeout"
    # The remote command and host are the last two argv elements.
    assert argv[-2:] == ["nus_hpc_cpu", "echo hi"]
    # Default timeout is non-None (so a hang becomes a TimeoutExpired, not forever).
    assert captured["timeout"] is not None, "default timeout must be non-None"


def test_run_ssh_explicit_timeout_override(monkeypatch):
    """A caller-provided timeout (e.g. check_connections=20) must be honored."""
    from src import hpc_backend

    captured: dict = {}
    monkeypatch.setattr(
        hpc_backend.subprocess, "run",
        lambda argv, **k: captured.update(timeout=k.get("timeout")) or _completed(),
    )
    b = hpc_backend.HpcBackend(BASE_CFG)
    b._run_ssh("nus_hpc_cpu", "true", timeout=20)
    assert captured["timeout"] == 20


def test_run_ssh_timeout_expired_becomes_hpc_error(monkeypatch):
    """A timed-out ssh must raise HpcError, not leak subprocess.TimeoutExpired."""
    from src import hpc_backend

    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(hpc_backend.subprocess, "run", fake_run)
    b = hpc_backend.HpcBackend(BASE_CFG)
    with pytest.raises(HpcError, match="timed out"):
        b._run_ssh("nus_hpc_cpu", "echo hi", timeout=5)


def test_run_rsync_passes_ssh_opts_and_io_timeout(monkeypatch):
    """rsync must run over ssh with the hardening opts and carry an
    --timeout (rsync-side idle cap)."""
    from src import hpc_backend

    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["timeout"] = kwargs.get("timeout")
        return _completed()

    monkeypatch.setattr(hpc_backend.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(hpc_backend.subprocess, "run", fake_run)
    b = hpc_backend.HpcBackend(BASE_CFG)
    b._run_rsync("nus_hpc_cpu:db/", "db/")

    argv = captured["argv"]
    assert argv[0] == "rsync"
    joined = " ".join(argv)
    assert "BatchMode=yes" in joined, "rsync -e ssh must carry BatchMode=yes"
    assert "ConnectTimeout=15" in joined
    assert "--timeout=" in joined, "rsync must carry an --timeout (idle cap)"
    # No hard subprocess timeout by default (large transfers are legitimately slow).
    assert captured["timeout"] is None


def test_run_rsync_scp_fallback_carries_ssh_opts(monkeypatch):
    """On Windows (no rsync), the scp fallback still gets BatchMode/ConnectTimeout."""
    from src import hpc_backend

    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        return _completed()

    # shutil.which('rsync') -> None, which('scp') -> a path.
    monkeypatch.setattr(
        hpc_backend.shutil, "which",
        lambda name: None if name == "rsync" else "/usr/bin/scp",
    )
    monkeypatch.setattr(hpc_backend.subprocess, "run", fake_run)
    b = hpc_backend.HpcBackend(BASE_CFG)
    b._run_rsync("nus_hpc_cpu:db/", "db/")

    joined = " ".join(captured["argv"])
    assert captured["argv"][0] == "scp"
    assert "BatchMode=yes" in joined
    assert "ConnectTimeout=15" in joined
