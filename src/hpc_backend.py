"""Delegate bulk ingestion/indexing to an HPC cluster over SSH.

This module is the "build half" of the web-app -> HPC delegation. The chat half
needs no code: point the Ollama client at the SSH tunnel. See
``docs/HPC_DELEGATION.md`` for the end-to-end architecture.

Design goals
------------
- **Drop-in for the local subprocess runner.** ``submit_ingest_index`` mirrors
  what ``RagJobQueue._run_pipeline_subprocess`` does locally: run the build and
  relay ``__RAG_PROGRESS__`` lines into a ``progress_callback`` so the SAME web UI
  progress bar works unchanged (parsed by ``src.progress_protocol``).
- **No new dependencies.** Uses plain ``ssh``/``rsync`` subprocesses, matching the
  tunnel daemon -- no paramiko/asyncssh to pin.
- **Pure and mockable.** Every cluster interaction goes through
  ``self._run_ssh(...)`` / ``self._run_rsync(...)``, which tests monkeypatch. No
  network in the unit tests.
- **Reuse, don't reinvent.** PBS scripts come from ``src.hpc.generate_pbs_script``
  (already validated, ``--cpu``-capable); progress parsing reuses
  ``src.progress_protocol.parse_progress_line``.

What this does NOT do (intentionally, see docs/HPC_DELEGATION.md "next steps"):
- It is not yet wired into ``_run_job_subprocess``. The web app calls it
  explicitly once ``cfg.hpc.enabled`` is honored at the job-queue seam.
- It does not rsync user-uploaded PDFs OUT to HPC (corpus is assumed pre-staged;
  upload-out is an additive ``rsync`` step documented separately).
- It does not invalidate the server's index caches after ``fetch_index`` -- that
  is the caller's responsibility (the known gap that needs ``/api/hpc/reload``).
"""

from __future__ import annotations

import logging
import posixpath
import re
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from src.config import HpcClusterConfig, HpcConfig
from src.hpc import generate_pbs_script, generate_serve_pbs_script
from src.progress_protocol import parse_progress_line

logger = logging.getLogger(__name__)

# qsub prints the job id on its own line, e.g. "12345[].aspsus01". Sometimes it
# is just "12345". Capture the first whitespace-delimited token of the line that
# looks like a number-ish id.
_QSUB_JOBID_RE = re.compile(r"^\s*(\S+)", re.MULTILINE)

# Terminal PBS job states we care about (PBS Pro). 'F' = finished (check exit
# code), 'E' = exiting. Anything else (Q/R/T/...) means keep polling.
_PBS_TERMINAL_STATES = {"F", "E"}
_SSH_HOST_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")
_PBS_JOB_TOKEN_RE = re.compile(r"^(\d+)")
_ACCEPTABLE_SSH_WARNING_FRAGMENTS = (
    "connection is not using a post-quantum key exchange algorithm",
    'session may be vulnerable to "store now, decrypt later" attacks',
    "server may need to be upgraded",
    "openssh.com/pq.html",
)

# SSH hardening options applied to every _run_ssh / _run_rsync invocation. This
# backend runs unattended (keys are installed non-interactively by setup_instance),
# so BatchMode=yes makes auth/host-key failures loud instead of hanging on an
# interactive prompt ("Are you sure you want to continue connecting?"). Connect
# Timeout caps the initial TCP/handshake; the ServerAlive keepalives let a wedged
# TCP connection be reaped rather than blocking the (single) job worker forever.
_SSH_BASE_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=4",
]
# Generous default for the subprocess.run timeout on cluster calls. qsub/qstat
# are quick; this exists only to convert a hung SSH (host-key prompt, MFA
# challenge, dead route) into a loud HpcError instead of a permanently blocked
# worker. Long-running transfers use rsync's own --timeout; _await_job polling
# passes explicit per-call timeouts.
_SSH_DEFAULT_TIMEOUT = 120.0


class HpcError(RuntimeError):
    """A cluster operation (qsub/qstat/rsync) failed."""


def _strip_acceptable_ssh_warnings(stderr: str) -> str:
    """Remove OpenSSH's informational post-quantum KEX warning block.

    The warning is emitted on stderr even when authentication and the remote
    command succeed. It must not mask a real missing prerequisite or turn a
    healthy preflight red.
    """
    retained: list[str] = []
    for line in str(stderr or "").splitlines():
        lowered = line.lower()
        if any(fragment in lowered for fragment in _ACCEPTABLE_SSH_WARNING_FRAGMENTS):
            continue
        retained.append(line)
    return "\n".join(retained).strip()


@dataclass
class PbsJobResult:
    """Outcome of a submitted PBS job."""

    job_id: str
    state: str            # final PBS state, e.g. "F"
    exit_code: int | None # parsed from the job's exit status; None if unknown
    remote_db_dir: str    # where the index landed on the HPC side


def _parse_qsub_output(stdout: str) -> str:
    """Extract the job id from ``qsub`` stdout. Raises HpcError if none found."""
    # qsub typically prints exactly one line: the job id. Be tolerant of leading
    # noise/blank lines but require at least one plausible token.
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _QSUB_JOBID_RE.match(line)
        if match:
            return match.group(1)
    raise HpcError(f"could not parse job id from qsub output: {stdout!r}")


def parse_qstat_job_state(qstat_output: str, job_id: str) -> str | None:
    """Parse ``qstat -f <jobid>`` output for the job's ``job_state``.

    Returns the state letter (e.g. 'R', 'Q', 'F') or None if the job is gone
    from the queue (which, for a finished job, means it completed). PBS ``qstat
    -f`` emits ``    job_state = F`` lines under the job's stanza.
    """
    if not qstat_output:
        return None
    # Look for the job_state attribute anywhere in the output. PBS -f output is
    # structured as "    attr = value" lines; a simple line scan is robust to
    # the exact stanza formatting across PBS versions.
    for line in qstat_output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("job_state") and "=" in stripped:
            value = stripped.split("=", 1)[1].strip()
            if value:
                return value
    # No job_state line + empty/non-matching output usually means "job unknown
    # to qstat" (already purged). Treat as None so the caller can decide.
    return None


def parse_job_exit_code(qstat_output: str) -> int | None:
    """Parse the ``Exit_status`` attribute from qstat -f output, if present."""
    if not qstat_output:
        return None
    for line in qstat_output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("exit_status") and "=" in stripped:
            value = stripped.split("=", 1)[1].strip()
            try:
                return int(value)
            except ValueError:
                return None
    return None


class HpcBackend:
    """SSH-driven orchestrator for PBS ingest/index/serve jobs.

    Construct with an ``HpcConfig`` (typically ``load_config().hpc``). Two
    clusters are supported because the CPU and GPU clusters are separate machines
    with separate login nodes: build work (ingest/index/fetch) routes to
    ``cfg.cpu``; the Ollama serving job routes to ``cfg.gpu``. Each cluster
    interaction runs through ``_run_ssh`` / ``_run_rsync``, which tests
    monkeypatch; nothing in this class touches the network directly.
    """

    def __init__(self, cfg: HpcConfig):
        self.cfg = cfg
        # Validate lazily per method, since a deployment may only use one
        # cluster (e.g. build-only, no chat serving). Each method checks the
        # cluster it needs via _require_cluster().

    # ------------------------------------------------------------------ #
    # Cluster interaction primitives (mockable seams).
    # ------------------------------------------------------------------ #

    def _run_ssh(self, host: str, remote_command: str, *, capture: bool = True,
                 check: bool = True, timeout: float | None = _SSH_DEFAULT_TIMEOUT) -> subprocess.CompletedProcess:
        """Run ``remote_command`` on the given SSH host. Returns CompletedProcess.

        Single argv: ``["ssh", *_SSH_BASE_OPTS, host, remote_command]``. We pass
        the remote command as one string so shell features (``cd a && qsub b``)
        work; this is the same pattern the tunnel daemon relies on. The base
        opts force non-interactive auth (BatchMode) and a connect timeout so a
        misconfigured cluster fails fast instead of wedging the job worker.
        ``timeout`` defaults to ``_SSH_DEFAULT_TIMEOUT`` (callers that poll, like
        ``_await_job``, pass smaller explicit values).
        """
        argv = ["ssh", *_SSH_BASE_OPTS, host, remote_command]
        logger.debug("ssh %s: %s", host, remote_command)
        try:
            result = subprocess.run(
                argv,
                capture_output=capture,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HpcError(
                f"ssh {host} timed out after {timeout}s: {remote_command}"
            ) from exc
        if check and result.returncode != 0:
            raise HpcError(
                f"ssh {host} failed (exit {result.returncode}): "
                f"{remote_command}\nstderr: {result.stderr.strip()}"
            )
        return result

    def _run_rsync(self, source: str, dest: str, *, check: bool = True,
                   extra_args: list[str] | None = None,
                   timeout: float | None = None) -> subprocess.CompletedProcess:
        """Run rsync. Source/dest already include any ``host:`` prefix.

        rsync uses ssh as its remote transport, so the same connect/keepalive
        hardening is passed via ``-e ssh <_SSH_BASE_OPTS>``; ``--timeout`` caps
        idle time on the rsync side. ``timeout`` (subprocess.run) defaults to
        None because transfers can legitimately be large/slow -- callers that
        want a hard cap (e.g. the connection check) pass one explicitly.
        """
        # ssh opts reused across rsync (-e) and the scp fallback.
        ssh_opt_str = " ".join(["ssh", *_SSH_BASE_OPTS])
        if shutil.which("rsync") is not None:
            argv = ["rsync", "-e", ssh_opt_str, "--timeout=120"]
            if extra_args:
                argv.extend(extra_args)
            argv.extend(["-az", "--partial", source, dest])
            tool = "rsync"
        elif shutil.which("scp") is not None:
            # Windows OpenSSH normally includes scp but not rsync. Modern scp
            # accepts ``host:/dir/.`` and recursively copies the directory
            # contents, which preserves the same source semantics used here.
            # scp spawns ssh itself, so BatchMode/ConnectTimeout apply via the
            # OpenSSH options (scp forwards them). No native idle-timeout flag.
            scp_source = source[:-1] + "/." if source.endswith("/") else source
            argv = ["scp", *_SSH_BASE_OPTS, "-r", scp_source, dest]
            tool = "scp"
        else:
            raise HpcError(
                "Neither rsync nor scp was found. Install OpenSSH (Windows "
                "Optional Features) or rsync, then run the setup check again."
            )
        logger.debug("%s %s -> %s", tool, source, dest)
        try:
            result = subprocess.run(argv, capture_output=True, text=True, check=False,
                                    timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise HpcError(
                f"{tool} {source} -> {dest} timed out after {timeout}s"
            ) from exc
        if check and result.returncode != 0:
            raise HpcError(
                f"{tool} {source} -> {dest} failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result

    @staticmethod
    def _require_cluster(cluster: HpcClusterConfig, name: str) -> None:
        """Validate a cluster sub-config before using it."""
        if not cluster.ssh_host:
            raise HpcError(f"hpc.{name}.ssh_host must be set for {name}-cluster operations")
        if not _SSH_HOST_RE.fullmatch(cluster.ssh_host):
            raise HpcError(
                f"hpc.{name}.ssh_host contains unsupported characters; use a hostname "
                "or an SSH config alias"
            )
        if not cluster.remote_repo_dir:
            raise HpcError(
                f"hpc.{name}.remote_repo_dir must be set relative to the SSH login directory"
            )
        repo = PurePosixPath(cluster.remote_repo_dir)
        if (
            repo.is_absolute()
            or cluster.remote_repo_dir.startswith("~")
            or ".." in repo.parts
            or any(character.isspace() for character in cluster.remote_repo_dir)
        ):
            raise HpcError(
                f"hpc.{name}.remote_repo_dir must be a safe path relative to the "
                "SSH login directory"
            )

    # ------------------------------------------------------------------ #
    # Public API.
    # ------------------------------------------------------------------ #

    def submit_ingest_index(
        self,
        input_dir_on_hpc: str | None = None,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> PbsJobResult:
        """Submit the ingest+index PBS job to the CPU cluster, relay progress.

        Mirrors the local ``_run_job_subprocess`` contract: blocks until the job
        finishes (or is cancelled), routing ``__RAG_PROGRESS__`` lines from the
        PBS job's stdout into ``progress_callback`` so the existing UI progress
        bar works unchanged. On success returns a ``PbsJobResult`` whose
        ``remote_db_dir`` is where the index landed on the CPU cluster.
        """
        cpu = self.cfg.cpu
        self._require_cluster(cpu, "cpu")

        input_dir = input_dir_on_hpc or self.cfg.remote_data_dir
        pbs_script = self._build_pbs_script(input_dir)
        remote_pbs_path = self._write_remote_pbs_script(pbs_script, cpu)

        if cancel_event is not None and cancel_event.is_set():
            raise HpcError("cancelled before qsub")

        job_id = self._qsub(remote_pbs_path, cpu)
        try:
            self._await_job(
                job_id, cpu,
                progress_callback=progress_callback,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )
        finally:
            # Best-effort cleanup of the generated PBS file on the CPU cluster.
            try:
                self._run_ssh(cpu.ssh_host, f"rm -f {remote_pbs_path}", check=False)
            except HpcError:
                pass

        # Confirm exit code so a silently-failed build doesn't look like success.
        exit_code = self._job_exit_code(job_id, cpu)
        if exit_code not in (None, 0):
            raise HpcError(f"PBS job {job_id} failed with exit code {exit_code}")

        return PbsJobResult(
            job_id=job_id,
            state="F",
            exit_code=exit_code,
            remote_db_dir=self._remote_db_abs_path(),
        )

    def fetch_index(self, remote_db_dir: str | None = None,
                    local_db_dir: str | Path = "db") -> Path:
        """rsync the built index from the CPU cluster into ``local_db_dir``.

        Returns the local path. The CALLER must then invalidate the server's
        index caches (``web_app._invalidate_index_caches``) -- this module does
        not depend on the web app, so that trigger lives one layer up.
        """
        cpu = self.cfg.cpu
        self._require_cluster(cpu, "cpu")
        remote = remote_db_dir or self._remote_db_abs_path()
        local = Path(local_db_dir)
        local.mkdir(parents=True, exist_ok=True)
        # Trailing slash on the source = "contents of", not the dir itself.
        self._run_rsync(f"{cpu.ssh_host}:{remote}/", str(local) + "/")
        return local

    def check_connections(self) -> dict[str, dict[str, Any]]:
        """Verify SSH, PBS, repo and container prerequisites on both clusters.

        This is intentionally read-only and powers the guided setup command.
        A deployment may omit either cluster; omitted clusters are reported as
        unconfigured rather than raising.
        """
        checks: dict[str, dict[str, Any]] = {}
        for name, cluster in (("cpu", self.cfg.cpu), ("gpu", self.cfg.gpu)):
            if not cluster.ssh_host or not cluster.remote_repo_dir:
                checks[name] = {"ok": False, "configured": False, "detail": "not configured"}
                continue
            try:
                self._require_cluster(cluster, name)
                repo = shlex.quote(cluster.remote_repo_dir)
                sif = shlex.quote(posixpath.join(cluster.remote_repo_dir, cluster.container_sif))
                command = (
                    "printf 'RAG_SSH_OK\\n'; "
                    f"test -d {repo} && printf 'RAG_REPO_OK\\n'; "
                    "command -v qsub >/dev/null 2>&1 && printf 'RAG_QSUB_OK\\n'; "
                    f"test -f {sif} && printf 'RAG_SIF_OK\\n'; "
                    "exit 0"
                )
                result = self._run_ssh(cluster.ssh_host, command, check=False, timeout=20)
                stdout = result.stdout or ""
                ssh_ok = result.returncode == 0 and "RAG_SSH_OK" in stdout
                missing: list[str] = []
                if ssh_ok:
                    if "RAG_REPO_OK" not in stdout:
                        missing.append(f"repo directory '{cluster.remote_repo_dir}'")
                    if "RAG_QSUB_OK" not in stdout:
                        missing.append("qsub command")
                    if "RAG_SIF_OK" not in stdout:
                        missing.append(f"container image '{cluster.container_sif}'")
                ok = ssh_ok and not missing
                cleaned_stderr = _strip_acceptable_ssh_warnings(result.stderr)
                if ok:
                    detail = "ready"
                elif not ssh_ok:
                    detail = cleaned_stderr or f"SSH connection failed (exit {result.returncode})"
                else:
                    detail = "missing " + ", ".join(missing)
                    if cleaned_stderr:
                        detail += f"; SSH diagnostic: {cleaned_stderr}"
                checks[name] = {"ok": ok, "configured": True, "detail": detail}
            except (HpcError, OSError, subprocess.SubprocessError) as exc:
                checks[name] = {"ok": False, "configured": True, "detail": str(exc)}
        return checks

    def submit_serve_job(self, *, ollama_host_file_on_hpc: str | None = None) -> str:
        """Submit the long-lived Ollama serving PBS job to the GPU cluster.

        Returns the job id. Non-blocking -- the serving job runs until its
        walltime. Caller then starts the tunnel daemon (pointed at the GPU
        login node via --jump-host) and points OLLAMA_HOST at the local tunnel
        port. See scripts/nus_hpc_serve.pbs and docs/HPC_DELEGATION.md.
        """
        gpu = self.cfg.gpu
        self._require_cluster(gpu, "gpu")
        ollama_host_file = ollama_host_file_on_hpc or "${HOME}/.rag_ollama_serving_host"
        pbs_script = generate_serve_pbs_script(
            container_sif=gpu.container_sif,
            ollama_host_file=ollama_host_file,
            **self._serve_overrides(gpu),
        )
        remote_pbs_path = self._remote_pbs_path(gpu, "serve")
        self._write_remote_pbs_script(pbs_script, gpu, remote_path=remote_pbs_path)
        try:
            return self._qsub(remote_pbs_path, gpu)
        finally:
            self._run_ssh(
                gpu.ssh_host,
                f"rm -f -- {shlex.quote(remote_pbs_path)}",
                check=False,
            )

    # ------------------------------------------------------------------ #
    # Internals.
    # ------------------------------------------------------------------ #

    def _build_pbs_script(self, input_dir: str) -> str:
        """Generate the ingest+index PBS script, merging cpu-cluster overrides."""
        overrides = dict(self.cfg.cpu.pbs_overrides)
        overrides.setdefault("input_data_dir", input_dir)
        overrides.setdefault("container_sif", self.cfg.cpu.container_sif)
        overrides.setdefault("storage_root", self.cfg.cpu.storage_root)
        accepted = {
            "job_name", "ncpus", "mem", "ngpus", "queue", "input_data_dir",
            "container_sif", "walltime", "ollama_models_dir", "storage_root",
        }
        kwargs = {k: v for k, v in overrides.items() if k in accepted}
        return generate_pbs_script(**kwargs)

    @staticmethod
    def _serve_overrides(gpu: HpcClusterConfig) -> dict:
        """Extract generate_serve_pbs_script kwargs from gpu.pbs_overrides."""
        accepted = {"job_name", "ncpus", "mem", "ngpus", "queue", "walltime",
                    "container_sif", "ollama_models_dir", "ollama_host_file",
                    "storage_root"}
        overrides = {
            k: v for k, v in gpu.pbs_overrides.items()
            if k in accepted and k != "container_sif"
        }
        overrides.setdefault("storage_root", gpu.storage_root)
        return overrides

    def _remote_pbs_path(self, cluster: HpcClusterConfig, kind: str = "ingest") -> str:
        return posixpath.join(
            cluster.remote_repo_dir,
            f".hpc_{kind}_{int(time.time())}_{threading.get_ident()}.pbs",
        )

    def _write_remote_pbs_script(self, pbs_script: str, cluster: HpcClusterConfig,
                                 *, remote_path: str | None = None) -> str:
        """Write the PBS script body to a remote path via ssh heredoc."""
        path = remote_path or self._remote_pbs_path(cluster)
        # Pipe the script body to a remote ``cat > path``. Use single-quoted
        # heredoc delimiter so the remote shell does not expand ${VAR} in the
        # script body (the PBS script is full of ${PBS_JOBID} etc.).
        # This is safe because generate_pbs_script output is validated and
        # contains no unescaped single quotes.
        remote_command = (
            f"cat > {shlex.quote(path)} <<'HPC_PBS_EOF'\n"
            f"{pbs_script}\nHPC_PBS_EOF"
        )
        self._run_ssh(cluster.ssh_host, remote_command)
        return path

    def _qsub(self, remote_pbs_path: str, cluster: HpcClusterConfig) -> str:
        """qsub the remote PBS script from the cluster's repo dir; return job id."""
        # qsub must run from the repo dir so ${PWD}:/app binds the right place
        # and relative output paths resolve. Submit only the basename because
        # remote_pbs_path is itself relative to the SSH login directory.
        pbs_name = posixpath.basename(remote_pbs_path)
        cmd = (
            f"cd {shlex.quote(cluster.remote_repo_dir)} "
            f"&& qsub {shlex.quote(pbs_name)}"
        )
        result = self._run_ssh(cluster.ssh_host, cmd)
        return _parse_qsub_output(result.stdout)

    def _await_job(
        self,
        job_id: str,
        cluster: HpcClusterConfig,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        log_callback: Callable[[str], None] | None,
        cancel_event: threading.Event | None,
    ) -> None:
        """Poll qstat until the job reaches a terminal state.

        Relays ``__RAG_PROGRESS__`` lines from the job's combined stdout file
        (``<jobname>.o<jobid>`` in the repo dir) into ``progress_callback``.
        On ``cancel_event``, qdel's the job and raises HpcError.
        """
        # Track how much of the output file we have already relayed so each poll
        # only emits NEW progress lines (the file grows as the job runs).
        relayed_bytes = 0
        missing_polls = 0
        seen_by_scheduler = False
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self._qdel(job_id, cluster)
                raise HpcError(f"PBS job {job_id} cancelled by user")

            state, qstat_text = self._query_job(job_id, cluster)
            relayed_bytes = self._relay_new_progress(
                job_id, cluster, relayed_bytes, progress_callback, log_callback
            )
            if state in _PBS_TERMINAL_STATES:
                return
            if state is None:
                missing_polls += 1
                # A completed job may disappear from qstat quickly. Once it has
                # been seen, one missing poll is sufficient; before that, allow
                # a short scheduler-registration race.
                if seen_by_scheduler or missing_polls >= 3:
                    return
            else:
                seen_by_scheduler = True
                missing_polls = 0
            time.sleep(max(1.0, float(self.cfg.poll_interval_seconds)))

    def _query_job(self, job_id: str, cluster: HpcClusterConfig) -> tuple[str | None, str]:
        """Return (state, raw_qstat_text). state may be None if job is purged."""
        quoted_job_id = shlex.quote(job_id)
        result = self._run_ssh(cluster.ssh_host, f"qstat -f {quoted_job_id}", check=False)
        if result.returncode != 0:
            # PBS Pro commonly retains completed jobs only in extended history.
            history = self._run_ssh(
                cluster.ssh_host,
                f"qstat -xf {quoted_job_id}",
                check=False,
            )
            if history.returncode == 0:
                return parse_qstat_job_state(history.stdout, job_id), history.stdout
            return None, ""
        state = parse_qstat_job_state(result.stdout, job_id)
        return state, result.stdout

    def _job_exit_code(self, job_id: str, cluster: HpcClusterConfig) -> int | None:
        """Best-effort exit-code read from qstat -f after the job finished."""
        quoted_job_id = shlex.quote(job_id)
        result = self._run_ssh(cluster.ssh_host, f"qstat -f {quoted_job_id}", check=False)
        if result.returncode != 0:
            result = self._run_ssh(
                cluster.ssh_host,
                f"qstat -xf {quoted_job_id}",
                check=False,
            )
        return parse_job_exit_code(result.stdout) if result.returncode == 0 else None

    def _relay_new_progress(
        self,
        job_id: str,
        cluster: HpcClusterConfig,
        already_relayed: int,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        log_callback: Callable[[str], None] | None,
    ) -> int:
        """Read new bytes from the job's stdout file and relay progress lines.

        Returns the new ``already_relayed`` offset. The PBS combined stdout file
        is named per PBS convention ``<jobname>.o<jobid>`` in the qsub cwd; the
        job name is rag_ingest_index (from generate_pbs_script). Reading it over
        ssh with ``tail -c +N`` gives us only the bytes since the last relay.
        """
        if progress_callback is None and log_callback is None:
            return already_relayed
        # tail -c +N prints from byte offset N (1-indexed), so +1 skips already-read.
        start = already_relayed + 1
        match = _PBS_JOB_TOKEN_RE.match(job_id)
        output_token = match.group(1) if match else job_id
        log_path = posixpath.join(
            cluster.remote_repo_dir,
            f"rag_ingest_index.o{output_token}",
        )
        result = self._run_ssh(
            cluster.ssh_host,
            f"tail -c +{start} {shlex.quote(log_path)} 2>/dev/null",
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            return already_relayed
        new_bytes = len(result.stdout.encode("utf-8"))
        for line in result.stdout.splitlines():
            text = line.rstrip()
            if not text:
                continue
            if progress_callback is not None:
                parsed = parse_progress_line(text)
                if parsed is not None:
                    progress_callback(parsed)
                    continue
            if log_callback is not None:
                log_callback(text)
        return already_relayed + new_bytes

    def _qdel(self, job_id: str, cluster: HpcClusterConfig) -> None:
        self._run_ssh(cluster.ssh_host, f"qdel {shlex.quote(job_id)}", check=False)

    def _remote_db_abs_path(self) -> str:
        """Resolve the index path relative to the SSH login directory.

        ``remote_repo_dir`` itself is login-relative. An explicitly absolute
        ``remote_db_dir`` remains supported for specialized cluster storage.
        """
        db = self.cfg.remote_db_dir
        if db.startswith("/"):
            return db
        return f"{self.cfg.cpu.remote_repo_dir.rstrip('/')}/{db}"
