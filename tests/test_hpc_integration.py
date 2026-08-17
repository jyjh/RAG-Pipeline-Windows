import os
import re
import subprocess
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.hpc import (
    generate_pbs_script,
    generate_serve_pbs_script,
    parse_hpc_args,
    main as hpc_main,
)
import src.hpc as hpc
import src.local_rag as local_rag
import src.web_app as web_app

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_singularity_def_exists_and_valid():
    singularity_path = ROOT_DIR / "Singularity.def"
    assert singularity_path.exists(), "Singularity.def must exist in repository root"
    
    content = singularity_path.read_text(encoding="utf-8")
    
    # Header assertions
    assert "Bootstrap: docker" in content
    assert "From: nvidia/cuda:12.1.1-devel-ubuntu22.04" in content
    
    # Section presence assertions
    assert "%labels" in content
    assert "%environment" in content
    assert "%post" in content
    assert "%runscript" in content
    
    # Labels assertion
    assert "Maintainer" in content
    assert "Version" in content
    assert "Description" in content
    
    # Environment assertions
    assert "export OLLAMA_HOST=127.0.0.1:11434" in content
    assert "export LC_ALL=C.UTF-8" in content
    assert "export LANG=C.UTF-8" in content
    assert "export PATH=/usr/local/bin:$PATH" in content
    
    # Post section dependency assertions
    assert "python3" in content
    assert "libgl1" in content
    assert "libglib2.0-0" in content
    assert "poppler-utils" in content
    assert "tesseract-ocr" in content
    assert "curl" in content
    assert "git" in content
    assert "ca-certificates" in content
    assert "https://download.pytorch.org/whl/cu121" in content
    # The CUDA 12.1.1 base is ubuntu:22.04 (Python 3.10), but the pinned deps
    # (onnxruntime~=1.27.0, numpy~=2.4.6) require >=3.11. The image must install
    # Python 3.14 via deadsnakes AND route the app-stack install through it, or
    # the build fails with "from versions: max 1.23.2" for onnxruntime.
    assert "ppa:deadsnakes/ppa" in content, "GPU image must add deadsnakes PPA for Python 3.14"
    assert "python3.14" in content, "GPU image must install python3.14"
    assert "python3.14 -m pip install --no-cache-dir torch" in content, \
        "GPU app-stack pip must run under python3.14, not the system 3.10"
    
    # Python package assertions
    assert "docling~=2.105.0" in content
    assert "lancedb~=0.33.0" in content
    assert "pylance~=7.0.0" in content
    assert "ollama~=0.6.2" in content
    assert "onnxruntime~=1.27.0" in content
    assert "fastapi~=0.138.0" in content
    assert "uvicorn[standard]~=0.49.0" in content
    
    # Ollama standalone binary installation assertion. The installer extracts a
    # .tar.zst, so zstd must be apt-installed; the fallback URL is the current
    # GitHub release asset (the old ollama.com/download/...tar.gz 404s).
    assert "ollama" in content.lower()
    assert "zstd" in content, "image must apt-install zstd (ollama installer needs it)"
    assert "install.sh" in content
    assert "ollama-linux-amd64.tar.zst" in content, \
        "fallback must be the current .tar.zst asset, not the dead .tar.gz URL"
    assert "tar --zstd -xf" in content, "fallback must extract with zstd, not gzip"


def test_nus_hpc_pbs_script_exists_and_valid():
    pbs_path = ROOT_DIR / "scripts" / "nus_hpc_ingest_index.pbs"
    assert pbs_path.exists(), "scripts/nus_hpc_ingest_index.pbs must exist"
    
    raw_bytes = pbs_path.read_bytes()
    
    # CRITICAL: Strict LF line endings check
    assert b"\r\n" not in raw_bytes, "scripts/nus_hpc_ingest_index.pbs MUST use LF line endings, no CRLF allowed"
    
    content = raw_bytes.decode("utf-8")
    
    # PBS Directives assertions
    assert "#PBS -N rag_ingest_index" in content
    assert "#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1" in content
    assert "#PBS -q gpu" in content
    assert "#PBS -j oe" in content
    
    # Execution Logic assertions
    assert "module load singularity" in content
    assert "/hpctmp/" in content
    assert "rag_scratch_" in content
    assert "trap" in content
    assert "singularity exec" in content
    assert "--nv" in content
    assert "-B /hpctmp/${USER}:/hpctmp/${USER}" in content
    # The CPU ingest job embeds via the SoCLAaS API (bge-m3) -- it no longer runs
    # `ollama serve`, pre-pulls models, or probes /api/version. Check the
    # command forms (the on-disk file documents this in comments).
    assert "ollama serve >" not in content
    assert "ollama pull" not in content
    assert "/api/version" not in content
    assert "scripts/bulk_ingest.py" in content
    assert "SOCLAAS_API_KEY" in content

    # Walltime must be set (a cold index takes hours; the cluster default is
    # often too short and kills the job).
    assert re.search(r"#PBS -l walltime=\d{1,4}:\d{2}:\d{2}", content), \
        "ingest .pbs must declare a walltime"


def test_tunnel_daemon_sh_exists_and_valid():
    sh_path = ROOT_DIR / "scripts" / "tunnel_daemon.sh"
    assert sh_path.exists(), "scripts/tunnel_daemon.sh must exist"
    
    raw_bytes = sh_path.read_bytes()
    assert b"\r\n" not in raw_bytes, "scripts/tunnel_daemon.sh MUST use LF line endings"
    
    content = raw_bytes.decode("utf-8")
    assert content.startswith("#!/bin/bash")
    assert "nus_hpc_gpu" in content
    assert "11434" in content
    assert "trap" in content
    assert "SIGINT" in content
    assert "ExitOnForwardFailure=yes" in content
    assert "ServerAliveInterval=15" in content
    assert "ServerAliveCountMax=3" in content
    assert "--help" in content
    # 2-hop support for reaching a GPU compute node via a login/jump host.
    assert "--jump-host" in content
    assert "--host-file" in content
    assert "JUMP_HOST" in content
    assert "HOST_FILE" in content
    # ProxyJump flag must be wired into the actual ssh command, not just parsed.
    assert '"-J"' in content or 'SSH_CMD+=(-J' in content


def test_tunnel_daemon_ps1_exists_and_valid():
    ps1_path = ROOT_DIR / "scripts" / "tunnel_daemon.ps1"
    assert ps1_path.exists(), "scripts/tunnel_daemon.ps1 must exist"
    
    content = ps1_path.read_text(encoding="utf-8")
    assert ".SYNOPSIS" in content
    assert ".DESCRIPTION" in content
    assert ".EXAMPLE" in content
    assert "param(" in content
    assert "nus_hpc_gpu" in content
    assert "11434" in content
    assert "ExitOnForwardFailure=yes" in content
    assert "ServerAliveInterval=15" in content
    assert "ServerAliveCountMax=3" in content
    assert "try" in content
    assert "finally" in content
    assert "Tunnel disconnected. Reconnecting in" in content
    # 2-hop support (PowerShell parameter names).
    assert "JumpHost" in content
    assert "HostFile" in content
    assert '"-J"' in content


def test_hpc_cli_parsing_and_pbs_template_generation():
    # parse_hpc_args uses None as a sentinel for "not specified" so main() can
    # apply mode-appropriate defaults (ingest vs serve differ, e.g. job name and
    # CPU count). Unspecified args are therefore None here; the resolution is
    # exercised via generate_pbs_script() / _build_script_from_args below.
    args = parse_hpc_args([])
    assert args.serve is False
    assert args.job_name is None
    assert args.ncpus is None
    assert args.mem is None
    assert args.output is None

    # Test CLI argument parsing overrides
    custom_args = parse_hpc_args([
        "--job-name", "custom_job",
        "--ncpus", "16",
        "--mem", "64gb",
        "--ngpus", "2",
        "--queue", "high_gpu",
        "--input-data-dir", "custom_data",
        "--container-sif", "custom_rag.sif",
    ])
    assert custom_args.job_name == "custom_job"
    assert custom_args.ncpus == 16
    assert custom_args.mem == "64gb"
    assert custom_args.ngpus == 2
    assert custom_args.queue == "high_gpu"
    assert custom_args.input_data_dir == "custom_data"
    assert custom_args.container_sif == "custom_rag.sif"

    # Generate PBS script with default options
    default_pbs = generate_pbs_script()
    assert "#PBS -N rag_ingest_index" in default_pbs
    assert "#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1" in default_pbs
    assert "#PBS -q gpu" in default_pbs
    assert "python3 scripts/bulk_ingest.py --input-dir \"data\"" in default_pbs

    # Generate PBS script with parameter overrides
    overridden_pbs = generate_pbs_script(
        job_name=custom_args.job_name,
        ncpus=custom_args.ncpus,
        mem=custom_args.mem,
        ngpus=custom_args.ngpus,
        queue=custom_args.queue,
        input_data_dir=custom_args.input_data_dir,
        container_sif=custom_args.container_sif,
    )
    assert "#PBS -N custom_job" in overridden_pbs
    assert "#PBS -l select=1:ncpus=16:mem=64gb:ngpus=2" in overridden_pbs
    assert "#PBS -q high_gpu" in overridden_pbs
    assert "CONTAINER_SIF=\"${CONTAINER_SIF:-custom_rag.sif}\"" in overridden_pbs
    assert "python3 scripts/bulk_ingest.py --input-dir \"custom_data\"" in overridden_pbs

    # The CPU ingest generator no longer runs Ollama: embeddings come from the
    # SoCLAaS API. Regenerated jobs must carry the API-key step + walltime and
    # must NOT regress to `ollama serve`/`ollama pull`.
    assert "SOCLAAS_API_KEY" in overridden_pbs
    assert re.search(r"#PBS -l walltime=\d{1,4}:\d{2}:\d{2}", overridden_pbs)
    assert "ollama pull" not in overridden_pbs
    assert "ollama serve" not in overridden_pbs

    # New CLI surface: --serve, --walltime, -o/--output.
    serve_args = parse_hpc_args(["--serve", "--walltime", "04:00:00", "-o", "x.pbs"])
    assert serve_args.serve is True
    assert serve_args.walltime == "04:00:00"
    assert serve_args.output == "x.pbs"


def test_ollama_config_host_resolution_precedence(monkeypatch, tmp_path):
    local_rag._ACTIVE_OLLAMA_HOST = None

    # Case 4: Default fallback when no env, no active host, no config file
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("RAG_PIPELINE_CONFIG", raising=False)
    assert local_rag._ollama_host() == "http://127.0.0.1:11434"

    # Case 3: Config file host
    toml_content = '[ollama]\nhost = "http://toml-gpu:11434"\n'
    cfg_path = tmp_path / "test_config.toml"
    cfg_path.write_text(toml_content, encoding="utf-8")
    monkeypatch.setenv("RAG_PIPELINE_CONFIG", str(cfg_path))
    assert local_rag._ollama_host() == "http://toml-gpu:11434"

    # Case 2: Active dynamic host overrides config file host
    local_rag._ACTIVE_OLLAMA_HOST = "http://dynamic-failover:11434"
    assert local_rag._ollama_host() == "http://dynamic-failover:11434"

    # Case 1: OLLAMA_HOST env var overrides active dynamic host & config file
    monkeypatch.setenv("OLLAMA_HOST", "http://env-override:11434")
    assert local_rag._ollama_host() == "http://env-override:11434"

    # Clean up state
    local_rag._ACTIVE_OLLAMA_HOST = None


def test_probe_ollama_endpoints_and_recovery(monkeypatch):
    def fake_healthy(host, timeout=3.0):
        return host == "http://healthy-backup:11434"

    monkeypatch.setattr(local_rag, "_ollama_server_healthy", fake_healthy)

    endpoints = ["http://unhealthy-1:11434", "http://healthy-backup:11434"]
    assert local_rag.probe_ollama_endpoints(endpoints) == "http://healthy-backup:11434"
    assert local_rag.probe_ollama_endpoints(["http://unhealthy-1:11434"]) is None

    local_rag._ACTIVE_OLLAMA_HOST = "http://dead-primary:11434"
    recovered = local_rag._wait_for_ollama_recovery(
        health_check_interval=0.01,
        max_lost_health_checks=1,
        candidate_hosts=["http://dead-primary:11434", "http://healthy-backup:11434"],
    )
    assert recovered is True
    assert local_rag._ACTIVE_OLLAMA_HOST == "http://healthy-backup:11434"

    # Clean up state
    local_rag._ACTIVE_OLLAMA_HOST = None


def test_web_app_health_and_metrics_endpoints_ollama_status(monkeypatch):
    # Exercise the dormant Ollama branch of the status snapshot (the default
    # SoCLAaS backend has its own coverage in tests/test_llm_api.py).
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setattr(
        local_rag,
        "_ollama_host",
        lambda: "http://127.0.0.1:11434",
    )
    monkeypatch.setattr(
        local_rag,
        "_get_ollama_candidate_hosts",
        lambda: ["http://127.0.0.1:11434", "http://remote-gpu:11434"],
    )
    monkeypatch.setattr(
        local_rag,
        "_ollama_server_healthy",
        lambda host=None, timeout=1.5: host == "http://127.0.0.1:11434",
    )

    client = TestClient(web_app.app)

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["ollama_active_host"] == "http://127.0.0.1:11434"
    assert data_health["ollama_candidate_hosts"] == ["http://127.0.0.1:11434", "http://remote-gpu:11434"]
    assert data_health["ollama_reachability"] == {
        "http://127.0.0.1:11434": True,
        "http://remote-gpu:11434": False,
    }

    res_metrics = client.get("/api/metrics")
    assert res_metrics.status_code == 200
    data_metrics = res_metrics.json()
    assert data_metrics["ollama_active_host"] == "http://127.0.0.1:11434"
    assert data_metrics["ollama_candidate_hosts"] == ["http://127.0.0.1:11434", "http://remote-gpu:11434"]
    assert data_metrics["ollama_reachability"] == {
        "http://127.0.0.1:11434": True,
        "http://remote-gpu:11434": False,
    }


# ==============================================================================
# Tests for the live-run + serving-job work (Part A/B/C of the HPC port).
# ==============================================================================


def test_nus_hpc_serve_pbs_script_exists_and_valid():
    """The long-lived Ollama serving job: keeps ollama alive, publishes its
    compute-node hostname so the 2-hop tunnel can find it, cleans up on exit."""
    pbs_path = ROOT_DIR / "scripts" / "nus_hpc_serve.pbs"
    assert pbs_path.exists(), "scripts/nus_hpc_serve.pbs must exist"

    raw_bytes = pbs_path.read_bytes()
    assert b"\r\n" not in raw_bytes, "scripts/nus_hpc_serve.pbs MUST use LF line endings"

    content = raw_bytes.decode("utf-8")

    # PBS directives (queue gpu, a GPU, a walltime so it stays alive).
    assert "#PBS -N rag_ollama_serve" in content
    assert re.search(r"#PBS -l select=1:ncpus=\d+:mem=\d+[gGmM][bB]?:ngpus=\d+", content)
    assert re.search(r"#PBS -l walltime=\d{1,4}:\d{2}:\d{2}", content)
    assert "#PBS -q gpu" in content
    assert "#PBS -j oe" in content

    # Hostname discovery: the serving job writes its compute node name to a
    # shared file that the tunnel daemon reads.
    assert "hostname -f" in content or "hostname" in content
    assert ".rag_ollama_serving_host" in content

    # Ollama is brought up and kept alive (not exited after a one-shot task).
    assert "ollama serve" in content
    assert "/api/version" in content
    assert "ollama pull" in content
    assert "/scratch/${USER}" in content
    assert "/hpctmp/${USER}" not in content

    # The keep-alive loop is what distinguishes a serving job from the ingest
    # job (which exits after bulk_ingest.py). There must be a blocking sleep.
    assert "sleep" in content and "while true" in content

    # Cleanup must remove the discovery file (so the tunnel stops targeting a
    # dead node) but must NOT wipe the persistent model store.
    assert "rm -f" in content and ".rag_ollama_serving_host" in content
    assert "trap" in content and "EXIT" in content


def test_hpc_generators_reject_injection_and_invalid_inputs():
    """generate_pbs_script must validate every interpolated field. The audit
    flagged raw f-string interpolation as allowing PBS-directive injection and
    shell breakout; these cases must now raise ValueError."""
    bad_inputs = [
        # PBS-directive injection via a newline in a name/queue field.
        {"job_name": "x\n#PBS -o /tmp/evil"},
        {"queue": "gpu\n#PBS -l walltime=999:99:99"},
        # Empty / boundary numeric values.
        {"ncpus": 0},
        {"ncpus": -5},
        {"ngpus": -1},
        # Bad memory specs.
        {"mem": "invalid"},
        {"mem": "32"},          # missing unit
        {"mem": "32 gb"},       # space
        # Shell-injection vectors through the path fields.
        {"input_data_dir": 'data"; touch /tmp/pwned #'},
        {"input_data_dir": "data$(touch /tmp/p)"},
        {"input_data_dir": "data`whoami`"},
        {"container_sif": 'x" || pwned'},
        {"ollama_models_dir": "$(evil)"},
        {"ollama_models_dir": "/tmp/a b"},
        # Bad walltime.
        {"walltime": "99"},
        {"walltime": "99:99"},
        {"storage_root": "/scratch/user;touch_bad"},
    ]
    for kw in bad_inputs:
        with pytest.raises(ValueError):
            generate_pbs_script(**kw)


def test_hpc_generators_accept_legitimate_values():
    """Sanity: the validation must not reject values the runbook actually uses,
    including shell-variable expansion like ${HOME}/ollama_models and
    /hpctmp/${USER}/... paths."""
    ok = generate_pbs_script(
        job_name="my_index",
        ncpus=16,
        mem="64gb",
        ngpus=2,
        queue="gpu",
        walltime="12:00:00",
        input_data_dir="/hpctmp/${USER}/pdfs",
        container_sif="rag_pipeline.sif",
        ollama_models_dir="${HOME}/ollama_models",
    )
    assert "#PBS -N my_index" in ok
    assert "#PBS -l select=1:ncpus=16:mem=64gb:ngpus=2" in ok
    assert "#PBS -l walltime=12:00:00" in ok
    assert "/hpctmp/${USER}/pdfs" in ok
    # ollama_models_dir is accepted (validated) for backward compat even though
    # the CPU template no longer interpolates it (embeddings now via SoCLAaS API).
    assert "SOCLAAS_API_KEY" in ok

    serve_ok = generate_serve_pbs_script(
        ollama_host_file="${HOME}/.rag_ollama_serving_host",
    )
    assert "rag_ollama_serve" in serve_ok
    assert "${HOME}/.rag_ollama_serving_host" in serve_ok
    assert 'STORAGE_ROOT="/scratch/${USER}"' in serve_ok


def test_hpc_module_has_runnable_main():
    """The CLI must actually emit a script. Before this work parse_hpc_args
    existed but had no __main__ block, so `python -m src.hpc` was dead code."""
    # Ingest mode -> stdout
    result = subprocess.run(
        [sys.executable, "-m", "src.hpc"],
        capture_output=True, text=True, cwd=str(ROOT_DIR),
    )
    assert result.returncode == 0, result.stderr
    assert "#!/bin/bash" in result.stdout
    assert "#PBS -N rag_ingest_index" in result.stdout

    # Serve mode -> stdout
    result = subprocess.run(
        [sys.executable, "-m", "src.hpc", "--serve"],
        capture_output=True, text=True, cwd=str(ROOT_DIR),
    )
    assert result.returncode == 0, result.stderr
    assert "#PBS -N rag_ollama_serve" in result.stdout

    # Bad input -> non-zero exit, no script emitted
    result = subprocess.run(
        [sys.executable, "-m", "src.hpc", "--ncpus", "0"],
        capture_output=True, text=True, cwd=str(ROOT_DIR),
    )
    assert result.returncode != 0
    assert "must be >=" in result.stderr


def test_gitattributes_enforces_lf_on_shell_scripts():
    """CRLF in *.sh/*.pbs/*.def breaks /bin/bash on the compute node (and the
    LF-only assertions above). .gitattributes must pin these to LF so a Windows
    checkout/commit cannot silently CRLF them."""
    ga = (ROOT_DIR / ".gitattributes").read_text(encoding="utf-8")
    assert re.search(r"\*\.sh\s+text eol=lf", ga)
    assert re.search(r"\*\.pbs\s+text eol=lf", ga)
    assert re.search(r"\*\.def\s+text eol=lf", ga)


def test_requirements_pins_urllib3_explicitly():
    """src/embeddings.py imports urllib3.PoolManager directly; it must not rely
    on transitive availability via requests (a latent supply-chain gap)."""
    req = (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^urllib3\s*~=", req, re.MULTILINE), \
        "urllib3 must be pinned explicitly in requirements.txt"


def test_hpc_main_writes_output_file(tmp_path, capsys):
    """`python -m src.hpc -o file` should write the script to the file."""
    out = tmp_path / "gen.pbs"
    rc = hpc_main(["-o", str(out)])
    assert rc == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "#!/bin/bash" in content
    assert "#PBS -N rag_ingest_index" in content
    # When writing to a file, the script body must NOT also go to stdout.
    captured = capsys.readouterr()
    assert "#!/bin/bash" not in captured.out


# ==============================================================================
# Tests for the dual-mode (GPU + CPU) generator work.
# ==============================================================================


def test_hpc_cpu_generator_omits_gpu_clauses():
    """ngpus=0 must produce a CPU-shaped script: no ':ngpus=' in the select
    clause (some CPU partitions reject the clause even at zero) and no '--nv'
    passed to singularity."""
    cpu = generate_pbs_script(ngpus=0, queue="cpu", container_sif="rag_pipeline_cpu.sif", ncpus=16)
    assert "#PBS -q cpu" in cpu
    assert "rag_pipeline_cpu.sif" in cpu
    # The select line must be GPU-clause-free.
    assert "select=1:ncpus=16:mem=32gb" in cpu
    assert ":ngpus=" not in cpu, "CPU script must not emit :ngpus= clause"
    # No singularity --nv anywhere in the script.
    assert "--nv " not in cpu, "CPU script must not pass --nv to singularity"
    # Pipeline invocation must still be intact.
    assert "python3 scripts/bulk_ingest.py --input-dir \"data\"" in cpu


def test_hpc_gpu_generator_regression_guard():
    """GPU mode (ngpus>0) must keep both the :ngpus= clause and --nv. This is a
    regression guard for the dual-mode refactor that made the clause conditional."""
    gpu = generate_pbs_script(ngpus=2)
    assert "select=1:ncpus=8:mem=32gb:ngpus=2" in gpu
    assert "singularity exec --nv " in gpu

    # Default args still mean GPU (the --cpu bundle is opt-in).
    default = generate_pbs_script()
    assert "select=1:ncpus=8:mem=32gb:ngpus=1" in default
    assert "singularity exec --nv " in default


def test_hpc_cpu_cli_bundle_and_override_semantics():
    """`--cpu` is a default bundle: CPU-shaped output by default, but an explicit
    arg always wins, so `--cpu --ngpus 2` still requests GPUs."""
    from src.hpc import _build_script_from_args

    # Default CPU bundle.
    cpu = _build_script_from_args(parse_hpc_args(["--cpu"]))
    assert ":ngpus=" not in cpu
    assert "--nv " not in cpu
    assert "#PBS -q cpu" in cpu
    assert "rag_pipeline_cpu.sif" in cpu
    assert "select=1:ncpus=16:mem=32gb" in cpu

    # CPU serving bundle.
    cpu_serve = _build_script_from_args(parse_hpc_args(["--cpu", "--serve"]))
    assert ":ngpus=" not in cpu_serve
    assert "--nv " not in cpu_serve
    assert "rag_pipeline_cpu.sif" in cpu_serve
    assert "rag_ollama_serve" in cpu_serve

    # Explicit args override the bundle: --cpu --ngpus 2 -> GPU-shaped.
    mixed = _build_script_from_args(parse_hpc_args(["--cpu", "--ngpus", "2"]))
    assert "select=1:ncpus=16:mem=32gb:ngpus=2" in mixed
    assert "singularity exec --nv " in mixed


def test_singularity_cpu_def_exists_and_valid():
    """A CPU-only container recipe must exist alongside the GPU one, differing
    only in base image + torch wheel index while keeping the same app stack."""
    def_path = ROOT_DIR / "Singularity.cpu.def"
    assert def_path.exists(), "Singularity.cpu.def must exist for CPU-only clusters"

    raw = def_path.read_bytes()
    assert b"\r\n" not in raw, "Singularity.cpu.def MUST use LF line endings"

    content = raw.decode("utf-8")
    assert "Bootstrap: docker" in content
    # CPU base is the official python image (Python 3.14), NOT ubuntu:22.04
    # (Py 3.10) and NOT the CUDA image. ubuntu:22.04's Python 3.10 can't install
    # the pinned onnxruntime~=1.27.0 / numpy~=2.4.6 (both require >=3.11).
    assert "From: python:3.14-slim" in content
    # Must not pull the CUDA base image or the CUDA torch wheel index. (The word
    # "cuda" may legitimately appear in a comment; assert the load-bearing lines.)
    assert "nvidia/cuda" not in content
    assert "whl/cu121" not in content and "whl/cu" not in content
    # build-essential so pip can build any cp314-less transitive dep from source.
    assert "build-essential" in content
    # The python:3.14-slim base ships python3 + pip already, so the apt block
    # must NOT apt-install python3-pip (which would pull the distro's 3.x and
    # shadow 3.14). Check the apt-install block, not comments that mention it.
    apt_block = content.split("apt-get install -y --no-install-recommends")[1].split("rm -rf /var/lib/apt/lists")[0]
    assert "python3-pip" not in apt_block, "CPU image must not apt-install python3-pip (base provides pip for 3.14)"

    # Same %sections as the GPU recipe.
    for section in ("%labels", "%environment", "%post", "%runscript"):
        assert section in content

    # CPU torch wheels (not the cu121 index the GPU recipe uses).
    assert "https://download.pytorch.org/whl/cpu" in content

    # The CPU ingest job embeds via the SoCLAaS API (bge-m3) -- it does NOT need
    # the `ollama` python package or the Ollama binary. urllib3 is the HTTP
    # transport for the API client.
    for dep in ("docling~=2.105.0", "lancedb~=0.33.0",
                "onnxruntime~=1.27.0", "fastapi~=0.138.0", "urllib3~=2.5.0", "python3"):
        assert dep in content, f"CPU recipe missing {dep}"
    assert "ollama~=0.6.2" not in content, "CPU image no longer pins ollama (SoCLAaS API path)"
    # OCR/PDF system deps. zstd is retained for Docling/general use.
    for pkg in ("poppler-utils", "tesseract-ocr", "libgl1", "libglib2.0-0", "zstd"):
        assert pkg in content
    # No Ollama binary install in the CPU image.
    assert "install.sh" not in content
    assert "ollama-linux-amd64.tar.zst" not in content
    assert "export OLLAMA_HOST" not in content
