"""PBS job-script generation for the NUS HPC deployment.

This module programmatically generates the same PBS scripts that live under
``scripts/`` (``nus_hpc_ingest_index.pbs`` and ``nus_hpc_serve.pbs``) so they
can be regenerated with overridden resource parameters without hand-editing the
files.

Security note: every interpolated value is validated against a strict
allow-list *before* it reaches the f-string template. The generated script
interpolates into several fixed literal contexts (an unquoted ``#PBS -N``, a
``mem=`` token inside a select spec, a double-quoted shell path), where naive
``shlex.quote`` would either fail to neutralize a directive-injection attempt
or mangle a safe value into a form the template no longer expects. Rejecting
anything outside the safe charset is both safer and format-preserving.

Run it::

    python -m src.hpc                         # ingest job, default params -> stdout
    python -m src.hpc --serve                 # serving job, default params
    python -m src.hpc --ngpus 2 --mem 64gb    # ingest job, overridden resources
    python -m src.hpc -o myjob.pbs            # write to a file instead
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- Validation allow-lists ---------------------------------------------------
# These are deliberately tight. Loosen only when a legitimate value is rejected.

# PBS job names and queue names: alphanumerics, underscore, dot, dash.
# (PBS itself rejects most punctuation in -N / -q anyway.)
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Memory spec inside the select chunk, e.g. "32gb", "16g", "512mb".
_MEM_RE = re.compile(r"^\d+[gGmM][bB]?$")

# PBS walltime, HH:MM:SS (hours may be 1-4 digits for multi-day jobs).
_WALLTIME_RE = re.compile(r"^\d{1,4}:\d{2}:\d{2}$")

# Filesystem paths interpolated into the template. Allows alphanumerics,
# underscore, dot, dash, forward slash, and the two shell-expansion sigils
# ${} so legitimate defaults like "${HOME}/ollama_models" pass. It still
# blocks command substitution -- $() and backticks -- because '(' ')' and
# '`' are excluded, along with spaces, quotes, ;, &, | and <>.
_PATH_RE = re.compile(r"^[A-Za-z0-9._/${}\-]+$")

# Default model tags the pipeline depends on (see config.toml [models]/*).
DEFAULT_MODELS_TO_PULL = "nomic-embed-text qwen2.5vl:7b gemma4 qwen2.5:1.5b"


def _strip(value: str, field: str) -> str:
    """Remove any embedded newlines/carriage returns and fail loudly if present.

    A newline in job_name/container_sif/queue is the primitive behind
    PBS-directive injection, so we treat it as a hard error rather than silently
    truncating.
    """
    if "\n" in value or "\r" in value:
        raise ValueError(
            f"{field} must not contain newlines (got {value!r}); "
            "this would allow PBS-directive injection in the generated script."
        )
    return value


def _validate_name(value: str, field: str) -> str:
    value = _strip(str(value), field)
    if not value:
        raise ValueError(f"{field} must be a non-empty string")
    if not _NAME_RE.match(value):
        raise ValueError(
            f"{field}={value!r} contains disallowed characters; "
            "only letters, digits, '.', '_' and '-' are permitted."
        )
    return value


def _validate_path(value: str, field: str) -> str:
    value = _strip(str(value), field)
    if not value:
        raise ValueError(f"{field} must be a non-empty string")
    if not _PATH_RE.match(value):
        raise ValueError(
            f"{field}={value!r} contains disallowed characters; "
            "only letters, digits, '.', '_', '-', '/' are permitted."
        )
    return value


def _validate_mem(value: str) -> str:
    value = _strip(str(value), "mem")
    if not _MEM_RE.match(value):
        raise ValueError(
            f"mem={value!r} is not a valid memory spec; expected e.g. '32gb', '16g', '512mb'."
        )
    return value


def _validate_walltime(value: str) -> str:
    value = _strip(str(value), "walltime")
    if not _WALLTIME_RE.match(value):
        raise ValueError(
            f"walltime={value!r} is not a valid HH:MM:SS duration."
        )
    return value


def _validate_int(value, field: str, minimum: int) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}={value!r} is not an integer") from exc
    if ivalue < minimum:
        raise ValueError(f"{field}={ivalue} must be >= {minimum}")
    return ivalue


# --- Resource-clause helpers (dual-mode: GPU vs CPU) -------------------------
#
# The single mode switch is ngpus. ngpus > 0  => GPU job (":ngpus=N" appended to
# the select clause, "--nv" passed to singularity for GPU passthrough). ngpus==0
# => CPU job (no ngpus clause at all -- some CPU partitions reject ":ngpus=0"
# -- and no --nv). This keeps both generators mode-agnostic: callers pick the
# mode by setting ngpus, and the --cpu CLI flag is just a default bundle on top
# (see _build_script_from_args). Explicit args always win, so `--cpu --ngpus 2`
# still requests 2 GPUs.

def _select_clause(ncpus: int, mem: str, ngpus: int) -> str:
    """Build the PBS ``select=`` chunk. Omits ``:ngpus=`` entirely on CPU jobs."""
    base = f"select=1:ncpus={ncpus}:mem={mem}"
    return f"{base}:ngpus={ngpus}" if ngpus > 0 else base


def _nv_flag(ngpus: int) -> str:
    """Return ``"--nv "`` (trailing space) for GPU jobs, empty for CPU.
    Interpolated directly before the bind-mount list in singularity commands."""
    return "--nv " if ngpus > 0 else ""


# --- Ingest/index PBS generator ----------------------------------------------


def generate_pbs_script(
    job_name: str = "rag_ingest_index",
    ncpus: int = 8,
    mem: str = "32gb",
    ngpus: int = 1,
    queue: str = "gpu",
    input_data_dir: str = "data",
    container_sif: str = "rag_pipeline.sif",
    walltime: str = "08:00:00",
    ollama_models_dir: str = "${HOME}/ollama_models",
) -> str:
    """Generate the ingest+index PBS script template with validated overrides.

    The emitted script mirrors ``scripts/nus_hpc_ingest_index.pbs``: it pins the
    Ollama model store to a *persistent* shared path (not the wiped per-job
    scratch), pre-pulls the pipeline's models, then runs ``bulk_ingest.py``.
    """
    job_name = _validate_name(job_name, "job_name")
    ncpus = _validate_int(ncpus, "ncpus", minimum=1)
    mem = _validate_mem(mem)
    ngpus = _validate_int(ngpus, "ngpus", minimum=0)
    queue = _validate_name(queue, "queue")
    input_data_dir = _validate_path(input_data_dir, "input_data_dir")
    container_sif = _validate_path(container_sif, "container_sif")
    walltime = _validate_walltime(walltime)
    ollama_models_dir = _validate_path(ollama_models_dir, "ollama_models_dir")

    select_clause = _select_clause(ncpus, mem, ngpus)
    nv = _nv_flag(ngpus)  # "--nv " on GPU, "" on CPU

    return f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l {select_clause}
#PBS -l walltime={walltime}
#PBS -q {queue}
#PBS -j oe

set -e

module load singularity

USER="${{USER:-$(whoami)}}"
PBS_JOBID="${{PBS_JOBID:-local_job}}"
HOME="${{HOME:-$(eval echo ~${{USER}})}}"
SCRATCH_DIR="/hpctmp2/${{USER}}/rag_scratch_${{PBS_JOBID}}"
OLLAMA_MODELS_DIR="${{OLLAMA_MODELS_DIR:-{ollama_models_dir}}}"

mkdir -p "${{SCRATCH_DIR}}"
mkdir -p "${{SCRATCH_DIR}}/tmp"
mkdir -p "${{OLLAMA_MODELS_DIR}}"

OLLAMA_PID=""
cleanup() {{
    echo "Stopping background services and cleaning up scratch directory..."
    if [ -n "${{OLLAMA_PID}}" ]; then
        kill -9 "${{OLLAMA_PID}}" 2>/dev/null || true
    fi
    rm -rf "${{SCRATCH_DIR}}"
    echo "Cleanup complete."
}}
trap cleanup EXIT

CONTAINER_SIF="${{CONTAINER_SIF:-{container_sif}}}"
if [ ! -f "${{CONTAINER_SIF}}" ] && [ -f "/hpctmp2/${{USER}}/{container_sif}" ]; then
    CONTAINER_SIF="/hpctmp2/${{USER}}/{container_sif}"
fi

BIND_MOUNTS="-B /hpctmp2/${{USER}}:/hpctmp2/${{USER}} -B ${{HOME}}:/srv/home -B ${{PWD}}:/app -B ${{OLLAMA_MODELS_DIR}}:/srv/ollama_models"

export OLLAMA_MODELS="/srv/ollama_models"
export TMPDIR="${{SCRATCH_DIR}}/tmp"

singularity exec {nv}${{BIND_MOUNTS}} "${{CONTAINER_SIF}}" ollama serve > "${{SCRATCH_DIR}}/ollama.log" 2>&1 &
OLLAMA_PID=$!

MAX_ATTEMPTS=30
ATTEMPT=0
READY=0
while [ ${{ATTEMPT}} -lt ${{MAX_ATTEMPTS}} ]; do
    if singularity exec ${{BIND_MOUNTS}} "${{CONTAINER_SIF}}" curl -s -f http://127.0.0.1:11434/api/version > /dev/null 2>&1; then
        READY=1
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    sleep 2
done
if [ ${{READY}} -ne 1 ]; then
    echo "ERROR: Ollama endpoint failed to become ready within timeout."
    cat "${{SCRATCH_DIR}}/ollama.log" || true
    exit 1
fi

DEFAULT_MODELS="{DEFAULT_MODELS_TO_PULL}"
OLLAMA_MODELS_TO_PULL="${{OLLAMA_MODELS_TO_PULL:-${{DEFAULT_MODELS}}}}"
PULL_FAIL=0
for model in ${{OLLAMA_MODELS_TO_PULL}}; do
    if ! singularity exec ${{BIND_MOUNTS}} "${{CONTAINER_SIF}}" ollama pull "${{model}}"; then
        echo "ERROR: ollama pull failed for '${{model}}'"
        PULL_FAIL=1
    fi
done
if [ ${{PULL_FAIL}} -ne 0 ]; then
    exit 1
fi

INPUT_DATA_DIR="${{INPUT_DATA_DIR:-{input_data_dir}}}"
singularity exec {nv}${{BIND_MOUNTS}} "${{CONTAINER_SIF}}" python3 scripts/bulk_ingest.py --input-dir "{input_data_dir}"
"""


# --- Serving PBS generator ----------------------------------------------------


def generate_serve_pbs_script(
    job_name: str = "rag_ollama_serve",
    ncpus: int = 4,
    mem: str = "16gb",
    ngpus: int = 1,
    queue: str = "gpu",
    walltime: str = "08:00:00",
    container_sif: str = "rag_pipeline.sif",
    ollama_models_dir: str = "${HOME}/ollama_models",
    ollama_host_file: str = "${HOME}/.rag_ollama_serving_host",
) -> str:
    """Generate the long-lived Ollama *serving* PBS script.

    Unlike the ingest job, this one does not run the pipeline: it keeps
    ``ollama serve`` alive on a GPU compute node, publishes that node's hostname
    to ``ollama_host_file`` (read by the SSH tunnel daemon), pre-pulls the chat
    models, and blocks until walltime. See ``scripts/nus_hpc_serve.pbs``.
    """
    job_name = _validate_name(job_name, "job_name")
    ncpus = _validate_int(ncpus, "ncpus", minimum=1)
    mem = _validate_mem(mem)
    ngpus = _validate_int(ngpus, "ngpus", minimum=0)
    queue = _validate_name(queue, "queue")
    walltime = _validate_walltime(walltime)
    container_sif = _validate_path(container_sif, "container_sif")
    ollama_models_dir = _validate_path(ollama_models_dir, "ollama_models_dir")
    ollama_host_file = _validate_path(ollama_host_file, "ollama_host_file")

    select_clause = _select_clause(ncpus, mem, ngpus)
    nv = _nv_flag(ngpus)  # "--nv " on GPU, "" on CPU

    return f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l {select_clause}
#PBS -l walltime={walltime}
#PBS -q {queue}
#PBS -j oe

set -e

module load singularity

USER="${{USER:-$(whoami)}}"
PBS_JOBID="${{PBS_JOBID:-local_job}}"
HOME="${{HOME:-$(eval echo ~${{USER}})}}"
SCRATCH_DIR="/hpctmp2/${{USER}}/rag_serve_scratch_${{PBS_JOBID}}"
OLLAMA_MODELS_DIR="${{OLLAMA_MODELS_DIR:-{ollama_models_dir}}}"
OLLAMA_HOST_FILE="${{OLLAMA_HOST_FILE:-{ollama_host_file}}}"

mkdir -p "${{SCRATCH_DIR}}/tmp"
mkdir -p "${{OLLAMA_MODELS_DIR}}"

COMPUTE_HOST="$(hostname -f 2>/dev/null || hostname)"
echo "${{COMPUTE_HOST}}" > "${{OLLAMA_HOST_FILE}}"

OLLAMA_PID=""
cleanup() {{
    if [ -n "${{OLLAMA_PID}}" ]; then
        kill -9 "${{OLLAMA_PID}}" 2>/dev/null || true
    fi
    rm -f "${{OLLAMA_HOST_FILE}}"
    rm -rf "${{SCRATCH_DIR}}"
}}
trap cleanup EXIT

CONTAINER_SIF="${{CONTAINER_SIF:-{container_sif}}}"
if [ ! -f "${{CONTAINER_SIF}}" ] && [ -f "/hpctmp2/${{USER}}/{container_sif}" ]; then
    CONTAINER_SIF="/hpctmp2/${{USER}}/{container_sif}"
fi

BIND_MOUNTS="-B /hpctmp2/${{USER}}:/hpctmp2/${{USER}} -B ${{HOME}}:/srv/home -B ${{PWD}}:/app -B ${{OLLAMA_MODELS_DIR}}:/srv/ollama_models"

export OLLAMA_MODELS="/srv/ollama_models"
export TMPDIR="${{SCRATCH_DIR}}/tmp"

singularity exec {nv}${{BIND_MOUNTS}} "${{CONTAINER_SIF}}" ollama serve > "${{SCRATCH_DIR}}/ollama.log" 2>&1 &
OLLAMA_PID=$!

MAX_ATTEMPTS=30
ATTEMPT=0
READY=0
while [ ${{ATTEMPT}} -lt ${{MAX_ATTEMPTS}} ]; do
    if singularity exec ${{BIND_MOUNTS}} "${{CONTAINER_SIF}}" curl -s -f http://127.0.0.1:11434/api/version > /dev/null 2>&1; then
        READY=1
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    sleep 2
done
if [ ${{READY}} -ne 1 ]; then
    echo "ERROR: Ollama endpoint failed to become ready within timeout."
    cat "${{SCRATCH_DIR}}/ollama.log" || true
    exit 1
fi

DEFAULT_MODELS="{DEFAULT_MODELS_TO_PULL}"
OLLAMA_MODELS_TO_PULL="${{OLLAMA_MODELS_TO_PULL:-${{DEFAULT_MODELS}}}}"
for model in ${{OLLAMA_MODELS_TO_PULL}}; do
    singularity exec ${{BIND_MOUNTS}} "${{CONTAINER_SIF}}" ollama pull "${{model}}" || \
        echo "WARNING: pull failed for '${{model}}'"
done

KEEPALIVE_INTERVAL=15
while true; do
    sleep "${{KEEPALIVE_INTERVAL}}"
    echo "${{COMPUTE_HOST}}" > "${{OLLAMA_HOST_FILE}}"
done
"""


# --- CLI ----------------------------------------------------------------------


def parse_hpc_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for PBS generation.

    ``--serve`` selects the long-lived Ollama serving template; otherwise the
    ingest+index template is generated. ``--cpu`` switches the default resource
    bundle to a CPU-only profile (0 GPUs, ``cpu`` queue, ``rag_pipeline_cpu.sif``,
    more cores). Explicit args always override ``--cpu``'s bundle. ``--output``
    writes the result to a file instead of stdout.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.hpc",
        description="Generate a PBS job script for the NUS HPC deployment.",
    )
    parser.add_argument("--serve", action="store_true", help="Generate the long-lived Ollama serving job (default: ingest+index).")
    parser.add_argument("--cpu", action="store_true", help="Generate a CPU-only job (0 GPUs, 'cpu' queue, rag_pipeline_cpu.sif). Explicit args still win.")
    parser.add_argument("--job-name", default=None, help="PBS job name (default: rag_ingest_index / rag_ollama_serve).")
    parser.add_argument("--ncpus", type=int, default=None, help="Number of CPUs (default: 8 ingest / 4 serve; 16/8 under --cpu).")
    parser.add_argument("--mem", default=None, help="Memory requirement, e.g. 32gb (default: 32gb ingest / 16gb serve).")
    parser.add_argument("--ngpus", type=int, default=None, help="Number of GPUs (default: 1; 0 under --cpu).")
    parser.add_argument("--queue", default=None, help="PBS queue name (default: gpu; cpu under --cpu).")
    parser.add_argument("--walltime", default=None, help="Job walltime HH:MM:SS (default: 08:00:00).")
    parser.add_argument("--container-sif", default=None, help="Singularity image filename (default: rag_pipeline.sif; rag_pipeline_cpu.sif under --cpu).")
    parser.add_argument("--ollama-models-dir", default=None, help="Persistent Ollama model store path (default: $HOME/ollama_models).")
    parser.add_argument("--ollama-host-file", dest="ollama_host_file", default=None, help="Serving discovery file (serving job only; default: $HOME/.rag_ollama_serving_host).")
    parser.add_argument("--input-data-dir", default=None, help="Ingest input directory (ingest job only; default: data).")
    parser.add_argument("-o", "--output", default=None, help="Write the generated script to this file (default: stdout).")
    return parser.parse_args(argv)


def _build_script_from_args(args: argparse.Namespace) -> str:
    """Dispatch to the selected generator, applying argparse defaults per mode.

    ``--cpu`` is a *default bundle*: when set, it changes the defaults for
    ngpus/queue/container_sif/ncpus to CPU-appropriate values, but only for
    fields the caller did not explicitly pass. So ``--cpu --ngpus 2`` still
    requests 2 GPUs (explicit arg wins), and ``--cpu --queue bigmem`` keeps the
    caller's queue.
    """
    if args.cpu:
        bundle = dict(
            ngpus=0 if args.ngpus is None else args.ngpus,
            queue="cpu" if args.queue is None else args.queue,
            container_sif="rag_pipeline_cpu.sif" if args.container_sif is None else args.container_sif,
            # CPU nodes are often core-rich; default to more workers than the
            # GPU defaults. Caller can override with --ncpus as usual.
            ncpus=16 if args.ncpus is None else args.ncpus,
            mem="32gb" if args.mem is None else args.mem,
        )
    else:
        bundle = dict(
            ngpus=1 if args.ngpus is None else args.ngpus,
            queue="gpu" if args.queue is None else args.queue,
            container_sif="rag_pipeline.sif" if args.container_sif is None else args.container_sif,
            ncpus=args.ncpus,
            mem=args.mem,
        )

    if args.serve:
        return generate_serve_pbs_script(
            job_name=args.job_name or "rag_ollama_serve",
            ncpus=bundle["ncpus"] if bundle["ncpus"] is not None else 4,
            mem=bundle["mem"] or "16gb",
            ngpus=bundle["ngpus"],
            queue=bundle["queue"],
            walltime=args.walltime or "08:00:00",
            container_sif=bundle["container_sif"],
            ollama_models_dir=args.ollama_models_dir or "${HOME}/ollama_models",
            ollama_host_file=args.ollama_host_file or "${HOME}/.rag_ollama_serving_host",
        )
    return generate_pbs_script(
        job_name=args.job_name or "rag_ingest_index",
        ncpus=bundle["ncpus"] if bundle["ncpus"] is not None else 8,
        mem=bundle["mem"] or "32gb",
        ngpus=bundle["ngpus"],
        queue=bundle["queue"],
        walltime=args.walltime or "08:00:00",
        container_sif=bundle["container_sif"],
        ollama_models_dir=args.ollama_models_dir or "${HOME}/ollama_models",
        input_data_dir=args.input_data_dir or "data",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_hpc_args(argv)
    try:
        script = _build_script_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        Path(args.output).write_text(script, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(script)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    raise SystemExit(main())
