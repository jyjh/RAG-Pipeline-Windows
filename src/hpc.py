"""PBS job-script generation for the NUS HPC deployment.

This module programmatically generates the same PBS script that lives under
``scripts/`` (``nus_hpc_ingest_index.pbs``) so it can be regenerated with
overridden resource parameters without hand-editing the file.

Security note: every interpolated value is validated against a strict
allow-list *before* it reaches the f-string template. The generated script
interpolates into several fixed literal contexts (an unquoted ``#PBS -N``, a
``mem=`` token inside a select spec, a double-quoted shell path), where naive
``shlex.quote`` would either fail to neutralize a directive-injection attempt
or mangle a safe value into a form the template no longer expects. Rejecting
anything outside the safe charset is both safer and format-preserving.

Run it::

    python -m src.hpc                         # ingest job, default params -> stdout
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


def _is_under(path: str, root: str) -> bool:
    """True when posix ``path`` equals or lies under posix ``root``."""
    if path == root:
        return True
    return path.startswith(root.rstrip("/") + "/")


def _extra_bind_mounts(storage_root: str, *dirs: str) -> str:
    """1:1 bind flags for absolute dirs the container must see at the same path.

    Only absolute paths need binding (repo-relative dirs already live under the
    ``${PWD}:/app`` bind), and anything under ``storage_root`` is already covered
    by its own bind. Order is preserved and duplicates collapse, so generated
    scripts stay deterministic for tests.
    """
    flags: list[str] = []
    seen = {"/", storage_root.rstrip("/")}
    for value in dirs:
        normalized = value.rstrip("/")
        if not normalized.startswith("/") or normalized in seen:
            continue
        if _is_under(normalized, storage_root.rstrip("/")):
            continue
        seen.add(normalized)
        flags.append(f"-B {normalized}:{normalized}")
    return " ".join(flags)


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
    processed_dir: str = "processed_docs",
    container_sif: str = "rag_pipeline.sif",
    walltime: str = "08:00:00",
    storage_root: str = "/hpctmp/${USER}",
    skip_index: bool = False,
) -> str:
    """Generate the ingest (+index) PBS script template with validated overrides.

    The CPU job does the heavy non-LLM work (Docling OCR/parsing, chunking,
    LanceDB writes, ANN build) and calls the hosted SoCLAaS API
    (``/v1/embeddings``) for vectors; it runs no local ``ollama serve``.
    Provision the API key at ``~/rag_soclaas_key`` on the login node
    (chmod 600), pass it via the qsub environment, or set
    ``[llm_api].api_key`` in the staged ``config.toml`` so the job can reach
    the embeddings endpoint.

    ``input_data_dir`` and ``processed_dir`` may be repo-relative (resolved
    against the qsub working directory, i.e. the repo bound at /app) or
    absolute. Absolute dirs must be visible inside the container at the SAME
    path, so they get their own 1:1 bind mount (unless already covered by the
    ``storage_root`` bind). This is what lets the corpus live OUTSIDE the
    provision-swapped repo directory under e.g. ``/hpctmp/<user>/rag-corpus``.

    With ``skip_index=True`` the job runs ``bulk_ingest.py --skip-index``: it
    stops after ingestion (Markdown under ``processed_dir``) and never calls
    the embeddings endpoint -- the index is then built locally with a local
    embedding model (see ``HpcBackend.fetch_processed_docs``). The SoCLAaS key
    remains needed only when vision enrichment is enabled in the staged
    ``[ingestion]`` config.
    """
    job_name = _validate_name(job_name, "job_name")
    ncpus = _validate_int(ncpus, "ncpus", minimum=1)
    mem = _validate_mem(mem)
    ngpus = _validate_int(ngpus, "ngpus", minimum=0)
    queue = _validate_name(queue, "queue")
    input_data_dir = _validate_path(input_data_dir, "input_data_dir")
    processed_dir = _validate_path(processed_dir, "processed_dir")
    container_sif = _validate_path(container_sif, "container_sif")
    walltime = _validate_walltime(walltime)
    storage_root = _validate_path(storage_root, "storage_root")

    select_clause = _select_clause(ncpus, mem, ngpus)
    nv = _nv_flag(ngpus)  # "--nv " on GPU, "" on CPU
    skip_flag = " --skip-index" if skip_index else ""
    extra_binds = _extra_bind_mounts(storage_root, input_data_dir, processed_dir)
    # Keep BIND_MOUNTS a single interpolatable token: the storage/home/app binds
    # are fixed, the corpus binds vary per invocation.
    bind_mounts = (
        f"-B ${{STORAGE_ROOT}}:${{STORAGE_ROOT}} -B ${{HOME}}:/srv/home -B ${{PWD}}:/app"
        + (f" {extra_binds}" if extra_binds else "")
    )

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
STORAGE_ROOT="{storage_root}"
SCRATCH_DIR="${{STORAGE_ROOT}}/rag_scratch_${{PBS_JOBID}}"

mkdir -p "${{SCRATCH_DIR}}"
mkdir -p "${{SCRATCH_DIR}}/tmp"

cleanup() {{
    echo "Cleaning up scratch directory..."
    rm -rf "${{SCRATCH_DIR}}"
    echo "Cleanup complete."
}}
trap cleanup EXIT

CONTAINER_SIF="${{CONTAINER_SIF:-{container_sif}}}"
if [ ! -f "${{CONTAINER_SIF}}" ] && [ -f "${{STORAGE_ROOT}}/{container_sif}" ]; then
    CONTAINER_SIF="${{STORAGE_ROOT}}/{container_sif}"
fi

BIND_MOUNTS="{bind_mounts}"

export TMPDIR="${{SCRATCH_DIR}}/tmp"

# SoCLAaS API key. Needed for /v1/embeddings when the job also indexes, and
# for vision enrichment of figures (qwen3-vl) whenever [ingestion]
# vision_enabled is true -- including ingest-only (--skip-index) runs.
# Provision it at ~/rag_soclaas_key on the login node (chmod 600), pass it
# via the qsub environment, or set [llm_api].api_key in the staged config.toml.
if [ -z "${{SOCLAAS_API_KEY:-}}" ] && [ -f "${{HOME}}/rag_soclaas_key" ]; then
    export SOCLAAS_API_KEY="$(cat "${{HOME}}/rag_soclaas_key")"
fi

# Both dirs honor their environment variables at qsub time (qsub -v). The
# defaults are baked in by the generator, so the script is self-contained.
INPUT_DATA_DIR="${{INPUT_DATA_DIR:-{input_data_dir}}}"
PROCESSED_DIR="${{PROCESSED_DIR:-{processed_dir}}}"
mkdir -p "${{PROCESSED_DIR}}"
singularity exec {nv}${{BIND_MOUNTS}} "${{CONTAINER_SIF}}" python3 scripts/bulk_ingest.py \\
    --input-dir "${{INPUT_DATA_DIR}}" \\
    --processed-dir "${{PROCESSED_DIR}}"{skip_flag}
"""


# --- CLI ----------------------------------------------------------------------


def parse_hpc_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for PBS generation.

    ``--cpu`` switches the default resource bundle to a CPU-only profile
    (0 GPUs, ``cpu`` queue, ``rag_pipeline_cpu.sif``, more cores). Explicit
    args always override ``--cpu``'s bundle. ``--output`` writes the result to
    a file instead of stdout.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.hpc",
        description="Generate a PBS job script for the NUS HPC deployment.",
    )
    parser.add_argument("--cpu", action="store_true", help="Generate a CPU-only job (0 GPUs, 'cpu' queue, rag_pipeline_cpu.sif). Explicit args still win.")
    parser.add_argument("--skip-index", action="store_true", help="Ingest-only job: run bulk_ingest.py --skip-index so the cluster produces processed_docs/ Markdown without embedding; index locally afterwards.")
    parser.add_argument("--job-name", default=None, help="PBS job name (default: rag_ingest_index).")
    parser.add_argument("--ncpus", type=int, default=None, help="Number of CPUs (default: 8; 16 under --cpu).")
    parser.add_argument("--mem", default=None, help="Memory requirement, e.g. 32gb (default: 32gb).")
    parser.add_argument("--ngpus", type=int, default=None, help="Number of GPUs (default: 1; 0 under --cpu).")
    parser.add_argument("--queue", default=None, help="PBS queue name (default: gpu; cpu under --cpu).")
    parser.add_argument("--walltime", default=None, help="Job walltime HH:MM:SS (default: 08:00:00).")
    parser.add_argument("--container-sif", default=None, help="Singularity image filename (default: rag_pipeline.sif; rag_pipeline_cpu.sif under --cpu).")
    parser.add_argument("--storage-root", default=None, help="Per-cluster scratch/storage root (CPU default: /hpctmp/$USER).")
    parser.add_argument("--input-data-dir", default=None, help="Ingest input directory (default: data; repo-relative or absolute).")
    parser.add_argument("--processed-dir", default=None, help="Ingest output directory for processed Markdown (default: processed_docs; repo-relative or absolute).")
    parser.add_argument("-o", "--output", default=None, help="Write the generated script to this file (default: stdout).")
    return parser.parse_args(argv)


def _build_script_from_args(args: argparse.Namespace) -> str:
    """Apply argparse defaults per mode and generate the ingest+index script.

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

    return generate_pbs_script(
        job_name=args.job_name or "rag_ingest_index",
        ncpus=bundle["ncpus"] if bundle["ncpus"] is not None else 8,
        mem=bundle["mem"] or "32gb",
        ngpus=bundle["ngpus"],
        queue=bundle["queue"],
        walltime=args.walltime or "08:00:00",
        container_sif=bundle["container_sif"],
        input_data_dir=args.input_data_dir or "data",
        processed_dir=args.processed_dir or "processed_docs",
        storage_root=args.storage_root or "/hpctmp/${USER}",
        skip_index=bool(args.skip_index),
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
