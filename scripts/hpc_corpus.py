"""One-command initial corpus deployment: zip -> HPC parse -> local index.

Complements ``setup_instance.py`` (which provisions the cluster and serves):
this module implements the ``--initial-corpus`` step of the initial
deployment. Given a zip of PDFs -- nested directories are fine -- it:

1. extracts the zip locally under ``data/corpus/<name>/`` (structure
   preserved; the pipeline's discovery is recursive and collision-safe);
2. verifies the prerequisites that would otherwise waste a multi-hour job
   (a usable SoCLAaS key for vision enrichment, local Ollama + embedding
   model for the index build, and that the web server is NOT running);
3. uploads the corpus to the cluster OUTSIDE the provision-swapped repo dir
   (``[hpc].remote_data_dir``, an absolute path under /hpctmp/<user>),
   submits the ingest-only PBS job via ``HpcBackend`` (which polls qstat,
   relays progress, and verifies the job's exit code), and fetches the
   processed Markdown home;
4. verifies the fetched corpus against the zip and reports per-file
   failures from ``.ingest_result.json``;
5. builds the local LanceDB index with ``main.py --mode index``.

Every cluster interaction goes through ``HpcBackend`` so tests can mock it,
and no third-party imports are required at module scope (``src.config`` /
``src.hpc_backend`` are stdlib-only) -- the flow runs before the project venv
is guaranteed to exist. Only the final index build needs the runtime venv.

Standalone use::

    python scripts/hpc_corpus.py --zip corpus.zip [--skip-index-build]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.fsae_ev_filter import filter_pdf_list
from src.hpc_backend import HpcBackend, HpcError

# NOTE: keep this module's import chain stdlib-only (src.config/src.hpc_backend
# qualify) so the flow can run before the project venv exists. Do not import
# src.ingestion (Docling parser stack) or src.pdf_registry (portalocker).

# Mirrors src.ingestion.INGEST_RESULT_FILENAME (kept literal here so this
# module stays importable before the project venv exists -- src.ingestion
# transitively imports the Docling parser stack).
INGEST_RESULT_FILENAME = ".ingest_result.json"

# Mirrors web_app.MAX_ZIP_ENTRIES: bounds a zip-bomb of tiny entries.
MAX_ZIP_ENTRIES = 10_000
# Characters Windows filenames cannot contain (zips built elsewhere may).
_UNSAFE_COMPONENT_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


class CorpusError(RuntimeError):
    """A user-correctable failure in the initial-corpus flow."""


def _log_line(text: str) -> None:
    print(str(text).rstrip(), flush=True)


def _sanitize_component(part: str) -> str:
    cleaned = _UNSAFE_COMPONENT_CHARS.sub("_", part).strip()
    return cleaned or "_"


def _safe_zip_relpath(name: str) -> PurePosixPath | None:
    """Sanitized relative path for one zip entry, or None if unsafe/junk.

    Mirrors the web-upload guards: rejects absolute paths and ``..``
    components (path traversal) and skips macOS/Windows junk files.
    """
    normalized = name.replace("\\", "/")
    if not normalized or normalized.endswith("/"):
        return None
    if normalized.startswith("/"):
        return None
    parts = [p for p in normalized.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    if any(p.startswith("__MACOSX") or p in (".DS_Store", "Thumbs.db") for p in parts):
        return None
    return PurePosixPath(*(_sanitize_component(p) for p in parts))


def extract_corpus_zip(
    zip_path: Path,
    dest_dir: Path,
    *,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_bytes: int = 0,
    log: Callable[[str], None] = _log_line,
) -> dict[str, Any]:
    """Extract a corpus zip PRESERVING directory structure; return a summary.

    Structure preservation is deliberate: the pipeline discovers PDFs
    recursively and already disambiguates duplicate Markdown stems with a path
    hash, whereas flattening (the old ``unzip -o -j`` recipe) silently
    overwrites same-named PDFs from different directories.
    """
    if not zip_path.is_file():
        raise CorpusError(f"Zip file not found: {zip_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "pdfs": [],
        "skipped_non_pdf": [],
        "skipped_unsafe": [],
        "dest": dest_dir,
    }
    try:
        with zipfile.ZipFile(zip_path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if len(infos) > max_entries:
                raise CorpusError(
                    f"Zip contains {len(infos)} entries, exceeding the cap of "
                    f"{max_entries}. Split the archive."
                )
            used_paths: set[str] = set()
            for info in infos:
                rel = _safe_zip_relpath(info.filename)
                if rel is None:
                    summary["skipped_unsafe"].append(info.filename)
                    continue
                if not rel.name.lower().endswith(".pdf"):
                    summary["skipped_non_pdf"].append(info.filename)
                    continue
                # De-duplicate case-insensitively: extraction may target a
                # case-insensitive filesystem (Windows/macOS) where two zip
                # entries differing only by case would collide.
                key = str(rel).casefold()
                unique = rel
                counter = 1
                while key in used_paths:
                    unique = rel.with_name(f"{rel.stem}__{counter}.pdf")
                    key = f"{str(rel).casefold()}#{counter}"
                    counter += 1
                used_paths.add(key)
                destination = dest_dir.joinpath(*unique.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                total = 0
                with archive.open(info, "r") as src, destination.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            dst.close()
                            destination.unlink(missing_ok=True)
                            raise CorpusError(
                                f"Zip entry '{info.filename}' exceeds the per-file "
                                f"size limit ({total} > {max_bytes} bytes)."
                            )
                        dst.write(chunk)
                summary["pdfs"].append(unique)
    except zipfile.BadZipFile as exc:
        raise CorpusError(f"Not a valid zip archive: {exc}") from exc

    log(f"Extracted {len(summary['pdfs'])} PDF(s) to {dest_dir}")
    if summary["skipped_non_pdf"]:
        log(f"  Skipped {len(summary['skipped_non_pdf'])} non-PDF entries.")
    if summary["skipped_unsafe"]:
        shown = ", ".join(summary["skipped_unsafe"][:5])
        log(f"  Skipped {len(summary['skipped_unsafe'])} unsafe/junk entries (e.g. {shown}).")
    return summary


# --- Preflight checks ---------------------------------------------------------


def _effective_embeddings_backend(config: Any) -> str:
    """Mirror ``src.embeddings.resolve_embeddings_backend`` (stdlib-only)."""
    env = os.environ.get("EMBEDDINGS_BACKEND", "").strip().lower()
    if env:
        return env
    value = str(getattr(config.embeddings, "backend", "") or "").strip().lower()
    if value not in ("", "soclaas", "ollama"):
        value = "ollama"
    return value or str(getattr(config.llm_api, "backend", "soclaas")).strip().lower()


def check_soclaas_vision_key(config: Any, backend: HpcBackend) -> None:
    """Fail fast when a vision-enabled parse has no usable SoCLAaS key.

    Without a key the cluster job still exits 0 while every figure
    description silently becomes ``[Image description failed]`` -- a
    multi-hour parse degraded with no error anywhere.
    """
    if not getattr(config.ingestion, "vision_enabled", False):
        return
    if str(getattr(config.llm_api, "backend", "soclaas")).lower() != "soclaas":
        return  # local Ollama vision path does not need the API key
    key_env = str(getattr(config.llm_api, "key_env", "SOCLAAS_API_KEY"))
    for name in (key_env, "LLM_API_KEY"):
        if os.environ.get(name, "").strip():
            return
    if str(getattr(config.llm_api, "api_key", "")).strip():
        return
    if backend.remote_file_nonempty("~/rag_soclaas_key"):
        return
    raise CorpusError(
        "Vision enrichment is enabled ([ingestion] vision_enabled = true) but "
        "no SoCLAaS API key is reachable. The cluster parse would silently "
        "degrade every figure description. Provide the key one of these ways:\n"
        "  * run setup with --set-api-key <key> (stored in config.toml), or\n"
        "  * export SOCLAAS_API_KEY before launching, or\n"
        "  * place the key at ~/rag_soclaas_key on the cluster login node "
        "(chmod 600).\n"
        "To proceed anyway with degraded descriptions, re-run with "
        "--allow-degraded-vision."
    )


def _http_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def check_local_ollama(config: Any) -> None:
    """Local Ollama + the configured embedding model are required pre-index.

    Embeddings run on the workstation (default ``all-minilm`` via Ollama);
    checking BEFORE the cluster job avoids parsing for hours only to fail at
    the first embedding call.
    """
    if _effective_embeddings_backend(config) != "ollama":
        return
    host = str(getattr(config.ollama, "host", "http://127.0.0.1:11434")).rstrip("/")
    try:
        _http_json(f"{host}/api/version")
    except (OSError, urllib.error.URLError) as exc:
        raise CorpusError(
            f"Local Ollama is not reachable at {host} but embeddings depend on "
            f"it. Start Ollama (https://ollama.com) and re-run. ({exc})"
        ) from exc
    model = str(getattr(config.models, "embedding_model", "all-minilm"))
    try:
        tags = _http_json(f"{host}/api/tags")
        names = {
            str(item.get("name", "")).split(":")[0]
            for item in (tags.get("models") or [])
            if isinstance(item, dict)
        }
    except (OSError, ValueError, urllib.error.URLError):
        return  # tags endpoint optional; version probe already passed
    if names and model not in names:
        raise CorpusError(
            f"Ollama is running but the embedding model '{model}' is not "
            f"installed. Run: ollama pull {model}"
        )


def _web_server_running(config: Any) -> bool:
    host = str(getattr(config.server, "host", "127.0.0.1"))
    port = int(getattr(config.server, "port", 8000))
    bind = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with urllib.request.urlopen(
            f"http://{bind}:{port}/api/health", timeout=2.0
        ) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


# --- Flow ---------------------------------------------------------------------


def _runtime_python() -> Path:
    candidate = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return candidate if candidate.exists() else Path(sys.executable)


def _verify_fetched_corpus(
    extracted: list[PurePosixPath], processed_dir: Path, log: Callable[[str], None]
) -> bool:
    """Compare the fetched Markdown against the extracted PDFs.

    Returns True when the corpus looks complete (or only partially failed with
    per-file reasons reported); returns False when nothing parsed at all.
    """
    result_path = processed_dir / INGEST_RESULT_FILENAME
    failed: list[Any] = []
    processed_count = skipped_count = failed_count = 0
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            processed_count = len(payload.get("processed") or [])
            skipped_count = len(payload.get("skipped") or [])
            failed = payload.get("failed") or []
            failed_count = len(failed)
        except (OSError, ValueError):
            log(f"  Warning: could not parse {result_path.name}; counts unavailable.")
    else:
        log(f"  Note: {result_path.name} not found (older deployment?); falling back to file counts.")

    total_pdfs = len(extracted)
    md_count = sum(1 for _ in processed_dir.glob("*.md"))
    log(
        f"Corpus verification: {total_pdfs} PDF(s) in the zip; cluster result: "
        f"{processed_count} processed, {skipped_count} skipped (already parsed), "
        f"{failed_count} failed; {md_count} Markdown file(s) now in {processed_dir}."
    )
    for entry in failed[:20]:
        if isinstance(entry, dict):
            log(f"  failed: {entry.get('file', '?')} - {entry.get('error', '')}")
    if failed_count > 20:
        log(f"  ... and {failed_count - 20} more failure(s).")
    if total_pdfs and processed_count + skipped_count < total_pdfs and not result_path.is_file():
        log(
            "  Warning: fewer Markdown files than PDFs. If the PBS job hit its "
            "walltime, raise [hpc.cpu.pbs_overrides] walltime and re-run "
            "(already-parsed PDFs are skipped, so a re-run is cheap)."
        )
    if total_pdfs and processed_count == 0 and skipped_count == 0:
        if md_count == 0:
            log("  ERROR: no Markdown was produced for any PDF.")
            return False
    return True


def run_initial_corpus(
    zip_path: str | Path,
    *,
    config_path: str | Path | None = None,
    allow_degraded_vision: bool = False,
    skip_index_build: bool = False,
    log: Callable[[str], None] = _log_line,
) -> int:
    zip_path = Path(zip_path).expanduser().resolve()
    resolved_config = Path(
        config_path or os.environ.get("RAG_PIPELINE_CONFIG") or (ROOT / "config.toml")
    ).resolve()
    # Child processes (main.py index build) must see the same config.
    os.environ["RAG_PIPELINE_CONFIG"] = str(resolved_config)

    config = load_config(resolved_config)
    if not config.hpc.enabled:
        raise CorpusError(
            "HPC mode is not enabled in config.toml. Run the guided setup once "
            "(setup.cmd / ./setup.sh, mode 'hpc') before using --initial-corpus."
        )
    if not zip_path.is_file() or zip_path.suffix.lower() != ".zip":
        raise CorpusError(f"Initial corpus must be an existing .zip file: {zip_path}")

    # 1. Extract locally, structure preserved.
    corpus_name = re.sub(r"[^A-Za-z0-9._-]+", "_", zip_path.stem).strip("._")[:60] or "corpus"
    local_corpus = ROOT / "data" / "corpus" / corpus_name
    if local_corpus.exists() and any(local_corpus.iterdir()):
        log(
            f"Note: {local_corpus} already has content; this run adds to it "
            "(already-parsed PDFs are skipped cluster-side)."
        )
    summary = extract_corpus_zip(zip_path, local_corpus, log=log)
    if not summary["pdfs"]:
        raise CorpusError(f"No PDF files were found inside {zip_path.name}.")

    # FSAE-EV relevance filter: drop combustion/outdated material BEFORE
    # spending cluster hours parsing it. Names + reasons go to the shared log.
    extracted_paths = [local_corpus / rel for rel in summary["pdfs"]]
    kept_paths, ignored = filter_pdf_list(
        extracted_paths,
        root=local_corpus,
        log_file=ROOT / "logs" / "ignored_documents.log",
    )
    if ignored:
        for path, _decision in ignored:
            Path(path).unlink(missing_ok=True)
        summary["pdfs"] = [p.relative_to(local_corpus) for p in kept_paths]
        log(
            f"FSAE-EV filter: ignored {len(ignored)} of {len(extracted_paths)} "
            "PDF(s); see logs/ignored_documents.log"
        )
    if not summary["pdfs"]:
        raise CorpusError(
            "No PDF files remained after the FSAE-EV relevance filter "
            "(everything was combustion/outdated material)."
        )

    backend = HpcBackend(config.hpc)

    # 2. Preflights: catch problems BEFORE the multi-hour job.
    if allow_degraded_vision:
        log("WARNING: proceeding without a SoCLAaS key; figure descriptions will degrade.")
    else:
        try:
            check_soclaas_vision_key(config, backend)
        except HpcError as exc:
            raise CorpusError(
                f"Could not reach the CPU cluster to verify prerequisites "
                f"(SSH/config problem?): {exc}"
            ) from exc
    if not skip_index_build:
        check_local_ollama(config)
        if _web_server_running(config):
            raise CorpusError(
                "The RAG web server is running. Stop it before rebuilding the "
                "index from the initial corpus, then re-run (--initial-corpus "
                "resumes: the cluster parse is skipped for already-parsed PDFs)."
            )

    # 3. Push corpus, run the ingest-only cluster job, fetch the Markdown.
    log(f"Uploading {len(summary['pdfs'])} PDF(s) to the CPU cluster...")
    try:
        remote_data = backend.push_corpus_dir(local_corpus)
        log(f"Corpus staged at {remote_data}; submitting ingest-only PBS job...")
        backend.submit_ingest_index(
            input_dir_on_hpc=remote_data,
            skip_index=True,
            log_callback=log,
            progress_callback=lambda payload: log(
                "  cluster: {phase} {done}/{total} {unit}".format(**{
                    "phase": payload.get("phase", ""),
                    "done": payload.get("done", "?"),
                    "total": payload.get("total", "?"),
                    "unit": payload.get("unit", ""),
                })
            ),
        )
        log("Fetching processed Markdown from the cluster...")
        backend.fetch_processed_docs(local_dir=ROOT / "processed_docs")
    except HpcError as exc:
        raise CorpusError(f"Cluster operation failed: {exc}") from exc

    # 4. Verify the fetched corpus against the zip.
    ok = _verify_fetched_corpus(summary["pdfs"], ROOT / "processed_docs", log)
    if not ok:
        raise CorpusError(
            "The cluster parse produced no output. Inspect the job log lines "
            "above; a missing container or a walltime kill are the usual causes."
        )

    if skip_index_build:
        log("Skipping the local index build (--skip-index-build).")
        log("Build it later with: python main.py --mode index --md_dir processed_docs --db_dir db")
        return 0

    # 5. Build the local index (workstation embeddings via local Ollama).
    log("Building the local index (this embeds with the local model and can take a while)...")
    command = [
        str(_runtime_python()),
        str(ROOT / "main.py"),
        "--mode", "index",
        "--md_dir", str(ROOT / "processed_docs"),
        "--db_dir", str(ROOT / "db"),
    ]
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise CorpusError(f"Local index build failed with exit code {result.returncode}.")
    log("Initial corpus deployed: parsed on the cluster, indexed locally.")
    log(f"Start the server with start.cmd / ./start.sh; documents are served from {ROOT / 'db'}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy an initial PDF corpus: parse on the HPC cluster, index locally."
    )
    parser.add_argument("--zip", required=True, help="Path to the corpus zip (nested directories are fine).")
    parser.add_argument("--config", default=None, help="Path to config.toml (default: RAG_PIPELINE_CONFIG or ./config.toml).")
    parser.add_argument(
        "--allow-degraded-vision",
        action="store_true",
        help="Proceed without a SoCLAaS key even though vision enrichment is enabled "
             "(figure descriptions will be [Image description failed] markers).",
    )
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        help="Stop after fetching processed_docs/; build the index later with main.py.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_initial_corpus(
            args.zip,
            config_path=args.config,
            allow_degraded_vision=args.allow_degraded_vision,
            skip_index_build=args.skip_index_build,
        )
    except CorpusError as exc:
        print(f"\nInitial-corpus deployment failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInitial-corpus deployment interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
