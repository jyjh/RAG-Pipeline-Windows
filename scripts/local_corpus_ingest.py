"""Resume-safe chunked local ingestion for very large corpora.

Why this exists: a single ``main.py --mode ingest`` run over thousands of
PDFs keeps multiprocessing workers alive for the whole run, and heavy
Docling/OCR documents accumulate memory until a worker dies
(``std::bad_alloc``). With a process pool every pending future then fails at
once (thousands of "process terminated abruptly" failures) while the run
still exits 0 -- silently indexing a fraction of the corpus.

Strategy:
* Split the corpus into flat batch directories of hardlinks (same volume,
  zero-copy), bounded file count. Each batch is a FRESH ``main.py --mode
  ingest`` invocation, so worker memory resets between batches.
* Markdown naming deduplicates duplicate file stems PER INVOCATION, and a
  plain ``<stem>.md`` is only safe when the stem is unique across the WHOLE
  corpus. The driver therefore hardlinks globally-unique stems under their
  original (sanitized) filename and duplicated stems under
  ``<stem>__<seq>.pdf`` -- so every batch sees only globally-unique stems
  and no two invocations can overwrite each other's Markdown.
* Re-run a stalled batch (with resume: already-parsed PDFs are skipped by
  content hash, so retries only cost a re-hash of that batch); if it keeps
  stalling, fall back to per-file invocations.
* Finally run one index build (vectors for already-indexed content are
  reused by content hash).

Standalone use::

    python scripts/local_corpus_ingest.py --data-dir "F:/FSAE Readings"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fsae_ev_filter import FilterDecision, filter_pdf_list, write_ignore_log

INGEST_RESULT_FILENAME = ".ingest_result.json"
ABRUPT_FAILURE_MARKER = "terminated abruptly"
MAX_STEM_LEN = 150
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def iter_pdfs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.pdf") if p.is_file())


def _safe_stem(name: str) -> str:
    cleaned = _UNSAFE_CHARS.sub("_", name).strip(" .")
    return (cleaned or "file")[:MAX_STEM_LEN]


def filter_expensive_scans(
    paths: list[Path],
    *,
    max_pages: int,
    max_bytes: int = 10 * 1024 * 1024,
    root: Path | None = None,
    log_file: Path | None = None,
    scanned_out: set[Path] | None = None,
) -> tuple[list[Path], list[tuple[Path, FilterDecision]]]:
    """Drop long image-only PDFs before they enter the multi-hour OCR path.

    Page count and five sampled text extractions are cheap compared with
    Docling OCR. Damaged or encrypted PDFs that cannot be inspected are also
    excluded and recorded in the ignore log; normal ingestion could not
    produce reliable Markdown for them either.
    """
    if max_pages <= 0 and max_bytes <= 0:
        return paths, []

    from pypdf import PdfReader

    # The corpus contains damaged cross-reference tables that make pypdf emit
    # thousands of repair warnings.  Strict mode below rejects those files;
    # keep the preflight output readable as well.
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)

    kept: list[Path] = []
    ignored: list[tuple[Path, FilterDecision]] = []
    for path in paths:
        try:
            reader = PdfReader(str(path), strict=True)
            page_count = len(reader.pages)
            over_page_limit = max_pages > 0 and page_count > max_pages
            over_size_limit = max_bytes > 0 and path.stat().st_size > max_bytes
            sample_count = min(5, page_count)
            indexes = sorted(
                {round(i * (page_count - 1) / max(sample_count - 1, 1)) for i in range(sample_count)}
            )
            usable = 0
            for index in indexes:
                text = reader.pages[index].extract_text() or ""
                visible = [char for char in text if not char.isspace()]
                if len(visible) >= 50:
                    alnum_ratio = sum(char.isalnum() for char in visible) / len(visible)
                    if alnum_ratio >= 0.35 and text.count("\ufffd") / len(visible) <= 0.01:
                        usable += 1
            if usable * 2 >= len(indexes):
                kept.append(path)
                continue
            if over_page_limit or over_size_limit:
                ignored.append(
                    (
                        path,
                        FilterDecision(
                            "expensive_scan",
                            "expensive scanned document "
                            f"({page_count} pages, {path.stat().st_size / (1024 * 1024):.1f} MiB)",
                        ),
                    )
                )
            else:
                kept.append(path)
                if scanned_out is not None:
                    scanned_out.add(path)
        except Exception as exc:
            ignored.append(
                (
                    path,
                    FilterDecision(
                        "unreadable",
                        f"unreadable or malformed PDF ({type(exc).__name__})",
                    ),
                )
            )

    if ignored and log_file is not None:
        write_ignore_log(ignored, log_file, root=root)
    return kept, ignored


def build_batches(
    data_dir: Path,
    staging_dir: Path,
    max_chunk_files: int,
    *,
    corpus_filter: bool = True,
    max_scanned_pages: int = 16,
    max_scanned_bytes: int = 10 * 1024 * 1024,
    max_scan_chunk_files: int = 25,
    ignore_log: Path | None = None,
) -> list[Path]:
    """Flat hardlink batches whose filenames are globally stem-unique.

    Hardlinks live under ``staging_dir`` (same volume as the corpus, so no
    data is copied). The staging dir is rebuilt from scratch each run; it
    holds only driver-created links. When ``corpus_filter`` is on, documents
    irrelevant/outdated for FSAE-EV usage are excluded (see
    ``src.fsae_ev_filter.py``) and recorded in ``ignore_log``.
    """
    if staging_dir.exists():
        # Best-effort cleanup of a previous run's links. Deletion can be
        # denied while an indexer/AV holds a handle through the repo junction;
        # that is fine because batch names are deterministic -- a leftover link
        # maps to the same source file, and the link step below skips existing
        # destinations.
        for _ in range(2):
            try:
                shutil.rmtree(staging_dir)
                break
            except OSError:
                time.sleep(1.0)
        if staging_dir.exists():
            print(
                "[driver] WARNING: could not fully clean stale staging dir "
                f"{staging_dir} (locked by another process); leaving it in place "
                "and using a fresh per-run directory.",
                flush=True,
            )
    staging_dir.mkdir(parents=True, exist_ok=True)
    # Each run gets its OWN subdirectory: leftover links from a previous run
    # may be locked (indexer/AV handles) and cannot always be deleted, and
    # merging them into reused batch_NNNN dirs would both inflate counts and
    # resurrect files the corpus filter has since excluded.
    from datetime import datetime

    run_dir = staging_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_pdfs = iter_pdfs(data_dir)
    scanned_paths: set[Path] = set()
    if corpus_filter:
        kept, ignored = filter_pdf_list(all_pdfs, root=data_dir, log_file=ignore_log)
        by_reason: Counter = Counter(d.reason for _, d in ignored)
        print(
            f"[driver] FSAE-EV filter: keeping {len(kept)}/{len(all_pdfs)} PDF(s); "
            f"ignored {len(ignored)}"
            + (f" ({'; '.join(f'{n}x {r}' for r, n in by_reason.most_common(8))})" if ignored else ""),
            flush=True,
        )
        if ignore_log is not None:
            print(f"[driver] ignore log: {ignore_log}", flush=True)
        all_pdfs = kept
        all_pdfs, scan_ignored = filter_expensive_scans(
            all_pdfs,
            max_pages=max_scanned_pages,
            max_bytes=max_scanned_bytes,
            root=data_dir,
            log_file=ignore_log,
            scanned_out=scanned_paths,
        )
        if scan_ignored:
            print(
                f"[driver] expensive-scan filter: ignored {len(scan_ignored)} "
                "image-only PDF(s) over the OCR cost limit",
                flush=True,
            )
    stem_counts = Counter(p.stem.casefold() for p in all_pdfs)
    per_stem_seq: Counter = Counter()
    used_names: set[str] = set()
    named: list[tuple[Path, str]] = []
    for pdf in all_pdfs:
        stem = pdf.stem.casefold()
        if stem_counts[stem] == 1:
            base = _safe_stem(pdf.stem)
        else:
            per_stem_seq[stem] += 1
            base = f"{_safe_stem(pdf.stem)}__{per_stem_seq[stem]:03d}"
        name = f"{base}.pdf"
        suffix = 1
        while name.casefold() in used_names:  # truncation edge: force uniqueness
            suffix += 1
            name = f"{base}~{suffix}.pdf"
        used_names.add(name.casefold())
        named.append((pdf, name))

    text_named = [item for item in named if item[0] not in scanned_paths]
    scan_named = [item for item in named if item[0] in scanned_paths]
    batches: list[Path] = []

    def materialize(items: list[tuple[Path, str]], *, prefix: str, chunk_size: int) -> None:
        for start in range(0, len(items), chunk_size):
            batch = run_dir / f"{prefix}_batch_{start // chunk_size + 1:04d}"
            batch.mkdir(parents=True, exist_ok=True)
            for source, name in items[start:start + chunk_size]:
                destination = batch / name
                if destination.exists():
                    continue
                try:
                    os.link(source, destination)
                except OSError:
                    if destination.exists():
                        continue
                    try:
                        shutil.copy2(source, destination)
                    except shutil.SameFileError:
                        continue
            batches.append(batch)

    materialize(text_named, prefix="text", chunk_size=max_chunk_files)
    materialize(scan_named, prefix="scan", chunk_size=max_scan_chunk_files)
    if scan_named:
        print(
            f"[driver] isolated {len(scan_named)} small scanned PDF(s) into "
            f"{math.ceil(len(scan_named) / max_scan_chunk_files)} serial OCR batch(es)",
            flush=True,
        )
    return batches


def run_ingest_chunk(
    chunk: Path,
    *,
    md_dir: Path,
    asset_dir: Path,
    workers: int,
    vision_enabled: bool,
    vision_model: str | None,
    asset_triggers: str,
    extra_args: list[str],
    python: str,
) -> tuple[int, int, int]:
    """One fresh-process ingest over ``chunk``. Returns (processed, skipped, failed)."""
    command = [
        python,
        str(ROOT / "main.py"),
        "--mode", "ingest",
        "--data_dir", str(chunk),
        "--md_dir", str(md_dir),
        "--asset_dir", str(asset_dir),
        "--ingestion_workers", str(workers),
        "--vision_enabled", "true" if vision_enabled else "false",
        "--asset_triggers", str(asset_triggers),
        "--no_progress",
        *extra_args,
    ]
    if vision_model:
        command.extend(["--vision_model", vision_model])
    completed = subprocess.run(command, cwd=str(ROOT))
    result_path = md_dir / INGEST_RESULT_FILENAME
    processed = skipped = failed = 0
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            processed = len(payload.get("processed") or [])
            skipped = len(payload.get("skipped") or [])
            failed = len(payload.get("failed") or [])
        except (OSError, ValueError):
            pass
    if completed.returncode != 0:
        failed = max(failed, 1)
    return processed, skipped, failed


def abrupt_failures(md_dir: Path) -> int:
    result_path = md_dir / INGEST_RESULT_FILENAME
    if not result_path.is_file():
        return 0
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return sum(
        1
        for entry in (payload.get("failed") or [])
        if isinstance(entry, dict) and ABRUPT_FAILURE_MARKER in str(entry.get("error", ""))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="Corpus root (scanned recursively for PDFs).")
    parser.add_argument("--md-dir", default=str(ROOT / "processed_docs"))
    parser.add_argument("--asset-dir", default=str(ROOT / "db" / "assets"))
    parser.add_argument(
        "--staging-dir",
        default=None,
        help="Temporary batch-link directory (default: a sibling of md-dir, outside processed output).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Ingestion worker processes per chunk (1 = safest).")
    parser.add_argument("--max-chunk-files", type=int, default=300)
    parser.add_argument("--max-scan-chunk-files", type=int, default=25)
    parser.add_argument(
        "--max-scanned-pages",
        type=int,
        default=16,
        help="Skip image-only PDFs above this page count (0 disables; default: 16).",
    )
    parser.add_argument(
        "--max-scanned-mb",
        type=float,
        default=10.0,
        help="Also skip image-only PDFs above this size in MiB (0 disables; default: 10).",
    )
    parser.add_argument("--chunk-retries", type=int, default=2)
    parser.add_argument("--vision-enabled", action="store_true", help="Enable vision enrichment (needs a working vision backend).")
    parser.add_argument(
        "--vision-model",
        default="qwen2.5vl:3b",
        help="Vision model used for scanned-page fallback (default: qwen2.5vl:3b).",
    )
    parser.add_argument(
        "--asset-triggers",
        default="none",
        help="Docling enrichment triggers for figure pages (default: none -- with local "
             "vision, per-figure description is far too slow for a full corpus; the "
             "scanned-page vision fallback still runs when vision is enabled). "
             "Use 'images' to enrich every figure page.",
    )
    parser.add_argument(
        "--no-corpus-filter",
        action="store_true",
        help="Disable the FSAE-EV relevance filter (ingest everything).",
    )
    parser.add_argument("--skip-index", action="store_true", help="Do not build the index at the end.")
    parser.add_argument("--embedding-timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true", help="List the chunk plan and exit.")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).resolve()
    md_dir = Path(args.md_dir).resolve()
    asset_dir = Path(args.asset_dir).resolve()
    staging_dir = (
        Path(args.staging_dir).resolve()
        if args.staging_dir
        else md_dir.parent / ".ingest_batches"
    )

    # A plan must be safe to run while another ingest owns the output tree.
    # The old implementation called build_batches first, so --dry-run deleted
    # and recreated staging links and could fail on a locked .batches folder.
    if args.dry_run:
        all_pdfs = iter_pdfs(data_dir)
        scanned_paths: set[Path] = set()
        if args.no_corpus_filter:
            kept, ignored = all_pdfs, []
        else:
            kept, ignored = filter_pdf_list(all_pdfs, root=data_dir, log_file=None)
            kept, scan_ignored = filter_expensive_scans(
                kept,
                max_pages=args.max_scanned_pages,
                max_bytes=max(0, int(args.max_scanned_mb * 1024 * 1024)),
                root=data_dir,
                log_file=None,
                scanned_out=scanned_paths,
            )
            ignored = [*ignored, *scan_ignored]
        text_count = len(kept) - len(scanned_paths)
        batch_count = (
            math.ceil(text_count / args.max_chunk_files)
            + math.ceil(len(scanned_paths) / args.max_scan_chunk_files)
        )
        print(
            f"[driver] dry run: keeping {len(kept)}/{len(all_pdfs)} PDF(s); "
            f"ignored {len(ignored)}; {len(scanned_paths)} serial OCR PDF(s); "
            f"{batch_count} batch(es)",
            flush=True,
        )
        return 0

    md_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_batches(
        data_dir,
        staging_dir,
        args.max_chunk_files,
        corpus_filter=not args.no_corpus_filter,
        max_scanned_pages=args.max_scanned_pages,
        max_scanned_bytes=max(0, int(args.max_scanned_mb * 1024 * 1024)),
        max_scan_chunk_files=args.max_scan_chunk_files,
        ignore_log=ROOT / "logs" / "ignored_documents.log",
    )
    total_pdfs = sum(len(iter_pdfs(c)) for c in chunks)
    print(f"[driver] {len(chunks)} batch(es), {total_pdfs} PDF(s) under {data_dir}", flush=True)
    python = sys.executable
    done_pdfs = 0
    stalled_chunks: list[Path] = []
    started = time.monotonic()
    for index, chunk in enumerate(chunks, 1):
        pdf_count = len(iter_pdfs(chunk))
        if pdf_count == 0:
            continue
        print(f"[driver] chunk {index}/{len(chunks)}: {pdf_count} PDF(s) in {chunk}", flush=True)
        chunk_workers = 1 if chunk.name.startswith("scan_batch_") else args.workers
        processed, skipped, failed = run_ingest_chunk(
            chunk,
            md_dir=md_dir,
            asset_dir=asset_dir,
            workers=chunk_workers,
            vision_enabled=args.vision_enabled,
            vision_model=args.vision_model,
            asset_triggers=args.asset_triggers,
            extra_args=[],
            python=python,
        )
        attempts = 1
        while failed > 0 and abrupt_failures(md_dir) > 0 and attempts <= args.chunk_retries:
            print(
                f"[driver] chunk {index}: {failed} failure(s) with worker death; "
                f"retrying (resume skips already-parsed files), attempt {attempts + 1}",
                flush=True,
            )
            processed2, skipped2, failed = run_ingest_chunk(
                chunk,
                md_dir=md_dir,
                asset_dir=asset_dir,
                workers=chunk_workers,
                vision_enabled=args.vision_enabled,
                vision_model=args.vision_model,
                asset_triggers=args.asset_triggers,
                extra_args=[],
                python=python,
            )
            processed += processed2
            skipped += skipped2
            attempts += 1
        remaining = pdf_count - processed - skipped
        if remaining > 0:
            stalled_chunks.append(chunk)
            print(f"[driver] chunk {index}: {remaining} PDF(s) still unprocessed.", flush=True)
        else:
            done_pdfs += pdf_count
            elapsed = time.monotonic() - started
            print(
                f"[driver] chunk {index} complete ({done_pdfs}/{total_pdfs} PDFs total, "
                f"{elapsed/60:.1f} min elapsed)",
                flush=True,
            )

    # Last resort for repeatedly-stalled chunks: one file per fresh process.
    for chunk in stalled_chunks:
        for pdf in iter_pdfs(chunk):
            processed, skipped, failed = run_ingest_chunk(
                pdf,
                md_dir=md_dir,
                asset_dir=asset_dir,
                workers=1,
                vision_enabled=args.vision_enabled,
                vision_model=args.vision_model,
                asset_triggers=args.asset_triggers,
                extra_args=[],
                python=python,
            )
            if failed:
                print(f"[driver] per-file fallback FAILED: {pdf}", flush=True)
            else:
                done_pdfs += 1

    print(
        f"[driver] ingestion pass finished: {done_pdfs}/{total_pdfs} PDF(s) have Markdown",
        flush=True,
    )

    if args.skip_index:
        print("[driver] --skip-index: build later with main.py --mode index.", flush=True)
        return 0

    print("[driver] building index (vectors for unchanged content are reused)...", flush=True)
    index_command = [
        python,
        str(ROOT / "main.py"),
        "--mode", "index",
        "--md_dir", str(md_dir),
        "--db_dir", str(ROOT / "db"),
        "--embedding_timeout", str(args.embedding_timeout),
        "--no_progress",
    ]
    completed = subprocess.run(index_command, cwd=str(ROOT))
    if completed.returncode != 0:
        print("[driver] index build failed; re-run main.py --mode index after fixing.", flush=True)
        return 1
    print("[driver] done: corpus parsed and indexed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
