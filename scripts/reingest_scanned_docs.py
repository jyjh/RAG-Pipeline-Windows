"""Re-ingest the scanned corpus with the configured scanned OCR engine.

Reads the ``scan_batch_*`` hardlink batches created by
``scripts/local_corpus_ingest.py`` and re-parses their image-only PDFs so they
benefit from a newly selected ``[ingestion] scanned_ocr_engine`` (e.g.
``vision_ocr``). Per batch:

  1. Hash-match each batch PDF against the processed-docs source map. Only
     docs whose CURRENT file hash matches the map entry are replaced -- this
     guarantees the existing Markdown really was produced from this exact
     file, and automatically excludes broken/stub source files whose good
     Markdown came from an earlier complete download.
  2. Under the index lock: delete the doc's vectors (by source hash), its
     Markdown + ``.pages.json`` sidecar, and its source-map entry -- the same
     deletion path as ``scripts/prune_processed_corpus.py``.
  3. Run ``run_ingestion`` over the batch dir (resume-safe: anything already
     parsed is skipped, so the driver can be re-run after an interruption).
  4. Rebuild the index so the corpus stays searchable while the batch runs
     (existing vectors are reused by content hash; only new/changed chunks
     are embedded).

Batch PDFs without a matching Markdown entry (never successfully ingested)
are parsed as new documents. Run it in a long-lived shell:

    python scripts/reingest_scanned_docs.py --batches-root F:/rag-fsae/.ingest_batches/run_20260826_163219
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pypdf import PdfReader

from src.file_lock import acquire_index_lock
from src.local_rag import update_index_manifest_sources
from src.pdf_registry import (
    load_source_map,
    remove_source_entries_by_hash,
    sha256_file,
)
from src.vector_store import default_store

MIN_PAGES = 3
_DELETE_CHUNK = 40


def _iter_batch_pdfs(batches_root: Path) -> list[Path]:
    return sorted(
        path
        for path in batches_root.glob("scan_batch_*/*.pdf")
        if path.is_file()
    )


def _usable_pdf(path: Path) -> int:
    """Return page count, or 0 when the file is unreadable/too small to OCR."""
    try:
        return len(PdfReader(path).pages)
    except Exception as exc:  # noqa: BLE001 - damaged/encrypted files are skipped
        print(f"[skip ] unreadable: {path.name}: {exc}", flush=True)
        return 0


def _delete_documents(db_dir: Path, processed_dir: Path, docs: dict[str, tuple[str, dict]]) -> dict:
    """Remove vectors + markdown + source-map entries for ``{hash: (md_name, entry)}``."""
    hashes = sorted(docs)
    stats = {"vectors": 0, "markdown": 0, "map_entries": 0}
    if not hashes:
        return stats
    with acquire_index_lock(db_dir):
        store = default_store(db_dir)
        if store.exists():
            for offset in range(0, len(hashes), _DELETE_CHUNK):
                result = store.delete_records_by_source_hash(
                    source_hashes=hashes[offset : offset + _DELETE_CHUNK]
                )
                stats["vectors"] += int(result.get("deleted", 0))
            if hashes:
                model, dim = store.metadata()
                update_index_manifest_sources(
                    db_dir,
                    {source_hash: [] for source_hash in hashes},
                    embedding_model=model,
                    embedding_dim=dim,
                )
        for md_name, entry in docs.values():
            raw_path = str(entry.get("processed_markdown_path") or (processed_dir / md_name))
            md_path = Path(raw_path).resolve()
            try:
                md_path.relative_to(processed_dir.resolve())
            except ValueError as exc:
                raise RuntimeError(f"Refusing to delete outside processed dir: {md_path}") from exc
            if md_path.exists():
                md_path.unlink()
                stats["markdown"] += 1
            md_path.with_suffix(".pages.json").unlink(missing_ok=True)
        stats["map_entries"] = len(remove_source_entries_by_hash(processed_dir, set(hashes)))
    return stats


def _ingestion_kwargs() -> dict:
    """run_ingestion kwargs mirroring the live config.toml (see main._ingestion_args)."""
    import main

    config = main._load_ingestion_config()
    from src.config import default_config_path, load_config

    cfg = load_config(default_config_path())
    kwargs = dict(config)
    kwargs["asset_dir"] = str(cfg.paths.asset_dir or config["asset_dir"])
    kwargs["progress_enabled"] = True
    kwargs["ingestion_workers"] = 1
    return kwargs


def _build_index(db_dir: Path, md_dir: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--mode",
        "index",
        "--md_dir",
        str(md_dir),
        "--db_dir",
        str(db_dir),
        "--no_progress",
    ]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"index build failed with exit code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-root", type=Path, required=True,
                        help="Directory containing the scan_batch_* hardlink batches")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "processed_docs")
    parser.add_argument("--db-dir", type=Path, default=ROOT / "db")
    parser.add_argument("--no-index", action="store_true",
                        help="Skip the per-batch index rebuild (run main.py --mode index manually)")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Stop after N re-ingested docs (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report the matched set and per-batch page counts")
    args = parser.parse_args()

    batches_root = args.batches_root.resolve()
    processed_dir = args.processed_dir.resolve()
    db_dir = args.db_dir.resolve()

    by_hash: dict[str, tuple[str, dict]] = {}
    for md_name, entry in load_source_map(processed_dir).get("documents", {}).items():
        source_hash = str(entry.get("source_hash") or "")
        if source_hash:
            by_hash[source_hash] = (md_name, entry)

    batches: dict[Path, list[tuple[Path, int, str | None]]] = {}
    for pdf_path in _iter_batch_pdfs(batches_root):
        pages = _usable_pdf(pdf_path)
        if pages < MIN_PAGES:
            if pages:
                print(f"[skip ] {pages} pages (<{MIN_PAGES}): {pdf_path.name}", flush=True)
            continue
        source_hash = sha256_file(pdf_path)
        match = by_hash.get(source_hash)
        batches.setdefault(pdf_path.parent, []).append((pdf_path, pages, match[0] if match else None))

    total_pages = sum(pages for docs in batches.values() for _, pages, _ in docs)
    total_docs = sum(len(docs) for docs in batches.values())
    replace_docs = sum(1 for docs in batches.values() for _, _, md in docs if md)
    print(
        json.dumps(
            {
                "batches": len(batches),
                "docs": total_docs,
                "pages": total_pages,
                "replacing_existing_markdown": replace_docs,
                "new_documents": total_docs - replace_docs,
                "estimated_hours_at_210s_per_page": round(total_pages * 210 / 3600, 1),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    started = time.time()
    ingested = 0
    for batch_dir in sorted(batches):
        docs = batches[batch_dir]
        # Rebuild hash -> (md_name, live entry) from the current source map so
        # deletion always targets what is on disk right now.
        current_map = load_source_map(processed_dir).get("documents", {})
        by_hash_now: dict[str, tuple[str, dict]] = {}
        for pdf_path, _pages, md_name in docs:
            if not md_name:
                continue
            entry = current_map.get(md_name)
            if isinstance(entry, dict) and entry.get("source_hash"):
                by_hash_now[str(entry["source_hash"])] = (md_name, entry)
        if by_hash_now:
            stats = _delete_documents(db_dir, processed_dir, by_hash_now)
            print(f"[del  ] {batch_dir.name}: {json.dumps(stats)}", flush=True)

        print(f"[ing  ] {batch_dir.name}: {len(docs)} doc(s), "
              f"{sum(p for _, p, _ in docs)} page(s)", flush=True)
        from src.ingestion import run_ingestion

        run_ingestion(str(batch_dir), str(processed_dir), **_ingestion_kwargs())

        ingested += len(docs)
        elapsed_min = (time.time() - started) / 60
        print(f"[done ] {batch_dir.name} at +{elapsed_min:.0f} min "
              f"({ingested}/{total_docs} docs)", flush=True)

        if not args.no_index:
            print(f"[index] rebuilding after {batch_dir.name}...", flush=True)
            _build_index(db_dir, processed_dir)

        if args.max_docs and ingested >= args.max_docs:
            print(f"[stop ] reached --max-docs {args.max_docs}", flush=True)
            break

    print(f"[all  ] finished {ingested} doc(s) in {(time.time() - started) / 3600:.1f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
