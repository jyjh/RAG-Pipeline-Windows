"""Remove already-processed documents rejected by the current corpus filter.

This reconciles an existing Markdown corpus after filter rules are tightened.
It never touches the source PDF library.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.file_lock import acquire_index_lock
from src.fsae_ev_filter import evaluate_pdf
from src.local_rag import update_index_manifest_sources
from src.pdf_registry import load_source_map, remove_source_entries_by_hash
from src.vector_store import default_store


def _candidates(processed_dir: Path, corpus_root: Path) -> list[tuple[str, dict, object]]:
    documents = load_source_map(processed_dir).get("documents", {})
    matches: list[tuple[str, dict, object]] = []
    for markdown_name, raw_entry in documents.items():
        if not isinstance(raw_entry, dict):
            continue
        pdf_name = str(raw_entry.get("source_pdf_name") or Path(markdown_name).with_suffix(".pdf").name)
        decision = evaluate_pdf(corpus_root / pdf_name, root=corpus_root)
        if decision is not None:
            matches.append((str(markdown_name), raw_entry, decision))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Perform the prune; otherwise only report it.")
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    matches = _candidates(processed_dir, args.corpus_root.resolve())
    counts = Counter(getattr(decision, "reason", "filtered") for _, _, decision in matches)
    print(json.dumps({"matched_documents": len(matches), "reasons": dict(counts)}, indent=2))
    if not args.apply or not matches:
        return 0

    hashes = sorted(
        {
            str(entry.get("source_hash") or "").strip()
            for _, entry, _ in matches
            if str(entry.get("source_hash") or "").strip()
        }
    )
    with acquire_index_lock(args.db_dir):
        store = default_store(args.db_dir)
        deleted_vectors = 0
        remaining_vectors = store.count() if store.exists() else 0
        for offset in range(0, len(hashes), 40):
            result = store.delete_records_by_source_hash(source_hashes=hashes[offset : offset + 40])
            deleted_vectors += int(result.get("deleted", 0))
            remaining_vectors = int(result.get("remaining", remaining_vectors))
        if store.exists() and hashes:
            model, dim = store.metadata()
            update_index_manifest_sources(
                args.db_dir,
                {source_hash: [] for source_hash in hashes},
                embedding_model=model,
                embedding_dim=dim,
            )

        deleted_markdown = 0
        for markdown_name, entry, _ in matches:
            raw_path = str(entry.get("processed_markdown_path") or (processed_dir / markdown_name))
            path = Path(raw_path).resolve()
            try:
                path.relative_to(processed_dir)
            except ValueError as exc:
                raise RuntimeError(f"Refusing to delete outside processed directory: {path}") from exc
            if path.exists():
                path.unlink()
                deleted_markdown += 1
            path.with_suffix(".pages.json").unlink(missing_ok=True)

        removed_entries = remove_source_entries_by_hash(processed_dir, set(hashes))

    print(
        json.dumps(
            {
                "deleted_markdown": deleted_markdown,
                "deleted_source_map_entries": len(removed_entries),
                "deleted_vectors": deleted_vectors,
                "remaining_vectors": remaining_vectors,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
