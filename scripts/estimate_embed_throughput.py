#!/usr/bin/env python3
"""Estimate embedding throughput for the cost-optimized CPU/GPU strategy.

This is the decision tool for "maximize free-CPU work, minimize paid-GPU work."
It measures the REAL per-node embedding rate (using the project's own chunker
and EmbeddingEngine, so the number matches production), then projects total
wall-clock across N nodes -- answering "can my free CPU cluster do the whole
cold index, and if so, how long does it take?"

Two modes:

  --count   : chunk the sample corpus only; report total chunk count + projected
              corpus-wide total. No model loading, no network. Use this first to
              size the job, and to sanity-check that the rate projections below
              will be meaningful.

  default   : embed the sample corpus, measure chunks/sec (the real rate for
              THIS node + THIS backend), and project:

                total_chunks / (chunks_per_sec * num_nodes)  ->  wall-clock

              across a range of node counts. Reports native (SentenceTransformers)
              and/or Ollama-HTTP backends depending on flags.

WHY this exists: the config's "weeks of embedding work on a single Ollama host"
is a SINGLE-HOST number. Native embeddings are 2-5x faster, and the corpus is
embarrassingly parallel across nodes. So one host = weeks, native on one host =
days, N free nodes = hours. This script turns that into concrete numbers for
YOUR corpus and YOUR nodes, so the CPU-vs-GPU spend decision is empirical.

Usage examples:

  # Size the job first (fast, no models):
  python scripts/estimate_embed_throughput.py --count --sample-dir processed_docs/

  # Measure native SentenceTransformers rate on this node, project to 8 nodes:
  python scripts/estimate_embed_throughput.py --native \\
      --sample-dir processed_docs/ --num-nodes 8

  # Measure the Ollama HTTP rate (requires `ollama serve` running with
  # nomic-embed-text pulled):
  python scripts/estimate_embed_throughput.py --ollama \\
      --sample-dir processed_docs/ --num-nodes 8

  # Compare both backends on the same sample:
  python scripts/estimate_embed_throughput.py --native --ollama \\
      --sample-dir processed_docs/ --num-nodes 1,4,8,16

  # Scale a small sample up to a corpus-size estimate (chunks per corpus is
  # roughly linear in source text; --scale says "my real corpus is Nx this
  # sample by chunk count"):
  python scripts/estimate_embed_throughput.py --native \\
      --sample-dir processed_docs/sample/ --scale 50 --num-nodes 16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Make `src.*` importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("estimate_embed_throughput")

# Match the production indexer's chunking defaults exactly (see
# src/local_rag_classes/local_vector_indexer.py:45-46 and src/indexing.py:24-25).
DEFAULT_CHUNK_TARGET_TOKENS = 900
DEFAULT_CHUNK_OVERLAP_TOKENS = 120

# Production embedding call shape (local_vector_indexer.py:110-114):
#   get_mrl_embeddings(batch, truncate_dim=768, prefix="search_document: ")
EMBED_TRUNCATE_DIM = 768
EMBED_PREFIX = "search_document: "

# Default batch size matches EmbeddingEngine default (embeddings.py:158-163,
# config [embeddings].batch_size default 128).
DEFAULT_BATCH_SIZE = 128


# --------------------------------------------------------------------------- #
# Pure projection math -- unit-testable without network/models.
# --------------------------------------------------------------------------- #


def project_wallclock(
    total_chunks: int,
    chunks_per_sec: float,
    num_nodes: int,
) -> float:
    """Seconds to embed ``total_chunks`` across ``num_nodes`` parallel workers
    at ``chunks_per_sec`` each. Embarrassingly-parallel linear scaling (the
    realistic model for sharding the corpus across nodes/processes).

    Returns 0.0 (not inf) for a zero corpus so callers don't have to special-case.
    """
    if total_chunks <= 0:
        return 0.0
    if chunks_per_sec <= 0 or num_nodes <= 0:
        return float("inf")
    return total_chunks / (chunks_per_sec * num_nodes)


def format_duration(seconds: float) -> str:
    """Human-readable wall-clock: '2.3d', '4h12m', '37m', '8s', 'n/a'."""
    if seconds == float("inf"):
        return "n/a (zero throughput)"
    if seconds <= 0:
        return "0s"
    s = int(round(seconds))
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    minutes, sec = divmod(s, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{sec}s"
    return f"{sec}s"


def parse_node_list(spec: str) -> list[int]:
    """Parse a node-count spec like '8' or '1,4,8,16' into a sorted unique list.

    Used by --num-nodes so the projection table can cover several scales at once.
    """
    nodes: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise ValueError(f"node count must be >= 1, got {n}")
        nodes.add(n)
    if not nodes:
        raise ValueError("no node counts parsed")
    return sorted(nodes)


# --------------------------------------------------------------------------- #
# Chunking (reuses the production chunker -- no OCR/Docling needed).
# --------------------------------------------------------------------------- #


def collect_chunks(
    sample_dir: Path,
    chunk_target_tokens: int,
    chunk_overlap_tokens: int,
) -> tuple[list[str], int]:
    """Chunk every ``.md`` under ``sample_dir`` using the production chunker.

    Returns (chunk_texts, file_count). Uses ``rglob`` so nested docs are covered;
    the production indexer uses top-level ``glob`` but nesting is common in a
    sample. Each returned text is the embeddable ``content`` of a section record
    (document/section summaries + chunks all get embedded in production).

    Pure text: sectioning never invokes Docling/OCR. If a sibling .pdf exists
    it uses pypdf text extraction (still no OCR); point at a .md-only dir for a
    fully deterministic pure-Markdown chunking.
    """
    from src.sectioning import build_section_records

    md_files = sorted(sample_dir.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"no .md files found under {sample_dir} -- ingest some PDFs first, "
            "or pass --raw-text to benchmark on synthetic chunks."
        )

    all_texts: list[str] = []
    for md in md_files:
        records = build_section_records(
            md,
            source_root=Path.cwd(),
            summary_mode="hybrid",
            chunk_target_tokens=chunk_target_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        for rec in records:
            content = rec.get("content")
            if content and content.strip():
                all_texts.append(content)
    return all_texts, len(md_files)


# --------------------------------------------------------------------------- #
# Synthetic fallback (no .md corpus needed; measures raw embedding rate only).
# --------------------------------------------------------------------------- #


def synthetic_chunks(count: int, chunk_target_tokens: int) -> list[str]:
    """Generate ``count`` realistic-length chunk strings without a corpus.

    ~4 chars/token * chunk_target_tokens, technical prose so the embeddings
    aren't trivially cacheable. For when you want to benchmark the embedder
    itself before ingesting anything.
    """
    words = (
        "the torque curve peak brake power thermal efficiency combustion chamber "
        "suspension geometry roll center kinematic camber recovery aerodynamic "
        "downforce coefficient wing element Reynolds number boundary layer "
        "friction coefficient tire slip ratio lateral force composite layup "
        "fiber orientation resin system cure cycle"
    ).split()
    import random
    rng = random.Random(0)  # deterministic
    target_chars = chunk_target_tokens * 4
    out: list[str] = []
    for _ in range(count):
        parts: list[str] = []
        n = 0
        while n < target_chars:
            w = rng.choice(words)
            parts.append(w)
            n += len(w) + 1
        out.append(" ".join(parts))
    return out


# --------------------------------------------------------------------------- #
# Rate measurement (reuses the production EmbeddingEngine).
# --------------------------------------------------------------------------- #


def _measure_rate_with_engine(engine, chunk_texts: list[str], label: str) -> float:
    """Time embedding all chunk_texts in production-sized batches; return chunks/sec.

    The warm-up batch (first batch) is excluded so cold-start / model-load time
    does not pollute the steady-state rate. Uses the exact production call shape.
    """
    if not chunk_texts:
        return 0.0
    batch_size = engine.ollama_batch_size or DEFAULT_BATCH_SIZE

    # Warm-up: load the model / establish the connection. Excluded from timing.
    warm = min(batch_size, len(chunk_texts))
    _ = engine.get_mrl_embeddings(chunk_texts[:warm], truncate_dim=EMBED_TRUNCATE_DIM, prefix=EMBED_PREFIX)

    start = time.perf_counter()
    done = 0
    total = len(chunk_texts)
    for i in range(0, total, batch_size):
        batch = chunk_texts[i:i + batch_size]
        _ = engine.get_mrl_embeddings(batch, truncate_dim=EMBED_TRUNCATE_DIM, prefix=EMBED_PREFIX)
        done += len(batch)
        if done % (batch_size * 10) == 0 or done == total:
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"  [{label}] {done}/{total} chunks in {format_duration(elapsed)} "
                  f"({rate:.1f} chunks/sec)", file=sys.stderr)
    elapsed = time.perf_counter() - start
    return done / elapsed if elapsed > 0 else 0.0


def measure_ollama_rate(chunk_texts: list[str], batch_size: int) -> float:
    """Measure chunks/sec for the Ollama-HTTP embedding backend.

    Requires `ollama serve` running with nomic-embed-text pulled. Uses the
    production EmbeddingEngine with the production call shape, so the rate is
    directly comparable to a real ingest job.
    """
    from src.embeddings import EmbeddingEngine

    engine = EmbeddingEngine(model_name="nomic-embed-text", ollama_batch_size=batch_size)
    return _measure_rate_with_engine(engine, chunk_texts, "ollama")


def measure_native_rate(chunk_texts: list[str], batch_size: int) -> float:
    """Measure chunks/sec for the native SentenceTransformers backend.

    Sets [models].native_embeddings = true for the run via a temporary config
    (the flag has no env var). First call downloads weights from HuggingFace
    (~274MB for nomic-embed-text-v1.5); the download + model load is EXCLUDED
    from the timed measurement so the rate reflects steady-state throughput.
    """
    # Save the env so a failure mid-run can't leak the temp config path.
    prior_cfg = os.environ.get("RAG_PIPELINE_CONFIG")
    tmp_name: str | None = None
    try:
        tmp_name = _write_native_config_override()
        os.environ["RAG_PIPELINE_CONFIG"] = tmp_name

        from src.embeddings import EmbeddingEngine
        engine = EmbeddingEngine(model_name="nomic-embed-text", ollama_batch_size=batch_size)
        if not engine.native_embeddings:
            raise RuntimeError(
                "native embedding model failed to load (see messages above). "
                "SentenceTransformers may be missing or the model download failed."
            )
        return _measure_rate_with_engine(engine, chunk_texts, "native")
    finally:
        # Restore prior env.
        if prior_cfg is not None:
            os.environ["RAG_PIPELINE_CONFIG"] = prior_cfg
        else:
            os.environ.pop("RAG_PIPELINE_CONFIG", None)
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _write_native_config_override() -> str:
    """Write a temp config enabling [models].native_embeddings = true.

    Reuses the user's config as the base (so model name etc. carry over) when
    available; otherwise a minimal config. Returns the temp file path.
    """
    import tempfile

    base_cfg_path = Path(os.environ.get("RAG_PIPELINE_CONFIG", "config.toml"))
    base_text = ""
    if base_cfg_path.exists():
        try:
            base_text = base_cfg_path.read_text(encoding="utf-8")
        except OSError:
            base_text = ""

    # Flip native_embeddings on. If the file has the line, replace it; if it has
    # a [models] table without it, insert it; otherwise append a minimal block.
    lines = base_text.splitlines()
    if any("native_embeddings" in ln for ln in lines):
        new_lines = []
        for ln in lines:
            if "native_embeddings" in ln:
                new_lines.append("native_embeddings = true")
            else:
                new_lines.append(ln)
        body = "\n".join(new_lines) + "\n"
    elif "[models]" in base_text:
        body = base_text.rstrip() + "\nnative_embeddings = true\n"
    else:
        body = (base_text.rstrip() + "\n\n"
                if base_text.strip() else "") + "[models]\nnative_embeddings = true\n"

    fd, name = tempfile.mkstemp(suffix=".toml", prefix="estimate_embed_")
    os.close(fd)
    Path(name).write_text(body, encoding="utf-8")
    return name


# --------------------------------------------------------------------------- #
# CLI + reporting.
# --------------------------------------------------------------------------- #


def _print_projection(
    total_chunks: int,
    chunks_per_sec: float,
    node_counts: list[int],
    backend_label: str,
    file=sys.stdout,
) -> None:
    print(f"\n--- {backend_label}: projection for {total_chunks:,} chunks "
          f"@ {chunks_per_sec:.1f} chunks/sec/node ---", file=file)
    print(f"{'nodes':>6}  {'wall-clock':>12}  {'chunks/node':>12}", file=file)
    for n in node_counts:
        secs = project_wallclock(total_chunks, chunks_per_sec, n)
        print(f"{n:>6}  {format_duration(secs):>12}  {total_chunks // n:>12,}", file=file)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="estimate_embed_throughput.py",
        description="Estimate embedding throughput and project wall-clock across N nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sample-dir", help="Directory of .md files to chunk + embed (your ingested corpus sample).")
    p.add_argument("--raw-text", type=int, metavar="N",
                   help="Skip chunking; generate N synthetic realistic-length chunks. Measures raw embedder rate.")
    p.add_argument("--count", action="store_true",
                   help="Only count chunks; report total + per-file breakdown. No model loading, no network.")
    p.add_argument("--native", action="store_true", help="Measure the native SentenceTransformers backend.")
    p.add_argument("--ollama", action="store_true", help="Measure the Ollama-HTTP backend (requires `ollama serve`).")
    p.add_argument("--num-nodes", default="1,4,8,16",
                   help="Comma-separated node counts for the projection table (default: 1,4,8,16).")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply measured-against-sample chunk count by this to size a larger corpus (e.g. --scale 50).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Embedding batch size (default {DEFAULT_BATCH_SIZE}; matches [embeddings].batch_size).")
    p.add_argument("--chunk-target-tokens", type=int, default=DEFAULT_CHUNK_TARGET_TOKENS)
    p.add_argument("--chunk-overlap-tokens", type=int, default=DEFAULT_CHUNK_OVERLAP_TOKENS)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.count and (args.native or args.ollama):
        print("error: --count is chunk-only; drop --native/--ollama to count", file=sys.stderr)
        return 2
    if not args.count and not (args.native or args.ollama):
        print("info: no backend selected and not --count; defaulting to --count (chunk-only).", file=sys.stderr)
        args.count = True

    # --- Gather chunks ---
    if args.raw_text:
        chunk_texts = synthetic_chunks(args.raw_text, args.chunk_target_tokens)
        file_count = 0
        source_label = f"{args.raw_text} synthetic chunks"
    elif args.sample_dir:
        sample_dir = Path(args.sample_dir)
        if not sample_dir.is_dir():
            print(f"error: --sample-dir {sample_dir} is not a directory", file=sys.stderr)
            return 2
        try:
            chunk_texts, file_count = collect_chunks(
                sample_dir, args.chunk_target_tokens, args.chunk_overlap_tokens,
            )
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        source_label = f"{file_count} .md file(s) under {sample_dir}"
    else:
        print("error: provide --sample-dir or --raw-text N", file=sys.stderr)
        return 2

    if not chunk_texts:
        print("error: no chunkable text produced from the sample", file=sys.stderr)
        return 2

    sample_chunk_count = len(chunk_texts)
    # Scale the SAMPLE chunk count up to the full-corpus estimate for projection.
    # (Chunks-per-corpus is roughly linear in source text; this is an estimate.)
    projected_total = int(round(sample_chunk_count * args.scale))
    try:
        node_counts = parse_node_list(args.num_nodes)
    except ValueError as e:
        print(f"error: bad --num-nodes: {e}", file=sys.stderr)
        return 2

    print(f"Sample: {source_label}", file=sys.stdout)
    print(f"Chunked into {sample_chunk_count:,} chunks "
          f"(target {args.chunk_target_tokens} tok, overlap {args.chunk_overlap_tokens} tok)", file=sys.stdout)
    if args.scale != 1.0:
        print(f"Scaling by {args.scale}x -> projecting {projected_total:,} chunks for the full corpus", file=sys.stdout)

    if args.count:
        print("\n--count only: skipping embedding. Re-run with --native and/or --ollama "
              "to measure the rate and project wall-clock.", file=sys.stdout)
        return 0

    # --- Measure + project ---
    backends = []
    if args.native:
        backends.append(("native (SentenceTransformers)", measure_native_rate))
    if args.ollama:
        backends.append(("ollama (HTTP)", measure_ollama_rate))

    summary = []
    for label, fn in backends:
        print(f"\nMeasuring {label} ...", file=sys.stderr)
        try:
            rate = fn(chunk_texts, args.batch_size)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue
        if rate <= 0:
            print(f"  {label}: measured zero throughput; cannot project.", file=sys.stdout)
            continue
        _print_projection(projected_total, rate, node_counts, label)
        summary.append((label, rate))

    if not summary:
        print("\nNo backend produced a usable rate. See errors above.", file=sys.stderr)
        return 1

    # Headline: cheapest path. Native is the recommended CPU path; if both ran,
    # show the speedup, which is the whole "why CPU native" argument made real.
    if len(summary) == 2:
        native_rate = next((r for l, r in summary if l.startswith("native")), None)
        ollama_rate = next((r for l, r in summary if l.startswith("ollama")), None)
        if native_rate and ollama_rate and ollama_rate > 0:
            print(f"\nNative is {native_rate / ollama_rate:.1f}x faster than Ollama on this node.",
                  file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
