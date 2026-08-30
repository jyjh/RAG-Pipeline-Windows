"""Retrieval evaluation harness.

Measures how reliably the live index retrieves the *expected* source for a set
of golden questions. Use it whenever retrieval-affecting settings change
(embedding model, chunking, RRF k, relevance floor) so quality moves are made
with numbers, not impressions.

Question file format (JSON)::

    [
      {
        "question": "What alloy family is most used for FSAE wheel uprights?",
        "expected": ["7075", "upright"]       # substrings matched against the
                                              # retrieved chunk's source file
                                              # name, file path, or content
      }
    ]

Usage (against a running local server)::

    python scripts/eval_retrieval.py --questions eval_questions.json --k 5
    python scripts/eval_retrieval.py --questions ... --base-url http://127.0.0.1:8000 --json

Metrics per question: hit@k (any expected source in top-k) and the best rank.
Reported overall: recall@k (share of questions with a hit) and MRR.
Exit code is 1 when recall@k falls below --min-recall (CI-friendly).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def _post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _matches_expected(row: dict, expected: list[str]) -> bool:
    haystacks = " ".join(
        str(row.get(field) or "")
        for field in ("source_pdf_name", "file_path", "content", "title", "section_path")
    ).lower()
    return any(str(needle).lower() in haystacks for needle in expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="JSON file with golden questions")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--k", type=int, default=5, help="hits considered (default 5)")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="relevance_floor override (default 0 = endpoint floor is bypassed; "
        "hybrid scores are typically < 0.2, so an unconfigured floor hides all results)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per-query timeout seconds")
    parser.add_argument("--min-recall", type=float, default=0.8, help="exit 1 below this recall@k")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if not questions:
        print("No questions in file.", file=sys.stderr)
        return 1

    payload_k = max(args.k, 10)
    rows_out = []
    hits = 0
    reciprocal_ranks = []

    for item in questions:
        question = str(item.get("question") or "").strip()
        expected = [str(e) for e in (item.get("expected") or [])]
        if not question or not expected:
            continue
        payload = {"query": question, "relevance_floor": args.min_score}
        try:
            result = _post(args.base_url, "/api/index/vector-search", payload, args.timeout)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            rows_out.append({"question": question, "error": str(exc)})
            continue
        rows = result.get("rows") or []
        best_rank = None
        for rank, row in enumerate(rows[: args.k], start=1):
            if _matches_expected(row, expected):
                best_rank = rank
                break
        hit = best_rank is not None
        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / best_rank)
        else:
            reciprocal_ranks.append(0.0)
        rows_out.append(
            {
                "question": question,
                "hit": hit,
                "rank": best_rank,
                "top_source": (rows[0].get("source_pdf_name") if rows else ""),
            }
        )

    evaluated = [row for row in rows_out if "error" not in row]
    recall = hits / len(evaluated) if evaluated else 0.0
    mrr = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0

    if args.json:
        print(
            json.dumps(
                {"k": args.k, "recall_at_k": recall, "mrr": mrr, "questions": rows_out},
                indent=2,
            )
        )
    else:
        for row in rows_out:
            if "error" in row:
                print(f"ERROR  {row['question'][:60]} -> {row['error'][:80]}")
            elif row["hit"]:
                print(f"HIT    rank {row['rank']:>2}  {row['question'][:60]}")
            else:
                print(f"MISS   rank --   {row['question'][:60]}  (top: {row['top_source'][:40]})")
        print(f"\nrecall@{args.k} = {recall:.2%}   MRR = {mrr:.3f}   ({hits}/{len(evaluated)} questions)")

    return 0 if recall >= args.min_recall else 1


if __name__ == "__main__":
    sys.exit(main())
