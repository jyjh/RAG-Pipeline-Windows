#!/usr/bin/env python3
"""Bootstrap golden retrieval questions from exported chat Markdown files.

The chat UI's per-chat "Export" button writes Markdown with `## Question` /
`## Answer` sections and a `### Sources` list. This script parses one or more
of those exports and emits an `eval_questions.json` suitable for
`scripts/eval_retrieval.py` — the questions teammates actually asked, with the
documents that answered them as expectations.

Usage::

    python scripts/bootstrap_eval_questions.py exported-chat.md [more.md ...] \
        --out eval/questions.json

Heuristics, on purpose: the output is a *draft* for human review, not ground
truth. Skips answers with no local sources (web-only answers say nothing
about the local index) and truncates very long questions.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_chat_markdown(text: str) -> list[dict[str, object]]:
    """Extract (question, expected_sources) pairs from an exported chat."""
    pairs: list[dict[str, object]] = []
    # Split on question headings.
    blocks = re.split(r"^## Question\s*$", text, flags=re.MULTILINE)
    for block in blocks[1:]:
        # Question text: up to the next heading.
        parts = re.split(r"^## Answer\s*$", block, maxsplit=1, flags=re.MULTILINE)
        question = parts[0].strip()
        if not parts[1:]:
            continue
        answer_section = parts[1]
        sources_match = re.search(r"^### Sources\s*$", answer_section, flags=re.MULTILINE)
        if not sources_match:
            # No sources block: likely web-only or interrupted. Skip.
            continue
        source_lines = answer_section[sources_match.end() :].splitlines()
        expected: list[str] = []
        for line in source_lines:
            entry = line.strip()
            if not entry.startswith("- "):
                if expected:
                    break
                continue
            entry = entry[2:].strip()
            # Strip the leading citation label ("S3 " / "W1 ") if present.
            entry = re.sub(r"^[SW]\d+\s+", "", entry)
            if entry.startswith(("http://", "https://")):
                continue  # web source
            entry = entry.strip()
            if entry:
                expected.append(entry)
        if question and expected:
            pairs.append({"question": question, "expected": expected})
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Exported chat .md files")
    parser.add_argument("--out", default="eval/questions.json", help="Output JSON path")
    parser.add_argument("--max-question-chars", type=int, default=300)
    args = parser.parse_args()

    seen: set[str] = set()
    questions: list[dict[str, object]] = []
    for name in args.files:
        text = Path(name).read_text(encoding="utf-8", errors="replace")
        for pair in parse_chat_markdown(text):
            question = str(pair["question"]).strip()
            if len(question) > args.max_question_chars:
                question = question[: args.max_question_chars].rsplit(" ", 1)[0] + "…"
            if question.lower() in seen:
                continue
            seen.add(question.lower())
            questions.append({"question": question, "expected": pair["expected"]})

    if not questions:
        print("No usable question/source pairs found. Did the exports include local Sources?", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(questions)} draft question(s) to {out}. Review expectations before evaluating.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
