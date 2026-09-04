"""LLM source-group auto-tagger (upload/trust plugin).

Manual tagging of every uploaded PDF into its source group (Official /
Student Research / Unofficial) does not scale past a few hundred documents.
This module asks the chat model (``gemma4:26b`` through the normal LLM
transport, i.e. SoCLAaS with the dormant Ollama fallback) to sort PDFs into
those existing categories. Decisions are applied through the SAME trust
registry as manual tags -- retrieval ranking sees an ordinary source group --
but each trust entry also carries ``auto_tagged: true`` plus the model,
confidence, reason, and timestamp so reviewers can audit and override them
(a manual tag always clears the flag).

Classification input is the filename plus a short text excerpt lifted from
the first pages of the PDF (best-effort pypdf; filename-only when the file is
unavailable or encrypted). Documents are classified in batches (one LLM call
per batch) and decisions below the confidence floor are dropped, leaving the
PDF ungrouped for a human.

Usage (web app integration)::

    from src.auto_tag import AutoTagInput, classify_documents, pdf_excerpt

    inputs = [AutoTagInput(h, filename, pdf_excerpt(path)) for ...]
    decisions = classify_documents(inputs, model="gemma4:26b")
    apply decisions via web_app.apply_auto_tag_decisions(decisions, model=...)

The module deliberately imports nothing from ``src.web_app`` (the web app
imports this); tests can stub the transport with ``chat_fn=``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.reliability import (
    SOURCE_GROUP_OFFICIAL,
    SOURCE_GROUP_STUDENT_RESEARCH,
    SOURCE_GROUP_UNOFFICIAL,
    normalize_source_group,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemma4:26b"
# Small batches keep the JSON array inside a local model's reliability (and
# generation budget): a 4-classification reply is ~200 tokens, while a
# 20-document batch elicits multi-thousand-token replies that overrun the
# request timeout on a 4 GB GPU and often come back malformed (see
# classify_documents: output budget scales with batch size).
DEFAULT_BATCH_SIZE = 5
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_EXCERPT_CHARS = 1200
DEFAULT_TIMEOUT_SECONDS = 120.0

AUTO_TAGGABLE_GROUPS = (SOURCE_GROUP_OFFICIAL, SOURCE_GROUP_STUDENT_RESEARCH, SOURCE_GROUP_UNOFFICIAL)

SYSTEM_PROMPT = """You classify documents in a Formula SAE Electric (FSAE-EV) engineering knowledge base.
For each document choose the single best source group:

- official: competition rules and regulations, standards (e.g. ISO, FIA), official team/org manuals, datasheets from component manufacturers, published textbooks and reference handbooks.
- student_research: student theses and final-year projects, university reports and coursework, team design reports and research papers written by students.
- unofficial: hobbyist/forum guides, magazine articles, blog posts, third-party tutorials, notes of unknown provenance.

Judge primarily by the document title/filename, secondarily by the excerpt.
Assign a confidence between 0.0 and 1.0. When the evidence is weak or ambiguous, lower the confidence rather than guessing.
Respond with ONLY a JSON array (no markdown, no prose), one object per input document, each exactly:
{"hash": "<hash exactly as given>", "source_group": "official" | "student_research" | "unofficial", "confidence": <number>, "reason": "<max 15 words>"}"""


@dataclass(frozen=True)
class AutoTagInput:
    """One document to classify: hash + filename + optional text excerpt."""

    source_hash: str
    filename: str
    excerpt: str = ""


@dataclass(frozen=True)
class AutoTagDecision:
    """A classification outcome for one source hash."""

    source_group: str
    confidence: float
    reason: str = ""


def pdf_excerpt(path: str | Path, *, max_chars: int = DEFAULT_EXCERPT_CHARS, max_pages: int = 2) -> str:
    """Best-effort text excerpt from the first pages of a PDF.

    Returns "" on any failure (missing/encrypted file, parse error, pypdf not
    installed) -- classification then falls back to the filename alone.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        remaining = max(0, int(max_chars))
        for page in reader.pages[: max(0, int(max_pages))]:
            if remaining <= 0:
                break
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text[:remaining])
                remaining -= len(text)
        return " ".join(part for part in parts if part).strip()
    except Exception:
        return ""


def build_user_prompt(items: list[AutoTagInput]) -> str:
    documents = [
        {"hash": item.source_hash, "filename": item.filename, "excerpt": item.excerpt}
        for item in items
    ]
    return (
        "Classify these documents. Reply with only the JSON array.\n\n"
        + json.dumps(documents, ensure_ascii=False)
    )


def _extract_json_array(text: str) -> list[Any] | None:
    """Pull the first JSON array out of an LLM reply.

    Tolerates markdown code fences and stray prose around the array. Returns
    None when no parseable array is present.
    """
    raw = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def parse_classification_response(
    text: str,
    expected_hashes: list[str] | set[str],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, AutoTagDecision]:
    """Parse the model reply into ``{source_hash: AutoTagDecision}``.

    Unknown hashes, invalid groups, and sub-floor confidences are dropped
    (the document simply stays ungrouped). Malformed entries are skipped
    rather than failing the whole batch.
    """
    expected = {str(value) for value in expected_hashes}
    rows = _extract_json_array(text)
    if rows is None:
        return {}
    decisions: dict[str, AutoTagDecision] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_hash = str(row.get("hash") or "").strip()
        if not source_hash or source_hash not in expected or source_hash in decisions:
            continue
        group = normalize_source_group(row.get("source_group"))
        if group not in AUTO_TAGGABLE_GROUPS:
            continue
        try:
            confidence = min(max(float(row.get("confidence")), 0.0), 1.0)
        except (TypeError, ValueError):
            continue
        if confidence < max(0.0, float(min_confidence)):
            continue
        decisions[source_hash] = AutoTagDecision(
            source_group=group,
            confidence=confidence,
            reason=str(row.get("reason") or "").strip()[:200],
        )
    return decisions


def _batch_doc_count(messages: list[dict[str, Any]]) -> int:
    """Number of documents in a classification user message (0 if unreadable)."""
    try:
        docs = json.loads(str(messages[-1].get("content") or "").split("\n\n", 1)[1])
    except (IndexError, KeyError, TypeError, ValueError):
        return 0
    return len(docs) if isinstance(docs, list) else 0


def _default_chat_fn(*, model: str, messages: list[dict[str, Any]], timeout: float | None):
    """Send one non-streaming chat request through the app's LLM transport.

    Lazy import: ``src.local_rag`` is heavy and this keeps ``src.auto_tag``
    importable standalone (tests, CLI). ``_llm_chat`` dispatches to SoCLAaS
    or the local Ollama fallback exactly like chat queries do, and
    ``_ollama_response_content`` extracts the assistant text from either
    response shape.
    """
    from src.local_rag import _llm_chat, _ollama_response_content

    response = _llm_chat(
        model=model,
        messages=messages,
        # Size the reply budget to this exact batch: a valid 5-document array
        # is ~300 tokens, so a batch-proportional cap stops derailed
        # generations well before the request timeout without truncating
        # well-formed replies.
        options={"temperature": 0.0, "num_predict": max_output_tokens(_batch_doc_count(messages))},
        stream=False,
        timeout=timeout,
    )
    return _ollama_response_content(response)


def max_output_tokens(batch_size: int) -> int:
    """Generation budget for one batch reply.

    Each decision object is ~40-60 tokens; ~200 per document is generous, the
    floor keeps room for a one-document batch, and the cap stops a confused
    model from generating to its context limit and blowing the request
    timeout (the batch would be dropped anyway as unparseable).
    """
    return min(2048, max(256, 200 * max(1, int(batch_size))))


def classify_documents(
    items: list[AutoTagInput],
    *,
    model: str = DEFAULT_MODEL,
    chat_fn: Callable[..., str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
    on_batch_failure: Callable[[int, Exception], None] | None = None,
) -> dict[str, AutoTagDecision]:
    """Classify documents in batches; return ``{source_hash: AutoTagDecision}``.

    ``chat_fn`` (keyword-only: ``model``, ``messages``, ``timeout``) defaults
    to the app LLM transport; tests substitute a stub. Per-batch failures
    are logged and skipped so one bad batch cannot lose the rest of a large
    run; ``on_batch_failure(batch_index, exc)`` lets callers surface the
    damage (e.g. in the web app's run status) without changing this return
    contract.
    """
    chat = chat_fn or _default_chat_fn
    size = max(1, int(batch_size))
    decisions: dict[str, AutoTagDecision] = {}
    for index, start in enumerate(range(0, len(items), size)):
        batch = items[start : start + size]
        try:
            reply = chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(batch)},
                ],
                timeout=timeout,
            )
            batch_decisions = parse_classification_response(
                reply, [item.source_hash for item in batch], min_confidence=min_confidence
            )
        except Exception as exc:
            logger.warning(
                "auto-tag batch failed (%d documents, model %s): %s", len(batch), model, exc
            )
            if on_batch_failure is not None:
                try:
                    on_batch_failure(index, exc)
                except Exception:
                    logger.debug("auto-tag on_batch_failure callback raised", exc_info=True)
            continue
        decisions.update(batch_decisions)
    return decisions
