from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src._class_module_support import import_split_class

# Precompiled regexes for the sectioning hot loops. Functions like
# ``find_heading_line``, ``clean_title``, ``normalized_heading``, and
# ``split_text`` run per section / per line during indexing, so compiling these
# patterns once at import avoids recompiling them on every call.
_RE_ONLY_DIGITS = re.compile(r"\d{1,4}")
_RE_TOC_DOT_LEADER = re.compile(
    r"^\s*(?P<num>(?:[A-Z]+|[A-Z]?\d+(?:\.\d+)*|[A-Z](?:\.\d+)*))\s+"
    r"(?P<title>.+?)\s+(?:[.\u2026]\s*){2,}(?P<page>\d{1,4})\s*$"
)
_RE_TOC_PLAIN = re.compile(
    r"^\s*(?P<num>[A-Z]|\d+(?:\.\d+)*)\s+"
    r"(?P<title>[A-Za-z][^\n.]{2,}?)\s+(?P<page>\d{1,4})\s*$"
)
_RE_HASH_HEADING = re.compile(r"(?m)^(?P<hashes>#{1,6})\s+(?P<title>[^\n#][^\n]{2,160})\s*$")
_RE_NUMBERED_HEADING = re.compile(
    r"(?m)^(?P<title>(?P<num>\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+[^\n]{3,120})$"
)
_RE_FRONT_MATTER_NUMBER = re.compile(r"^(?:\[begin lecture [^\]]+\]\s*)?\(?\d+\)?$")
_RE_FRONT_MATTER_NUMBERED = re.compile(r"^\d+(?:\.\d+)*\s+[A-Za-z][^\n.]{2,}$")
_RE_SPLIT_PARAGRAPH = re.compile(r"\n\s*\n")
_RE_TAG_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
_RE_TITLE_WHITESPACE = re.compile(r"\s+")
_RE_TITLE_TRAILING_NUM = re.compile(r"\s+\d{1,4}$")
_RE_HEADING_LEADING_NUM = re.compile(r"^\d+(?:\.\d+)*\s+")
_RE_HEADING_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_CLASS_MODULE_PROXY_FUNCTIONS = (
    "has_enriched_markdown",
    "build_section_records",
    "records_for_section",
    "find_source_pdf",
    "sections_from_pdf",
    "outline_sections",
    "toc_sections_from_pages",
    "parse_toc_entries",
    "sections_from_markdown",
    "markdown_headings",
    "strip_front_matter",
    "content_for_section",
    "trim_to_heading",
    "find_heading_line",
    "split_text",
    "normalize_page_text",
    "collapse_repeated_edge_lines",
    "is_noise_line",
    "deterministic_summary",
    "tags_for",
    "flatten_sections",
    "format_page_range",
    "clean_title",
    "normalized_heading",
    "pages_sidecar_path",
    "write_pages_sidecar",
    "read_pages_sidecar",
    "stable_id",
)


DEFAULT_CHUNK_TARGET_TOKENS = 900
DEFAULT_CHUNK_OVERLAP_TOKENS = 120
CHARS_PER_TOKEN = 4
ENRICHED_MARKDOWN_MARKERS = ("[Vision Analysis]", "[Page Image Analysis]")


PageText = import_split_class("src.sectioning_classes.page_text", "PageText")
PageText.__module__ = __name__


SectionNode = import_split_class("src.sectioning_classes.section_node", "SectionNode")
SectionNode.__module__ = __name__


SectionChunk = import_split_class("src.sectioning_classes.section_chunk", "SectionChunk")
SectionChunk.__module__ = __name__


def has_enriched_markdown(content: str) -> bool:
    return any(marker in content for marker in ENRICHED_MARKDOWN_MARKERS)


def build_section_records(
    markdown_path: Path,
    *,
    source_root: Path | None = None,
    summary_mode: str = "hybrid",
    chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    source_root = source_root or Path.cwd()
    from src.pdf_registry import source_entry_for_markdown

    markdown_content = markdown_path.read_text(encoding="utf-8")
    source_entry = source_entry_for_markdown(markdown_path)
    mapped_pdf_value = str(source_entry.get("source_pdf_path", ""))
    mapped_pdf_path = Path(mapped_pdf_value) if mapped_pdf_value else None
    pdf_path = mapped_pdf_path if mapped_pdf_path is not None and mapped_pdf_path.exists() else None
    if pdf_path is None:
        pdf_path = find_source_pdf(markdown_path, source_root=source_root)
    max_chars = max(800, chunk_target_tokens * CHARS_PER_TOKEN)
    overlap_chars = max(0, chunk_overlap_tokens * CHARS_PER_TOKEN)
    use_markdown_content = has_enriched_markdown(markdown_content)
    if pdf_path is not None:
        # Prefer the page-text sidecar written at ingestion time to skip the
        # expensive pypdf extract_text() loop on every page. Falls back to None
        # (full re-extraction) if the sidecar is missing or stale -- so old /
        # manually-added Markdown still indexes identically.
        cached_pages = read_pages_sidecar(markdown_path)
        sections, pages = sections_from_pdf(pdf_path, cached_pages=cached_pages)
        source_path = str(pdf_path)
        doc_title = pdf_path.stem
        if use_markdown_content or not any(page.text.strip() for page in pages):
            sections, pages = sections_from_markdown(doc_title, markdown_content)
    else:
        sections, pages = sections_from_markdown(markdown_path.stem, markdown_content)
        source_path = str(markdown_path)
        doc_title = markdown_path.stem

    source_hash = str(source_entry.get("source_hash", ""))
    source_pdf_name = str(source_entry.get("source_pdf_name", pdf_path.name if pdf_path else ""))
    source_pdf_path = str(source_entry.get("source_pdf_path", pdf_path if pdf_path else ""))

    if not sections:
        sections, pages = sections_from_markdown(markdown_path.stem, markdown_content)
        source_path = str(markdown_path)
        doc_title = markdown_path.stem

    doc_id = stable_id("doc", markdown_path.stem)
    records: list[SectionChunk] = []
    chunk_counter = 0
    page_count = len(pages)
    all_titles = [section.title for section in flatten_sections(sections)]
    doc_tags = tags_for([doc_title, *all_titles])
    doc_summary = deterministic_summary(
        title=doc_title,
        section_path=doc_title,
        page_start=1 if page_count else None,
        page_end=page_count or None,
        tags=doc_tags,
        summary_mode=summary_mode,
    )
    records.append(
        SectionChunk(
            doc_id=doc_id,
            node_id=doc_id,
            parent_id="",
            node_type="document_summary",
            title=doc_title,
            section_path=doc_title,
            page_start=1 if page_count else None,
            page_end=page_count or None,
            content=doc_summary + "\n\nSections:\n" + "\n".join(f"- {title}" for title in all_titles[:80]),
            summary=doc_summary,
            tags=doc_tags,
            chunk_index=-1,
            source_path=source_path,
            source_hash=source_hash,
            source_pdf_name=source_pdf_name,
            source_pdf_path=source_pdf_path,
        )
    )

    for top_index, section in enumerate(sections):
        section_records = records_for_section(
            section,
            pages=pages,
            doc_id=doc_id,
            doc_title=doc_title,
            source_path=source_path,
            path_titles=[],
            summary_parent_id=doc_id,
            chunk_counter_start=chunk_counter,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            summary_mode=summary_mode,
            top_index=top_index,
            source_hash=source_hash,
            source_pdf_name=source_pdf_name,
            source_pdf_path=source_pdf_path,
        )
        records.extend(section_records)
        chunk_counter += sum(1 for record in section_records if record.node_type == "chunk")

    return [record.to_record() for record in records]


def records_for_section(
    section: SectionNode,
    *,
    pages: list[PageText],
    doc_id: str,
    doc_title: str,
    source_path: str,
    path_titles: list[str],
    summary_parent_id: str,
    chunk_counter_start: int,
    max_chars: int,
    overlap_chars: int,
    summary_mode: str,
    top_index: int,
    source_hash: str = "",
    source_pdf_name: str = "",
    source_pdf_path: str = "",
) -> list[SectionChunk]:
    section_path_titles = [*path_titles, section.title]
    section_path = " > ".join(section_path_titles)
    tags = tags_for([doc_title, *section_path_titles])
    summary = deterministic_summary(
        title=section.title,
        section_path=section_path,
        page_start=section.page_start,
        page_end=section.page_end,
        tags=tags,
        summary_mode=summary_mode,
    )
    section_summary_id = stable_id(doc_id, "summary", section.node_id)
    records: list[SectionChunk] = []

    create_summary = section.level <= 1 or bool(section.children)
    active_summary_parent = summary_parent_id
    if create_summary:
        records.append(
            SectionChunk(
                doc_id=doc_id,
                node_id=section_summary_id,
                parent_id=summary_parent_id,
                node_type="section_summary",
                title=section.title,
                section_path=section_path,
                page_start=section.page_start,
                page_end=section.page_end,
                content=summary,
                summary=summary,
                tags=tags,
                chunk_index=-1,
                source_path=source_path,
                source_hash=source_hash,
                source_pdf_name=source_pdf_name,
                source_pdf_path=source_pdf_path,
            )
        )
        active_summary_parent = section_summary_id

    if section.children:
        chunk_counter = chunk_counter_start
        for child_index, child in enumerate(section.children):
            child_records = records_for_section(
                child,
                pages=pages,
                doc_id=doc_id,
                doc_title=doc_title,
                source_path=source_path,
                path_titles=section_path_titles,
                summary_parent_id=active_summary_parent,
                chunk_counter_start=chunk_counter,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                summary_mode=summary_mode,
                top_index=top_index + child_index,
                source_hash=source_hash,
                source_pdf_name=source_pdf_name,
                source_pdf_path=source_pdf_path,
            )
            chunk_counter += sum(1 for record in child_records if record.node_type == "chunk")
            records.extend(child_records)
        return records

    raw_content = content_for_section(section, pages)
    chunks = split_text(raw_content, max_chars=max_chars, overlap_chars=overlap_chars)
    chunk_counter = chunk_counter_start
    for offset, chunk in enumerate(chunks):
        chunk_index = chunk_counter
        chunk_counter += 1
        prefixed = (
            f"Section: {section_path}\n"
            f"Pages: {format_page_range(section.page_start, section.page_end)}\n\n"
            f"{chunk}"
        ).strip()
        records.append(
            SectionChunk(
                doc_id=doc_id,
                node_id=stable_id(doc_id, "chunk", section.node_id, str(offset)),
                parent_id=active_summary_parent,
                node_type="chunk",
                title=section.title,
                section_path=section_path,
                page_start=section.page_start,
                page_end=section.page_end,
                content=prefixed,
                summary=summary,
                tags=tags,
                chunk_index=chunk_index,
                source_path=source_path,
                source_hash=source_hash,
                source_pdf_name=source_pdf_name,
                source_pdf_path=source_pdf_path,
            )
        )
    return records


def find_source_pdf(markdown_path: Path, *, source_root: Path) -> Path | None:
    candidates = [
        source_root / "data" / f"{markdown_path.stem}.pdf",
        markdown_path.with_suffix(".pdf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    data_dir = source_root / "data"
    if data_dir.exists():
        matches = sorted(data_dir.rglob(f"{markdown_path.stem}.pdf"))
        if matches:
            return matches[-1]
    return None


# --- Page-text sidecar -------------------------------------------------------
# Every PDF is pypdf-parsed twice in the pipeline: once during ingestion (to
# produce Markdown) and once during indexing (sections_from_pdf re-extracts
# every page). At 100GB scale that doubles pypdf CPU. The sidecar persists the
# normalized per-page text next to the Markdown output so the indexing pass can
# skip the expensive extract_text() loop and only pay for the cheap outline
# read. Stored values are already normalize_page_text()-ed -- byte-identical to
# what sections_from_pdf would compute -- so consuming them directly cannot
# change indexing output.

_PAGES_SIDECAR_VERSION = 1


def pages_sidecar_path(markdown_path: Path) -> Path:
    """Sidecar path for a given Markdown output (``<file>.md`` -> ``<file>.pages.json``)."""
    return markdown_path.with_suffix(".pages.json")


def write_pages_sidecar(markdown_path: Path, page_texts: Iterable[str]) -> None:
    """Persist normalized per-page text next to the Markdown output.

    Called from ingestion after page extraction. ``page_texts`` are raw
    extracted strings; this applies ``normalize_page_text`` so the values match
    what ``sections_from_pdf`` would produce, guaranteeing index-time parity.
    Writes atomically so a crash cannot leave a half-written sidecar.
    """
    import json

    sidecar = pages_sidecar_path(markdown_path)
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    # Stream the page array so a very large PDF does not create a second full
    # normalized-page list just to serialize the sidecar. The reader accepts
    # the same JSON shape, including the exact page_count field.
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write('{"version":%d,"page_count":' % _PAGES_SIDECAR_VERSION)
        count_position = handle.tell()
        handle.write("0".ljust(20))
        handle.write(',"pages":[')
        count = 0
        for text in page_texts:
            if count:
                handle.write(",")
            json.dump(normalize_page_text(text), handle, ensure_ascii=False)
            count += 1
        handle.write("]}")
        end_position = handle.tell()
        handle.seek(count_position)
        handle.write(str(count).ljust(20))
        handle.seek(end_position)
    tmp.replace(sidecar)


def read_pages_sidecar(markdown_path: Path) -> list[str] | None:
    """Read the page-text sidecar, or ``None`` if absent/invalid.

    Returns the list of normalized page texts (1-based by position). Any
    schema/version mismatch returns ``None`` so the caller falls back to
    re-extraction -- a missing sidecar must never change indexing output.
    """
    import json

    sidecar = pages_sidecar_path(markdown_path)
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _PAGES_SIDECAR_VERSION:
        return None
    pages = payload.get("pages")
    if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
        return None
    return pages


def sections_from_pdf(
    pdf_path: Path,
    *,
    cached_pages: list[str] | None = None,
) -> tuple[list[SectionNode], list[PageText]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    if cached_pages is not None and len(cached_pages) == len(reader.pages):
        # Skip the expensive extract_text() loop -- the caller (build_section_records)
        # already loaded normalized page text from the sidecar written at ingest time.
        pages = [PageText(index + 1, text) for index, text in enumerate(cached_pages)]
    else:
        pages = [
            PageText(index + 1, normalize_page_text(page.extract_text() or ""))
            for index, page in enumerate(reader.pages)
        ]
    outline = getattr(reader, "outline", None) or []
    sections = outline_sections(reader, outline, page_count=len(pages))
    if not sections:
        sections = toc_sections_from_pages(pages, doc_id_seed=pdf_path.stem)
    if not sections:
        sections, _ = sections_from_markdown(pdf_path.stem, "\n\n".join(page.text for page in pages))
    return sections, pages


def outline_sections(reader: Any, outline: list[Any], *, page_count: int) -> list[SectionNode]:
    flat: list[tuple[int, int, str, int]] = []

    def walk(items: list[Any], level: int, order: list[int]) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1, order)
                continue
            title = clean_title(str(getattr(item, "title", item)))
            if not title:
                continue
            try:
                page_no = int(reader.get_destination_page_number(item)) + 1
            except Exception:
                page_no = 1
            order[0] += 1
            flat.append((level, order[0], title, max(1, min(page_no, page_count or 1))))

    walk(outline, 0, [0])
    if not flat:
        return []

    nodes: list[SectionNode] = []
    stack: list[SectionNode] = []
    all_nodes: list[SectionNode] = []
    for level, order, title, page_start in flat:
        node = SectionNode(
            node_id=stable_id("section", str(order), title),
            parent_id="",
            title=title,
            level=level,
            page_start=page_start,
        )
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            node.parent_id = stack[-1].node_id
            stack[-1].children.append(node)
        else:
            nodes.append(node)
        stack.append(node)
        all_nodes.append(node)

    for index, node in enumerate(all_nodes):
        node.page_end = page_count or node.page_start
        node.next_title = ""
        for later in all_nodes[index + 1 :]:
            if later.level <= node.level:
                node.page_end = max(node.page_start or 1, (later.page_start or 1) - 1)
                node.next_title = later.title if later.page_start == node.page_start else ""
                break
        if node.page_end is None:
            node.page_end = page_count
    return nodes


def toc_sections_from_pages(pages: list[PageText], *, doc_id_seed: str) -> list[SectionNode]:
    toc_text = "\n".join(page.text for page in pages[:8])
    entries = parse_toc_entries(toc_text)
    if not entries:
        return []
    roots: list[SectionNode] = []
    stack: list[SectionNode] = []
    nodes: list[SectionNode] = []
    for order, entry in enumerate(entries):
        number, title, page_no = entry
        level = number.count(".")
        node = SectionNode(
            node_id=stable_id("toc", doc_id_seed, str(order), number, title),
            parent_id="",
            title=f"{number} {title}".strip(),
            level=level,
            page_start=page_no,
        )
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            node.parent_id = stack[-1].node_id
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
        nodes.append(node)
    page_count = len(pages)
    for index, node in enumerate(nodes):
        node.page_end = page_count
        for later in nodes[index + 1 :]:
            if later.level <= node.level:
                node.page_end = max(node.page_start or 1, (later.page_start or 1) - 1)
                node.next_title = later.title if later.page_start == node.page_start else ""
                break
    return roots


def parse_toc_entries(text: str) -> list[tuple[str, str, int]]:
    entries: list[tuple[str, str, int]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower() in {"contents", "table of contents"}:
            continue
        match = _RE_TOC_DOT_LEADER.match(stripped) or _RE_TOC_PLAIN.match(stripped)
        if not match:
            continue
        title = clean_title(match.group("title"))
        if not title or title.isdigit():
            continue
        try:
            page_no = int(match.group("page"))
        except ValueError:
            continue
        entries.append((match.group("num"), title, page_no))
    return entries


def sections_from_markdown(doc_title: str, content: str) -> tuple[list[SectionNode], list[PageText]]:
    clean_content = strip_front_matter(content)
    headings = markdown_headings(clean_content)
    if not headings:
        root = SectionNode(
            node_id=stable_id("markdown", doc_title),
            parent_id="",
            title=doc_title,
            level=0,
            page_start=1,
            page_end=1,
        )
        return [root], [PageText(1, normalize_page_text(clean_content))]

    roots: list[SectionNode] = []
    stack: list[SectionNode] = []
    node_pages: list[tuple[SectionNode, str]] = []
    for index, (offset, level, title) in enumerate(headings):
        end_offset = headings[index + 1][0] if index + 1 < len(headings) else len(clean_content)
        text = clean_content[offset:end_offset].strip()
        node = SectionNode(
            node_id=stable_id("markdown", doc_title, str(index), title),
            parent_id="",
            title=title,
            level=level,
            page_start=index + 1,
            page_end=index + 1,
        )
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            node.parent_id = stack[-1].node_id
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
        node_pages.append((node, text))

    pages = [
        PageText(index + 1, text)
        for index, (_, text) in enumerate(node_pages)
    ]
    return roots, pages


def markdown_headings(content: str) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    seen_offsets: set[int] = set()
    for match in _RE_HASH_HEADING.finditer(content):
        title = clean_title(match.group("title"))
        if is_noise_line(title):
            continue
        level = len(match.group("hashes")) - 1
        headings.append((match.start(), level, title))
        seen_offsets.add(match.start())
    for match in _RE_NUMBERED_HEADING.finditer(content):
        if match.start() in seen_offsets:
            continue
        title = clean_title(match.group("title"))
        if is_noise_line(title):
            continue
        number = match.group("num")
        level = number.count(".")
        headings.append((match.start(), level, title))
    return sorted(headings, key=lambda item: item[0])


def strip_front_matter(content: str) -> str:
    lines = content.replace("\r\n", "\n").splitlines()
    content_markers = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _RE_FRONT_MATTER_NUMBER.match(stripped.lower()):
            continue
        if _RE_FRONT_MATTER_NUMBERED.match(stripped):
            content_markers.append(index)
            if len(content_markers) >= 2:
                return "\n".join(lines[index:]).strip()
    return content.strip()


def content_for_section(section: SectionNode, pages: list[PageText]) -> str:
    if section.page_start is None or section.page_end is None:
        page_text = ""
    else:
        selected = [
            page.text
            for page in pages
            if section.page_start <= page.page_no <= section.page_end
        ]
        page_text = "\n\n".join(selected)
    page_text = trim_to_heading(page_text, section.title, next_title=section.next_title)
    return page_text.strip()


def trim_to_heading(text: str, title: str, *, next_title: str = "") -> str:
    if not text:
        return ""
    lines = text.splitlines()
    start = find_heading_line(lines, title)
    if start is not None:
        lines = lines[start:]
    if next_title:
        end = find_heading_line(lines[1:], next_title)
        if end is not None:
            lines = lines[: end + 1]
    return "\n".join(lines).strip()


def find_heading_line(lines: list[str], title: str) -> int | None:
    target = normalized_heading(title)
    if not target:
        return None
    for index, line in enumerate(lines):
        candidate = normalized_heading(line)
        if target and (target in candidate or candidate in target):
            return index
    return None


def split_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in _RE_SPLIT_PARAGRAPH.split(text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(start + 1, end - overlap_chars)
            continue
        projected = current_len + len(paragraph) + (2 if current else 0)
        if current and projected > max_chars:
            flush()
        current.append(paragraph)
        current_len += len(paragraph) + (2 if current_len else 0)
    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def normalize_page_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").splitlines()]
    cleaned = [line for line in lines if not is_noise_line(line)]
    return "\n".join(collapse_repeated_edge_lines(cleaned)).strip()


def collapse_repeated_edge_lines(lines: list[str]) -> list[str]:
    if len(lines) < 4:
        return lines
    cleaned = list(lines)
    for index in (0, len(cleaned) - 1):
        value = cleaned[index].strip()
        if value and sum(1 for line in cleaned if line.strip() == value) > 2:
            cleaned[index] = ""
    return [line for line in cleaned if line.strip()]


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _RE_ONLY_DIGITS.fullmatch(stripped):
        return True
    if stripped.lower() in {"contents", "table of contents"}:
        return True
    return False


def deterministic_summary(
    *,
    title: str,
    section_path: str,
    page_start: int | None,
    page_end: int | None,
    tags: list[str],
    summary_mode: str,
) -> str:
    tag_text = ", ".join(tags[:8]) if tags else "general content"
    mode_note = "outline/heading-derived"
    if summary_mode == "llm":
        mode_note = "deterministic fallback"
    return (
        f"{title}. {mode_note} summary for {section_path}. "
        f"Pages {format_page_range(page_start, page_end)}. Tags: {tag_text}."
    )


def tags_for(parts: list[str]) -> list[str]:
    stop = {
        "and",
        "for",
        "the",
        "with",
        "from",
        "into",
        "this",
        "that",
        "lecture",
        "optional",
        "reading",
        "section",
        "summary",
    }
    tags: list[str] = []
    for part in parts:
        for word in _RE_TAG_WORD.findall(part.lower()):
            if word in stop or word in tags:
                continue
            tags.append(word)
            if len(tags) >= 12:
                return tags
    return tags


def flatten_sections(sections: list[SectionNode]) -> list[SectionNode]:
    flat: list[SectionNode] = []
    for section in sections:
        flat.append(section)
        flat.extend(flatten_sections(section.children))
    return flat


def format_page_range(page_start: int | None, page_end: int | None) -> str:
    if page_start is None and page_end is None:
        return "unknown"
    if page_start == page_end or page_end is None:
        return str(page_start)
    if page_start is None:
        return str(page_end)
    return f"{page_start}-{page_end}"


def clean_title(title: str) -> str:
    title = _RE_TITLE_WHITESPACE.sub(" ", title.replace("\ufb01", "fi").replace("\ufb02", "fl")).strip()
    title = _RE_TITLE_TRAILING_NUM.sub("", title).strip()
    return title


def normalized_heading(value: str) -> str:
    value = clean_title(value)
    value = _RE_HEADING_LEADING_NUM.sub("", value)
    return _RE_HEADING_NON_ALNUM.sub("", value.lower())


def stable_id(*parts: str) -> str:
    raw = "\0".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
