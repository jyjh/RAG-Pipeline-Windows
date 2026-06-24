from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CHUNK_TARGET_TOKENS = 900
DEFAULT_CHUNK_OVERLAP_TOKENS = 120
CHARS_PER_TOKEN = 4


@dataclass
class PageText:
    page_no: int
    text: str


@dataclass
class SectionNode:
    node_id: str
    parent_id: str
    title: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    children: list["SectionNode"] = field(default_factory=list)
    next_title: str = ""


@dataclass
class SectionChunk:
    doc_id: str
    node_id: str
    parent_id: str
    node_type: str
    title: str
    section_path: str
    page_start: int | None
    page_end: int | None
    content: str
    summary: str
    tags: list[str]
    chunk_index: int
    source_path: str

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "doc_id": self.doc_id,
            "parent_id": self.parent_id,
            "node_type": self.node_type,
            "file_path": self.source_path,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "title": self.title,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "summary": self.summary,
            "tags": list(self.tags),
        }


def build_section_records(
    markdown_path: Path,
    *,
    source_root: Path | None = None,
    summary_mode: str = "hybrid",
    chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    source_root = source_root or Path.cwd()
    pdf_path = find_source_pdf(markdown_path, source_root=source_root)
    max_chars = max(800, chunk_target_tokens * CHARS_PER_TOKEN)
    overlap_chars = max(0, chunk_overlap_tokens * CHARS_PER_TOKEN)
    if pdf_path is not None:
        sections, pages = sections_from_pdf(pdf_path)
        source_path = str(pdf_path)
        doc_title = pdf_path.stem
    else:
        content = markdown_path.read_text(encoding="utf-8")
        sections, pages = sections_from_markdown(markdown_path.stem, content)
        source_path = str(markdown_path)
        doc_title = markdown_path.stem

    if not sections:
        content = markdown_path.read_text(encoding="utf-8")
        sections, pages = sections_from_markdown(markdown_path.stem, content)
        source_path = str(markdown_path)
        doc_title = markdown_path.stem

    doc_id = stable_id("doc", markdown_path.stem)
    records: list[SectionChunk] = []
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
        )
    )

    for top_index, section in enumerate(sections):
        records.extend(
            records_for_section(
                section,
                pages=pages,
                doc_id=doc_id,
                doc_title=doc_title,
                source_path=source_path,
                path_titles=[],
                summary_parent_id=doc_id,
                chunk_counter_start=len([record for record in records if record.node_type == "chunk"]),
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                summary_mode=summary_mode,
                top_index=top_index,
            )
        )

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
            )
        )
        active_summary_parent = section_summary_id

    if section.children:
        chunk_counter = chunk_counter_start + len([record for record in records if record.node_type == "chunk"])
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
            )
            chunk_counter += len([record for record in child_records if record.node_type == "chunk"])
            records.extend(child_records)
        return records

    raw_content = content_for_section(section, pages)
    chunks = split_text(raw_content, max_chars=max_chars, overlap_chars=overlap_chars)
    for offset, chunk in enumerate(chunks):
        chunk_index = chunk_counter_start + len([record for record in records if record.node_type == "chunk"])
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


def sections_from_pdf(pdf_path: Path) -> tuple[list[SectionNode], list[PageText]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
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
    dot_leader = re.compile(
        r"^\s*(?P<num>(?:[A-Z]+|[A-Z]?\d+(?:\.\d+)*|[A-Z](?:\.\d+)*))\s+"
        r"(?P<title>.+?)\s+(?:[.\u2026]\s*){2,}(?P<page>\d{1,4})\s*$"
    )
    plain = re.compile(
        r"^\s*(?P<num>[A-Z]|\d+(?:\.\d+)*)\s+"
        r"(?P<title>[A-Za-z][^\n.]{2,}?)\s+(?P<page>\d{1,4})\s*$"
    )
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower() in {"contents", "table of contents"}:
            continue
        match = dot_leader.match(stripped) or plain.match(stripped)
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
    pattern = re.compile(
        r"(?m)^(?:#{1,6}\s+)?(?P<title>(?P<num>\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+[^\n]{3,120})$"
    )
    for match in pattern.finditer(content):
        title = clean_title(match.group("title"))
        if is_noise_line(title):
            continue
        number = match.group("num")
        level = number.count(".")
        headings.append((match.start(), level, title))
    return headings


def strip_front_matter(content: str) -> str:
    lines = content.replace("\r\n", "\n").splitlines()
    content_markers = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^(?:\[begin lecture [^\]]+\]\s*)?\(?\d+\)?$", stripped.lower()):
            continue
        if re.match(r"^\d+(?:\.\d+)*\s+[A-Za-z][^\n.]{2,}$", stripped):
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
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
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
    if re.fullmatch(r"\d{1,4}", stripped):
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
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", part.lower()):
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
    title = re.sub(r"\s+", " ", title.replace("\ufb01", "fi").replace("\ufb02", "fl")).strip()
    title = re.sub(r"\s+\d{1,4}$", "", title).strip()
    return title


def normalized_heading(value: str) -> str:
    value = clean_title(value)
    value = re.sub(r"^\d+(?:\.\d+)*\s+", "", value)
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def stable_id(*parts: str) -> str:
    raw = "\0".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
