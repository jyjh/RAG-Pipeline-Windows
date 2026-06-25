import shutil
import uuid
from pathlib import Path

from src.sectioning import (
    PageText,
    SectionNode,
    build_section_records,
    content_for_section,
    outline_sections,
    parse_toc_entries,
    split_text,
    toc_sections_from_pages,
)
from src.pdf_registry import write_source_entry


class FakeDestination:
    def __init__(self, title):
        self.title = title


class FakeReader:
    def __init__(self, pages_by_item):
        self.pages_by_item = pages_by_item

    def get_destination_page_number(self, item):
        return self.pages_by_item[item]


def test_parse_toc_entries_handles_dot_leaders():
    entries = parse_toc_entries("7.2 Changes at test time . . . . . . . . . 206")

    assert entries == [("7.2", "Changes at test time", 206)]


def test_markdown_hash_headings_create_sections():
    root = Path.cwd() / f".tmp_test_sectioning_{uuid.uuid4().hex}"
    try:
        markdown_path = root / "notes.md"
        root.mkdir(parents=True)
        markdown_path.write_text("# Overview\n\nalpha\n\n## Details\n\nbeta", encoding="utf-8")

        records = build_section_records(markdown_path, source_root=root)

        paths = {record["section_path"] for record in records}
        assert "Overview" in paths
        assert "Overview > Details" in paths
        assert any(record["node_type"] == "chunk" and "beta" in record["content"] for record in records)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_outline_extraction_builds_nested_page_ranges():
    overview = FakeDestination("Overview")
    review = FakeDestination("Review")
    test_time = FakeDestination("Changes at test time")
    next_major = FakeDestination("Appendix")
    reader = FakeReader(
        {
            overview: 3,
            review: 3,
            test_time: 5,
            next_major: 9,
        }
    )

    sections = outline_sections(
        reader,
        [overview, [review, test_time], next_major],
        page_count=12,
    )

    assert sections[0].title == "Overview"
    assert sections[0].page_start == 4
    assert sections[0].page_end == 9
    assert sections[0].children[0].page_start == 4
    assert sections[0].children[0].page_end == 5
    assert sections[0].children[1].page_start == 6
    assert sections[0].children[1].page_end == 9


def test_toc_sections_exclude_front_matter_pages():
    pages = [
        PageText(1, "Title page\nContents\n1 Intro . . . . . 3\n1.1 Scope . . . . . 4"),
        PageText(2, "More contents"),
        PageText(3, "1 Intro\nreal content"),
        PageText(4, "1.1 Scope\nscoped content"),
    ]

    sections = toc_sections_from_pages(pages, doc_id_seed="doc")

    assert sections[0].page_start == 3
    assert sections[0].children[0].page_start == 4


def test_content_for_section_trims_before_next_same_page_heading():
    section = toc_sections_from_pages(
        [PageText(1, "Contents\n7.1 Review . . . . . 5\n7.2 Changes at test time . . . . . 5")],
        doc_id_seed="doc",
    )[0]
    section.page_start = 5
    section.page_end = 5
    section.next_title = "7.2 Changes at test time"
    pages = [
        PageText(
            5,
            "7.1 Review\nfirst section details\n7.2 Changes at test time\nsecond section details",
        )
    ]

    content = content_for_section(section, pages)

    assert "first section details" in content
    assert "second section details" not in content


def test_oversized_section_splits_without_crossing_sections():
    text = "alpha " * 500

    chunks = split_text(text, max_chars=200, overlap_chars=20)

    assert len(chunks) > 1
    assert all("beta" not in chunk for chunk in chunks)


def test_section_records_include_parent_pdf_source_metadata():
    root = Path.cwd() / f".tmp_test_sectioning_{uuid.uuid4().hex}"
    try:
        markdown_path = root / "processed" / "doc.md"
        markdown_path.parent.mkdir(parents=True)
        markdown_path.write_text("# Intro\n\nalpha content", encoding="utf-8")
        write_source_entry(
            processed_dir=markdown_path.parent,
            markdown_path=markdown_path,
            source_hash="hash-a",
            source_pdf_name="doc.pdf",
            source_pdf_path=root / "uploads" / "doc.pdf",
        )

        records = build_section_records(markdown_path, source_root=root)

        assert records
        assert {record["source_hash"] for record in records} == {"hash-a"}
        assert {record["source_pdf_name"] for record in records} == {"doc.pdf"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_section_records_index_enriched_markdown_when_source_pdf_has_no_text(monkeypatch):
    root = Path.cwd() / f".tmp_test_sectioning_{uuid.uuid4().hex}"
    try:
        pdf_path = root / "uploads" / "scan.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-1.4")
        markdown_path = root / "processed" / "scan.md"
        markdown_path.parent.mkdir(parents=True)
        markdown_path.write_text(
            "## Page 1\n\n> [Page Image Analysis]: viscous damping diagram and $mx'' + cx' + kx = 0$",
            encoding="utf-8",
        )
        write_source_entry(
            processed_dir=markdown_path.parent,
            markdown_path=markdown_path,
            source_hash="hash-scan",
            source_pdf_name="scan.pdf",
            source_pdf_path=pdf_path,
        )

        def fake_sections_from_pdf(path):
            return (
                [
                    SectionNode(
                        node_id="pdf-section",
                        parent_id="",
                        title="PDF Outline",
                        level=0,
                        page_start=1,
                        page_end=1,
                    )
                ],
                [PageText(1, "")],
            )

        monkeypatch.setattr("src.sectioning.sections_from_pdf", fake_sections_from_pdf)

        records = build_section_records(markdown_path, source_root=root)

        chunks = [record for record in records if record["node_type"] == "chunk"]
        assert chunks
        assert any("viscous damping diagram" in record["content"] for record in chunks)
        assert {record["source_pdf_path"] for record in records} == {str(pdf_path)}
    finally:
        shutil.rmtree(root, ignore_errors=True)
