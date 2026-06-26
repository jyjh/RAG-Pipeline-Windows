from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.sectioning as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


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
    source_hash: str = ""
    source_pdf_name: str = ""
    source_pdf_path: str = ""

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
            "source_hash": self.source_hash,
            "source_pdf_name": self.source_pdf_name,
            "source_pdf_path": self.source_pdf_path,
        }

SectionChunk.__module__ = _source_module.__name__
finalize_split_class(_source_module, SectionChunk)

