from __future__ import annotations

from typing import Protocol

import src.ingestion as _source_module
from src._class_module_support import finalize_split_class


class PdfParser(Protocol):
    def parse(self, file_path: str) -> str:
        """Return enriched Markdown for a PDF."""

PdfParser.__module__ = _source_module.__name__
finalize_split_class(_source_module, PdfParser)

