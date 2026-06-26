from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class HybridPdfParser:
    def __init__(
        self,
        *,
        manual_parser: ManualTextPdfParser,
        docling_parser: DoclingPdfParser,
        scanned_page_parser: ScannedPageImageParser | None = None,
    ):
        self.manual_parser = manual_parser
        self.docling_parser = docling_parser
        self.scanned_page_parser = scanned_page_parser

    def parse(self, file_path: str) -> str:
        page_texts = self.manual_parser.extract_page_texts(file_path)
        if self.manual_parser.is_text_usable(page_texts):
            if hasattr(self.manual_parser, "asset_enrichment_page_hints"):
                asset_hints = self.manual_parser.asset_enrichment_page_hints(file_path, page_texts)
                asset_pages = set(asset_hints)
                enriched_asset_pages = self.manual_parser._enriched_pages_from_hints(asset_hints)
            else:
                asset_pages = self.manual_parser.asset_enrichment_pages(file_path, page_texts)
                enriched_asset_pages = set()
            return self.manual_parser.parse(
                file_path,
                include_assets=bool(asset_pages),
                asset_pages=asset_pages,
                enriched_asset_pages=enriched_asset_pages,
                page_texts=page_texts,
            )

        logger.info("Extracted PDF text is missing or low quality; using full Docling parsing: %s", file_path)
        try:
            content = self.docling_parser.parse(file_path)
        except Exception as exc:
            logger.warning("Docling parsing failed for scanned PDF; using page-image fallback: %s", exc)
            return self._parse_scanned_pages(file_path)

        if content.strip():
            return content

        logger.warning("Docling parsing returned empty content for scanned PDF; using page-image fallback: %s", file_path)
        return self._parse_scanned_pages(file_path)

    def _parse_scanned_pages(self, file_path: str) -> str:
        if self.scanned_page_parser is None:
            raise RuntimeError(f"Scanned PDF parsing failed and no page-image fallback is configured: {file_path}")
        return self.scanned_page_parser.parse(file_path)

HybridPdfParser.__module__ = _source_module.__name__
finalize_split_class(_source_module, HybridPdfParser)

