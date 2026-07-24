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
        # Cheap usability probe: extract a spread of pages (first/middle/last +
        # a few between) instead of every page up front. For a clearly scanned
        # PDF this lets us route straight to page-at-a-time Docling WITHOUT ever
        # materializing the full pypdf page-text list -- a meaningful memory win
        # on a large dense PDF (the whole-document pypdf extraction builds a
        # list[str] of every page). For a born-digital PDF the sample passes the
        # usability check and we proceed to the (unchanged) full extraction; the
        # reader is cached so the probe adds no extra parse cost there.
        sample = self.manual_parser.extract_sampled_page_texts(file_path)
        if not self.manual_parser.is_text_usable(sample):
            logger.info(
                "Sampled text probe indicates a scanned/low-quality PDF; using Docling parsing: %s",
                file_path,
            )
            try:
                content = self.docling_parser.parse(file_path)
            except Exception as exc:
                logger.warning(
                    "Docling parsing failed for scanned PDF; using page-image fallback: %s", exc
                )
                return self._parse_scanned_pages(file_path)
            if content.strip():
                return content
            logger.warning(
                "Docling parsing returned empty content for scanned PDF; "
                "using page-image fallback: %s",
                file_path,
            )
            return self._parse_scanned_pages(file_path)

        # Born-digital: the sample was usable, so extract the full text. This is
        # the fast path and is unchanged from before -- we just deferred it past
        # the probe so scanned PDFs never pay this cost.
        page_texts = self.manual_parser.extract_page_texts(file_path)
        # Re-check the full text in case the sample passed but the document as a
        # whole is borderline (e.g. only the sampled pages had text). If the
        # full extraction is not usable after all, fall through to Docling.
        if not self.manual_parser.is_text_usable(page_texts):
            logger.info(
                "Full text extraction was not usable despite a passing sample; "
                "using Docling parsing: %s",
                file_path,
            )
            try:
                content = self.docling_parser.parse(file_path)
            except Exception as exc:
                logger.warning(
                    "Docling parsing failed for scanned PDF; using page-image fallback: %s", exc
                )
                return self._parse_scanned_pages(file_path)
            if content.strip():
                return content
            return self._parse_scanned_pages(file_path)

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

    def _parse_scanned_pages(self, file_path: str) -> str:
        if self.scanned_page_parser is None:
            raise RuntimeError(f"Scanned PDF parsing failed and no page-image fallback is configured: {file_path}")
        return self.scanned_page_parser.parse(file_path)

HybridPdfParser.__module__ = _source_module.__name__
finalize_split_class(_source_module, HybridPdfParser)

