from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class DocumentProcessor:
    def __init__(
        self,
        vision_model: str = DEFAULT_VISION_MODEL,
        parser_mode: str = DEFAULT_PDF_PARSER_MODE,
        accelerator: str | Any = DEFAULT_DOCLING_ACCELERATOR,
        num_threads: int = 8,
        vision_enabled: bool = DEFAULT_VISION_ENABLED,
        min_text_chars: int = 20,
        asset_triggers: str = DEFAULT_ASSET_TRIGGERS,
        asset_dir: str | Path = DEFAULT_ASSET_DIR,
        code_enrichment: bool = DEFAULT_CODE_ENRICHMENT,
        formula_enrichment: bool = DEFAULT_FORMULA_ENRICHMENT,
        ocr_backend: str = DEFAULT_OCR_BACKEND,
        ocr_langs: str | list[str] | tuple[str, ...] | None = None,
        ocr_force_full_page: bool = DEFAULT_OCR_FORCE_FULL_PAGE,
        ocr_bitmap_area_threshold: float = DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
        rapidocr_backend: str = DEFAULT_RAPIDOCR_BACKEND,
        tesseract_cmd: str = DEFAULT_TESSERACT_CMD,
        tesseract_data_path: str | None = DEFAULT_TESSERACT_DATA_PATH,
        tesseract_psm: int | str | None = DEFAULT_TESSERACT_PSM,
        progress_enabled: bool = True,
        max_pages_whole_doc: int = 50,
    ):
        self.vision_model = vision_model
        vision_enabled = _normalize_bool(vision_enabled, DEFAULT_VISION_ENABLED)
        code_enrichment = _normalize_bool(code_enrichment, DEFAULT_CODE_ENRICHMENT)
        formula_enrichment = _normalize_bool(formula_enrichment, DEFAULT_FORMULA_ENRICHMENT)
        if vision_enabled:
            self.vision_describer: VisionDescriber = OllamaVisionDescriber(vision_model=vision_model)
        else:
            self.vision_describer = DisabledVisionDescriber()
        from src.asset_store import ImageAssetStore

        self.image_asset_store = ImageAssetStore(asset_dir)

        self.docling_parser = DoclingPdfParser(
            vision_describer=self.vision_describer,
            image_asset_store=self.image_asset_store,
            accelerator=accelerator,
            num_threads=num_threads,
            code_enrichment=code_enrichment,
            formula_enrichment=formula_enrichment,
            ocr_enabled=True,
            ocr_backend=ocr_backend,
            ocr_langs=ocr_langs,
            ocr_force_full_page=ocr_force_full_page,
            ocr_bitmap_area_threshold=ocr_bitmap_area_threshold,
            rapidocr_backend=rapidocr_backend,
            tesseract_cmd=tesseract_cmd,
            tesseract_data_path=tesseract_data_path,
            tesseract_psm=tesseract_psm,
            progress_enabled=progress_enabled,
            max_pages_whole_doc=max_pages_whole_doc,
        )
        self.asset_docling_parser = DoclingPdfParser(
            vision_describer=self.vision_describer,
            image_asset_store=self.image_asset_store,
            accelerator=accelerator,
            num_threads=num_threads,
            ocr_enabled=False,
            ocr_backend=ocr_backend,
            ocr_langs=ocr_langs,
            ocr_force_full_page=ocr_force_full_page,
            ocr_bitmap_area_threshold=ocr_bitmap_area_threshold,
            rapidocr_backend=rapidocr_backend,
            tesseract_cmd=tesseract_cmd,
            tesseract_data_path=tesseract_data_path,
            tesseract_psm=tesseract_psm,
            progress_enabled=progress_enabled,
            max_pages_whole_doc=max_pages_whole_doc,
        )
        self.enriched_asset_docling_parser = DoclingPdfParser(
            vision_describer=self.vision_describer,
            image_asset_store=self.image_asset_store,
            accelerator=accelerator,
            num_threads=num_threads,
            code_enrichment=code_enrichment,
            formula_enrichment=formula_enrichment,
            ocr_enabled=False,
            ocr_backend=ocr_backend,
            ocr_langs=ocr_langs,
            ocr_force_full_page=ocr_force_full_page,
            ocr_bitmap_area_threshold=ocr_bitmap_area_threshold,
            rapidocr_backend=rapidocr_backend,
            tesseract_cmd=tesseract_cmd,
            tesseract_data_path=tesseract_data_path,
            tesseract_psm=tesseract_psm,
            progress_enabled=progress_enabled,
            max_pages_whole_doc=max_pages_whole_doc,
        )
        self.manual_parser = ManualTextPdfParser(
            docling_parser=self.asset_docling_parser,
            enriched_docling_parser=self.enriched_asset_docling_parser,
            min_text_chars=min_text_chars,
            asset_triggers=asset_triggers,
            progress_enabled=progress_enabled,
        )
        self.scanned_page_parser = ScannedPageImageParser(
            vision_describer=self.vision_describer,
            vision_enabled=vision_enabled,
            progress_enabled=progress_enabled,
        )
        self.parser = self._select_parser(parser_mode)

    def set_source_context(self, *, source_hash: str = "", source_pdf_name: str = "") -> None:
        for parser in (self.docling_parser, self.asset_docling_parser, self.enriched_asset_docling_parser):
            parser.set_source_context(source_hash=source_hash, source_pdf_name=source_pdf_name)

    def _select_parser(self, parser_mode: str) -> PdfParser:
        mode = parser_mode.lower()
        if mode == "hybrid":
            return HybridPdfParser(
                manual_parser=self.manual_parser,
                docling_parser=self.docling_parser,
                scanned_page_parser=self.scanned_page_parser,
            )
        if mode == "manual":
            return self.manual_parser
        if mode == "docling":
            return self.docling_parser
        raise ValueError("parser_mode must be one of: hybrid, manual, docling")

    def process_images(self, image_data: bytes) -> str:
        return self.vision_describer.describe(image_data)

    def _item_to_markdown(self, item) -> str:
        return DoclingMarkdownRenderer.item_to_markdown(item)

    def process_pdf(self, file_path: str) -> str:
        return self.parser.parse(file_path)

DocumentProcessor.__module__ = _source_module.__name__
finalize_split_class(_source_module, DocumentProcessor)

