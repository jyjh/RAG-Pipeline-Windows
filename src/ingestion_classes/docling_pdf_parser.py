from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


@dataclass
class DoclingPdfParser:
    vision_describer: VisionDescriber
    accelerator: str | Any = DEFAULT_DOCLING_ACCELERATOR
    num_threads: int = 8
    table_structure: bool = True
    code_enrichment: bool = False
    formula_enrichment: bool = False
    ocr_enabled: bool = True
    ocr_backend: str = DEFAULT_OCR_BACKEND
    ocr_langs: str | list[str] | tuple[str, ...] | None = None
    ocr_force_full_page: bool = DEFAULT_OCR_FORCE_FULL_PAGE
    ocr_bitmap_area_threshold: float = DEFAULT_OCR_BITMAP_AREA_THRESHOLD
    rapidocr_backend: str = DEFAULT_RAPIDOCR_BACKEND
    tesseract_cmd: str = DEFAULT_TESSERACT_CMD
    tesseract_data_path: str | None = DEFAULT_TESSERACT_DATA_PATH
    tesseract_psm: int | str | None = DEFAULT_TESSERACT_PSM
    progress_enabled: bool = False
    reader_factory: Callable[[str], Any] = _default_pdf_reader
    image_asset_store: Any | None = None
    converter: Any | None = None

    def __post_init__(self) -> None:
        self.renderer = DoclingMarkdownRenderer(self.vision_describer, image_asset_store=self.image_asset_store)

    def set_source_context(self, *, source_hash: str = "", source_pdf_name: str = "") -> None:
        self.renderer.set_source_context(source_hash=source_hash, source_pdf_name=source_pdf_name)

    def parse(self, file_path: str) -> str:
        logger.info("Parsing PDF with Docling: %s", file_path)
        _progress_status(f"Docling parse: {Path(file_path).name}", enabled=self.progress_enabled)
        if self.progress_enabled:
            return self.parse_by_page(file_path)

        doc = self.convert(file_path).document
        return self.renderer.render_document(doc)

    def parse_by_page(self, file_path: str) -> str:
        try:
            reader = self.reader_factory(file_path)
        except Exception as exc:
            logger.warning("Could not open PDF for Docling page progress; parsing whole file: %s", exc)
            doc = self.convert(file_path).document
            return self.renderer.render_document(doc)

        page_count = len(getattr(reader, "pages", []))
        if page_count == 0:
            return ""

        parts: list[str] = []
        desc = f"Docling pages: {Path(file_path).name}"
        pages = enumerate(reader.pages, start=1)
        with tempfile.TemporaryDirectory(prefix="rag_docling_pages_") as temp_dir:
            for page_no, page in _iter_with_progress(
                pages,
                enabled=self.progress_enabled,
                total=page_count,
                desc=desc,
                unit="page",
            ):
                page_path = Path(temp_dir) / f"page_{page_no}.pdf"
                try:
                    writer = _new_pdf_writer()
                    writer.add_page(page)
                    with page_path.open("wb") as handle:
                        writer.write(handle)
                    page_doc = self.convert(str(page_path)).document
                    page_md = self.renderer.render_document(page_doc, page_no_override=page_no)
                except Exception as exc:
                    logger.warning("Docling failed on page %s; continuing: %s", page_no, exc)
                    page_md = ""

                if page_md:
                    parts.append(page_md)

        return "\n\n".join(parts)

    def convert(self, file_path: str):
        self._ensure_converter()
        assert self.converter is not None
        try:
            return self.converter.convert(file_path)
        except Exception as exc:
            if _accelerator_value(self.accelerator) == "cpu" or not _looks_like_accelerator_failure(exc):
                raise

            logger.warning(
                "Docling accelerator '%s' failed; retrying with CPU. Error: %s",
                _accelerator_value(self.accelerator),
                exc,
            )
            self.accelerator = "cpu"
            self.converter = _build_docling_converter(
                accelerator="cpu",
                num_threads=self.num_threads,
                table_structure=self.table_structure,
                code_enrichment=self.code_enrichment,
                formula_enrichment=self.formula_enrichment,
                ocr_enabled=self.ocr_enabled,
                ocr_backend=self.ocr_backend,
                ocr_langs=self.ocr_langs,
                ocr_force_full_page=self.ocr_force_full_page,
                ocr_bitmap_area_threshold=self.ocr_bitmap_area_threshold,
                rapidocr_backend=self.rapidocr_backend,
                tesseract_cmd=self.tesseract_cmd,
                tesseract_data_path=self.tesseract_data_path,
                tesseract_psm=self.tesseract_psm,
            )
            return self.converter.convert(file_path)

    def _ensure_converter(self) -> None:
        if self.converter is not None:
            return

        logger.info("Initializing Docling converter with accelerator=%s", _accelerator_value(self.accelerator))
        _progress_status(
            f"Initializing Docling converter ({_accelerator_value(self.accelerator)})...",
            enabled=self.progress_enabled,
        )
        self.converter = _build_docling_converter(
            accelerator=self.accelerator,
            num_threads=self.num_threads,
            table_structure=self.table_structure,
            code_enrichment=self.code_enrichment,
            formula_enrichment=self.formula_enrichment,
            ocr_enabled=self.ocr_enabled,
            ocr_backend=self.ocr_backend,
            ocr_langs=self.ocr_langs,
            ocr_force_full_page=self.ocr_force_full_page,
            ocr_bitmap_area_threshold=self.ocr_bitmap_area_threshold,
            rapidocr_backend=self.rapidocr_backend,
            tesseract_cmd=self.tesseract_cmd,
            tesseract_data_path=self.tesseract_data_path,
            tesseract_psm=self.tesseract_psm,
        )

DoclingPdfParser.__module__ = _source_module.__name__
finalize_split_class(_source_module, DoclingPdfParser)

