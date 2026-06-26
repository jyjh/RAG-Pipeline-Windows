from __future__ import annotations

import base64
import io
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from src.defaults import (
    DEFAULT_ASSET_DIR,
    DEFAULT_ASSET_TRIGGERS,
    DEFAULT_CODE_ENRICHMENT,
    DEFAULT_DOCLING_ACCELERATOR,
    DEFAULT_FORMULA_ENRICHMENT,
    DEFAULT_OCR_BACKEND,
    DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
    DEFAULT_OCR_FORCE_FULL_PAGE,
    DEFAULT_OCR_LANGS,
    DEFAULT_PDF_PARSER_MODE,
    DEFAULT_RAPIDOCR_BACKEND,
    DEFAULT_TESSERACT_CMD,
    DEFAULT_TESSERACT_DATA_PATH,
    DEFAULT_TESSERACT_PSM,
    DEFAULT_VISION_ENABLED,
    DEFAULT_VISION_MODEL,
    SUPPORTED_OCR_BACKENDS,
    SUPPORTED_RAPIDOCR_BACKENDS,
)

from src._class_module_support import import_split_class

_CLASS_MODULE_PROXY_FUNCTIONS = (
    "_usable_vision_description",
    "_legacy_run_ingestion",
    "_ollama_generate",
    "_pdf_components",
    "_default_pdf_reader",
    "_new_pdf_writer",
    "_tqdm",
    "_docling_components",
    "_normalize_bool",
    "_normalize_optional_int",
    "_normalize_ocr_langs",
    "_build_ocr_options",
    "_coerce_accelerator",
    "_accelerator_value",
    "_looks_like_accelerator_failure",
    "_iter_with_progress",
    "_progress_status",
    "_build_docling_converter",
    "_iter_pdf_paths",
    "run_ingestion",
)

logger = logging.getLogger(__name__)

STEM_VISION_PROMPT = """You are analyzing a STEM document image. Provide a precise technical description:
- Mathematical equations or formulas: provide the exact LaTeX representation (e.g. $E = mc^2$)
- Charts and graphs: state axis labels, units, scale, data series names, and specific key values or inflection points
- Diagrams and schematics: label all components, describe connections, signal/data flow directions
- Tables: transcribe the structure, headers, and representative values
Be machine-readable and exact. Avoid subjective language."""

SCANNED_PAGE_VISION_PROMPT = """You are analyzing a scanned STEM textbook page. Extract searchable technical context:
- Transcribe visible headings, body text, captions, labels, and equations as accurately as possible.
- For diagrams, identify components, arrows, dimensions, symbols, and relationships between labeled parts.
- Preserve page order and technical terminology.
- Use LaTeX for equations when possible.
Be concise but complete enough for retrieval."""

VISION_ANALYSIS_NUM_CTX = 8192
FAILED_VISION_DESCRIPTIONS = {
    "[Image description failed]",
    "[Image description empty]",
    "[Vision analysis disabled]",
}


def _usable_vision_description(description: str) -> bool:
    text = (description or "").strip()
    return bool(text) and text not in FAILED_VISION_DESCRIPTIONS


_LegacyDocumentProcessor = import_split_class("src.ingestion_classes._legacy_document_processor", "_LegacyDocumentProcessor")
_LegacyDocumentProcessor.__module__ = __name__


def _legacy_run_ingestion(input_dir: str, output_dir: str):
    processor = _LegacyDocumentProcessor()
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".pdf"):
            input_path = os.path.join(input_dir, filename)
            md_content = processor.process_pdf(input_path)

            output_path = os.path.join(output_dir, f"{Path(filename).stem}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Processed {filename} -> {output_path}")

PdfParser = import_split_class("src.ingestion_classes.pdf_parser", "PdfParser")
PdfParser.__module__ = __name__


VisionDescriber = import_split_class("src.ingestion_classes.vision_describer", "VisionDescriber")
VisionDescriber.__module__ = __name__


OllamaVisionDescriber = import_split_class("src.ingestion_classes.ollama_vision_describer", "OllamaVisionDescriber")
OllamaVisionDescriber.__module__ = __name__


DisabledVisionDescriber = import_split_class("src.ingestion_classes.disabled_vision_describer", "DisabledVisionDescriber")
DisabledVisionDescriber.__module__ = __name__


def _ollama_generate(*args, **kwargs):
    import ollama

    return ollama.generate(*args, **kwargs)


def _pdf_components():
    from pypdf import PdfReader, PdfWriter

    return PdfReader, PdfWriter


def _default_pdf_reader(file_path: str):
    PdfReader, _ = _pdf_components()
    return PdfReader(file_path)


def _new_pdf_writer():
    _, PdfWriter = _pdf_components()
    return PdfWriter()


def _tqdm():
    from tqdm import tqdm

    return tqdm


def _docling_components():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    return InputFormat, AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions, DocumentConverter, PdfFormatOption


def _normalize_bool(value: bool | str | int | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_optional_int(value: int | str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_ocr_langs(value: str | list[str] | tuple[str, ...] | None, *, backend: str) -> list[str]:
    if isinstance(value, str):
        langs = [part.strip() for part in value.split(",") if part.strip()]
    elif value is None:
        langs = []
    else:
        langs = [str(part).strip() for part in value if str(part).strip()]
    if langs:
        return langs
    if backend == "rapidocr":
        return list(DEFAULT_OCR_LANGS)
    if backend in {"tesseract", "tesseract_cli"}:
        return ["eng"]
    if backend == "easyocr":
        return ["en"]
    return []


def _build_ocr_options(
    *,
    ocr_backend: str = DEFAULT_OCR_BACKEND,
    ocr_langs: str | list[str] | tuple[str, ...] | None = None,
    ocr_force_full_page: bool = DEFAULT_OCR_FORCE_FULL_PAGE,
    ocr_bitmap_area_threshold: float = DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
    rapidocr_backend: str = DEFAULT_RAPIDOCR_BACKEND,
    tesseract_cmd: str = DEFAULT_TESSERACT_CMD,
    tesseract_data_path: str | None = DEFAULT_TESSERACT_DATA_PATH,
    tesseract_psm: int | str | None = DEFAULT_TESSERACT_PSM,
) -> Any:
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrAutoOptions,
        RapidOcrOptions,
        TesseractCliOcrOptions,
        TesseractOcrOptions,
    )

    backend = str(ocr_backend or DEFAULT_OCR_BACKEND).lower()
    if backend not in SUPPORTED_OCR_BACKENDS:
        choices = ", ".join(SUPPORTED_OCR_BACKENDS)
        raise ValueError(f"Unsupported OCR backend '{ocr_backend}'. Use one of: {choices}")

    force_full_page = _normalize_bool(ocr_force_full_page, DEFAULT_OCR_FORCE_FULL_PAGE)
    bitmap_area_threshold = float(ocr_bitmap_area_threshold or DEFAULT_OCR_BITMAP_AREA_THRESHOLD)
    langs = _normalize_ocr_langs(ocr_langs, backend=backend)

    common = {
        "lang": langs,
        "force_full_page_ocr": force_full_page,
        "bitmap_area_threshold": bitmap_area_threshold,
    }
    if backend == "auto":
        return OcrAutoOptions(**common)
    if backend == "rapidocr":
        rapid_backend = str(rapidocr_backend or DEFAULT_RAPIDOCR_BACKEND).lower()
        if rapid_backend not in SUPPORTED_RAPIDOCR_BACKENDS:
            choices = ", ".join(SUPPORTED_RAPIDOCR_BACKENDS)
            raise ValueError(f"Unsupported RapidOCR backend '{rapidocr_backend}'. Use one of: {choices}")
        return RapidOcrOptions(**common, backend=rapid_backend)
    if backend == "tesseract_cli":
        path = str(tesseract_data_path or "").strip() or None
        return TesseractCliOcrOptions(
            **common,
            tesseract_cmd=str(tesseract_cmd or DEFAULT_TESSERACT_CMD),
            path=path,
            psm=_normalize_optional_int(tesseract_psm),
        )
    if backend == "tesseract":
        path = str(tesseract_data_path or "").strip() or None
        return TesseractOcrOptions(
            **common,
            path=path,
            psm=_normalize_optional_int(tesseract_psm),
        )
    return EasyOcrOptions(**common)


def _coerce_accelerator(accelerator: str | Any) -> Any:
    _, AcceleratorDevice, _, _, _, _ = _docling_components()
    if isinstance(accelerator, AcceleratorDevice):
        return accelerator
    try:
        return AcceleratorDevice(str(accelerator).lower())
    except ValueError as exc:
        choices = ", ".join(device.value for device in AcceleratorDevice)
        raise ValueError(f"Unsupported accelerator '{accelerator}'. Use one of: {choices}") from exc


def _accelerator_value(accelerator: str | Any) -> str:
    return str(getattr(accelerator, "value", accelerator)).lower()


def _looks_like_accelerator_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return "accelerator" in text or "cuda is not available" in text


def _iter_with_progress(
    iterable,
    *,
    enabled: bool,
    total: int | None,
    desc: str,
    unit: str,
):
    if not enabled:
        return iterable
    return _tqdm()(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        ascii=True,
    )


def _progress_status(message: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    print(message, file=sys.stderr, flush=True)


def _build_docling_converter(
    *,
    accelerator: str | Any = DEFAULT_DOCLING_ACCELERATOR,
    num_threads: int = 8,
    generate_picture_images: bool = True,
    table_structure: bool = True,
    code_enrichment: bool = False,
    formula_enrichment: bool = False,
    ocr_enabled: bool = True,
    ocr_backend: str = DEFAULT_OCR_BACKEND,
    ocr_langs: str | list[str] | tuple[str, ...] | None = None,
    ocr_force_full_page: bool = DEFAULT_OCR_FORCE_FULL_PAGE,
    ocr_bitmap_area_threshold: float = DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
    rapidocr_backend: str = DEFAULT_RAPIDOCR_BACKEND,
    tesseract_cmd: str = DEFAULT_TESSERACT_CMD,
    tesseract_data_path: str | None = DEFAULT_TESSERACT_DATA_PATH,
    tesseract_psm: int | str | None = DEFAULT_TESSERACT_PSM,
) -> Any:
    InputFormat, _, AcceleratorOptions, PdfPipelineOptions, DocumentConverter, PdfFormatOption = _docling_components()
    device = _coerce_accelerator(accelerator)
    accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
    pipeline_options = PdfPipelineOptions(generate_picture_images=generate_picture_images)
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = ocr_enabled
    if ocr_enabled:
        pipeline_options.ocr_options = _build_ocr_options(
            ocr_backend=ocr_backend,
            ocr_langs=ocr_langs,
            ocr_force_full_page=ocr_force_full_page,
            ocr_bitmap_area_threshold=ocr_bitmap_area_threshold,
            rapidocr_backend=rapidocr_backend,
            tesseract_cmd=tesseract_cmd,
            tesseract_data_path=tesseract_data_path,
            tesseract_psm=tesseract_psm,
        )
    pipeline_options.do_table_structure = table_structure
    pipeline_options.do_code_enrichment = code_enrichment
    pipeline_options.do_formula_enrichment = formula_enrichment

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


DoclingMarkdownRenderer = import_split_class("src.ingestion_classes.docling_markdown_renderer", "DoclingMarkdownRenderer")
DoclingMarkdownRenderer.__module__ = __name__


DoclingPdfParser = import_split_class("src.ingestion_classes.docling_pdf_parser", "DoclingPdfParser")
DoclingPdfParser.__module__ = __name__


ManualTextPdfParser = import_split_class("src.ingestion_classes.manual_text_pdf_parser", "ManualTextPdfParser")
ManualTextPdfParser.__module__ = __name__


ScannedPageImageParser = import_split_class("src.ingestion_classes.scanned_page_image_parser", "ScannedPageImageParser")
ScannedPageImageParser.__module__ = __name__


HybridPdfParser = import_split_class("src.ingestion_classes.hybrid_pdf_parser", "HybridPdfParser")
HybridPdfParser.__module__ = __name__



DocumentProcessor = import_split_class("src.ingestion_classes.document_processor", "DocumentProcessor")
DocumentProcessor.__module__ = __name__


def _iter_pdf_paths(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def run_ingestion(
    input_dir: str,
    output_dir: str,
    *,
    parser_mode: str = DEFAULT_PDF_PARSER_MODE,
    accelerator: str | Any = DEFAULT_DOCLING_ACCELERATOR,
    num_threads: int = 8,
    asset_triggers: str = DEFAULT_ASSET_TRIGGERS,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    code_enrichment: bool = DEFAULT_CODE_ENRICHMENT,
    formula_enrichment: bool = DEFAULT_FORMULA_ENRICHMENT,
    vision_model: str = DEFAULT_VISION_MODEL,
    vision_enabled: bool = DEFAULT_VISION_ENABLED,
    ocr_backend: str = DEFAULT_OCR_BACKEND,
    ocr_langs: str | list[str] | tuple[str, ...] | None = None,
    ocr_force_full_page: bool = DEFAULT_OCR_FORCE_FULL_PAGE,
    ocr_bitmap_area_threshold: float = DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
    rapidocr_backend: str = DEFAULT_RAPIDOCR_BACKEND,
    tesseract_cmd: str = DEFAULT_TESSERACT_CMD,
    tesseract_data_path: str | None = DEFAULT_TESSERACT_DATA_PATH,
    tesseract_psm: int | str | None = DEFAULT_TESSERACT_PSM,
    progress_enabled: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    from src.asset_store import ImageAssetStore
    from src.pdf_registry import sha256_file, write_source_entry

    _progress_status(f"Discovering PDFs in {input_dir}...", enabled=progress_enabled)
    input_paths = _iter_pdf_paths(input_dir)
    _progress_status(f"Found {len(input_paths)} PDF(s).", enabled=progress_enabled)
    processor: DocumentProcessor | None = None
    for input_path in _iter_with_progress(
        input_paths,
        enabled=progress_enabled,
        total=len(input_paths),
        desc="Ingest documents",
        unit="doc",
    ):
        _progress_status(f"Starting ingest: {input_path.name}", enabled=progress_enabled)
        source_hash = sha256_file(input_path)
        if processor is None:
            _progress_status("Preparing PDF parser...", enabled=progress_enabled)
            processor = DocumentProcessor(
                vision_model=vision_model,
                parser_mode=parser_mode,
                accelerator=accelerator,
                num_threads=num_threads,
                vision_enabled=vision_enabled,
                asset_triggers=asset_triggers,
                asset_dir=asset_dir,
                code_enrichment=code_enrichment,
                formula_enrichment=formula_enrichment,
                ocr_backend=ocr_backend,
                ocr_langs=ocr_langs,
                ocr_force_full_page=ocr_force_full_page,
                ocr_bitmap_area_threshold=ocr_bitmap_area_threshold,
                rapidocr_backend=rapidocr_backend,
                tesseract_cmd=tesseract_cmd,
                tesseract_data_path=tesseract_data_path,
                tesseract_psm=tesseract_psm,
                progress_enabled=progress_enabled,
            )

        ImageAssetStore(asset_dir).remove_source_assets(source_hash)
        processor.set_source_context(source_hash=source_hash, source_pdf_name=input_path.name)
        md_content = processor.process_pdf(str(input_path))
        if not md_content.strip():
            raise RuntimeError(
                f"Ingestion produced empty Markdown for {input_path.name}. "
                "Check OCR dependencies or enable vision-based scanned-page fallback."
            )

        output_path = Path(output_dir) / f"{input_path.stem}.md"
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(md_content)
        write_source_entry(
            processed_dir=output_dir,
            markdown_path=output_path,
            source_hash=source_hash,
            source_pdf_name=input_path.name,
            source_pdf_path=input_path,
        )
        logger.info("Processed %s -> %s", input_path.name, output_path)


if __name__ == "__main__":
    run_ingestion("data", "processed_docs")
