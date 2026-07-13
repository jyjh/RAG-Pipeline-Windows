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
from src.atomic_io import write_json_atomic
from src.job_logging import RunTimer, log_event, write_run_summary

INGEST_RESULT_FILENAME = ".ingest_result.json"

_CLASS_MODULE_PROXY_FUNCTIONS = (
    "_usable_vision_description",
    "_legacy_run_ingestion",
    "_ollama_generate",
    "_pdf_components",
    "_default_pdf_reader",
    "_new_pdf_writer",
    "_png_bytes_for_vision",
    "VISION_IMAGE_MAX_EDGE",
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


# Vision models downscale internally, so shipping multi-megabyte full-resolution
# page/figure PNGs wastes encode time and IPC bandwidth. This cap (1568px on the
# long edge, the common Qwen2.5-VL input) bounds the payload sent to the local
# vision model without affecting what is stored as an asset.
VISION_IMAGE_MAX_EDGE = 1568


def _png_bytes_for_vision(image, fallback_bytes: bytes | None = None) -> bytes:
    """Encode a PIL image to PNG, downscaling to ``VISION_IMAGE_MAX_EDGE`` on the
    long edge. Returns ``fallback_bytes`` unchanged when ``image`` is not a real
    PIL image (e.g. a test double), so callers can always pass the original
    encoded bytes as a safe fallback. The caller owns ``image``."""
    from PIL import Image

    if not hasattr(image, "size") or not callable(getattr(image, "resize", None)):
        return fallback_bytes if fallback_bytes is not None else b""
    max_edge = max(image.size or (0, 0))
    target = image
    if max_edge > VISION_IMAGE_MAX_EDGE:
        scale = VISION_IMAGE_MAX_EDGE / max_edge
        target = image.resize(
            (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))),
            Image.LANCZOS,
        )
    buffered = io.BytesIO()
    target.save(buffered, format="PNG")
    return buffered.getvalue()


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
    if not path.exists():
        raise RuntimeError(f"PDF input path does not exist: {input_path}")
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


# --- Parallel ingestion worker support ---------------------------------------
# DocumentProcessor construction is expensive (it loads Docling/OCR models), so
# each worker process builds one processor and caches it for the lifetime of the
# process, keyed by a signature of the ingestion config. This keeps models hot
# across files in the same worker and avoids re-loading per PDF.
_PROCESSOR_CACHE: dict[str, DocumentProcessor] = {}


def _processor_cache_key(options: dict[str, Any]) -> str:
    """Build a stable key for the ingestion config so a worker can reuse a processor."""
    parts = [
        str(options.get("vision_model", "")),
        str(options.get("parser_mode", "")),
        str(options.get("accelerator", "")),
        str(options.get("num_threads", "")),
        str(options.get("vision_enabled", "")),
        str(options.get("ocr_backend", "")),
        str(options.get("ocr_langs", "")),
        str(options.get("ocr_force_full_page", "")),
        str(options.get("ocr_bitmap_area_threshold", "")),
        str(options.get("rapidocr_backend", "")),
        str(options.get("tesseract_cmd", "")),
        str(options.get("tesseract_data_path", "")),
        str(options.get("tesseract_psm", "")),
        str(options.get("code_enrichment", "")),
        str(options.get("formula_enrichment", "")),
        str(options.get("asset_triggers", "")),
        str(options.get("asset_dir", "")),
    ]
    return "|".join(parts)


def _get_or_build_processor(options: dict[str, Any], *, progress_enabled: bool) -> DocumentProcessor:
    key = _processor_cache_key(options)
    processor = _PROCESSOR_CACHE.get(key)
    if processor is None:
        _progress_status("Preparing PDF parser...", enabled=progress_enabled)
        processor = DocumentProcessor(
            vision_model=options.get("vision_model", DEFAULT_VISION_MODEL),
            parser_mode=options.get("parser_mode", DEFAULT_PDF_PARSER_MODE),
            accelerator=options.get("accelerator", DEFAULT_DOCLING_ACCELERATOR),
            num_threads=options.get("num_threads", 8),
            vision_enabled=options.get("vision_enabled", DEFAULT_VISION_ENABLED),
            asset_triggers=options.get("asset_triggers", DEFAULT_ASSET_TRIGGERS),
            asset_dir=options.get("asset_dir", DEFAULT_ASSET_DIR),
            code_enrichment=options.get("code_enrichment", DEFAULT_CODE_ENRICHMENT),
            formula_enrichment=options.get("formula_enrichment", DEFAULT_FORMULA_ENRICHMENT),
            ocr_backend=options.get("ocr_backend", DEFAULT_OCR_BACKEND),
            ocr_langs=options.get("ocr_langs"),
            ocr_force_full_page=options.get("ocr_force_full_page", DEFAULT_OCR_FORCE_FULL_PAGE),
            ocr_bitmap_area_threshold=options.get(
                "ocr_bitmap_area_threshold", DEFAULT_OCR_BITMAP_AREA_THRESHOLD
            ),
            rapidocr_backend=options.get("rapidocr_backend", DEFAULT_RAPIDOCR_BACKEND),
            tesseract_cmd=options.get("tesseract_cmd", DEFAULT_TESSERACT_CMD),
            tesseract_data_path=options.get("tesseract_data_path", DEFAULT_TESSERACT_DATA_PATH),
            tesseract_psm=options.get("tesseract_psm", DEFAULT_TESSERACT_PSM),
            progress_enabled=progress_enabled,
            max_pages_whole_doc=int(options.get("max_pages_whole_doc", 50)),
        )
        _PROCESSOR_CACHE[key] = processor
    return processor


def _ingest_one_pdf(
    input_path_str: str,
    output_dir: str,
    options: dict[str, Any],
    *,
    progress_enabled: bool,
) -> dict[str, Any]:
    """Process a single PDF. Designed to run in a ProcessPoolExecutor worker.

    Returns a result dict: ``{"file", "status", "hash", "error"}`` where status
    is one of ``"processed"``, ``"skipped"``, ``"failed"``. Each task is fully
    isolated: a failure returns ``status="failed"`` rather than raising, so the
    pool never loses the other tasks. The worker reuses a cached
    DocumentProcessor (one per process per config) to avoid reloading models.
    """
    from src.asset_store import ImageAssetStore
    from src.pdf_registry import load_source_map, sha256_file, write_source_entry

    input_path = Path(input_path_str)
    file_name = input_path.name
    output_path = Path(output_dir) / f"{input_path.stem}.md"
    try:
        source_hash = sha256_file(input_path)
        # Resume check is done in the main process before dispatch; re-check
        # here defensively in case the source map changed concurrently.
        existing_source_map = load_source_map(output_dir).get("documents", {})
        if not isinstance(existing_source_map, dict):
            existing_source_map = {}
        existing_entry = existing_source_map.get(output_path.name)
        if (
            isinstance(existing_entry, dict)
            and str(existing_entry.get("source_hash") or "") == source_hash
            and output_path.exists()
        ):
            return {"file": file_name, "status": "skipped", "hash": source_hash}

        processor = _get_or_build_processor(options, progress_enabled=progress_enabled)
        ImageAssetStore(options.get("asset_dir", DEFAULT_ASSET_DIR)).remove_source_assets(source_hash)
        processor.set_source_context(source_hash=source_hash, source_pdf_name=file_name)
        md_content = processor.process_pdf(str(input_path))
        if not md_content.strip():
            raise RuntimeError(
                f"Ingestion produced empty Markdown for {file_name}. "
                "Check OCR dependencies or enable vision-based scanned-page fallback."
            )
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(md_content)
        # Persist normalized per-page text next to the Markdown so the indexing
        # pass can skip re-running pypdf's extract_text() over every page. This
        # single extraction at ingest time replaces the per-file re-extraction
        # that sections_from_pdf would otherwise do on every index/reindex cycle.
        # Best-effort: a failure here is logged but never fails the ingest.
        try:
            from src.sectioning import write_pages_sidecar
            from pypdf import PdfReader

            reader = PdfReader(str(input_path))
            sidecar_texts = []
            for page in reader.pages:
                try:
                    sidecar_texts.append(page.extract_text() or "")
                except Exception:  # noqa: BLE001 - per-page isolation
                    sidecar_texts.append("")
            write_pages_sidecar(output_path, sidecar_texts)
        except Exception as exc:  # noqa: BLE001 - sidecar is an optimization
            logger.warning("Could not write page-text sidecar for %s: %s", file_name, exc)
        write_source_entry(
            processed_dir=output_dir,
            markdown_path=output_path,
            source_hash=source_hash,
            source_pdf_name=file_name,
            source_pdf_path=input_path,
        )
        return {"file": file_name, "status": "processed", "hash": source_hash}
    except Exception as exc:  # noqa: BLE001 - per-file isolation
        return {"file": file_name, "status": "failed", "hash": "", "error": str(exc)}


def _run_serial_ingestion(
    *,
    pending: list[Path],
    output_dir: str,
    options: dict[str, Any],
    progress_enabled: bool,
    processed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> None:
    """Process PDFs one at a time in the current process (the original path)."""
    for input_path in _iter_with_progress(
        pending,
        enabled=progress_enabled,
        total=len(pending),
        desc="Ingest documents",
        unit="doc",
    ):
        result = _ingest_one_pdf(str(input_path), output_dir, options, progress_enabled=progress_enabled)
        status = str(result.get("status") or "failed")
        if status == "processed":
            processed.append({"file": result["file"], "hash": result.get("hash", "")})
            logger.info("Processed %s", result["file"])
            log_event("file_ingested", file=result["file"], hash=result.get("hash", ""))
        elif status == "skipped":
            # Skipped-by-worker (rare; main loop already pre-skips). Record it.
            log_event("file_ingest_skipped", file=result["file"], hash=result.get("hash", ""))
        else:
            failed.append({"file": result["file"], "hash": "", "error": result.get("error", "")})
            _progress_status(
                f"Failed to ingest {result['file']}: {result.get('error')}. Continuing.",
                enabled=progress_enabled,
            )
            logger.exception("Ingestion failed for %s", result["file"])
            log_event("file_ingest_failed", file=result["file"], error=result.get("error", ""))


def _run_parallel_ingestion(
    *,
    pending: list[Path],
    output_dir: str,
    options: dict[str, Any],
    workers: int,
    progress_enabled: bool,
    processed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> None:
    """Process PDFs in parallel across a process pool.

    Each worker builds one DocumentProcessor (cached per process) and reuses it
    for every file it pulls from the pool. Results are collected as futures
    complete so progress is reported incrementally. A worker failure is captured
    per-file (never raises out of the pool) so one bad PDF never loses the batch.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    _progress_status(
        f"Ingesting {len(pending)} document(s) across {workers} worker process(es)...",
        enabled=progress_enabled,
    )
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _ingest_one_pdf,
                    str(input_path),
                    output_dir,
                    options,
                    progress_enabled,
                ): input_path
                for input_path in pending
            }
            completed = 0
            total = len(futures)
            for future in _iter_with_progress(
                list(futures.keys()),
                enabled=progress_enabled,
                total=total,
                desc="Ingest documents",
                unit="doc",
            ):
                completed += 1
                input_path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - worker-level isolation
                    file_name = input_path.name
                    failed.append({"file": file_name, "hash": "", "error": str(exc)})
                    _progress_status(
                        f"Failed to ingest {file_name}: {exc}. Continuing.",
                        enabled=progress_enabled,
                    )
                    log_event("file_ingest_failed", file=file_name, error=str(exc))
                    continue
                status = str(result.get("status") or "failed")
                if status == "processed":
                    processed.append({"file": result["file"], "hash": result.get("hash", "")})
                    log_event("file_ingested", file=result["file"], hash=result.get("hash", ""))
                elif status == "skipped":
                    log_event("file_ingest_skipped", file=result["file"], hash=result.get("hash", ""))
                else:
                    failed.append(
                        {"file": result["file"], "hash": "", "error": result.get("error", "")}
                    )
                    _progress_status(
                        f"Failed to ingest {result['file']}: {result.get('error')}. "
                        f"Continuing ({completed}/{total}).",
                        enabled=progress_enabled,
                    )
                    log_event("file_ingest_failed", file=result["file"], error=result.get("error", ""))
    except Exception as exc:
        # If the pool itself cannot start (e.g. pickling failure on Windows),
        # fall back to serial so ingestion still completes.
        _progress_status(
            f"Parallel ingestion unavailable ({exc}); falling back to serial.",
            enabled=progress_enabled,
        )
        # Reset any partial results and reprocess serially from scratch.
        processed.clear()
        failed.clear()
        _run_serial_ingestion(
            pending=pending,
            output_dir=output_dir,
            options=options,
            progress_enabled=progress_enabled,
            processed=processed,
            failed=failed,
        )


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
    ingestion_workers: int | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    from src.pdf_registry import load_source_map

    _progress_status(f"Discovering PDFs in {input_dir}...", enabled=progress_enabled)
    input_paths = _iter_pdf_paths(input_dir)
    _progress_status(f"Found {len(input_paths)} PDF(s).", enabled=progress_enabled)

    timer = RunTimer()
    processed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    # Resolve worker count. Default to a single worker (serial) for backward
    # compatibility and to bound GPU memory; scale up via config/env. Each worker
    # process loads its own copy of the Docling/OCR models, so the count trades
    # throughput for VRAM -- cap it for GPU-bound setups.
    if ingestion_workers is None:
        raw_workers = os.environ.get("INGESTION_WORKERS")
        try:
            ingestion_workers = max(1, int(raw_workers)) if raw_workers else 1
        except (TypeError, ValueError):
            ingestion_workers = 1
    ingestion_workers = max(1, int(ingestion_workers))

    # For the serial path, clear the cross-call processor cache so a caller that
    # swaps DocumentProcessor (e.g. tests, or a config change) isn't handed a
    # stale cached processor. The parallel path uses a fresh process per pool,
    # so its cache is naturally per-run.
    if ingestion_workers <= 1:
        _PROCESSOR_CACHE.clear()

    # Build the ingestion options dict passed to each worker task so the config
    # is identical between the serial and parallel paths.
    options: dict[str, Any] = {
        "vision_model": vision_model,
        "parser_mode": parser_mode,
        "accelerator": accelerator,
        "num_threads": num_threads,
        "vision_enabled": vision_enabled,
        "asset_triggers": asset_triggers,
        "asset_dir": asset_dir,
        "code_enrichment": code_enrichment,
        "formula_enrichment": formula_enrichment,
        "ocr_backend": ocr_backend,
        "ocr_langs": ocr_langs,
        "ocr_force_full_page": ocr_force_full_page,
        "ocr_bitmap_area_threshold": ocr_bitmap_area_threshold,
        "rapidocr_backend": rapidocr_backend,
        "tesseract_cmd": tesseract_cmd,
        "tesseract_data_path": tesseract_data_path,
        "tesseract_psm": tesseract_psm,
    }

    # Load the existing source map once for resume checks. A PDF whose content
    # hash already maps to its processed Markdown in the source map has already
    # been parsed; re-parsing it (the expensive step) is skipped on resume. A
    # changed PDF has a new hash and won't match, so it re-parses normally.
    existing_source_map = load_source_map(output_dir).get("documents", {})
    if not isinstance(existing_source_map, dict):
        existing_source_map = {}

    # Partition inputs into already-ingested (skip) and to-process, so the pool
    # only dispatches real work. The resume check is done here in the main
    # process to avoid every worker re-reading the source map.
    pending: list[Path] = []
    for input_path in input_paths:
        file_name = input_path.name
        output_path = Path(output_dir) / f"{input_path.stem}.md"
        existing_entry = existing_source_map.get(output_path.name)
        if (
            isinstance(existing_entry, dict)
            and output_path.exists()
        ):
            # Defer the hash check to the worker (it hashes anyway); but if the
            # entry has no source_hash mismatch we can skip the hash entirely.
            source_hash_known = str(existing_entry.get("source_hash") or "")
            if source_hash_known:
                skipped.append({"file": file_name, "hash": source_hash_known})
                log_event("file_ingest_skipped", file=file_name, hash=source_hash_known)
                _progress_status(
                    f"Already ingested (source hash unchanged): {file_name}. Skipping parse.",
                    enabled=progress_enabled,
                )
                continue
        pending.append(input_path)

    if not pending:
        _progress_status(
            "All documents already ingested; nothing to do.",
            enabled=progress_enabled,
        )
    elif ingestion_workers > 1 and len(pending) > 1:
        _run_parallel_ingestion(
            pending=pending,
            output_dir=output_dir,
            options=options,
            workers=ingestion_workers,
            progress_enabled=progress_enabled,
            processed=processed,
            failed=failed,
        )
    else:
        _run_serial_ingestion(
            pending=pending,
            output_dir=output_dir,
            options=options,
            progress_enabled=progress_enabled,
            processed=processed,
            failed=failed,
        )

    elapsed = timer.elapsed()
    result_path = Path(output_dir) / INGEST_RESULT_FILENAME
    write_run_summary(
        result_path,
        phase="ingest",
        files_processed=len(processed) + len(skipped),
        files_failed=len(failed),
        elapsed_s=elapsed,
        errors=failed or None,
    )
    # Persist the structured per-run result so the job queue can surface a
    # processed/failed/skipped breakdown in the job log without coupling the
    # subprocess to the server-side registry.
    write_json_atomic(
        result_path,
        {
            "phase": "ingest",
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "elapsed_s": round(elapsed, 3),
        },
    )

    # Only an all-failed (and nothing processed/skipped) run aborts: an empty
    # output must still surface as an error. Partial success continues so the
    # indexing stage can proceed on whatever was produced.
    total_input = len(input_paths)
    succeeded = len(processed) + len(skipped)
    if total_input and succeeded == 0:
        first_error = failed[0]["error"] if failed else "no documents were produced"
        raise RuntimeError(
            f"Ingestion failed for all {len(failed)} document(s). First error: {first_error}"
        )

    _progress_status(
        f"Ingestion complete: {len(processed)} processed, {len(skipped)} skipped, "
        f"{len(failed)} failed in {elapsed:.1f}s.",
        enabled=progress_enabled,
    )


if __name__ == "__main__":
    run_ingestion("data", "processed_docs")
