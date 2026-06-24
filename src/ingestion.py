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

logger = logging.getLogger(__name__)

STEM_VISION_PROMPT = """You are analyzing a STEM document image. Provide a precise technical description:
- Mathematical equations or formulas: provide the exact LaTeX representation (e.g. $E = mc^2$)
- Charts and graphs: state axis labels, units, scale, data series names, and specific key values or inflection points
- Diagrams and schematics: label all components, describe connections, signal/data flow directions
- Tables: transcribe the structure, headers, and representative values
Be machine-readable and exact. Avoid subjective language."""


class _LegacyDocumentProcessor:
    def __init__(self, vision_model="qwen2.5vl:7b"):
        self.vision_model = vision_model
        self._vision_model_loaded = False

        self.converter = _build_docling_converter(
            accelerator="auto",
            num_threads=8,
            generate_picture_images=True,
        )

    def _ensure_vision_loaded(self):
        """
        Lazily warms up the vision model on first figure encounter.
        Avoids consuming ~8GB VRAM on documents that contain no images.
        """
        if not self._vision_model_loaded:
            logger.info(f"First figure detected — loading vision model: {self.vision_model}")
            try:
                _ollama_generate(model=self.vision_model, prompt="Hello", keep_alive="30m")
            except Exception as e:
                logger.warning(f"Could not pre-warm vision model: {e}")
            self._vision_model_loaded = True

    def process_images(self, image_data: bytes):
        """
        Sends image data to qwen2.5vl via Ollama for STEM-focused description.
        """
        self._ensure_vision_loaded()
        try:
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            response = _ollama_generate(
                model=self.vision_model,
                prompt=STEM_VISION_PROMPT,
                images=[encoded_image]
            )
            # ollama.generate returns a GenerateResponse object, not a dict
            return response.response or '[Image description empty]'
        except Exception as e:
            logger.error(f"Error processing image with vision model: {e}")
            return "[Image description failed]"

    def _item_to_markdown(self, item) -> str:
        """
        Extracts text/markdown from a single non-figure Docling item.
        Tries the item-level export method first (preserves table formatting),
        then falls back to the raw text property.
        Note: item.export_to_markdown() is distinct from doc.export_to_markdown(item)
        — the latter is not valid Docling API.
        """
        # Item-level export works for tables and richly formatted items in Docling 2.x
        if hasattr(item, "export_to_markdown") and callable(item.export_to_markdown):
            try:
                md = item.export_to_markdown()
                if md:
                    return md
            except Exception:
                pass
        # Fallback: raw text content (TextItem, SectionHeaderItem, ListItem, FormulaItem)
        if hasattr(item, "text") and item.text:
            return item.text
        return ""

    def process_pdf(self, file_path: str):
        """
        Parses a PDF using Docling and assembles enriched markdown with vision
        descriptions injected inline at each figure's original document position.

        Inline injection is critical for RAG context quality: a figure describing
        a "Boltzmann probability density plot" must appear in the same chunk as the
        surrounding text about "Boltzmann distribution" so that local retrieval and
        the vector embeddings capture the relationship. Appending figures at the end
        of the document breaks this — vision descriptions would land in isolated chunks
        disconnected from the text that gives them meaning.
        """
        logger.info(f"Starting ingestion of: {file_path}")
        result = self.converter.convert(file_path)
        doc = result.document

        enriched_md = []

        for item, _ in doc.iterate_items():
            label_str = str(item.label).lower() if hasattr(item, "label") else ""

            if any(t in label_str for t in ("picture", "figure", "chart")):
                # Inject vision description inline at the figure's position in the document.
                # This keeps the figure description in the same 1200-token chunk as the
                # surrounding text, preserving the semantic context for both NER and ANN.
                try:
                    if hasattr(item, "get_image"):
                        img = item.get_image(doc)
                        if img is not None:
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            description = self.process_images(buffered.getvalue())
                            enriched_md.append(f"\n> [Vision Analysis]: {description}\n")
                except Exception as e:
                    logger.warning(f"Could not process figure: {e}")
            else:
                content = self._item_to_markdown(item)
                if content:
                    enriched_md.append(content)

        return "\n\n".join(enriched_md)


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

class PdfParser(Protocol):
    def parse(self, file_path: str) -> str:
        """Return enriched Markdown for a PDF."""


class VisionDescriber(Protocol):
    def describe(self, image_data: bytes) -> str:
        """Return a technical description for an image."""


class OllamaVisionDescriber:
    def __init__(self, vision_model: str = "qwen2.5vl:7b"):
        self.vision_model = vision_model
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """
        Lazily warm up the vision model on first figure encounter.
        Avoids consuming VRAM on documents that contain no images.
        """
        if self._loaded:
            return

        logger.info("First figure detected - loading vision model: %s", self.vision_model)
        try:
            _ollama_generate(model=self.vision_model, prompt="Hello", keep_alive="30m")
        except Exception as exc:
            logger.warning("Could not pre-warm vision model: %s", exc)
        self._loaded = True

    def describe(self, image_data: bytes) -> str:
        self._ensure_loaded()
        try:
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            response = _ollama_generate(
                model=self.vision_model,
                prompt=STEM_VISION_PROMPT,
                images=[encoded_image],
            )
            return response.response or "[Image description empty]"
        except Exception as exc:
            logger.error("Error processing image with vision model: %s", exc)
            return "[Image description failed]"


class DisabledVisionDescriber:
    def describe(self, image_data: bytes) -> str:
        return "[Vision analysis disabled]"


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
    accelerator: str | Any = "auto",
    num_threads: int = 8,
    generate_picture_images: bool = True,
    table_structure: bool = True,
    formula_enrichment: bool = False,
    ocr_enabled: bool = True,
) -> Any:
    InputFormat, _, AcceleratorOptions, PdfPipelineOptions, DocumentConverter, PdfFormatOption = _docling_components()
    device = _coerce_accelerator(accelerator)
    accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
    pipeline_options = PdfPipelineOptions(generate_picture_images=generate_picture_images)
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = ocr_enabled
    pipeline_options.do_table_structure = table_structure
    pipeline_options.do_formula_enrichment = formula_enrichment

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


class DoclingMarkdownRenderer:
    PICTURE_LABELS = ("picture", "figure", "chart")
    ASSET_LABELS = PICTURE_LABELS + ("table", "formula", "equation")

    def __init__(self, vision_describer: VisionDescriber):
        self.vision_describer = vision_describer

    def render_document(self, doc) -> str:
        parts: list[str] = []
        for item, _ in doc.iterate_items():
            content = self.render_item(item, doc)
            if content:
                parts.append(content)
        return "\n\n".join(parts)

    def render_assets_by_page(self, doc) -> dict[int, list[str]]:
        pages: dict[int, list[str]] = {}
        for item, _ in doc.iterate_items():
            if not self.is_asset_item(item):
                continue

            content = self.render_item(item, doc)
            if not content:
                continue

            page_no = self.item_page_no(item)
            if page_no is None:
                page_no = 1
            pages.setdefault(page_no, []).append(content)
        return pages

    def render_item(self, item, doc) -> str:
        if self.is_picture_item(item):
            return self._render_picture(item, doc)
        return self.item_to_markdown(item)

    def is_asset_item(self, item) -> bool:
        label = self.item_label(item)
        return any(token in label for token in self.ASSET_LABELS)

    def is_picture_item(self, item) -> bool:
        label = self.item_label(item)
        return any(token in label for token in self.PICTURE_LABELS)

    @staticmethod
    def item_label(item) -> str:
        return str(getattr(item, "label", "")).lower()

    @staticmethod
    def item_page_no(item) -> int | None:
        for attr in ("page_no", "page_number"):
            value = getattr(item, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        for prov in getattr(item, "prov", []) or []:
            value = getattr(prov, "page_no", None)
            if value is None and isinstance(prov, dict):
                value = prov.get("page_no")
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def item_to_markdown(item) -> str:
        """
        Extract text/markdown from a Docling item.
        Item-level export preserves table formatting where Docling supports it.
        """
        export = getattr(item, "export_to_markdown", None)
        if callable(export):
            try:
                markdown = export()
                if markdown:
                    return str(markdown)
            except Exception:
                pass

        text = getattr(item, "text", None)
        if text:
            return str(text)
        return ""

    def _render_picture(self, item, doc) -> str:
        try:
            get_image = getattr(item, "get_image", None)
            if not callable(get_image):
                return ""

            image = get_image(doc)
            if image is None:
                return ""

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            description = self.vision_describer.describe(buffered.getvalue())
            return f"\n> [Vision Analysis]: {description}\n"
        except Exception as exc:
            logger.warning("Could not process figure: %s", exc)
            return ""


@dataclass
class DoclingPdfParser:
    vision_describer: VisionDescriber
    accelerator: str | Any = "auto"
    num_threads: int = 8
    table_structure: bool = True
    formula_enrichment: bool = False
    ocr_enabled: bool = True
    progress_enabled: bool = False
    reader_factory: Callable[[str], Any] = _default_pdf_reader
    converter: Any | None = None

    def __post_init__(self) -> None:
        self.renderer = DoclingMarkdownRenderer(self.vision_describer)

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
                    page_md = self.renderer.render_document(page_doc)
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
                formula_enrichment=self.formula_enrichment,
                ocr_enabled=self.ocr_enabled,
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
            formula_enrichment=self.formula_enrichment,
            ocr_enabled=self.ocr_enabled,
        )


class ManualTextPdfParser:
    ASSET_TRIGGER_NONE = "none"
    ASSET_TRIGGER_IMAGES = "images"
    ASSET_TRIGGER_ALL = "all"
    ASSET_TRIGGERS = {ASSET_TRIGGER_NONE, ASSET_TRIGGER_IMAGES, ASSET_TRIGGER_ALL}

    EQUATION_MARKERS = ("=", "+", "-", "*", "/", "^", "_", "∑", "∫", "√", "≤", "≥", "≈", "≠")

    def __init__(
        self,
        *,
        docling_parser: DoclingPdfParser,
        reader_factory: Callable[[str], Any] = _default_pdf_reader,
        extraction_mode: str = "plain",
        min_text_chars: int = 20,
        min_alnum_ratio: float = 0.35,
        asset_triggers: str = ASSET_TRIGGER_NONE,
        progress_enabled: bool = False,
    ):
        self.docling_parser = docling_parser
        self.reader_factory = reader_factory
        self.extraction_mode = extraction_mode
        self.min_text_chars = min_text_chars
        self.min_alnum_ratio = min_alnum_ratio
        self.asset_triggers = self._normalize_asset_triggers(asset_triggers)
        self.progress_enabled = progress_enabled

    def has_extractable_text(self, file_path: str) -> bool:
        return self.is_text_usable(self.extract_page_texts(file_path))

    def parse(
        self,
        file_path: str,
        *,
        include_assets: bool = False,
        asset_pages: set[int] | None = None,
        page_texts: list[str] | None = None,
    ) -> str:
        page_texts = page_texts if page_texts is not None else self.extract_page_texts(file_path)
        if asset_pages is not None and not asset_pages:
            include_assets = False

        if not include_assets:
            logger.info("Parsing PDF with pypdf text extraction only: %s", file_path)
            return self.render_page_texts(page_texts)

        if asset_pages is None:
            logger.info("Parsing PDF with pypdf text plus Docling assets: %s", file_path)
            assets_by_page = self.extract_docling_assets(file_path)
        else:
            logger.info(
                "Parsing PDF with pypdf text plus Docling assets for pages %s: %s",
                sorted(asset_pages),
                file_path,
            )
            assets_by_page = self.extract_docling_assets_for_pages(file_path, asset_pages)

        return self.render_page_texts(page_texts, assets_by_page=assets_by_page)

    def extract_docling_assets(self, file_path: str) -> dict[int, list[str]]:
        try:
            _progress_status(
                f"Docling asset extraction: {Path(file_path).name}",
                enabled=self.progress_enabled,
            )
            doc = self.docling_parser.convert(file_path).document
            return self.docling_parser.renderer.render_assets_by_page(doc)
        except Exception as exc:
            logger.warning(
                "Docling asset extraction failed; continuing with manual text only: %s",
                exc,
            )
            return {}

    def extract_docling_assets_for_pages(
        self,
        file_path: str,
        page_numbers: set[int],
    ) -> dict[int, list[str]]:
        try:
            reader = self.reader_factory(file_path)
        except Exception as exc:
            logger.warning("Could not open PDF for page-level Docling assets: %s", exc)
            return {}

        assets_by_page: dict[int, list[str]] = {}
        page_count = len(getattr(reader, "pages", []))
        desc = f"Docling asset pages: {Path(file_path).name}"
        with tempfile.TemporaryDirectory(prefix="rag_pdf_assets_") as temp_dir:
            page_iter = _iter_with_progress(
                sorted(page_numbers),
                enabled=self.progress_enabled,
                total=len(page_numbers),
                desc=desc,
                unit="page",
            )
            for page_no in page_iter:
                if page_no < 1 or page_no > page_count:
                    continue

                page_path = Path(temp_dir) / f"page_{page_no}.pdf"
                try:
                    writer = _new_pdf_writer()
                    writer.add_page(reader.pages[page_no - 1])
                    with page_path.open("wb") as handle:
                        writer.write(handle)
                except Exception as exc:
                    logger.warning("Could not isolate PDF page %s for Docling assets: %s", page_no, exc)
                    continue

                page_assets = self.extract_docling_assets(str(page_path))
                for assets in page_assets.values():
                    assets_by_page.setdefault(page_no, []).extend(assets)

        return assets_by_page

    def render_page_texts(
        self,
        page_texts: list[str],
        *,
        assets_by_page: dict[int, list[str]] | None = None,
    ) -> str:
        assets_by_page = assets_by_page or {}
        page_parts: list[str] = []
        page_count = max(len(page_texts), max(assets_by_page.keys(), default=0))

        for index in range(page_count):
            page_no = index + 1
            parts: list[str] = []
            if index < len(page_texts) and page_texts[index]:
                parts.append(page_texts[index])
            parts.extend(assets_by_page.get(page_no, []))
            if parts:
                page_parts.append("\n\n".join(parts))

        return "\n\n".join(page_parts)

    def is_text_usable(self, page_texts: list[str]) -> bool:
        text = "\n".join(page_texts)
        compact = "".join(text.split())
        if len(compact) < self.min_text_chars:
            return False

        visible = [char for char in text if not char.isspace()]
        if not visible:
            return False

        replacement_ratio = text.count("\ufffd") / max(len(visible), 1)
        if replacement_ratio > 0.01:
            return False

        alnum_ratio = sum(char.isalnum() for char in visible) / len(visible)
        return alnum_ratio >= self.min_alnum_ratio

    def needs_asset_enrichment(self, file_path: str, page_texts: list[str]) -> bool:
        return bool(self.asset_enrichment_pages(file_path, page_texts))

    def asset_enrichment_pages(self, file_path: str, page_texts: list[str]) -> set[int]:
        if self.asset_triggers == self.ASSET_TRIGGER_NONE:
            return set()

        pages = self.image_pages(file_path)
        if self.asset_triggers == self.ASSET_TRIGGER_ALL:
            page_iter = _iter_with_progress(
                enumerate(page_texts, start=1),
                enabled=self.progress_enabled,
                total=len(page_texts),
                desc=f"Inspect text assets: {Path(file_path).name}",
                unit="page",
            )
            for index, text in page_iter:
                if self.text_suggests_table(text) or self.text_suggests_equation(text):
                    pages.add(index)
        return pages

    def has_images(self, file_path: str) -> bool:
        return bool(self.image_pages(file_path))

    def image_pages(self, file_path: str) -> set[int]:
        try:
            reader = self.reader_factory(file_path)
        except Exception as exc:
            logger.warning("Could not inspect PDF images with pypdf: %s", exc)
            return set()

        pages: set[int] = set()
        page_list = getattr(reader, "pages", [])
        page_iter = _iter_with_progress(
            enumerate(page_list, start=1),
            enabled=self.progress_enabled,
            total=len(page_list),
            desc=f"Inspect pages: {Path(file_path).name}",
            unit="page",
        )
        for index, page in page_iter:
            if self._page_has_image(page):
                pages.add(index)
        return pages

    @classmethod
    def _page_has_image(cls, page) -> bool:
        try:
            resources = page.get("/Resources", {})
            resources = cls._resolve_pdf_object(resources) or {}
            xobjects = cls._resolve_pdf_object(resources.get("/XObject", {})) or {}
            for obj in xobjects.values():
                obj = cls._resolve_pdf_object(obj)
                if obj is not None and obj.get("/Subtype") == "/Image":
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def _resolve_pdf_object(value):
        get_object = getattr(value, "get_object", None)
        if callable(get_object):
            return get_object()
        return value

    @classmethod
    def _normalize_asset_triggers(cls, asset_triggers: str) -> str:
        mode = asset_triggers.lower()
        if mode == "auto":
            mode = cls.ASSET_TRIGGER_IMAGES
        if mode not in cls.ASSET_TRIGGERS:
            raise ValueError("asset_triggers must be one of: none, images, all")
        return mode

    @staticmethod
    def text_suggests_table(text: str) -> bool:
        table_like_lines = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "|" in stripped or "\t" in stripped:
                return True
            if len(re.findall(r"\S+\s{2,}\S+", stripped)) >= 2:
                table_like_lines += 1
            elif len(re.findall(r"\d+(?:\.\d+)?", stripped)) >= 3 and re.search(r"\s{2,}", stripped):
                table_like_lines += 1
        return table_like_lines >= 3

    @classmethod
    def text_suggests_equation(cls, text: str) -> bool:
        for line in text.splitlines():
            stripped = line.strip()
            if len(stripped) < 6:
                continue
            marker_count = sum(stripped.count(marker) for marker in cls.EQUATION_MARKERS)
            has_variable = bool(re.search(r"[A-Za-z]", stripped))
            has_number = bool(re.search(r"\d", stripped))
            if marker_count >= 2 and (has_variable or has_number):
                return True
            if re.search(r"\b[A-Za-z]\s*=\s*[^=]+", stripped):
                return True
        return False

    def extract_page_texts(self, file_path: str) -> list[str]:
        _progress_status(f"Opening PDF with pypdf: {Path(file_path).name}", enabled=self.progress_enabled)
        reader = self.reader_factory(file_path)
        texts: list[str] = []
        page_iter = _iter_with_progress(
            enumerate(reader.pages, start=1),
            enabled=self.progress_enabled,
            total=len(reader.pages),
            desc=f"Extract text: {Path(file_path).name}",
            unit="page",
        )
        for page_no, page in page_iter:
            try:
                text = page.extract_text(extraction_mode=self.extraction_mode) or ""
            except TypeError:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Manual text extraction failed on page %s: %s", page_no, exc)
                text = ""
            texts.append(self._normalize_text(text))
        return texts

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        normalized: list[str] = []
        previous_blank = False
        for line in lines:
            blank = not line.strip()
            if blank and previous_blank:
                continue
            normalized.append(line)
            previous_blank = blank
        return "\n".join(normalized).strip()


class HybridPdfParser:
    def __init__(self, *, manual_parser: ManualTextPdfParser, docling_parser: DoclingPdfParser):
        self.manual_parser = manual_parser
        self.docling_parser = docling_parser

    def parse(self, file_path: str) -> str:
        page_texts = self.manual_parser.extract_page_texts(file_path)
        if self.manual_parser.is_text_usable(page_texts):
            asset_pages = self.manual_parser.asset_enrichment_pages(file_path, page_texts)
            return self.manual_parser.parse(
                file_path,
                include_assets=bool(asset_pages),
                asset_pages=asset_pages,
                page_texts=page_texts,
            )

        logger.info("Extracted PDF text is missing or low quality; using full Docling parsing: %s", file_path)
        return self.docling_parser.parse(file_path)


class DocumentProcessor:
    def __init__(
        self,
        vision_model: str = "qwen2.5vl:7b",
        parser_mode: str = "hybrid",
        accelerator: str | Any = "auto",
        num_threads: int = 8,
        vision_enabled: bool = True,
        min_text_chars: int = 20,
        asset_triggers: str = ManualTextPdfParser.ASSET_TRIGGER_NONE,
        progress_enabled: bool = True,
    ):
        self.vision_model = vision_model
        if vision_enabled:
            self.vision_describer: VisionDescriber = OllamaVisionDescriber(vision_model=vision_model)
        else:
            self.vision_describer = DisabledVisionDescriber()

        self.docling_parser = DoclingPdfParser(
            vision_describer=self.vision_describer,
            accelerator=accelerator,
            num_threads=num_threads,
            ocr_enabled=True,
            progress_enabled=progress_enabled,
        )
        self.asset_docling_parser = DoclingPdfParser(
            vision_describer=self.vision_describer,
            accelerator=accelerator,
            num_threads=num_threads,
            ocr_enabled=False,
            progress_enabled=progress_enabled,
        )
        self.manual_parser = ManualTextPdfParser(
            docling_parser=self.asset_docling_parser,
            min_text_chars=min_text_chars,
            asset_triggers=asset_triggers,
            progress_enabled=progress_enabled,
        )
        self.parser = self._select_parser(parser_mode)

    def _select_parser(self, parser_mode: str) -> PdfParser:
        mode = parser_mode.lower()
        if mode == "hybrid":
            return HybridPdfParser(
                manual_parser=self.manual_parser,
                docling_parser=self.docling_parser,
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


def _iter_pdf_paths(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def run_ingestion(
    input_dir: str,
    output_dir: str,
    *,
    parser_mode: str = "hybrid",
    accelerator: str | Any = "auto",
    asset_triggers: str = ManualTextPdfParser.ASSET_TRIGGER_NONE,
    progress_enabled: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

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
        if processor is None:
            _progress_status("Preparing PDF parser...", enabled=progress_enabled)
            processor = DocumentProcessor(
                parser_mode=parser_mode,
                accelerator=accelerator,
                asset_triggers=asset_triggers,
                progress_enabled=progress_enabled,
            )

        md_content = processor.process_pdf(str(input_path))

        output_path = Path(output_dir) / f"{input_path.stem}.md"
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(md_content)
        logger.info("Processed %s -> %s", input_path.name, output_path)


if __name__ == "__main__":
    run_ingestion("data", "processed_docs")
