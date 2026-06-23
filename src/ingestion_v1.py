from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PipelineConfig
from src.schema import AssetRecord, BlockRecord, DocumentRecord, PageRecord, normalize_text, stable_id
from src.store import SQLiteBlockStore

logger = logging.getLogger(__name__)


STEM_VISION_PROMPT = """You are analyzing a STEM document image for a local RAG system.
Return concise, machine-readable content:
- equations as LaTeX
- charts with axis labels, units, scales, series names, and key values
- diagrams with components, labels, and connections
- tables with headers and representative values
Avoid subjective wording. If text is unreadable, say exactly what is unreadable."""


class StructuredIngestor:
    def __init__(self, config: PipelineConfig, store: SQLiteBlockStore):
        self.config = config
        self.store = store
        self.asset_dir = Path(config.paths.asset_dir)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self._vision_loaded = False

    def ingest_path(self, source: str | os.PathLike[str]) -> list[str]:
        path = Path(source)
        if path.is_dir():
            return self.ingest_directory(path)
        if path.suffix.lower() == ".pdf":
            return [self.ingest_pdf(path)]
        if path.suffix.lower() in {".md", ".markdown", ".txt"}:
            return [self.ingest_text_file(path)]
        raise ValueError(f"Unsupported input path: {path}")

    def ingest_directory(self, directory: str | os.PathLike[str]) -> list[str]:
        root = Path(directory)
        doc_ids = []
        for path in sorted(root.iterdir()):
            if path.suffix.lower() == ".pdf":
                doc_ids.append(self.ingest_pdf(path))
            elif path.suffix.lower() in {".md", ".markdown", ".txt"}:
                doc_ids.append(self.ingest_text_file(path))
        return doc_ids

    def ingest_pdf(self, path: str | os.PathLike[str]) -> str:
        pdf_path = Path(path)
        fingerprint = _sha256_file(pdf_path)
        doc_id = stable_id("doc", pdf_path.resolve(), fingerprint)
        page_profiles = classify_pdf_pages(pdf_path)
        doc = DocumentRecord(
            doc_id=doc_id,
            title=pdf_path.stem,
            source_path=str(pdf_path),
            source_sha256=fingerprint,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="ingesting",
            page_count=max(page_profiles.keys(), default=0),
            metadata={"kind": "pdf"},
        )
        self.store.upsert_document(doc)
        for page_num, profile in page_profiles.items():
            self.store.upsert_page(PageRecord(doc_id=doc_id, page_num=page_num, **profile))

        try:
            block_count = self._ingest_pdf_with_docling(pdf_path, doc_id, page_profiles)
            status = "ready" if block_count else "empty"
        except Exception as exc:
            logger.exception("Docling ingestion failed for %s", pdf_path)
            status = "failed"
            doc.metadata["error"] = str(exc)
            block_count = 0

        doc.status = status
        doc.metadata["block_count"] = block_count
        self.store.upsert_document(doc)
        return doc_id

    def ingest_text_file(self, path: str | os.PathLike[str]) -> str:
        text_path = Path(path)
        text = text_path.read_text(encoding="utf-8", errors="replace")
        fingerprint = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        doc_id = stable_id("doc", text_path.resolve(), fingerprint)
        doc = DocumentRecord(
            doc_id=doc_id,
            title=text_path.stem,
            source_path=str(text_path),
            source_sha256=fingerprint,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="ready",
            page_count=1,
            metadata={"kind": "text"},
        )
        self.store.upsert_document(doc)
        self.store.upsert_page(
            PageRecord(
                doc_id=doc_id,
                page_num=1,
                page_type="digital",
                text_chars=len(text),
                image_count=0,
            )
        )
        block_count = 0
        for order, paragraph in enumerate(_split_markdown_blocks(text), start=1):
            modality = "section_header" if _looks_like_heading(paragraph) else "text"
            block = BlockRecord(
                block_id=stable_id("block", doc_id, 1, order, paragraph[:80]),
                doc_id=doc_id,
                page_num=1,
                modality=modality,
                reading_order=order,
                text=paragraph,
                markdown=paragraph,
                metadata={"source": "markdown_fallback"},
            )
            self.store.upsert_block(block)
            block_count += 1
        doc.metadata["block_count"] = block_count
        self.store.upsert_document(doc)
        return doc_id

    def _ingest_pdf_with_docling(
        self,
        pdf_path: Path,
        doc_id: str,
        page_profiles: dict[int, dict[str, Any]],
    ) -> int:
        converter = self._build_docling_converter(page_profiles)
        result = converter.convert(str(pdf_path))
        dl_doc = result.document
        block_count = 0
        known_pages = set(page_profiles)

        for reading_order, (item, _) in enumerate(dl_doc.iterate_items(), start=1):
            page_num, bbox = _item_provenance(item)
            page_num = page_num or 1
            known_pages.add(page_num)
            modality = _modality_for_item(item)
            content = _item_content(item)
            asset_id = ""

            if modality in {"figure", "chart", "image"} and self.config.ingestion.figure_crop_enabled:
                asset_id, vision_text = self._capture_picture(item, dl_doc, doc_id, page_num, bbox)
                if vision_text:
                    content = normalize_text("\n\n".join([content, f"[Vision Analysis]\n{vision_text}"]))

            if not content and modality not in {"figure", "chart", "image"}:
                continue

            block = BlockRecord(
                block_id=stable_id("block", doc_id, page_num, reading_order, modality, content[:100]),
                doc_id=doc_id,
                page_num=page_num,
                modality=modality,
                reading_order=reading_order,
                text=getattr(item, "text", "") or content,
                markdown=content,
                latex=content if modality == "equation" else "",
                bbox=bbox,
                confidence=_confidence(item),
                asset_id=asset_id,
                metadata={"docling_label": str(getattr(item, "label", ""))},
            )
            self.store.upsert_block(block)
            if asset_id:
                self.store.set_asset_block(asset_id, block.block_id)
            block_count += 1

        for page_num in known_pages:
            if page_num not in page_profiles:
                self.store.upsert_page(PageRecord(doc_id=doc_id, page_num=page_num))

        if block_count == 0 and hasattr(dl_doc, "export_to_markdown"):
            markdown = dl_doc.export_to_markdown()
            if markdown:
                for order, paragraph in enumerate(_split_markdown_blocks(markdown), start=1):
                    block = BlockRecord(
                        block_id=stable_id("block", doc_id, 1, order, paragraph[:80]),
                        doc_id=doc_id,
                        page_num=1,
                        modality="text",
                        reading_order=order,
                        text=paragraph,
                        markdown=paragraph,
                        metadata={"source": "docling_markdown_fallback"},
                    )
                    self.store.upsert_block(block)
                    block_count += 1

        return block_count

    def _build_docling_converter(self, page_profiles: dict[int, dict[str, Any]]):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        if hasattr(pipeline_options, "generate_picture_images"):
            pipeline_options.generate_picture_images = True

        if self.config.ingestion.table_structure and hasattr(pipeline_options, "do_table_structure"):
            pipeline_options.do_table_structure = True
            try:
                from docling.datamodel.pipeline_options import TableStructureOptions

                pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
            except Exception:
                pass

        if self.config.ingestion.formula_enrichment:
            for attr in ("do_formula_enrichment", "do_code_enrichment"):
                if hasattr(pipeline_options, attr):
                    setattr(pipeline_options, attr, True)

        try:
            from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions

            device_name = self.config.ingestion.accelerator.lower()
            device = AcceleratorDevice.CUDA if device_name == "cuda" else AcceleratorDevice.CPU
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=self.config.ingestion.num_threads,
                device=device,
            )
        except Exception:
            pass

        if _should_force_full_page_ocr(self.config.ingestion.ocr_strategy, page_profiles):
            ocr_options = _make_ocr_options(self.config.ingestion.ocr_backend)
            if ocr_options is not None:
                pipeline_options.ocr_options = ocr_options

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def _capture_picture(self, item, dl_doc, doc_id: str, page_num: int, bbox: list[float] | None) -> tuple[str, str]:
        if not hasattr(item, "get_image"):
            return "", ""
        try:
            image = item.get_image(dl_doc)
        except Exception as exc:
            logger.warning("Could not extract figure crop: %s", exc)
            return "", ""
        if image is None:
            return "", ""

        asset_id = stable_id("asset", doc_id, page_num, getattr(item, "self_ref", ""), bbox)
        path = self.asset_dir / f"{asset_id}.png"
        image.save(path, format="PNG")
        self.store.upsert_asset(
            AssetRecord(
                asset_id=asset_id,
                doc_id=doc_id,
                block_id="",
                page_num=page_num,
                asset_type="figure_crop",
                path=str(path),
                bbox=bbox,
            )
        )
        vision_text = self._describe_image(path) if self.config.ingestion.vision_enabled else ""
        return asset_id, vision_text

    def _describe_image(self, image_path: Path) -> str:
        try:
            import ollama

            if not self._vision_loaded:
                try:
                    ollama.generate(model=self.config.models.vision_model, prompt="ready", keep_alive="30m")
                except Exception:
                    pass
                self._vision_loaded = True

            encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            response = ollama.generate(
                model=self.config.models.vision_model,
                prompt=STEM_VISION_PROMPT,
                images=[encoded],
                keep_alive="30m",
            )
            return getattr(response, "response", "") or response.get("response", "")
        except Exception as exc:
            logger.warning("Vision enrichment failed for %s: %s", image_path, exc)
            return ""


def classify_pdf_pages(path: str | os.PathLike[str]) -> dict[int, dict[str, Any]]:
    pdf_path = Path(path)
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        profiles: dict[int, dict[str, Any]] = {}
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            image_count = _pypdf_image_count(page)
            if len(text.strip()) < 30 and image_count > 0:
                page_type = "scanned"
            elif len(text.strip()) >= 30 and image_count > 0:
                page_type = "mixed"
            elif len(text.strip()) >= 30:
                page_type = "digital"
            else:
                page_type = "unknown"
            width = float(page.mediabox.width) if page.mediabox else None
            height = float(page.mediabox.height) if page.mediabox else None
            profiles[index] = {
                "page_type": page_type,
                "width": width,
                "height": height,
                "text_chars": len(text),
                "image_count": image_count,
                "metadata": {},
            }
        return profiles
    except Exception:
        return {1: {"page_type": "unknown", "text_chars": 0, "image_count": 0, "metadata": {}}}


def _pypdf_image_count(page) -> int:
    try:
        return len(page.images)
    except Exception:
        return 0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_markdown_blocks(text: str) -> list[str]:
    blocks = [normalize_text(part) for part in re.split(r"\n\s*\n", text)]
    return [block for block in blocks if block]


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("#") or (len(stripped) < 120 and not stripped.endswith("."))


def _should_force_full_page_ocr(strategy: str, page_profiles: dict[int, dict[str, Any]]) -> bool:
    strategy = (strategy or "auto").lower()
    if strategy in {"always", "full_page", "force"}:
        return True
    if strategy in {"never", "disabled", "hybrid"}:
        return False
    return any(profile.get("page_type") == "scanned" for profile in page_profiles.values())


def _make_ocr_options(backend: str):
    backend = (backend or "tesseract_cli").lower()
    try:
        from docling.datamodel import pipeline_options as opts

        mapping = {
            "tesseract_cli": "TesseractCliOcrOptions",
            "tesseract": "TesseractOcrOptions",
            "easyocr": "EasyOcrOptions",
            "rapidocr": "RapidOcrOptions",
            "surya": "SuryaOcrOptions",
        }
        cls = getattr(opts, mapping.get(backend, "TesseractCliOcrOptions"))
        return cls(force_full_page_ocr=True)
    except Exception as exc:
        logger.warning("Could not configure OCR backend %s: %s", backend, exc)
        return None


def _item_provenance(item) -> tuple[int | None, list[float] | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None, None
    first = prov[0]
    page_num = getattr(first, "page_no", None)
    bbox = getattr(first, "bbox", None)
    return page_num, _bbox_to_list(bbox)


def _bbox_to_list(bbox) -> list[float] | None:
    if bbox is None:
        return None
    for attrs in (("l", "t", "r", "b"), ("left", "top", "right", "bottom")):
        if all(hasattr(bbox, attr) for attr in attrs):
            return [float(getattr(bbox, attr)) for attr in attrs]
    if hasattr(bbox, "as_tuple"):
        return [float(v) for v in bbox.as_tuple()]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(v) for v in bbox]
    return None


def _modality_for_item(item) -> str:
    label = str(getattr(item, "label", "")).lower()
    cls = item.__class__.__name__.lower()
    value = f"{label} {cls}"
    if "table" in value:
        return "table"
    if "formula" in value or "equation" in value:
        return "equation"
    if "chart" in value:
        return "chart"
    if "picture" in value or "figure" in value or "image" in value:
        return "figure"
    if "section" in value or "heading" in value or "title" in value:
        return "section_header"
    return "text"


def _item_content(item) -> str:
    if hasattr(item, "export_to_markdown") and callable(item.export_to_markdown):
        try:
            markdown = item.export_to_markdown()
            if markdown:
                return markdown
        except Exception:
            pass
    return getattr(item, "text", "") or ""


def _confidence(item) -> float | None:
    for attr in ("confidence", "conf"):
        value = getattr(item, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None
