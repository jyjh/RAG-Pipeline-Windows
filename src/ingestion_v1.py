from __future__ import annotations

import base64
import hashlib
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

STEM_VISION_PROMPT = """Analyze this STEM document image for local RAG. Return equations as LaTeX, chart axes/units/key values, diagram labels/connections, and table structure. Be concise and machine-readable."""


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
            ids = []
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}:
                    ids.extend(self.ingest_path(child))
            return ids
        if path.suffix.lower() == ".pdf":
            return [self.ingest_pdf(path)]
        if path.suffix.lower() in {".md", ".markdown", ".txt"}:
            return [self.ingest_text_file(path)]
        raise ValueError(f"Unsupported input path: {path}")

    def ingest_pdf(self, path: str | os.PathLike[str]) -> str:
        pdf_path = Path(path)
        fingerprint = _sha256_file(pdf_path)
        doc_id = stable_id("doc", pdf_path.resolve(), fingerprint)
        profiles = classify_pdf_pages(pdf_path)
        doc = DocumentRecord(doc_id, pdf_path.stem, str(pdf_path), fingerprint, datetime.now(timezone.utc).isoformat(),
                             "ingesting", max(profiles.keys(), default=0), {"kind": "pdf"})
        self.store.upsert_document(doc)
        for page_num, profile in profiles.items():
            self.store.upsert_page(PageRecord(doc_id=doc_id, page_num=page_num, **profile))
        try:
            blocks = self._ingest_pdf_with_docling(pdf_path, doc_id, profiles)
            doc.status = "ready" if blocks else "empty"
            doc.metadata["block_count"] = blocks
        except Exception as exc:
            logger.exception("PDF ingestion failed for %s", pdf_path)
            doc.status = "failed"
            doc.metadata["error"] = str(exc)
        self.store.upsert_document(doc)
        return doc_id

    def ingest_text_file(self, path: str | os.PathLike[str]) -> str:
        text_path = Path(path)
        text = text_path.read_text(encoding="utf-8", errors="replace")
        fingerprint = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        doc_id = stable_id("doc", text_path.resolve(), fingerprint)
        doc = DocumentRecord(doc_id, text_path.stem, str(text_path), fingerprint, datetime.now(timezone.utc).isoformat(),
                             "ready", 1, {"kind": "text"})
        self.store.upsert_document(doc)
        self.store.upsert_page(PageRecord(doc_id, 1, "digital", text_chars=len(text), image_count=0))
        count = 0
        for order, paragraph in enumerate(_split_blocks(text), start=1):
            modality = "section_header" if _looks_like_heading(paragraph) else "text"
            self.store.upsert_block(BlockRecord(stable_id("block", doc_id, 1, order, paragraph[:80]), doc_id, 1,
                                                modality, order, paragraph, paragraph,
                                                metadata={"source": "markdown_fallback"}))
            count += 1
        doc.metadata["block_count"] = count
        self.store.upsert_document(doc)
        return doc_id

    def _ingest_pdf_with_docling(self, pdf_path: Path, doc_id: str, profiles: dict[int, dict[str, Any]]) -> int:
        converter = self._build_converter(profiles)
        result = converter.convert(str(pdf_path))
        dl_doc = result.document
        count = 0
        for order, (item, _) in enumerate(dl_doc.iterate_items(), start=1):
            page_num, bbox = _item_provenance(item)
            page_num = page_num or 1
            modality = _modality_for_item(item)
            content = _item_content(item)
            asset_id = ""
            if modality in {"figure", "chart", "image"} and self.config.ingestion.figure_crop_enabled:
                asset_id, vision = self._capture_picture(item, dl_doc, doc_id, page_num, bbox)
                if vision:
                    content = normalize_text("\n\n".join([content, f"[Vision Analysis]\n{vision}"]))
            if not content and modality not in {"figure", "chart", "image"}:
                continue
            block = BlockRecord(stable_id("block", doc_id, page_num, order, modality, content[:100]), doc_id, page_num,
                                modality, order, getattr(item, "text", "") or content, content,
                                content if modality == "equation" else "", bbox=bbox, asset_id=asset_id,
                                metadata={"docling_label": str(getattr(item, "label", ""))})
            self.store.upsert_block(block)
            if asset_id:
                self.store.set_asset_block(asset_id, block.block_id)
            count += 1
        return count

    def _build_converter(self, profiles: dict[int, dict[str, Any]]):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = True
        if hasattr(options, "generate_picture_images"):
            options.generate_picture_images = True
        if hasattr(options, "do_table_structure"):
            options.do_table_structure = self.config.ingestion.table_structure
        if _should_force_full_page_ocr(self.config.ingestion.ocr_strategy, profiles):
            ocr = _make_ocr_options(self.config.ingestion.ocr_backend)
            if ocr is not None:
                options.ocr_options = ocr
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})

    def _capture_picture(self, item, dl_doc, doc_id: str, page_num: int, bbox: list[float] | None) -> tuple[str, str]:
        if not hasattr(item, "get_image"):
            return "", ""
        try:
            image = item.get_image(dl_doc)
        except Exception:
            return "", ""
        if image is None:
            return "", ""
        asset_id = stable_id("asset", doc_id, page_num, getattr(item, "self_ref", ""), bbox)
        path = self.asset_dir / f"{asset_id}.png"
        image.save(path, format="PNG")
        self.store.upsert_asset(AssetRecord(asset_id, doc_id, "", page_num, "figure_crop", str(path), bbox=bbox))
        return asset_id, self._describe_image(path) if self.config.ingestion.vision_enabled else ""

    def _describe_image(self, path: Path) -> str:
        try:
            import ollama

            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            response = ollama.generate(model=self.config.models.vision_model, prompt=STEM_VISION_PROMPT,
                                       images=[encoded], keep_alive="30m")
            return getattr(response, "response", "") or response.get("response", "")
        except Exception as exc:
            logger.warning("Vision enrichment failed: %s", exc)
            return ""


def classify_pdf_pages(path: str | os.PathLike[str]) -> dict[int, dict[str, Any]]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        out = {}
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            try:
                images = len(page.images)
            except Exception:
                images = 0
            page_type = "scanned" if len(text.strip()) < 30 and images else "mixed" if text.strip() and images else "digital" if text.strip() else "unknown"
            out[index] = {"page_type": page_type, "width": float(page.mediabox.width), "height": float(page.mediabox.height),
                          "text_chars": len(text), "image_count": images, "metadata": {}}
        return out
    except Exception:
        return {1: {"page_type": "unknown", "text_chars": 0, "image_count": 0, "metadata": {}}}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_blocks(text: str) -> list[str]:
    return [normalize_text(x) for x in re.split(r"\n\s*\n", text) if normalize_text(x)]


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("#") or (len(stripped) < 120 and not stripped.endswith("."))


def _should_force_full_page_ocr(strategy: str, profiles: dict[int, dict[str, Any]]) -> bool:
    strategy = (strategy or "auto").lower()
    return strategy in {"always", "full_page", "force"} or (
        strategy == "auto" and any(p.get("page_type") == "scanned" for p in profiles.values())
    )


def _make_ocr_options(backend: str):
    try:
        from docling.datamodel import pipeline_options as opts

        names = {"tesseract_cli": "TesseractCliOcrOptions", "tesseract": "TesseractOcrOptions",
                 "easyocr": "EasyOcrOptions", "rapidocr": "RapidOcrOptions", "surya": "SuryaOcrOptions"}
        return getattr(opts, names.get((backend or "").lower(), "TesseractCliOcrOptions"))(force_full_page_ocr=True)
    except Exception:
        return None


def _item_provenance(item) -> tuple[int | None, list[float] | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None, None
    bbox = getattr(prov[0], "bbox", None)
    values = None
    if bbox is not None:
        for attrs in (("l", "t", "r", "b"), ("left", "top", "right", "bottom")):
            if all(hasattr(bbox, a) for a in attrs):
                values = [float(getattr(bbox, a)) for a in attrs]
                break
    return getattr(prov[0], "page_no", None), values


def _modality_for_item(item) -> str:
    value = f"{getattr(item, 'label', '')} {item.__class__.__name__}".lower()
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
            return item.export_to_markdown() or getattr(item, "text", "") or ""
        except Exception:
            pass
    return getattr(item, "text", "") or ""

