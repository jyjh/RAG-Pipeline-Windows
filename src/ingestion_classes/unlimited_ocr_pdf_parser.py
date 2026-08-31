from __future__ import annotations

import io
import re

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)

# Vendor prompt for single-image document parsing (baidu/Unlimited-OCR README).
UNLIMITED_OCR_PROMPT = "document parsing."

# Generic vision OCR: verbatim transcription beats the description-style STEM
# prompt for scanned-page OCR (measured in eval/ocr; description prompts drift
# into summarization on dense pages).
VISION_OCR_PROMPT = (
    "Transcribe ALL text on this scanned page verbatim, preserving the "
    "original reading order. Output only the transcription."
)

# Detection markers emitted by Unlimited-OCR: either the raw <|det|> form from
# the transformers/llama.cpp stacks or the flattened "text [x1, y1, x2, y2]"
# line prefixes that survive the Ollama conversion. The anchored form is tried
# first (safe against prose like "see [1, 2, 3, 4]"); the anywhere form only
# runs after raw-tag removal, when a leading tag left the prefix mid-line.
_UNLIMITED_DET_LINE_RE = re.compile(r"(?m)^\s*[A-Za-z_]+\s*\[\s*\d+(?:\s*,\s*\d+){3}\s*\]\s*")
_UNLIMITED_DET_LINE_ANYWHERE_RE = re.compile(r"\b[A-Za-z_]+\s*\[\s*\d+(?:\s*,\s*\d+){3}\s*\]\s*")
_UNLIMITED_DET_RAW_RE = re.compile(r"<\|/?det\|?>")


class UnlimitedOcrPdfParser:
    """Scanned-PDF OCR via Baidu's Unlimited-OCR vision model on local Ollama.

    Unlimited-OCR (baidu/Unlimited-OCR, MIT) is a 3B document-parsing VLM; the
    Ollama import (``frob/unlimited-ocr``) accepts a page image plus the
    vendor prompt ``document parsing.`` and emits per-region text with
    detection markers that are stripped here. Pages are rendered with
    pypdfium2 at ``dpi`` (300 = the vendor-recommended scan resolution) and
    OCR'd one page per request, producing the same ``## Page N`` structure as
    :class:`ScannedPageImageParser` so downstream page sidecars and sectioning
    behave identically.

    Unlike the Docling OCR path this engine makes a network call per page to
    the local Ollama host, so failures are retried briefly and then raised to
    let :class:`HybridPdfParser` fall back to Docling.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_UNLIMITED_OCR_MODEL,
        dpi: int = DEFAULT_UNLIMITED_OCR_DPI,
        num_ctx: int = DEFAULT_UNLIMITED_OCR_NUM_CTX,
        prompt: str = UNLIMITED_OCR_PROMPT,
        strip_markers: bool = True,
        retries: int = 2,
        progress_enabled: bool = False,
        max_pages_per_window: int = 4,
    ):
        if dpi < 72:
            raise ValueError(f"unlimited_ocr dpi must be >= 72, got {dpi}")
        self.model = model
        self.dpi = int(dpi)
        self.num_ctx = max(2048, int(num_ctx))
        self.prompt = prompt
        self.strip_markers = strip_markers
        self.retries = max(1, int(retries))
        self.progress_enabled = progress_enabled
        # Peak PNG bytes held in memory during rendering, same rationale as
        # ScannedPageImageParser.max_pages_per_window: bound memory on
        # multi-thousand-page scans (300 dpi pages are ~0.5-1 MB each).
        self.max_pages_per_window = max(1, int(max_pages_per_window))

    def parse(self, file_path: str) -> str:
        pages: list[tuple[int, str]] = []
        window: list[tuple[int, bytes]] = []

        def flush_window() -> None:
            rendered = window[:]
            window.clear()
            for page_no, png_bytes in rendered:
                pages.append((page_no, self._ocr_page_png(png_bytes, file_path)))

        for page_no, png_bytes in self._iter_page_pngs(file_path):
            window.append((page_no, png_bytes))
            if len(window) >= self.max_pages_per_window:
                flush_window()
        flush_window()

        parts: list[str] = []
        for page_no, text in pages:
            text = (text or "").strip()
            if not text:
                continue
            parts.append(f"## Page {page_no}\n\n{text}")
        if not parts:
            raise RuntimeError(
                f"Unlimited-OCR produced no text for scanned PDF: {Path(file_path).name}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Overridable seams (tests substitute these; production uses pdfium +
    # the local Ollama host).

    def _iter_page_pngs(self, file_path: str):
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(file_path)
        total = len(pdf)
        for page_no in range(total):
            _progress_status(
                f"Rendering page {page_no + 1}/{total} for Unlimited-OCR: {Path(file_path).name}",
                enabled=self.progress_enabled,
            )
            bitmap = pdf[page_no].render(scale=self.dpi / 72)
            buffer = io.BytesIO()
            bitmap.to_pil().save(buffer, format="PNG")
            yield page_no + 1, buffer.getvalue()

    def _ocr_page_png(self, png_bytes: bytes, file_path: str) -> str:
        encoded = base64.b64encode(png_bytes).decode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                import ollama

                response = ollama.generate(
                    model=self.model,
                    prompt=self.prompt,
                    images=[encoded],
                    options={"num_ctx": self.num_ctx},
                )
                text = response.response or ""
                return strip_detection_markers(text) if self.strip_markers else text.strip()
            except Exception as exc:  # noqa: BLE001 - retried then raised to caller
                last_error = exc
                logger.warning(
                    "Unlimited-OCR request failed (attempt %d/%d) on %s: %s",
                    attempt,
                    self.retries,
                    Path(file_path).name,
                    exc,
                )
        raise RuntimeError(f"Unlimited-OCR failed after {self.retries} attempt(s): {last_error}")


def strip_detection_markers(text: str) -> str:
    """Remove Unlimited-OCR detection markup, leaving plain text/markdown.

    The model tags each recognized region with a bounding box -- either the
    raw ``<|det|>type [x1,y1,x2,y2]<|/det|>`` form or, after the Ollama
    conversion, a flattened ``type [x1, y1, x2, y2]`` line prefix.
    """
    if _UNLIMITED_DET_RAW_RE.search(text):
        text = _UNLIMITED_DET_RAW_RE.sub("", text)
        text = _UNLIMITED_DET_LINE_ANYWHERE_RE.sub("", text)
    else:
        text = _UNLIMITED_DET_LINE_RE.sub("", text)
    return text.strip()


UnlimitedOcrPdfParser.__module__ = _source_module.__name__


class VisionOcrPdfParser(UnlimitedOcrPdfParser):
    """Scanned-PDF OCR via a general vision-language model on local Ollama.

    Same page-render + per-page OCR flow as :class:`UnlimitedOcrPdfParser`,
    but tuned for the local qwen2.5-vl: full-resolution page renders (that
    model tiles images natively, so 300 dpi works), a verbatim-transcription
    prompt, no detection markers to strip, and the standard vision context
    window. Measured on real corpus scans this is the most accurate local OCR
    available (eval/ocr/RESULTS.md) at roughly 3-4 min/page on a 4 GB GPU.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_VISION_OCR_MODEL,
        dpi: int = DEFAULT_VISION_OCR_DPI,
        num_ctx: int = DEFAULT_VISION_OCR_NUM_CTX,
        retries: int = 2,
        progress_enabled: bool = False,
        max_pages_per_window: int = 4,
    ):
        super().__init__(
            model=model,
            dpi=dpi,
            num_ctx=num_ctx,
            prompt=VISION_OCR_PROMPT,
            strip_markers=False,
            retries=retries,
            progress_enabled=progress_enabled,
            max_pages_per_window=max_pages_per_window,
        )


VisionOcrPdfParser.__module__ = _source_module.__name__
finalize_split_class(_source_module, VisionOcrPdfParser)
