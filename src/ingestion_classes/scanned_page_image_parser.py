from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class ScannedPageImageParser:
    def __init__(
        self,
        *,
        vision_describer: VisionDescriber,
        vision_enabled: bool,
        reader_factory: Callable[[str], Any] = _default_pdf_reader,
        progress_enabled: bool = False,
    ):
        self.vision_describer = vision_describer
        self.vision_enabled = vision_enabled
        self.reader_factory = reader_factory
        self.progress_enabled = progress_enabled

    def parse(self, file_path: str) -> str:
        if not self.vision_enabled:
            raise RuntimeError(
                f"Scanned PDF fallback requires vision analysis, but vision is disabled: {Path(file_path).name}"
            )

        try:
            reader = self.reader_factory(file_path)
        except Exception as exc:
            raise RuntimeError(f"Could not open scanned PDF for page-image fallback: {file_path}") from exc

        pages = getattr(reader, "pages", [])
        page_iter = _iter_with_progress(
            enumerate(pages, start=1),
            enabled=self.progress_enabled,
            total=len(pages),
            desc=f"Analyze page images: {Path(file_path).name}",
            unit="page",
        )
        parts: list[str] = []
        for page_no, page in page_iter:
            image_bytes = self._page_image_bytes(page)
            if not image_bytes:
                continue
            page_parts = [f"## Page {page_no}"]
            for image_data in image_bytes:
                description = self.vision_describer.describe(
                    image_data,
                    prompt=SCANNED_PAGE_VISION_PROMPT,
                )
                if self._is_usable_description(description):
                    page_parts.append(f"> [Page Image Analysis]: {description.strip()}")
            if len(page_parts) > 1:
                parts.append("\n\n".join(page_parts))

        if not parts:
            raise RuntimeError(
                f"No usable page-image analysis was produced for scanned PDF fallback: {Path(file_path).name}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _is_usable_description(description: str) -> bool:
        text = (description or "").strip()
        return bool(text) and text not in FAILED_VISION_DESCRIPTIONS

    def _page_image_bytes(self, page) -> list[bytes]:
        images = list(getattr(page, "images", []) or [])
        output: list[bytes] = []
        for image in images:
            if getattr(image, "is_displayed", True) is False:
                continue
            try:
                output.append(self._image_to_png_bytes(image))
            except Exception as exc:
                logger.warning("Could not extract page image for fallback: %s", exc)
        return output

    @staticmethod
    def _image_to_png_bytes(image) -> bytes:
        source_image = getattr(image, "image", None)
        if source_image is None:
            data = getattr(image, "data", None)
            if not data:
                raise ValueError("PDF image has no pixel data")
            from PIL import Image

            source_image = Image.open(io.BytesIO(data))

        buffered = io.BytesIO()
        source_image.save(buffered, format="PNG")
        return buffered.getvalue()

ScannedPageImageParser.__module__ = _source_module.__name__
finalize_split_class(_source_module, ScannedPageImageParser)

