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
        max_pages_per_window: int = 16,
    ):
        self.vision_describer = vision_describer
        self.vision_enabled = vision_enabled
        self.reader_factory = reader_factory
        self.progress_enabled = progress_enabled
        # Maximum number of pages whose rendered images are held in memory at
        # once. The original implementation gathered EVERY page image before
        # describing any, which OOMs on a large scanned PDF (e.g. a 5000-page
        # book at ~1-2 MB PNG/page = several GB of raw bytes). The windowed
        # approach keeps the same concurrency benefit within each batch while
        # bounding peak memory to ~max_pages_per_window pages of image bytes.
        self.max_pages_per_window = max(1, int(max_pages_per_window))

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

        # Re-group descriptions by page (accumulated across all windows).
        per_page: dict[int, list[str]] = {}
        page_order: list[int] = []
        any_described = False

        # Process pages in streaming windows: accumulate up to
        # max_pages_per_window pages' worth of images, describe them
        # concurrently, merge into per_page, then free the window before
        # moving on. This bounds peak memory regardless of document size.
        window: list[tuple[int, list[bytes]]] = []

        def flush_window() -> bool:
            nonlocal any_described
            if not window:
                return False
            # Flatten the window's images, tracking (page_no, img_idx) so the
            # descriptions map back to the right page.
            flat_images: list[bytes] = []
            flat_index: list[tuple[int, int]] = []
            for page_no, image_bytes in window:
                for img_idx, image_data in enumerate(image_bytes):
                    flat_images.append(image_data)
                    flat_index.append((page_no, img_idx))

            window_page_count = len(window)
            window.clear()

            if not flat_images:
                return False

            _progress_status(
                f"Describing {len(flat_images)} page image(s) for {window_page_count} page(s) "
                f"for scanned PDF fallback: {Path(file_path).name}",
                enabled=self.progress_enabled,
            )
            descriptions = self.vision_describer.describe_many(
                flat_images,
                prompt=SCANNED_PAGE_VISION_PROMPT,
            )
            for (page_no, _img_idx), description in zip(flat_index, descriptions):
                if self._is_usable_description(description):
                    per_page.setdefault(page_no, []).append(description.strip())
            any_described = True
            return True

        for page_no, page in enumerate(pages, start=1):
            image_bytes = self._page_image_bytes(page)
            if image_bytes:
                window.append((page_no, image_bytes))
                page_order.append(page_no)
            # Flush when the window reaches capacity (always forward-progress:
            # a page with a single huge image still flushes as its own window).
            if len(window) >= self.max_pages_per_window:
                flush_window()

        # Flush any remaining pages in the final partial window.
        flush_window()

        if not any_described:
            raise RuntimeError(
                f"No usable page-image analysis was produced for scanned PDF fallback: {Path(file_path).name}"
            )

        parts: list[str] = []
        for page_no in page_order:
            descs = per_page.get(page_no)
            if descs:
                page_parts = [f"## Page {page_no}"]
                page_parts.extend(f"> [Page Image Analysis]: {d}" for d in descs)
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
        opened = False
        if source_image is None:
            data = getattr(image, "data", None)
            if not data:
                raise ValueError("PDF image has no pixel data")
            from PIL import Image

            source_image = Image.open(io.BytesIO(data))
            opened = True
        try:
            # Real PIL images get downscaled for the vision model; non-PIL image
            # objects (e.g. docling wrappers or test doubles without .size) fall
            # back to a plain PNG encode of whatever ``.save`` produces.
            if hasattr(source_image, "size") and callable(getattr(source_image, "resize", None)):
                return _png_bytes_for_vision(source_image)
            buffered = io.BytesIO()
            source_image.save(buffered, format="PNG")
            return buffered.getvalue()
        finally:
            if opened:
                try:
                    source_image.close()
                except Exception:
                    pass

ScannedPageImageParser.__module__ = _source_module.__name__
finalize_split_class(_source_module, ScannedPageImageParser)

