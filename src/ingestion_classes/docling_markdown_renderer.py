from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class DoclingMarkdownRenderer:
    PICTURE_LABELS = ("picture", "figure", "chart")
    CODE_LABELS = ("code",)
    FORMULA_LABELS = ("formula", "equation")
    ASSET_LABELS = PICTURE_LABELS + CODE_LABELS + FORMULA_LABELS + ("table",)

    def __init__(self, vision_describer: VisionDescriber, image_asset_store: Any | None = None):
        self.vision_describer = vision_describer
        self.image_asset_store = image_asset_store
        self.source_hash = ""
        self.source_pdf_name = ""

    def set_source_context(self, *, source_hash: str = "", source_pdf_name: str = "") -> None:
        self.source_hash = str(source_hash or "")
        self.source_pdf_name = str(source_pdf_name or "")

    def render_document(self, doc, *, page_no_override: int | None = None) -> str:
        self._begin_asset_batch()
        try:
            parts: list[str] = []
            for item, _ in doc.iterate_items():
                content = self.render_item(item, doc, page_no_override=page_no_override)
                if content:
                    parts.append(content)
            return "\n\n".join(parts)
        finally:
            self._commit_asset_batch()

    def render_assets_by_page(self, doc, *, page_no_override: int | None = None) -> dict[int, list[str]]:
        self._begin_asset_batch()
        try:
            pages: dict[int, list[str]] = {}
            for item, _ in doc.iterate_items():
                if not self.is_asset_item(item):
                    continue

                content = self.render_item(item, doc, page_no_override=page_no_override)
                if not content:
                    continue

                page_no = page_no_override or self.item_page_no(item)
                if page_no is None:
                    page_no = 1
                pages.setdefault(page_no, []).append(content)
            return pages
        finally:
            self._commit_asset_batch()

    def _begin_asset_batch(self) -> None:
        # Defer per-image manifest rewrites so a picture-heavy document does a
        # single manifest flush instead of one growing rewrite per image.
        store = self.image_asset_store
        if store is not None and hasattr(store, "begin_batch"):
            store.begin_batch()

    def _commit_asset_batch(self) -> None:
        store = self.image_asset_store
        if store is not None and hasattr(store, "commit_batch"):
            store.commit_batch()

    def render_item(self, item, doc, *, page_no_override: int | None = None) -> str:
        if self.is_picture_item(item):
            return self._render_picture(item, doc, page_no_override=page_no_override)
        if self.is_code_item(item):
            return self._render_code(item)
        if self.is_formula_item(item):
            return self._render_formula(item)
        return self.item_to_markdown(item)

    def is_asset_item(self, item) -> bool:
        label = self.item_label(item)
        return any(token in label for token in self.ASSET_LABELS)

    def is_picture_item(self, item) -> bool:
        label = self.item_label(item)
        return any(token in label for token in self.PICTURE_LABELS)

    def is_code_item(self, item) -> bool:
        label = self.item_label(item)
        return any(token in label for token in self.CODE_LABELS)

    def is_formula_item(self, item) -> bool:
        label = self.item_label(item)
        return any(token in label for token in self.FORMULA_LABELS)

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

    def _render_picture(self, item, doc, *, page_no_override: int | None = None) -> str:
        try:
            get_image = getattr(item, "get_image", None)
            if not callable(get_image):
                return ""

            image = get_image(doc)
            if image is None:
                return ""

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_data = buffered.getvalue()
            # Vision models downscale internally, so describe a downscaled copy
            # rather than the full-resolution PNG when ``image`` is a real PIL
            # image. The original ``image_data`` is still what gets stored as the
            # asset for source fidelity, and is used as the fallback for non-PIL
            # image objects (e.g. docling wrappers without PIL semantics).
            description = self.vision_describer.describe(_png_bytes_for_vision(image, fallback_bytes=image_data))
            marker = ""
            if image_data and _usable_vision_description(description) and self.image_asset_store is not None:
                from src.asset_store import image_asset_marker

                page_no = page_no_override or self.item_page_no(item) or 1
                asset = self.image_asset_store.save_image(
                    image_data=image_data,
                    source_hash=self.source_hash,
                    source_pdf_name=self.source_pdf_name,
                    page_no=page_no,
                    description=description,
                )
                marker = f"\n{image_asset_marker(str(asset.get('asset_id') or ''))}\n"
            return f"\n> [Vision Analysis]: {description}\n{marker}"
        except Exception as exc:
            logger.warning("Could not process figure: %s", exc)
            return ""

    def _render_code(self, item) -> str:
        content = self.item_to_markdown(item).strip()
        if not content:
            content = str(getattr(item, "orig", "") or "").strip()
        if not content:
            return ""
        if content.startswith("```"):
            return content

        language = self._code_language(item)
        return f"```{language}\n{content}\n```"

    @staticmethod
    def _code_language(item) -> str:
        raw_language = getattr(item, "code_language", None)
        value = str(getattr(raw_language, "value", raw_language) or "").strip().lower()
        if not value or value == "unknown":
            return ""
        return re.sub(r"[^a-z0-9_+.#-]", "", value)

    def _render_formula(self, item) -> str:
        content = self.item_to_markdown(item).strip()
        if not content:
            return ""
        if content.startswith(("$", r"\(", r"\[")):
            return content
        if "\n" in content:
            return f"$$\n{content}\n$$"
        return f"${content}$"

DoclingMarkdownRenderer.__module__ = _source_module.__name__
finalize_split_class(_source_module, DoclingMarkdownRenderer)

