from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class ManualTextPdfParser:
    ASSET_TRIGGER_NONE = "none"
    ASSET_TRIGGER_IMAGES = "images"
    ASSET_TRIGGER_AUTO = "auto"
    ASSET_TRIGGER_ALL = "all"
    ASSET_TRIGGERS = {ASSET_TRIGGER_NONE, ASSET_TRIGGER_IMAGES, ASSET_TRIGGER_AUTO, ASSET_TRIGGER_ALL}
    LATEX_MARKERS = (
        r"\frac",
        r"\sum",
        r"\int",
        r"\sqrt",
        r"\begin{equation",
        r"\begin{align",
        r"\left",
        r"\right",
    )
    CODE_KEYWORD_PATTERNS = (
        r"^\s*(def|class|import|from|return|async\s+def|await)\b",
        r"^\s*(for|while|if|elif|else|try|except|finally|with)\b.*:",
        r"^\s*(const|let|var|function|return|export|import)\b",
        r"^\s*(public|private|protected|static|class|interface|void|int|double|float|bool|string)\b",
        r"^\s*#\s*include\b",
        r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b",
        r"^\s*(fn|let|mut|impl|pub|use|struct|enum|trait)\b",
    )

    EQUATION_MARKERS = ("=", "+", "-", "*", "/", "^", "_", "∑", "∫", "√", "≤", "≥", "≈", "≠")

    def __init__(
        self,
        *,
        docling_parser: DoclingPdfParser,
        enriched_docling_parser: DoclingPdfParser | None = None,
        reader_factory: Callable[[str], Any] = _default_pdf_reader,
        extraction_mode: str = "plain",
        min_text_chars: int = 20,
        min_alnum_ratio: float = 0.35,
        asset_triggers: str = ASSET_TRIGGER_NONE,
        progress_enabled: bool = False,
    ):
        self.docling_parser = docling_parser
        self.enriched_docling_parser = enriched_docling_parser or docling_parser
        self.reader_factory = reader_factory
        self.extraction_mode = extraction_mode
        self.min_text_chars = min_text_chars
        self.min_alnum_ratio = min_alnum_ratio
        self.asset_triggers = self._normalize_asset_triggers(asset_triggers)
        self.progress_enabled = progress_enabled
        # A single parse opens the same PDF up to three times (text extraction,
        # image detection, page-asset isolation). Memoize the reader per file path
        # so they share one parsed structure instead of re-reading it repeatedly.
        self._reader_cache: dict[str, Any] = {}

    def _get_reader(self, file_path: str):
        reader = self._reader_cache.get(file_path)
        if reader is None:
            reader = self.reader_factory(file_path)
            self._reader_cache[file_path] = reader
        return reader

    def clear_reader_cache(self, file_path: str | None = None) -> None:
        """Drop cached PDF readers. Pass ``file_path`` to drop one, or nothing to drop all."""
        if file_path is None:
            self._reader_cache.clear()
        else:
            self._reader_cache.pop(file_path, None)

    def has_extractable_text(self, file_path: str) -> bool:
        return self.is_text_usable(self.extract_page_texts(file_path))

    def parse(
        self,
        file_path: str,
        *,
        include_assets: bool = False,
        asset_pages: set[int] | None = None,
        enriched_asset_pages: set[int] | None = None,
        page_texts: list[str] | None = None,
    ) -> str:
        # Drop any readers cached from a previous document so this parse starts
        # fresh (and so the cache cannot grow unbounded across a large batch).
        self.clear_reader_cache()
        page_texts = page_texts if page_texts is not None else self.extract_page_texts(file_path)
        if not include_assets and asset_pages is None and self.asset_triggers != self.ASSET_TRIGGER_NONE:
            hints = self.asset_enrichment_page_hints(file_path, page_texts)
            asset_pages = set(hints)
            if enriched_asset_pages is None:
                enriched_asset_pages = self._enriched_pages_from_hints(hints)
            include_assets = bool(asset_pages)

        if asset_pages is not None and not asset_pages:
            include_assets = False

        if not include_assets:
            logger.info("Parsing PDF with pypdf text extraction only: %s", file_path)
            return self.render_page_texts(page_texts)

        if asset_pages is None:
            logger.info("Parsing PDF with pypdf text plus Docling assets: %s", file_path)
            hints = self.asset_enrichment_page_hints(file_path, page_texts)
            enriched = self._has_code_or_formula_hints(hints)
            assets_by_page = self.extract_docling_assets(file_path, enriched=enriched)
        else:
            logger.info(
                "Parsing PDF with pypdf text plus Docling assets for pages %s: %s",
                sorted(asset_pages),
                file_path,
            )
            assets_by_page = self.extract_docling_assets_for_pages(
                file_path,
                asset_pages,
                enriched_asset_pages=enriched_asset_pages,
            )

        return self.render_page_texts(page_texts, assets_by_page=assets_by_page)

    def extract_docling_assets(
        self,
        file_path: str,
        *,
        enriched: bool = False,
        page_no_override: int | None = None,
    ) -> dict[int, list[str]]:
        parser = self.enriched_docling_parser if enriched else self.docling_parser
        try:
            _progress_status(
                f"Docling asset extraction: {Path(file_path).name}",
                enabled=self.progress_enabled,
            )
            doc = parser.convert(file_path).document
            return parser.renderer.render_assets_by_page(doc, page_no_override=page_no_override)
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
        *,
        enriched_asset_pages: set[int] | None = None,
    ) -> dict[int, list[str]]:
        try:
            reader = self._get_reader(file_path)
        except Exception as exc:
            logger.warning("Could not open PDF for page-level Docling assets: %s", exc)
            return {}

        assets_by_page: dict[int, list[str]] = {}
        enriched_asset_pages = enriched_asset_pages or set()
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

                page_assets = self.extract_docling_assets(
                    str(page_path),
                    enriched=page_no in enriched_asset_pages,
                    page_no_override=page_no,
                )
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
        return set(self.asset_enrichment_page_hints(file_path, page_texts))

    def asset_enrichment_page_hints(self, file_path: str, page_texts: list[str]) -> dict[int, set[str]]:
        if self.asset_triggers == self.ASSET_TRIGGER_NONE:
            return {}

        hints: dict[int, set[str]] = {}
        if self.asset_triggers in {
            self.ASSET_TRIGGER_IMAGES,
            self.ASSET_TRIGGER_AUTO,
            self.ASSET_TRIGGER_ALL,
        }:
            for page_no in self.image_pages(file_path):
                hints.setdefault(page_no, set()).add("picture")

        if self.asset_triggers in {self.ASSET_TRIGGER_AUTO, self.ASSET_TRIGGER_ALL}:
            page_iter = _iter_with_progress(
                enumerate(page_texts, start=1),
                enabled=self.progress_enabled,
                total=len(page_texts),
                desc=f"Inspect text assets: {Path(file_path).name}",
                unit="page",
            )
            for index, text in page_iter:
                if self.text_suggests_formula(text):
                    hints.setdefault(index, set()).add("formula")
                if self.text_suggests_code(text):
                    hints.setdefault(index, set()).add("code")
                if self.asset_triggers == self.ASSET_TRIGGER_ALL and self.text_suggests_table(text):
                    hints.setdefault(index, set()).add("table")
        return hints

    @staticmethod
    def _has_code_or_formula_hints(hints: dict[int, set[str]]) -> bool:
        return any({"code", "formula"} & page_hints for page_hints in hints.values())

    @staticmethod
    def _enriched_pages_from_hints(hints: dict[int, set[str]]) -> set[int]:
        return {
            page_no
            for page_no, page_hints in hints.items()
            if {"code", "formula"} & page_hints
        }

    def has_images(self, file_path: str) -> bool:
        return bool(self.image_pages(file_path))

    def image_pages(self, file_path: str) -> set[int]:
        try:
            reader = self._get_reader(file_path)
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
        if mode not in cls.ASSET_TRIGGERS:
            raise ValueError("asset_triggers must be one of: none, images, auto, all")
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
            if len(_TABLE_SPACED_RUN.findall(stripped)) >= 2:
                table_like_lines += 1
            elif len(_NUMBER_RUN.findall(stripped)) >= 3 and re.search(r"\s{2,}", stripped):
                table_like_lines += 1
        return table_like_lines >= 3

    @classmethod
    def text_suggests_equation(cls, text: str) -> bool:
        for line in text.splitlines():
            stripped = line.strip()
            if len(stripped) < 6:
                continue
            marker_count = sum(stripped.count(marker) for marker in cls.EQUATION_MARKERS)
            has_variable = bool(_VARIABLE_RE.search(stripped))
            has_number = bool(_NUMBER_ANYWHERE_RE.search(stripped))
            if marker_count >= 2 and (has_variable or has_number):
                return True
            if _EQUATION_ASSIGN_RE.search(stripped):
                return True
        return False

    @classmethod
    def text_suggests_formula(cls, text: str) -> bool:
        if any(marker in text for marker in cls.LATEX_MARKERS):
            return True
        if _INLINE_FORMULA_RE.search(text):
            return True
        if _DISPLAY_FORMULA_RE.search(text):
            return True
        if cls.text_suggests_equation(text):
            return True
        return cls.text_suggests_broken_formula_area(text)

    @staticmethod
    def text_suggests_broken_formula_area(text: str) -> bool:
        for line in text.splitlines():
            stripped = line.strip()
            if len(stripped) < 6:
                continue
            if "�" in stripped or "\ufffd" in stripped or "□" in stripped:
                if _BROKEN_FORMULA_CHAR_RE.search(stripped):
                    return True
            compact = stripped.replace(" ", "")
            if len(compact) < 4:
                continue
            operator_count = len(_OPERATOR_RUN_RE.findall(compact))
            symbol_count = sum(not char.isalnum() for char in compact)
            if operator_count >= 2 and symbol_count / max(len(compact), 1) >= 0.2:
                return True
        return False

    @classmethod
    def text_suggests_code(cls, text: str) -> bool:
        if _CODE_FENCE_RE.search(text):
            return True
        if _CODE_PROMPT_RE.search(text):
            return True

        nonblank_lines = [line for line in text.splitlines() if line.strip()]
        if not nonblank_lines:
            return False

        code_like_lines = 0
        indented_code_lines = 0
        for line in nonblank_lines:
            stripped = line.strip()
            if cls._line_suggests_code(stripped):
                code_like_lines += 1
            if line.startswith(("    ", "\t")) and cls._line_suggests_code(stripped, allow_plain_assignment=True):
                indented_code_lines += 1

        if code_like_lines >= 3:
            return True
        if code_like_lines >= 2 and len(nonblank_lines) <= 12:
            return True
        return indented_code_lines >= 2

    @classmethod
    def _line_suggests_code(cls, stripped: str, *, allow_plain_assignment: bool = False) -> bool:
        if not stripped:
            return False
        if any(pattern.search(stripped) for pattern in _CODE_KEYWORD_RES):
            return True
        if _CODE_BRACE_TAIL_RE.search(stripped) and _CODE_HAS_IDENT_RE.search(stripped):
            return True
        if _CODE_OP_RE.search(stripped):
            return True
        if _CODE_TAG_RE.search(stripped):
            return True
        if allow_plain_assignment and _CODE_ASSIGN_RE.search(stripped):
            return True
        return False

    def extract_page_texts(self, file_path: str) -> list[str]:
        _progress_status(f"Opening PDF with pypdf: {Path(file_path).name}", enabled=self.progress_enabled)
        reader = self._get_reader(file_path)
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

    def extract_sampled_page_texts(self, file_path: str, *, max_pages: int = 5) -> list[str]:
        """Extract text from a spread of pages for a cheap usability probe.

        Used by the hybrid parser to decide born-digital vs scanned WITHOUT
        materializing every page's text for a large PDF. Samples up to
        ``max_pages`` pages spread across the document (first, middle, last, and
        evenly-spaced pages between). Each sampled page's text is extracted and
        normalized exactly as in :meth:`extract_page_texts`, so the result is
        directly usable with :meth:`is_text_usable`.

        The reader is cached (via :meth:`_get_reader`) so opening it here does
        not add a second full parse -- the subsequent full extraction reuses the
        same parsed structure. Returns an empty list if the PDF has no pages.
        """
        reader = self._get_reader(file_path)
        total = len(reader.pages)
        if total == 0:
            return []
        if total <= max_pages:
            # Small enough that sampling buys nothing; fall through to the
            # full extraction path (the caller will do that anyway).
            page_indices = list(range(total))
        else:
            # Evenly-spaced spread that always includes the first and last page
            # so a leading/guard-page anomaly or a tail watermark is seen.
            step = (total - 1) / max(1, max_pages - 1)
            page_indices = sorted({int(round(i * step)) for i in range(max_pages)})
        texts: list[str] = []
        for index in page_indices:
            page = reader.pages[index]
            try:
                text = page.extract_text(extraction_mode=self.extraction_mode) or ""
            except TypeError:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Manual text extraction failed on page %s: %s", index + 1, exc)
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


# Precompiled regexes used by the per-page asset-hint heuristics above. These
# functions run over every line of every page during ingestion, so compiling the
# patterns once at import avoids recompiling them on every call (Python's regex
# cache helps, but explicit module-level compiles are faster and clearer).
_TABLE_SPACED_RUN = re.compile(r"\S+\s{2,}\S+")
_NUMBER_RUN = re.compile(r"\d+(?:\.\d+)?")
_VARIABLE_RE = re.compile(r"[A-Za-z]")
_NUMBER_ANYWHERE_RE = re.compile(r"\d")
_EQUATION_ASSIGN_RE = re.compile(r"\b[A-Za-z]\s*=\s*[^=]+")
_INLINE_FORMULA_RE = re.compile(r"(?<!\$)\$[^$\n]{2,}\$(?!\$)")
_DISPLAY_FORMULA_RE = re.compile(r"\\\[[\s\S]{2,}?\\\]|\\\([\s\S]{2,}?\\\)")
_BROKEN_FORMULA_CHAR_RE = re.compile(r"[A-Za-z0-9=+\-*/^_<>()[\]{}]")
_OPERATOR_RUN_RE = re.compile(r"(?:<=|>=|!=|==|[=+\-*/^_<>])")
_CODE_FENCE_RE = re.compile(r"(?m)^\s*(```|~~~)")
_CODE_PROMPT_RE = re.compile(r"(?m)^\s*(>>>|\.\.\.|In \[\d+\]:|\$ )")
_CODE_KEYWORD_RES = tuple(
    re.compile(pattern, flags=re.IGNORECASE) for pattern in ManualTextPdfParser.CODE_KEYWORD_PATTERNS
)
_CODE_BRACE_TAIL_RE = re.compile(r"[{};]$")
_CODE_HAS_IDENT_RE = re.compile(r"[A-Za-z_]")
_CODE_OP_RE = re.compile(r"(->|=>|::|:=|==|!=|&&|\|\|)")
_CODE_TAG_RE = re.compile(r"^\s*</?[A-Za-z][A-Za-z0-9-]*(\s+[^>]*)?>")
_CODE_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.$\[\]]*\s*=\s*[^=]")

ManualTextPdfParser.__module__ = _source_module.__name__
finalize_split_class(_source_module, ManualTextPdfParser)

