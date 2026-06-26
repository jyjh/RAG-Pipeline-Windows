import hashlib
from types import SimpleNamespace

import pytest

from src.asset_store import ImageAssetStore, image_asset_ids
from src.ingestion import (
    DisabledVisionDescriber,
    DoclingMarkdownRenderer,
    DoclingPdfParser,
    HybridPdfParser,
    ManualTextPdfParser,
    OllamaVisionDescriber,
    ScannedPageImageParser,
    VISION_ANALYSIS_NUM_CTX,
    _build_docling_converter,
    _build_ocr_options,
    run_ingestion,
)


class FakePage:
    def __init__(self, text, resources=None, images=None):
        self.text = text
        self.resources = resources or {}
        self.images = images or []

    def extract_text(self, **kwargs):
        return self.text

    def get(self, key, default=None):
        if key == "/Resources":
            return self.resources
        return default


class FakeReader:
    def __init__(self, page_texts, resources_by_page=None, images_by_page=None):
        resources_by_page = resources_by_page or {}
        images_by_page = images_by_page or {}
        self.pages = [
            FakePage(
                text,
                resources_by_page.get(index + 1),
                images_by_page.get(index + 1),
            )
            for index, text in enumerate(page_texts)
        ]


class FakeItem:
    def __init__(self, label, *, text="", markdown=None, page_no=1, code_language=None, orig=None):
        self.label = label
        self.text = text
        self.markdown = markdown
        self.code_language = code_language
        self.orig = text if orig is None else orig
        self.prov = [SimpleNamespace(page_no=page_no)]

    def export_to_markdown(self):
        return self.markdown


class FakeDoc:
    def __init__(self, items):
        self.items = items

    def iterate_items(self):
        for item in self.items:
            yield item, None


class FakeConverter:
    def __init__(self, doc):
        self.doc = doc

    def convert(self, file_path):
        return SimpleNamespace(document=self.doc)


class FailingConverter:
    def convert(self, file_path):
        raise RuntimeError("docling unavailable")


def reader_factory(page_texts, resources_by_page=None, images_by_page=None):
    return lambda file_path: FakeReader(page_texts, resources_by_page, images_by_page)


def docling_parser_for(items):
    return DoclingPdfParser(
        vision_describer=DisabledVisionDescriber(),
        converter=FakeConverter(FakeDoc(items)),
    )


class FakeImage:
    is_displayed = True

    def __init__(self):
        self.image = self

    def save(self, handle, format):
        handle.write(b"fake-png")


class FakePictureItem(FakeItem):
    def __init__(self, *, page_no=1, image=None):
        super().__init__("picture", page_no=page_no)
        self._image = image or FakeImage()

    def get_image(self, doc):
        return self._image


class RecordingVisionDescriber:
    def __init__(self, response="described scanned vibration diagram"):
        self.response = response
        self.calls = []

    def describe(self, image_data, *, prompt=None):
        self.calls.append({"image_data": image_data, "prompt": prompt})
        return self.response


def test_docling_converter_defaults_to_rapidocr_onnx_full_page():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import RapidOcrOptions

    converter = _build_docling_converter()
    options = converter.format_to_options[InputFormat.PDF].pipeline_options
    ocr_options = options.ocr_options

    assert isinstance(ocr_options, RapidOcrOptions)
    assert ocr_options.backend == "onnxruntime"
    assert ocr_options.lang == ["english"]
    assert ocr_options.force_full_page_ocr is True
    assert options.do_code_enrichment is False
    assert options.do_formula_enrichment is False


def test_docling_converter_sets_code_and_formula_enrichment_flags():
    from docling.datamodel.base_models import InputFormat

    converter = _build_docling_converter(code_enrichment=True, formula_enrichment=True)
    options = converter.format_to_options[InputFormat.PDF].pipeline_options

    assert options.do_code_enrichment is True
    assert options.do_formula_enrichment is True


def test_docling_renderer_formats_code_and_formula_items():
    renderer = DoclingMarkdownRenderer(DisabledVisionDescriber())
    doc = FakeDoc(
        [
            FakeItem(
                "code",
                text="def solve(value):\n    return value",
                code_language=SimpleNamespace(value="python"),
            ),
            FakeItem("formula", text="E = m c^2"),
        ]
    )

    output = renderer.render_document(doc)

    assert "```python\ndef solve(value):\n    return value\n```" in output
    assert "$E = m c^2$" in output


def test_docling_renderer_stores_described_picture_assets(safe_tmp_path):
    store = ImageAssetStore(safe_tmp_path / "assets")
    vision = RecordingVisionDescriber("A plotted suspension response with labeled axes.")
    renderer = DoclingMarkdownRenderer(vision, image_asset_store=store)
    renderer.set_source_context(source_hash="source-hash-a", source_pdf_name="report.pdf")

    output = renderer.render_document(FakeDoc([FakePictureItem(page_no=3)]))

    asset_ids = image_asset_ids(output)
    assert len(asset_ids) == 1
    assert "[Vision Analysis]: A plotted suspension response with labeled axes." in output
    asset = store.get_asset(asset_ids[0])
    assert asset is not None
    assert asset["source_hash"] == "source-hash-a"
    assert asset["source_pdf_name"] == "report.pdf"
    assert asset["page_no"] == 3
    assert asset["description"] == "A plotted suspension response with labeled axes."
    assert asset["mime_type"] == "image/png"
    assert asset["image_sha"] == hashlib.sha256(b"fake-png").hexdigest()
    assert store.asset_path(asset_ids[0]).read_bytes() == b"fake-png"


def test_docling_renderer_skips_failed_picture_asset_descriptions(safe_tmp_path):
    store = ImageAssetStore(safe_tmp_path / "assets")
    renderer = DoclingMarkdownRenderer(
        RecordingVisionDescriber("[Image description failed]"),
        image_asset_store=store,
    )
    renderer.set_source_context(source_hash="source-hash-a", source_pdf_name="report.pdf")

    output = renderer.render_document(FakeDoc([FakePictureItem(page_no=2)]))

    assert "[Vision Analysis]: [Image description failed]" in output
    assert "[Image Asset:" not in output
    assert store.load_manifest()["assets"] == {}


def test_docling_ocr_options_support_tesseract_and_easyocr():
    from docling.datamodel.pipeline_options import EasyOcrOptions, TesseractCliOcrOptions

    tesseract = _build_ocr_options(
        ocr_backend="tesseract_cli",
        ocr_langs=["eng"],
        tesseract_cmd="C:/Tools/tesseract.exe",
        tesseract_data_path="C:/Tools/tessdata",
        tesseract_psm=6,
    )
    easyocr = _build_ocr_options(ocr_backend="easyocr", ocr_langs=["en"])

    assert isinstance(tesseract, TesseractCliOcrOptions)
    assert tesseract.tesseract_cmd == "C:/Tools/tesseract.exe"
    assert tesseract.path == "C:/Tools/tessdata"
    assert tesseract.psm == 6
    assert isinstance(easyocr, EasyOcrOptions)
    assert easyocr.lang == ["en"]


def test_unsupported_ocr_backend_is_rejected():
    with pytest.raises(ValueError, match="Unsupported OCR backend"):
        _build_ocr_options(ocr_backend="unknown")


def test_ollama_vision_describer_uses_prompt_and_larger_context(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(response="vision text")

    monkeypatch.setattr("src.ingestion._ollama_generate", fake_generate)

    describer = OllamaVisionDescriber(vision_model="vision-test")
    result = describer.describe(b"image", prompt="custom prompt")

    assert result == "vision text"
    assert calls[0]["prompt"] == "Hello"
    assert calls[1]["model"] == "vision-test"
    assert calls[1]["prompt"] == "custom prompt"
    assert calls[1]["options"] == {"num_ctx": VISION_ANALYSIS_NUM_CTX}


def test_docling_parser_builds_converter_lazily(monkeypatch):
    calls = []

    def fake_build_docling_converter(**kwargs):
        calls.append(kwargs)
        return FakeConverter(FakeDoc([]))

    monkeypatch.setattr("src.ingestion._build_docling_converter", fake_build_docling_converter)

    parser = DoclingPdfParser(vision_describer=DisabledVisionDescriber())

    assert calls == []
    assert parser.convert("fake.pdf").document.items == []
    assert len(calls) == 1


def test_manual_text_detection_uses_pypdf_text_threshold():
    parser = ManualTextPdfParser(
        docling_parser=docling_parser_for([]),
        reader_factory=reader_factory(["Short", "Enough extracted text"]),
        min_text_chars=10,
    )

    assert parser.has_extractable_text("fake.pdf")


def test_manual_text_detection_rejects_empty_pdf_text():
    parser = ManualTextPdfParser(
        docling_parser=docling_parser_for([]),
        reader_factory=reader_factory(["   ", ""]),
        min_text_chars=1,
    )

    assert not parser.has_extractable_text("fake.pdf")


def test_manual_parser_keeps_page_text_and_docling_assets_without_duplicate_text():
    items = [
        FakeItem("text", markdown="Docling duplicate text", page_no=1),
        FakeItem("table", markdown="| Col |\n| --- |\n| 1 |", page_no=1),
        FakeItem("formula", markdown="$E = mc^2$", page_no=2),
    ]
    parser = ManualTextPdfParser(
        docling_parser=docling_parser_for(items),
        reader_factory=reader_factory(["Manual page one", "Manual page two"]),
        min_text_chars=1,
    )

    output = parser.parse("fake.pdf", include_assets=True)

    assert "Manual page one" in output
    assert "Manual page two" in output
    assert "| Col |" in output
    assert "$E = mc^2$" in output
    assert "Docling duplicate text" not in output
    assert output.index("Manual page one") < output.index("| Col |")
    assert output.index("Manual page two") < output.index("$E = mc^2$")


def test_manual_parser_returns_text_when_docling_asset_pass_fails():
    parser = ManualTextPdfParser(
        docling_parser=DoclingPdfParser(
            vision_describer=DisabledVisionDescriber(),
            converter=FailingConverter(),
        ),
        reader_factory=reader_factory(["Manual text survives"]),
        min_text_chars=1,
    )

    assert parser.parse("fake.pdf", include_assets=True) == "Manual text survives"


def test_manual_parser_defaults_to_pypdf_text_without_docling_assets():
    parser = ManualTextPdfParser(
        docling_parser=DoclingPdfParser(
            vision_describer=DisabledVisionDescriber(),
            converter=FailingConverter(),
        ),
        reader_factory=reader_factory(["Clean extracted text"]),
        min_text_chars=1,
    )

    assert parser.parse("fake.pdf") == "Clean extracted text"


def test_manual_parser_detects_images_from_pypdf_resources():
    resources = {
        1: {
            "/XObject": {
                "/Im1": {
                    "/Subtype": "/Image",
                }
            }
        }
    }
    parser = ManualTextPdfParser(
        docling_parser=docling_parser_for([]),
        reader_factory=reader_factory(["Clean extracted text"], resources),
        min_text_chars=1,
        asset_triggers="images",
    )

    assert parser.image_pages("fake.pdf") == {1}
    assert parser.needs_asset_enrichment("fake.pdf", ["Clean extracted text"])


def test_manual_parser_auto_detects_picture_code_and_formula_pages():
    resources = {
        1: {
            "/XObject": {
                "/Im1": {
                    "/Subtype": "/Image",
                }
            }
        }
    }
    page_texts = [
        "Clean caption text",
        "def solve(value):\n    return value\nclass Solver:",
        "E = m * c ^ 2",
    ]
    parser = ManualTextPdfParser(
        docling_parser=docling_parser_for([]),
        reader_factory=reader_factory(page_texts, resources),
        min_text_chars=1,
        asset_triggers="auto",
    )

    hints = parser.asset_enrichment_page_hints("fake.pdf", page_texts)

    assert hints == {
        1: {"picture"},
        2: {"code"},
        3: {"formula"},
    }
    assert parser.asset_enrichment_pages("fake.pdf", page_texts) == {1, 2, 3}


def test_manual_parser_images_trigger_remains_picture_only():
    resources = {
        1: {
            "/XObject": {
                "/Im1": {
                    "/Subtype": "/Image",
                }
            }
        }
    }
    page_texts = [
        "Clean caption text",
        "def solve(value):\n    return value\nclass Solver:",
        "E = m * c ^ 2",
    ]
    parser = ManualTextPdfParser(
        docling_parser=docling_parser_for([]),
        reader_factory=reader_factory(page_texts, resources),
        min_text_chars=1,
        asset_triggers="images",
    )

    assert parser.asset_enrichment_page_hints("fake.pdf", page_texts) == {1: {"picture"}}


def test_manual_parser_detects_table_and_equation_text():
    parser = ManualTextPdfParser(
        docling_parser=docling_parser_for([]),
        reader_factory=reader_factory(["unused"]),
        min_text_chars=1,
    )

    table_text = "A  B  C\n1  2  3\n4  5  6\n7  8  9"
    equation_text = "E = m * c ^ 2"

    assert parser.text_suggests_table(table_text)
    assert parser.text_suggests_equation(equation_text)
    assert not parser.needs_asset_enrichment("fake.pdf", [table_text, equation_text])


def test_manual_parser_can_opt_into_table_and_equation_asset_pages():
    parser = ManualTextPdfParser(
        docling_parser=docling_parser_for([]),
        reader_factory=reader_factory(["unused"]),
        min_text_chars=1,
        asset_triggers="all",
    )

    table_text = "A  B  C\n1  2  3\n4  5  6\n7  8  9"
    equation_text = "E = m * c ^ 2"

    assert parser.asset_enrichment_pages("fake.pdf", [table_text, equation_text]) == {1, 2}
    assert parser.asset_enrichment_page_hints("fake.pdf", [table_text, equation_text]) == {
        1: {"table"},
        2: {"formula"},
    }


def test_manual_parser_uses_enriched_parser_when_requested():
    light_parser = docling_parser_for([FakeItem("table", markdown="| A |\n| - |", page_no=1)])
    enriched_parser = docling_parser_for(
        [
            FakeItem(
                "code",
                text="def solve(value):\n    return value",
                code_language=SimpleNamespace(value="python"),
                page_no=1,
            )
        ]
    )
    parser = ManualTextPdfParser(
        docling_parser=light_parser,
        enriched_docling_parser=enriched_parser,
        reader_factory=reader_factory(["unused"]),
        min_text_chars=1,
    )

    assert parser.extract_docling_assets("fake.pdf", enriched=False) == {1: ["| A |\n| - |"]}
    assert parser.extract_docling_assets("fake.pdf", enriched=True) == {
        1: ["```python\ndef solve(value):\n    return value\n```"]
    }


def test_manual_parser_direct_parse_auto_includes_detected_assets():
    calls = {}
    page_texts = [
        "Plain text",
        "def solve(value):\n    return value\nclass Solver:",
        "E = m * c ^ 2",
    ]

    class RecordingManualParser(ManualTextPdfParser):
        def extract_docling_assets_for_pages(self, file_path, page_numbers, *, enriched_asset_pages=None):
            calls["page_numbers"] = page_numbers
            calls["enriched_asset_pages"] = enriched_asset_pages
            return {2: ["```python\nreturn value\n```"], 3: ["$E = mc^2$"]}

    parser = RecordingManualParser(
        docling_parser=docling_parser_for([]),
        reader_factory=reader_factory(page_texts),
        min_text_chars=1,
        asset_triggers="auto",
    )

    output = parser.parse("fake.pdf")

    assert calls == {"page_numbers": {2, 3}, "enriched_asset_pages": {2, 3}}
    assert "Plain text" in output
    assert "```python\nreturn value\n```" in output
    assert "$E = mc^2$" in output


def test_hybrid_parser_routes_to_manual_text_only_when_clean_text_has_no_assets():
    class Manual:
        def extract_page_texts(self, file_path):
            return ["clean text"]

        def is_text_usable(self, page_texts):
            return True

        def asset_enrichment_pages(self, file_path, page_texts):
            return set()

        def parse(self, file_path, *, include_assets=False, asset_pages=None, enriched_asset_pages=None, page_texts=None):
            assert not include_assets
            assert asset_pages == set()
            assert enriched_asset_pages == set()
            assert page_texts == ["clean text"]
            return "manual text"

    class Docling:
        def parse(self, file_path):
            return "docling"

    parser = HybridPdfParser(manual_parser=Manual(), docling_parser=Docling())

    assert parser.parse("fake.pdf") == "manual text"


def test_hybrid_parser_routes_to_manual_with_assets_when_asset_detected():
    class Manual:
        def extract_page_texts(self, file_path):
            return ["clean text"]

        def is_text_usable(self, page_texts):
            return True

        def asset_enrichment_pages(self, file_path, page_texts):
            return {1}

        def parse(self, file_path, *, include_assets=False, asset_pages=None, enriched_asset_pages=None, page_texts=None):
            assert include_assets
            assert asset_pages == {1}
            assert enriched_asset_pages == set()
            return "manual plus assets"

    class Docling:
        def parse(self, file_path):
            return "docling"

    parser = HybridPdfParser(manual_parser=Manual(), docling_parser=Docling())

    assert parser.parse("fake.pdf") == "manual plus assets"


def test_hybrid_parser_routes_to_docling_when_text_is_garbage():
    class Manual:
        def extract_page_texts(self, file_path):
            return ["����"]

        def is_text_usable(self, page_texts):
            return False

        def parse(self, file_path):
            return "manual"

    class Docling:
        def parse(self, file_path):
            return "docling"

    parser = HybridPdfParser(manual_parser=Manual(), docling_parser=Docling())

    assert parser.parse("fake.pdf") == "docling"


def test_hybrid_parser_falls_back_to_page_image_analysis_when_docling_fails():
    vision = RecordingVisionDescriber()
    fallback = ScannedPageImageParser(
        vision_describer=vision,
        vision_enabled=True,
        reader_factory=reader_factory([""], images_by_page={1: [FakeImage()]}),
    )

    class Manual:
        def extract_page_texts(self, file_path):
            return [""]

        def is_text_usable(self, page_texts):
            return False

    class Docling:
        def parse(self, file_path):
            raise RuntimeError("ocr failed")

    parser = HybridPdfParser(
        manual_parser=Manual(),
        docling_parser=Docling(),
        scanned_page_parser=fallback,
    )

    output = parser.parse("fake.pdf")

    assert "## Page 1" in output
    assert "[Page Image Analysis]" in output
    assert "described scanned vibration diagram" in output
    assert vision.calls
    assert "scanned STEM textbook page" in vision.calls[0]["prompt"]


def test_hybrid_parser_falls_back_to_page_image_analysis_when_docling_is_empty():
    vision = RecordingVisionDescriber("visible equation and mechanism diagram")
    fallback = ScannedPageImageParser(
        vision_describer=vision,
        vision_enabled=True,
        reader_factory=reader_factory([""], images_by_page={1: [FakeImage()]}),
    )

    class Manual:
        def extract_page_texts(self, file_path):
            return [""]

        def is_text_usable(self, page_texts):
            return False

    class Docling:
        def parse(self, file_path):
            return "   "

    parser = HybridPdfParser(
        manual_parser=Manual(),
        docling_parser=Docling(),
        scanned_page_parser=fallback,
    )

    assert "visible equation and mechanism diagram" in parser.parse("fake.pdf")


def test_scanned_page_fallback_requires_vision_enabled():
    fallback = ScannedPageImageParser(
        vision_describer=DisabledVisionDescriber(),
        vision_enabled=False,
        reader_factory=reader_factory([""], images_by_page={1: [FakeImage()]}),
    )

    with pytest.raises(RuntimeError, match="vision is disabled"):
        fallback.parse("fake.pdf")


def test_scanned_page_fallback_rejects_failed_vision_descriptions():
    fallback = ScannedPageImageParser(
        vision_describer=RecordingVisionDescriber("[Image description failed]"),
        vision_enabled=True,
        reader_factory=reader_factory([""], images_by_page={1: [FakeImage()]}),
    )

    with pytest.raises(RuntimeError, match="No usable page-image analysis"):
        fallback.parse("fake.pdf")


def test_run_ingestion_refuses_empty_markdown(monkeypatch, safe_tmp_path):
    input_pdf = safe_tmp_path / "scan.pdf"
    input_pdf.write_bytes(b"%PDF-1.4")
    output_dir = safe_tmp_path / "processed"

    class EmptyProcessor:
        def __init__(self, **kwargs):
            pass

        def set_source_context(self, **kwargs):
            pass

        def process_pdf(self, file_path):
            return ""

    monkeypatch.setattr("src.ingestion.DocumentProcessor", EmptyProcessor)

    with pytest.raises(RuntimeError, match="empty Markdown"):
        run_ingestion(str(input_pdf), str(output_dir), progress_enabled=False)

    assert not (output_dir / "scan.md").exists()
    assert not (output_dir / ".source_map.json").exists()


def test_run_ingestion_removes_stale_assets_before_processing(monkeypatch, safe_tmp_path):
    input_pdf = safe_tmp_path / "source.pdf"
    input_pdf.write_bytes(b"%PDF-1.4 source")
    source_hash = hashlib.sha256(input_pdf.read_bytes()).hexdigest()
    output_dir = safe_tmp_path / "processed"
    asset_dir = safe_tmp_path / "assets"
    store = ImageAssetStore(asset_dir)
    stale = store.save_image(
        image_data=b"stale-png",
        source_hash=source_hash,
        source_pdf_name="source.pdf",
        page_no=1,
        description="stale graph",
    )
    calls = {}

    class Processor:
        def __init__(self, **kwargs):
            calls["asset_dir"] = kwargs["asset_dir"]

        def set_source_context(self, **kwargs):
            calls["context"] = kwargs

        def process_pdf(self, file_path):
            calls["asset_exists_during_process"] = store.asset_path(stale["asset_id"]) is not None
            return "fresh markdown"

    monkeypatch.setattr("src.ingestion.DocumentProcessor", Processor)

    run_ingestion(str(input_pdf), str(output_dir), asset_dir=asset_dir, progress_enabled=False)

    assert calls["asset_dir"] == asset_dir
    assert calls["context"] == {"source_hash": source_hash, "source_pdf_name": "source.pdf"}
    assert calls["asset_exists_during_process"] is False
    assert store.load_manifest()["assets"] == {}
    assert (output_dir / "source.md").read_text(encoding="utf-8") == "fresh markdown"
