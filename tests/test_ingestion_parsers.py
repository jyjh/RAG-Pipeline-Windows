from types import SimpleNamespace

from src.ingestion import (
    DisabledVisionDescriber,
    DoclingPdfParser,
    HybridPdfParser,
    ManualTextPdfParser,
)


class FakePage:
    def __init__(self, text, resources=None):
        self.text = text
        self.resources = resources or {}

    def extract_text(self, **kwargs):
        return self.text

    def get(self, key, default=None):
        if key == "/Resources":
            return self.resources
        return default


class FakeReader:
    def __init__(self, page_texts, resources_by_page=None):
        resources_by_page = resources_by_page or {}
        self.pages = [
            FakePage(text, resources_by_page.get(index + 1))
            for index, text in enumerate(page_texts)
        ]


class FakeItem:
    def __init__(self, label, *, text="", markdown=None, page_no=1):
        self.label = label
        self.text = text
        self.markdown = markdown
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


def reader_factory(page_texts, resources_by_page=None):
    return lambda file_path: FakeReader(page_texts, resources_by_page)


def docling_parser_for(items):
    return DoclingPdfParser(
        vision_describer=DisabledVisionDescriber(),
        converter=FakeConverter(FakeDoc(items)),
    )


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


def test_hybrid_parser_routes_to_manual_text_only_when_clean_text_has_no_assets():
    class Manual:
        def extract_page_texts(self, file_path):
            return ["clean text"]

        def is_text_usable(self, page_texts):
            return True

        def asset_enrichment_pages(self, file_path, page_texts):
            return set()

        def parse(self, file_path, *, include_assets=False, asset_pages=None, page_texts=None):
            assert not include_assets
            assert asset_pages == set()
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

        def parse(self, file_path, *, include_assets=False, asset_pages=None, page_texts=None):
            assert include_assets
            assert asset_pages == {1}
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
