from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.local_rag as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class DuckDuckGoLiteParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._link: dict[str, Any] | None = None
        self._snippet: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key.lower(): value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "a" and attr.get("href") and (
            "result-link" in classes
            or "result__a" in classes
            or "result-title-a" in classes
            or "uddg=" in attr.get("href", "")
        ):
            self._link = {"href": attr["href"], "text": []}
            return
        if "result-snippet" in classes or "result__snippet" in classes:
            self._snippet = []

    def handle_data(self, data: str) -> None:
        if self._link is not None:
            self._link["text"].append(data)
        if self._snippet is not None:
            self._snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._link is not None:
            title = " ".join("".join(self._link["text"]).split())
            url = normalize_search_url(str(self._link["href"]))
            if title and url and not any(item["url"] == url for item in self.results):
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._link = None
            return
        if self._snippet is not None:
            snippet = " ".join("".join(self._snippet).split())
            if snippet and self.results and not self.results[-1].get("snippet"):
                self.results[-1]["snippet"] = snippet
            self._snippet = None

DuckDuckGoLiteParser.__module__ = _source_module.__name__
finalize_split_class(_source_module, DuckDuckGoLiteParser)

