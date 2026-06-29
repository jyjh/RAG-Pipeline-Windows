from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.local_rag as _source_module
from src.reliability import SOURCE_GROUP_UNGROUPED

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class CitationRegistry:
    def __init__(self, *, asset_store: Any | None = None):
        self.asset_store = asset_store
        self._local_by_record_id: dict[str, dict[str, Any]] = {}
        self._web_by_url: dict[str, dict[str, Any]] = {}
        self._local_sources: list[dict[str, Any]] = []
        self._web_sources: list[dict[str, Any]] = []

    def _asset_url(self, asset_id: str) -> str:
        encoded = urllib.parse.quote(asset_id, safe="")
        return f"/api/assets/{encoded}"

    def _local_source(self, record: dict[str, Any], index: int) -> dict[str, Any]:
        record_id = str(record.get("id") or "")
        source_hash = str(record.get("source_hash") or "")
        page_start = int(record.get("page_start") or 0)
        source = {
            "id": f"S{index}",
            "kind": "local",
            "label": f"[S{index}]",
            "chunk_id": record_id,
            "doc_id": str(record.get("doc_id") or ""),
            "source_hash": source_hash,
            "source_pdf_name": _source_pdf_name(record),
            "source_pdf_path": str(record.get("source_pdf_path") or ""),
            "file_path": str(record.get("file_path") or ""),
            "section_path": str(record.get("section_path") or ""),
            "page_start": page_start,
            "page_end": int(record.get("page_end") or 0),
            "page_label": _page_label(record),
            "score": round(float(record.get("score") or 0.0), 4),
            "vector_score": round(float(record.get("vector_score") or 0.0), 4),
            "lexical_score": round(float(record.get("lexical_score") or 0.0), 4),
            "hybrid_score": round(float(record.get("hybrid_score") or 0.0), 4),
            "reliability_modifier": round(float(record.get("reliability_modifier") or 0.0), 4),
            "source_group": str(record.get("source_group") or SOURCE_GROUP_UNGROUPED),
            "snippet": _short_snippet(str(record.get("content") or "")),
            "open_url": _pdf_source_url(source_hash, mode="view", page=page_start),
            "download_url": _pdf_source_url(source_hash, mode="download"),
        }
        if self.asset_store is not None:
            assets = self.asset_store.assets_for_text(
                str(record.get("content") or ""),
                url_for=self._asset_url,
            )
            if assets:
                source["assets"] = assets
        return source

    def preview_local(self, record: dict[str, Any]) -> dict[str, Any]:
        record_id = str(record.get("id") or "")
        if record_id in self._local_by_record_id:
            return self._local_by_record_id[record_id]
        return self._local_source(record, len(self._local_sources) + 1)

    def add_local(self, record: dict[str, Any]) -> dict[str, Any]:
        record_id = str(record.get("id") or "")
        if record_id in self._local_by_record_id:
            return self._local_by_record_id[record_id]
        source = self._local_source(record, len(self._local_sources) + 1)
        self._local_by_record_id[record_id] = source
        self._local_sources.append(source)
        return source

    def _web_source(self, result: dict[str, Any], index: int) -> dict[str, Any]:
        url = str(result.get("url") or "")
        return {
            "id": f"W{index}",
            "kind": "web",
            "label": f"[W{index}]",
            "title": str(result.get("title") or url),
            "url": url,
            "snippet": _short_snippet(str(result.get("snippet") or ""), limit=300),
            "provider": str(result.get("provider") or "duckduckgo_lite"),
        }

    def preview_web(self, result: dict[str, Any]) -> dict[str, Any]:
        url = str(result.get("url") or "")
        if url in self._web_by_url:
            return self._web_by_url[url]
        return self._web_source(result, len(self._web_sources) + 1)

    def add_web(self, result: dict[str, Any]) -> dict[str, Any]:
        url = str(result.get("url") or "")
        if url in self._web_by_url:
            return self._web_by_url[url]
        source = self._web_source(result, len(self._web_sources) + 1)
        self._web_by_url[url] = source
        self._web_sources.append(source)
        return source

    def local_record_ids(self) -> set[str]:
        return set(self._local_by_record_id)

    def all_sources(self) -> list[dict[str, Any]]:
        return [*self._local_sources, *self._web_sources]

    def valid_labels(self) -> set[str]:
        return {source["label"] for source in self.all_sources()}

CitationRegistry.__module__ = _source_module.__name__
finalize_split_class(_source_module, CitationRegistry)

