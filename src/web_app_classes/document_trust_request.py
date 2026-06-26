from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class DocumentTrustRequest(BaseModel):
    review_status: str | None = None
    source_type: str | None = None
    publication_year: int | None = None
    expires_at: str | None = None
    reviewed_by: str | None = None
    notes: str | None = None

DocumentTrustRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, DocumentTrustRequest)

