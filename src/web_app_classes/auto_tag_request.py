from __future__ import annotations

from pydantic import BaseModel, Field

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class AutoTagRequest(BaseModel):
    """Manual auto-tag sweep: specific hashes (must be ungrouped) or all."""

    source_hashes: list[str] = Field(default_factory=list)
    limit: int | None = None


AutoTagRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, AutoTagRequest)
