from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.sectioning as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


@dataclass
class PageText:
    page_no: int
    text: str

PageText.__module__ = _source_module.__name__
finalize_split_class(_source_module, PageText)

