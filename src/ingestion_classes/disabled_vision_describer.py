from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class DisabledVisionDescriber:
    def describe(self, image_data: bytes, *, prompt: str | None = None) -> str:
        return "[Vision analysis disabled]"

DisabledVisionDescriber.__module__ = _source_module.__name__
finalize_split_class(_source_module, DisabledVisionDescriber)

