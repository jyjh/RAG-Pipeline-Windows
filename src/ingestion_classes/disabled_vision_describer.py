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

    def describe_many(
        self,
        images: list[bytes],
        *,
        prompt: str | None = None,
        max_workers: int = 4,
    ) -> list[str]:
        return ["[Vision analysis disabled]"] * len(images)

DisabledVisionDescriber.__module__ = _source_module.__name__
finalize_split_class(_source_module, DisabledVisionDescriber)

