from __future__ import annotations

import src.ingestion as _source_module
from src._class_module_support import finalize_split_class


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

