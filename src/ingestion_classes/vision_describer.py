from __future__ import annotations

from typing import Protocol

import src.ingestion as _source_module
from src._class_module_support import finalize_split_class


class VisionDescriber(Protocol):
    def describe(self, image_data: bytes, *, prompt: str | None = None) -> str:
        """Return a technical description for an image."""

    def describe_many(
        self,
        images: list[bytes],
        *,
        prompt: str | None = None,
        max_workers: int = 4,
    ) -> list[str]:
        """Return descriptions for multiple images, concurrently where supported."""

VisionDescriber.__module__ = _source_module.__name__
finalize_split_class(_source_module, VisionDescriber)

