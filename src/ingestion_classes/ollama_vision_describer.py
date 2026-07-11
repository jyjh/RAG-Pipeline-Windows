from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class OllamaVisionDescriber:
    def __init__(self, vision_model: str = "qwen2.5vl:7b"):
        self.vision_model = vision_model
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """
        Lazily warm up the vision model on first figure encounter.
        Avoids consuming VRAM on documents that contain no images.
        """
        if self._loaded:
            return

        logger.info("First figure detected - loading vision model: %s", self.vision_model)
        try:
            _ollama_generate(model=self.vision_model, prompt="Hello", keep_alive="30m")
        except Exception as exc:
            logger.warning("Could not pre-warm vision model: %s", exc)
        self._loaded = True

    def describe(self, image_data: bytes, *, prompt: str | None = None) -> str:
        self._ensure_loaded()
        try:
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            response = _ollama_generate(
                model=self.vision_model,
                prompt=prompt or STEM_VISION_PROMPT,
                images=[encoded_image],
                options={"num_ctx": VISION_ANALYSIS_NUM_CTX},
            )
            return response.response or "[Image description empty]"
        except Exception as exc:
            logger.error("Error processing image with vision model: %s", exc)
            return "[Image description failed]"

    def describe_many(
        self,
        images: list[bytes],
        *,
        prompt: str | None = None,
        max_workers: int = 4,
    ) -> list[str]:
        """Describe multiple images concurrently.

        Each image is an independent Ollama HTTP call, so a bounded thread pool
        overlaps them -- a scanned-document with N pages describes N page images
        in roughly ``ceil(N / max_workers)`` round-trips instead of N. Concurrency
        is capped to avoid overloading Ollama. Falls back to serial on error.
        """
        if not images:
            return []
        if len(images) == 1 or max_workers <= 1:
            return [self.describe(img, prompt=prompt) for img in images]
        self._ensure_loaded()
        from concurrent.futures import ThreadPoolExecutor

        max_workers = max(1, min(int(max_workers), len(images)))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(lambda img: self.describe(img, prompt=prompt), images))
        except Exception:
            # Fall back to serial; concurrency is best-effort.
            return [self.describe(img, prompt=prompt) for img in images]

OllamaVisionDescriber.__module__ = _source_module.__name__
finalize_split_class(_source_module, OllamaVisionDescriber)

