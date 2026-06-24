from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def unload_model(model_name: str):
    """Unload a model from VRAM via Ollama."""
    try:
        import ollama

        logger.info("Unloading model: %s", model_name)
        ollama.generate(model=model_name, prompt=" ", keep_alive="0")
    except Exception as exc:
        logger.error("Failed to unload model %s: %s", model_name, exc)


def load_model(model_name: str):
    """Pre-load a model into VRAM via Ollama."""
    try:
        import ollama

        logger.info("Loading model: %s", model_name)
        ollama.generate(model=model_name, prompt="Hello", keep_alive="30m")
    except Exception as exc:
        logger.error("Failed to load model %s: %s", model_name, exc)


def manage_vram(active_model: str, model_to_unload: str = None):
    """Switch between Ollama models to optimize VRAM usage."""
    if model_to_unload:
        unload_model(model_to_unload)
    load_model(active_model)
