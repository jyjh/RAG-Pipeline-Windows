"""Shared value-coercion helpers.

One implementation of the bool/int/float/str-list parsers previously copied
(with slight divergence) across main.py, web_app.py, ingestion.py, local_rag.py,
and embeddings.py. All of them are total functions: an unparseable or
out-of-range value falls back to ``default`` instead of raising, because they
sit on the config-file / CLI-arg / env-var boundary where bad input must
degrade to a sane default, not crash a long ingest run.
"""
from __future__ import annotations

from typing import Any

_TRUE_TOKENS = {"1", "true", "yes", "on"}
_FALSE_TOKENS = {"0", "false", "no", "off"}


def as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def as_positive_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
    """Parse ``value`` as a float strictly above ``minimum``; else ``default``."""
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > minimum else default


def as_optional_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_bounded_float(value: Any, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, as_float(value, default)))


def as_string_list(value: Any, default: list[str] | tuple[str, ...]) -> list[str]:
    """Parse a comma-separated string or an iterable into a clean str list."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or list(default)
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return parts or list(default)
    return list(default)
