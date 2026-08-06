from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import Any


class _PendingSplitInstance:
    """Sentinel returned when a split class could not be resolved at import.

    A real instance of this (rather than the intended class) means the split
    module failed to define/attribute its class -- usually a NameError during
    the module body under a circular-import timing edge. Attribute access is
    intercepted so the failure surfaces at the point of use with an actionable
    message, instead of a cryptic downstream ``AttributeError``.
    """

    def __init__(self, class_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]):
        object.__setattr__(self, "_pending_class_name", class_name)
        object.__setattr__(self, "_pending_args", args)
        object.__setattr__(self, "_pending_kwargs", kwargs)

    @property
    def class_name(self) -> str:
        return object.__getattribute__(self, "_pending_class_name")

    @property
    def args(self) -> tuple[Any, ...]:
        return object.__getattribute__(self, "_pending_args")

    @property
    def kwargs(self) -> dict[str, Any]:
        return object.__getattribute__(self, "_pending_kwargs")

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"'{type(self).__name__}' for class "
            f"'{object.__getattribute__(self, '_pending_class_name')!r}' is a "
            f"pending split-class stub: the real class was never finalized into "
            f"its module (the split module likely raised during import). "
            f"Accessed attribute {name!r}."
        )


def _pending_split_class(class_name: str):
    try:
        from pydantic import BaseModel
    except Exception:
        bases = (object,)
    else:
        bases = (BaseModel,)

    def __new__(cls, *args, **kwargs):
        return _PendingSplitInstance(class_name, args, kwargs)

    return type(
        class_name,
        bases,
        {
            "__module__": __name__,
            "__new__": __new__,
            "_split_pending_class_name": class_name,
        },
    )


def import_split_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name, _pending_split_class(class_name))


def finalize_split_class(module: ModuleType, cls: type) -> None:
    setattr(module, cls.__name__, cls)
    for name, value in list(module.__dict__.items()):
        if isinstance(value, _PendingSplitInstance) and value.class_name == cls.__name__:
            setattr(module, name, cls(*value.args, **value.kwargs))


def bind_module_namespace(
    module: ModuleType,
    target_globals: dict[str, Any],
    *,
    proxy_functions: Iterable[str] = (),
) -> None:
    """Bind a split class module to its original compatibility module."""
    for name, value in module.__dict__.items():
        if not name.startswith("__"):
            target_globals.setdefault(name, value)

    for name in proxy_functions:
        target_globals[name] = _module_function_proxy(module, name)


def _module_function_proxy(module: ModuleType, name: str):
    def proxy(*args, **kwargs):
        return getattr(module, name)(*args, **kwargs)

    proxy.__name__ = name
    proxy.__qualname__ = name
    proxy.__module__ = module.__name__
    return proxy
