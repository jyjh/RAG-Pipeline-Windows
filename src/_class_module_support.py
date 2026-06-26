from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import Any


class _PendingSplitInstance:
    def __init__(self, class_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]):
        self.class_name = class_name
        self.args = args
        self.kwargs = kwargs


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
