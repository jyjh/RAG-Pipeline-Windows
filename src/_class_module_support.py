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


def _pending_split_class(class_name: str, module_name: str | None = None):
    try:
        from pydantic import BaseModel
    except Exception:
        bases = (object,)
    else:
        bases = (BaseModel,)

    def __new__(cls, *args, **kwargs):
        # The split module was mid-import when this stub was handed out
        # (circular import, e.g. ``python -m src.web_app``). By call time it
        # has finished; resolve the real class and construct that instead.
        if module_name:
            module = importlib.import_module(module_name)
            real = vars(module).get(class_name)
            if (
                isinstance(real, type)
                and real is not cls
                and not hasattr(real, "_split_pending_class_name")
            ):
                return real(*args, **kwargs)
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
    return getattr(
        module, class_name, _pending_split_class(class_name, module_name)
    )


# Split-class stubs placed into borrowing modules' globals by
# bind_module_namespace, as (target_globals, name). Under a circular import
# (e.g. ``python -m src.web_app``: queue_job -> web_app -> queue_job), a
# borrower like rag_job_queue can bind the pending stub for a class whose
# split module is still mid-import; setdefault would keep that dead stub
# forever. finalize_split_class uses this registry to rebind the real class.
_STUB_BINDINGS: list[tuple[dict[str, Any], str]] = []


def _is_pending_stub_class(value: Any) -> bool:
    return isinstance(value, type) and hasattr(value, "_split_pending_class_name")


def finalize_split_class(module: ModuleType, cls: type) -> None:
    setattr(module, cls.__name__, cls)
    for name, value in list(module.__dict__.items()):
        if isinstance(value, _PendingSplitInstance) and value.class_name == cls.__name__:
            setattr(module, name, cls(*value.args, **value.kwargs))
    remaining: list[tuple[dict[str, Any], str]] = []
    for target_globals, name in _STUB_BINDINGS:
        current = target_globals.get(name)
        if current is cls:
            continue
        if (
            _is_pending_stub_class(current)
            and current._split_pending_class_name == cls.__name__
        ):
            target_globals[name] = cls
        else:
            remaining.append((target_globals, name))
    _STUB_BINDINGS[:] = remaining


def bind_module_namespace(
    module: ModuleType,
    target_globals: dict[str, Any],
    *,
    proxy_functions: Iterable[str] = (),
) -> None:
    """Bind a split class module to its original compatibility module."""
    for name, value in module.__dict__.items():
        if not name.startswith("__"):
            if _is_pending_stub_class(value):
                _STUB_BINDINGS.append((target_globals, name))
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
