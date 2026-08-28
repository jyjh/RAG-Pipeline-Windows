from __future__ import annotations

from pydantic import BaseModel, Field

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class AdminApiKeyCreateRequest(BaseModel):
    label: str = ""
    role: str = "user"
    expires_at: str | None = None
    expires_in_days: float | None = Field(default=None, gt=0)
    rate_limit_per_minute: int | None = Field(default=None, gt=0)


class AdminApiKeyStatusRequest(BaseModel):
    status: str


class AdminApiKeyRoleRequest(BaseModel):
    role: str

AdminApiKeyCreateRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, AdminApiKeyCreateRequest)
AdminApiKeyStatusRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, AdminApiKeyStatusRequest)
AdminApiKeyRoleRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, AdminApiKeyRoleRequest)
