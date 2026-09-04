from __future__ import annotations

from typing import Union

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
    # Permission set wins over the legacy two-value ``role`` when provided.
    permission_set: str | None = None
    expires_at: str | None = None
    # Upper bound: timedelta/datetime arithmetic overflows at ~999999999 days
    # (and much earlier against datetime.max), which surfaced as an unhandled
    # 500 instead of a 422 on the admin endpoint.
    expires_in_days: float | None = Field(default=None, gt=0, le=36500)
    rate_limit_per_minute: int | None = Field(default=None, gt=0)


class AdminApiKeyStatusRequest(BaseModel):
    status: str


class AdminApiKeyRoleRequest(BaseModel):
    role: str


class AdminApiKeyPermissionSetRequest(BaseModel):
    permission_set: str


class PermissionSetCreateRequest(BaseModel):
    name: str
    label: str = ""
    # ``["*"]`` (or "*") = every category, including ones created later; a list
    # of slugs otherwise. Comma-separated strings are accepted from the CLI.
    categories: Union[list[str], str, None] = None
    can_write: bool = True
    admin: bool = False


class PermissionSetUpdateRequest(BaseModel):
    label: str | None = None
    categories: Union[list[str], str, None] = None
    can_write: bool | None = None
    admin: bool | None = None

AdminApiKeyCreateRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, AdminApiKeyCreateRequest)
AdminApiKeyStatusRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, AdminApiKeyStatusRequest)
AdminApiKeyRoleRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, AdminApiKeyRoleRequest)
