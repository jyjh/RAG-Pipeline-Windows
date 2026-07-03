from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class IndexUpdateRequest(BaseModel):
    record_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    embedding_model: str | None = None
    embedding_batch_size: int | None = Field(DEFAULT_EMBEDDING_BATCH_SIZE, ge=1, le=256)
    embedding_timeout: float | None = Field(DEFAULT_EMBEDDING_TIMEOUT, gt=0)

IndexUpdateRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, IndexUpdateRequest)

