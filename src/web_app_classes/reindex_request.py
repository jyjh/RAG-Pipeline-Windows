from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class ReindexRequest(BaseModel):
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int | None = Field(DEFAULT_EMBEDDING_BATCH_SIZE, ge=1, le=256)
    embedding_timeout: float | None = Field(DEFAULT_EMBEDDING_TIMEOUT, gt=0)
    index_backend: str = DEFAULT_INDEX_BACKEND
    summary_mode: str = DEFAULT_SUMMARY_MODE
    chunk_target_tokens: int = Field(DEFAULT_CHUNK_TARGET_TOKENS, ge=100)
    chunk_overlap_tokens: int = Field(DEFAULT_CHUNK_OVERLAP_TOKENS, ge=0)

ReindexRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, ReindexRequest)

