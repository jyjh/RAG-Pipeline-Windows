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
    embedding_batch_size: int | None = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_timeout: float | None = DEFAULT_EMBEDDING_TIMEOUT
    index_backend: str = DEFAULT_INDEX_BACKEND
    summary_mode: str = DEFAULT_SUMMARY_MODE
    chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS

ReindexRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, ReindexRequest)

