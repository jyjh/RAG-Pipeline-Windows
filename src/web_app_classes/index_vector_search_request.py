from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class IndexVectorSearchRequest(BaseModel):
    query: str
    relevance_floor: float = DEFAULT_INDEX_VECTOR_RELEVANCE_FLOOR
    embedding_model: str | None = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int | None = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_timeout: float | None = DEFAULT_EMBEDDING_TIMEOUT

IndexVectorSearchRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, IndexVectorSearchRequest)

