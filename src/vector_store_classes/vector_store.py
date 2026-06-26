from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.vector_store as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class VectorStore(Protocol):
    def exists(self) -> bool: ...

    def count(self) -> int: ...

    def write_records(
        self,
        records: list[dict[str, Any]],
        *,
        embedding_model: str,
        embedding_dim: int,
    ) -> None: ...

    def list_records(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        search: str = "",
    ) -> dict[str, Any]: ...

    def iter_record_batches(
        self,
        *,
        batch_size: int = 250,
        search: str = "",
        workers: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]: ...

    def metadata(self) -> tuple[str, int]: ...

    def all_records(self) -> list[dict[str, Any]]: ...

    def get_record(self, record_id: str) -> dict[str, Any]: ...

    def update_record(
        self,
        *,
        record_id: str,
        content: str,
        vector: list[float],
        embedding_model: str,
        embedding_dim: int,
    ) -> dict[str, Any]: ...

    def delete_records(self, *, record_ids: list[str]) -> dict[str, Any]: ...

    def delete_records_by_source_hash(
        self,
        *,
        source_hashes: list[str],
        legacy_file_paths: list[str] | None = None,
        legacy_doc_ids: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def search(self, vector: list[float], *, top_k: int) -> list[dict[str, Any]]: ...

    def child_chunks(self, parent: dict[str, Any], *, limit: int) -> list[dict[str, Any]]: ...

VectorStore.__module__ = _source_module.__name__
finalize_split_class(_source_module, VectorStore)

