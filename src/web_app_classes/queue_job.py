from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


@dataclass
class QueueJob:
    id: str
    kind: str
    status: str = "queued"
    phase: str = "queued"
    filenames: list[str] = field(default_factory=list)
    uploads: list[dict[str, Any]] = field(default_factory=list)
    force_duplicate_hashes: list[str] = field(default_factory=list)
    staging_dir: str | None = None
    upload_dir: str | None = None
    resume_status: str | None = None
    recovered: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(default_factory=_utcnow)
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "phase": self.phase,
            "filenames": list(self.filenames),
            "uploads": [dict(item) for item in self.uploads],
            "force_duplicate_hashes": list(self.force_duplicate_hashes),
            "staging_dir": self.staging_dir,
            "upload_dir": self.upload_dir,
            "resume_status": self.resume_status,
            "recovered": self.recovered,
            "options": dict(self.options),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

QueueJob.__module__ = _source_module.__name__
finalize_split_class(_source_module, QueueJob)

