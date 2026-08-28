from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module


def _utcnow() -> str:
    """ISO-8601 UTC timestamp. Local copy so this module's @dataclass can
    resolve its default_factory without depending on borrowed web_app globals
    (which are only injected by bind_module_namespace below and may not be
    present yet under a circular-import timing edge on some platforms)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    source_hashes: list[str] = field(default_factory=list)
    backup_name: str | None = None
    staging_dir: str | None = None
    upload_dir: str | None = None
    resume_status: str | None = None
    recovered: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)
    log_line_count: int = 0
    # Live structured progress parsed from the subprocess's __RAG_PROGRESS__
    # lines (see src/progress_protocol.py). None until the first progress line
    # arrives; replaced on each subsequent line so /api/jobs always returns the
    # freshest done/total/rate snapshot. Cleared on terminal state.
    progress: dict[str, Any] | None = None
    created_at: str = field(default_factory=_utcnow)
    started_at: str | None = None
    finished_at: str | None = None
    _cancel_event: Any = field(default_factory=threading.Event, repr=False, compare=False)

    def to_dict(self, *, include_log_tail: bool = True) -> dict[str, Any]:
        """Serialize for API responses.

        ``include_log_tail=False`` skips joining the (up to 200-line) tail.
        The list endpoint drops the tail anyway, so leaving it out there saves
        rebuilding every job's tail on each 2s active-job poll; the detail
        endpoint keeps it.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "phase": self.phase,
            "filenames": list(self.filenames),
            "uploads": [dict(item) for item in self.uploads],
            "force_duplicate_hashes": list(self.force_duplicate_hashes),
            "source_hashes": list(self.source_hashes),
            "backup_name": self.backup_name,
            "staging_dir": self.staging_dir,
            "upload_dir": self.upload_dir,
            "resume_status": self.resume_status,
            "recovered": self.recovered,
            "options": dict(self.options),
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "log_tail": "\n".join(self.log_tail) if include_log_tail else "",
            "log_line_count": self.log_line_count,
            "progress": dict(self.progress) if self.progress else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

QueueJob.__module__ = _source_module.__name__
finalize_split_class(_source_module, QueueJob)

