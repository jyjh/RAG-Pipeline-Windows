"""Disk-space guards for large ingestion/indexing runs.

Ingestion can triple its on-disk footprint: uploads + processed_docs + a
staged LanceDB copy + a backup copytree. Without free-space checks, disk
exhaustion surfaces as a mid-write crash, which is exactly the corruption
vector this overhaul targets. These helpers let callers fail fast with a clear
message before touching disk.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class DiskSpaceError(RuntimeError):
    """Raised when an operation would exceed available free disk space."""

    def __init__(self, path: str | Path, required_bytes: int, free_bytes: int):
        self.path = str(path)
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)
        super().__init__(
            f"Insufficient disk space at {self.path}: "
            f"required {required_bytes} bytes ({_humanize(required_bytes)}), "
            f"only {free_bytes} bytes ({_humanize(free_bytes)}) free."
        )


def _humanize(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def free_disk_bytes(path: str | Path) -> int:
    """Return free bytes available on the filesystem holding ``path``."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(str(resolved)).free


def estimate_dir_bytes(path: str | Path) -> int:
    """Return the walked on-disk size (sum of file sizes) under ``path``.

    Missing paths report 0. Symlinks are not followed.
    """
    base = Path(path)
    if not base.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(base):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def check_disk_space(path: str | Path, required_bytes: int) -> None:
    """Raise :class:`DiskSpaceError` if ``path``'s filesystem lacks ``required_bytes``.

    A negative or zero ``required_bytes`` is a no-op. The check is best-effort:
    if ``disk_usage`` itself fails we let the caller proceed (the subsequent
    write will raise a normal ``OSError``).
    """
    required = int(required_bytes)
    if required <= 0:
        return
    try:
        free = free_disk_bytes(path)
    except OSError:
        return
    if free < required:
        raise DiskSpaceError(path, required, free)
