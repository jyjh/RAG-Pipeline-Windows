"""Atomic filesystem helpers.

All metadata JSON files (registry, source-map, index manifest, overrides,
trust registry, asset manifest) are mutated by read-modify-write code paths.
A crash between ``open(..., "w")`` truncating the file and the write finishing
leaves a half-written file that loaders would otherwise silently swallow as
empty -- losing the entire registry/index manifest on the next read.

These helpers write to a sibling temp file, fsync it, and then ``os.replace``
to swap it into place. ``os.replace`` is atomic on the same filesystem, so a
reader always sees either the old or the new file in full, never a partial one.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: str | Path, data: str, *, encoding: str = "utf-8") -> None:
    """Write ``data`` to ``path`` atomically.

    The temp file is created in the same directory as ``path`` so the final
    ``os.replace`` is a same-filesystem rename (atomic on POSIX and Windows).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile keeps a unique name and is cleaned up on failure.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        # Best-effort cleanup of the temp file; never raise over the real error.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def write_bytes_atomic(path: str | Path, data: bytes) -> None:
    """Write raw ``data`` bytes to ``path`` atomically (fsync + os.replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Serialize ``payload`` to JSON and write it to ``path`` atomically."""
    data = json.dumps(payload, indent=indent, sort_keys=True)
    write_text_atomic(path, data)
