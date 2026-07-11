from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _connect_host(host: str) -> str:
    if host in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((_connect_host(host), port), timeout=0.3):
            return True
    except OSError:
        return False


def _wait_for_shutdown(*, old_pid: int, host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _process_alive(old_pid) and not _port_open(host, port):
            return
        time.sleep(0.25)


def _start_server(*, root: Path, app: str, host: str, port: int, log_path: Path) -> subprocess.Popen:
    # Bound the restart-log size across many restarts: if the existing log has
    # grown past the ceiling, trim it down to its tail before appending. This is
    # a raw stdout redirect (not a logging.Handler), so rotating-file semantics
    # don't apply; a tail-trim on each restart keeps it bounded.
    _trim_log_tail(log_path, max_bytes=RESTART_LOG_MAX_BYTES)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        app,
        "--host",
        host,
        "--port",
        str(port),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    log_handle = log_path.open("ab")
    try:
        return subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=False if os.name == "nt" else True,
            creationflags=creationflags,
        )
    finally:
        log_handle.close()


# Ceiling for the restart log (raw uvicorn stdout redirect). Trimmed on each
# restart so repeated restarts can't grow it without bound.
RESTART_LOG_MAX_BYTES = 5 * 1024 * 1024


def _trim_log_tail(log_path: Path, *, max_bytes: int) -> None:
    """Keep only the trailing ``max_bytes`` of ``log_path`` if it exceeds the cap.

    Preserves recent context (the most useful part of a restart log) while
    preventing unbounded growth across many restarts. No-op if the file is
    missing, empty, or already under the cap.
    """
    try:
        size = log_path.stat().st_size
    except OSError:
        return
    if size <= max_bytes:
        return
    try:
        with log_path.open("rb") as handle:
            handle.seek(-max_bytes, os.SEEK_END)
            _ = handle.read(1)  # discard partial leading line
            tail = handle.read()
        with log_path.open("wb") as handle:
            handle.write(tail)
    except OSError:
        # Best-effort: never let log trimming break a restart.
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restart the Local FSAE RAG web server after an update.")
    parser.add_argument("--root", required=True, help="Repository root directory.")
    parser.add_argument("--old-pid", required=True, type=int, help="PID of the server process being replaced.")
    parser.add_argument("--host", required=True, help="Host to pass to uvicorn.")
    parser.add_argument("--port", required=True, type=int, help="Port to pass to uvicorn.")
    parser.add_argument("--app", default="src.web_app:app", help="ASGI app import path.")
    parser.add_argument("--timeout", default=90.0, type=float, help="Seconds to wait for the old server to stop.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "update-restart.log"

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{_utcnow()} waiting for PID {args.old_pid} and port {args.port}\n")

    _wait_for_shutdown(
        old_pid=args.old_pid,
        host=args.host,
        port=args.port,
        timeout_seconds=max(1.0, args.timeout),
    )

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{_utcnow()} starting uvicorn {args.app} on {args.host}:{args.port}\n")

    process = _start_server(
        root=root,
        app=args.app,
        host=args.host,
        port=args.port,
        log_path=log_path,
    )

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{_utcnow()} spawned PID {process.pid}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
