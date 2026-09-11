"""Guided, one-command setup for the web server and HPC connections.

Run interactively with ``python scripts/setup_instance.py``.  The command uses
only the Python standard library, so it can configure and diagnose a fresh
checkout before the project's optional runtime dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.toml"
EXAMPLE_CONFIG = ROOT / "config.example.toml"
DEFAULT_SSH_CONFIG = Path.home() / ".ssh" / "config"
_SECTION_RE = re.compile(r"^\s*\[([^\]]+)]\s*(?:#.*)?$")
_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_-]+)\s*=")
_SSH_ALIAS_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SSH_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SSH_USER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_REPO_ARCHIVE_EXCLUDED_NAMES = {
    ".git",
    ".venv",
    ".zcode",
    "__pycache__",
    "data",
    "db",
    "db_out",
    "logs",
    "processed_docs",
    # Local-only config backups (written by update_toml_sections). Never useful
    # remotely, and shipping them would leak the PREVIOUS API key to the cluster.
    "config.toml.bak",
    "config.toml.tmp",
}


@dataclass
class SetupValues:
    mode: str
    server_host: str
    server_port: int
    manage_ssh_aliases: bool = False
    cpu_host: str = ""
    cpu_hostname: str = ""
    cpu_user: str = ""
    cpu_identity_file: str = ""
    cpu_repo: str = ""
    cpu_storage_root: str = ""
    local_ollama_port: int = 11434
    llm_api_key: str = ""


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def update_toml_sections(path: Path, updates: dict[str, dict[str, Any]]) -> None:
    """Update selected TOML keys while preserving every unrelated setting."""
    if path.exists():
        text = path.read_text(encoding="utf-8")
    elif EXAMPLE_CONFIG.exists():
        text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    else:
        text = ""

    lines = text.splitlines()
    section_ranges: dict[str, tuple[int, int]] = {}
    starts: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        match = _SECTION_RE.match(line)
        if match:
            starts.append((match.group(1).strip(), index))
    for offset, (name, start) in enumerate(starts):
        end = starts[offset + 1][1] if offset + 1 < len(starts) else len(lines)
        section_ranges[name] = (start, end)

    for section, values in updates.items():
        if section not in section_ranges:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(f"[{section}]")
            lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items())
            # Rebuild ranges because this new section may be followed by more
            # sections added during the same call.
            section_ranges[section] = (len(lines) - len(values) - 1, len(lines))
            continue

        start, end = section_ranges[section]
        existing: dict[str, int] = {}
        for index in range(start + 1, end):
            match = _KEY_RE.match(lines[index])
            if match:
                existing[match.group(2)] = index
        insert_at = end
        added = 0
        for key, value in values.items():
            rendered = f"{key} = {_toml_value(value)}"
            if key in existing:
                lines[existing[key]] = rendered
            else:
                lines.insert(insert_at + added, rendered)
                added += 1
        if added:
            for name, (other_start, other_end) in list(section_ranges.items()):
                if other_start >= end and name != section:
                    section_ranges[name] = (other_start + added, other_end + added)
            section_ranges[section] = (start, end + added)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _ssh_host_block(lines: list[str], alias: str) -> tuple[int, int] | None:
    """Return the exact ``Host alias`` block, ignoring wildcard/group blocks."""
    for start, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.lower().startswith("host "):
            continue
        patterns = stripped.split()[1:]
        if patterns != [alias]:
            continue
        end = len(lines)
        for index in range(start + 1, len(lines)):
            candidate = lines[index].strip().lower()
            if candidate.startswith("host ") or candidate == "match" or candidate.startswith("match "):
                end = index
                break
        return start, end
    return None


def read_ssh_alias(path: Path, alias: str) -> dict[str, str]:
    """Read fields from an exact SSH Host block, if one exists."""
    if not alias or not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    bounds = _ssh_host_block(lines, alias)
    if bounds is None:
        return {}
    start, end = bounds
    result: dict[str, str] = {}
    for line in lines[start + 1:end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) == 2:
            result[parts[0].lower()] = parts[1].strip().strip('"')
    return result


def _ssh_config_value(value: str) -> str:
    if any(character.isspace() for character in value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def upsert_ssh_alias(
    path: Path,
    *,
    alias: str,
    hostname: str,
    user: str,
    identity_file: str = "",
    create_backup: bool = True,
) -> None:
    """Create or safely update one exact SSH alias, retaining extra options."""
    if not _SSH_ALIAS_RE.fullmatch(alias):
        raise ValueError(f"invalid SSH alias: {alias!r}")
    if not _SSH_HOSTNAME_RE.fullmatch(hostname):
        raise ValueError(f"invalid SSH hostname for {alias!r}")
    if not _SSH_USER_RE.fullmatch(user):
        raise ValueError(f"invalid SSH user for {alias!r}")

    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    desired = {
        "hostname": ("HostName", hostname),
        "user": ("User", user),
        "serveraliveinterval": ("ServerAliveInterval", "15"),
        "serveralivecountmax": ("ServerAliveCountMax", "3"),
    }
    if identity_file:
        desired["identityfile"] = ("IdentityFile", identity_file)

    bounds = _ssh_host_block(lines, alias)
    if bounds is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"Host {alias}")
        lines.extend(
            f"    {key} {_ssh_config_value(value)}"
            for key, value in desired.values()
        )
    else:
        start, end = bounds
        found: set[str] = set()
        rewritten = [lines[start]]
        for line in lines[start + 1:end]:
            stripped = line.strip()
            parts = stripped.split(None, 1)
            normalized = parts[0].lower() if parts else ""
            if normalized in desired:
                if normalized not in found:
                    key, value = desired[normalized]
                    rewritten.append(f"    {key} {_ssh_config_value(value)}")
                    found.add(normalized)
                continue
            rewritten.append(line)
        for normalized, (key, value) in desired.items():
            if normalized not in found:
                rewritten.append(f"    {key} {_ssh_config_value(value)}")
        lines[start:end] = rewritten

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        path.parent.chmod(0o700)
    if path.exists() and create_backup:
        shutil.copy2(path, path.with_name(path.name + ".rag-setup.bak"))
    temporary = path.with_name(path.name + ".rag-setup.tmp")
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    if os.name != "nt":
        path.chmod(0o600)


def configure_ssh_aliases(path: Path, values: SetupValues) -> None:
    if values.mode != "hpc" or not values.manage_ssh_aliases:
        return
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".rag-setup.bak"))
    upsert_ssh_alias(
        path,
        alias=values.cpu_host,
        hostname=values.cpu_hostname,
        user=values.cpu_user,
        identity_file=values.cpu_identity_file,
        create_backup=False,
    )


def _private_key_path(raw_path: str) -> Path:
    return Path(os.path.expandvars(raw_path)).expanduser().resolve()


def ensure_ssh_private_key(raw_path: str, alias: str) -> tuple[Path, Path]:
    """Create a dedicated Ed25519 key, or recover its missing public key."""
    private_key = _private_key_path(raw_path)
    public_key = Path(str(private_key) + ".pub")
    ssh_keygen = shutil.which("ssh-keygen")
    if not ssh_keygen:
        raise RuntimeError(
            "ssh-keygen was not found. Install the Windows OpenSSH Client or "
            "your platform's OpenSSH package."
        )
    private_key.parent.mkdir(parents=True, exist_ok=True)
    if not private_key.exists():
        if public_key.exists():
            raise RuntimeError(
                f"Public key exists but its private key is missing: {public_key}"
            )
        result = subprocess.run(
            [
                ssh_keygen,
                "-t", "ed25519",
                "-a", "64",
                "-f", str(private_key),
                "-N", "",
                "-C", f"rag-pipeline-{alias}",
            ],
            check=False,
        )
        if result.returncode != 0 or not private_key.exists() or not public_key.exists():
            raise RuntimeError(f"failed to generate SSH key {private_key}")
        print(f"Generated SSH key: {private_key}")
    elif not public_key.exists():
        result = subprocess.run(
            [ssh_keygen, "-y", "-f", str(private_key)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                f"could not derive public key from {private_key}: {result.stderr.strip()}"
            )
        temporary = public_key.with_name(public_key.name + ".tmp")
        temporary.write_text(result.stdout.strip() + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, public_key)
        print(f"Recovered public key: {public_key}")
    if os.name != "nt":
        private_key.chmod(0o600)
        public_key.chmod(0o644)
    return private_key, public_key


def install_ssh_public_key(
    *,
    hostname: str,
    user: str,
    private_key: Path,
    public_key: Path,
) -> None:
    """Install one public key remotely, prompting for password/MFA if needed."""
    ssh = shutil.which("ssh")
    if not ssh:
        raise RuntimeError(
            "ssh was not found. Install the Windows OpenSSH Client or your "
            "platform's OpenSSH package."
        )
    key_text = public_key.read_text(encoding="utf-8").strip()
    if not key_text.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")):
        raise RuntimeError(f"unsupported or malformed SSH public key: {public_key}")
    quoted_key = shlex.quote(key_text)
    remote_command = (
        'umask 077; mkdir -p "$HOME/.ssh"; '
        'touch "$HOME/.ssh/authorized_keys"; '
        f'grep -qxF {quoted_key} "$HOME/.ssh/authorized_keys" '
        f"|| printf '%s\\n' {quoted_key} >> \"$HOME/.ssh/authorized_keys\"; "
        'chmod 700 "$HOME/.ssh"; chmod 600 "$HOME/.ssh/authorized_keys"'
    )
    target = f"{user}@{hostname}"
    print(f"Installing public key on {target} (password/MFA may be requested)...")
    result = subprocess.run(
        [
            ssh,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "IdentitiesOnly=no",
            "-i", str(private_key),
            target,
            remote_command,
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not install the SSH public key on {target}. Confirm password/"
            "MFA login is allowed, or ask the cluster administrator to install "
            f"{public_key}."
        )

    verify = subprocess.run(
        [
            ssh,
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-i", str(private_key),
            target,
            "printf RAG_SSH_KEY_OK",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.returncode != 0 or "RAG_SSH_KEY_OK" not in verify.stdout:
        raise RuntimeError(f"SSH key verification failed for {target}: {verify.stderr.strip()}")


def setup_ssh_credentials(values: SetupValues, *, install_public_keys: bool) -> None:
    """Generate and optionally authorize the dedicated key for the CPU cluster."""
    if values.mode != "hpc" or not values.manage_ssh_aliases:
        return
    alias, hostname, user, raw_key = (
        values.cpu_host,
        values.cpu_hostname,
        values.cpu_user,
        values.cpu_identity_file,
    )
    private_key, public_key = ensure_ssh_private_key(raw_key, alias)
    if install_public_keys:
        install_ssh_public_key(
            hostname=hostname,
            user=user,
            private_key=private_key,
            public_key=public_key,
        )


def _repo_archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = PurePosixPath(info.name).parts
    if any(part in _REPO_ARCHIVE_EXCLUDED_NAMES for part in parts):
        return None
    if any(
        part.startswith((".index_build_", ".tmp_test_", ".pytest_"))
        for part in parts
    ):
        return None
    if info.name.lower().endswith(".sif"):
        return None
    return info


# Name of the content-fingerprint file embedded in every provisioned archive and
# deposited at the remote repo root after activation. Comparing this file lets a
# subsequent run skip a redundant re-upload/re-extract when the source is
# unchanged. Format: one "<relative_path>\0<sha256_hex>" line per regular file.
_REPO_MANIFEST_NAME = ".rag_manifest"


def _build_repo_manifest(
    source_root: Path = ROOT,
    *,
    exclude: frozenset[str] | set[str] = frozenset(),
) -> str:
    """Return a deterministic content fingerprint of the archived source tree.

    Walks ``source_root`` with the SAME exclusions as the archive filter so the
    manifest exactly describes the bytes that get shipped. Each regular file is
    one line "<posix_rel_path>\\0<sha256>", sorted by path -- stable across runs
    and independent of filesystem walk order or mtimes. The whole buffer is then
    SHA-256'd into a header line so a single comparison suffices to detect any
    change, but the per-file lines are retained for diagnosing what differed.

    ``exclude`` drops additional root-relative files from the fingerprint. The
    freshness check uses it for ``config.toml`` (see ``_manifest_fingerprint``
    callers): config edits (API key, ports) must not trigger a full re-upload of
    the source + multi-GB SIF when nothing else changed.
    """
    entries: list[tuple[str, str]] = []
    excluded = _REPO_ARCHIVE_EXCLUDED_NAMES

    def _excluded(rel: PurePosixPath) -> bool:
        parts = rel.parts
        if any(p in excluded for p in parts):
            return True
        if any(p.startswith((".index_build_", ".tmp_test_", ".pytest_")) for p in parts):
            return True
        if len(parts) == 1 and parts[0] in exclude:
            return True
        return False

    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = PurePosixPath(path.relative_to(source_root))
        if str(rel) == ".":
            continue
        if _excluded(rel):
            continue
        if path.name.lower().endswith(".sif"):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append((str(rel).replace(os.sep, "/"), digest))

    body = "".join(f"{name}\0{digest}\n" for name, digest in entries)
    overall = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"# sha256={overall}\n{body}"


def _manifest_fingerprint(manifest: str) -> str:
    """Extract the aggregate '# sha256=...' header from a manifest buffer."""
    for line in manifest.splitlines():
        if line.startswith("# sha256="):
            return line.split("=", 1)[1].strip()
    return ""


# Files whose changes must NOT mark the deployed source stale: config.toml is
# rewritten by every setup run (API key, ports), but the deploy only needs the
# new file pushed -- not a full re-upload of the source archive and the
# multi-GB SIF over a non-resumable Windows scp.
_CONFIG_SYNC_EXCLUDE = frozenset({"config.toml"})


def _manifest_without(manifest: str, names: frozenset[str] | set[str]) -> str:
    """Drop root-level file lines from a manifest buffer and re-header it."""
    kept = [
        line
        for line in manifest.splitlines()
        if not line.startswith("# sha256=") and line.split("\0", 1)[0] not in names
    ]
    body = "".join(f"{line}\n" for line in kept)
    overall = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"# sha256={overall}\n{body}"


def _file_sha256(path: Path, *, chunk: int = 1024 * 1024) -> str:
    """Streaming SHA-256 of a (possibly multi-GB) file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _remote_sha256(alias: str, remote_path: str) -> str:
    """First sha256sum token of ``remote_path`` on the cluster, or ""."""
    ssh = shutil.which("ssh")
    if not ssh:
        return ""
    try:
        result = subprocess.run(
            [
                ssh,
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=20",
                alias,
                f"sha256sum {shlex.quote(remote_path)} 2>/dev/null | cut -d' ' -f1",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip().splitlines()[0].strip() if result.returncode == 0 and result.stdout.strip() else ""


def _sync_config_only(alias: str, target_repo: str, manifest: str, staging_dir: Path) -> None:
    """Push just config.toml + a fresh manifest; skip archive/SIF entirely."""
    local_config = ROOT / "config.toml"
    if not local_config.is_file():
        return
    _upload_provision_file(local_config, alias, posixpath.join(target_repo, "config.toml"))
    manifest_file = staging_dir / _REPO_MANIFEST_NAME
    manifest_file.write_text(manifest, encoding="utf-8", newline="\n")
    _upload_provision_file(manifest_file, alias, posixpath.join(target_repo, _REPO_MANIFEST_NAME))


def create_repository_archive(destination: Path, source_root: Path = ROOT) -> Path:
    """Package the exact working-tree source without local data/index artifacts.

    Embeds a ``.rag_manifest`` content fingerprint at the archive root so the
    remote provisioning step can skip a redundant re-upload when the source is
    unchanged on a subsequent run.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _build_repo_manifest(source_root).encode("utf-8")
    with tarfile.open(destination, "w:gz", compresslevel=6) as archive:
        info = tarfile.TarInfo(name=_REPO_MANIFEST_NAME)
        info.size = len(manifest_bytes)
        info.mtime = time.time()
        archive.addfile(info, io.BytesIO(manifest_bytes))
        archive.add(source_root, arcname=".", filter=_repo_archive_filter)
    return destination


def _run_provision_ssh(
    alias: str,
    remote_command: str,
    *,
    timeout: float | None = 120.0,
) -> None:
    ssh = shutil.which("ssh")
    if not ssh:
        raise RuntimeError("ssh was not found; install the OpenSSH Client")
    result = subprocess.run(
        [
            ssh,
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
            alias,
            remote_command,
        ],
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"remote provisioning command failed on {alias} "
            f"(exit {result.returncode})"
        )


def _reuse_remote_sif_if_unchanged(
    alias: str, existing_sif: str, staged_sif: str, local_image: Path
) -> bool:
    """Copy the deployed SIF into the staged repo when it matches the local one.

    A re-provision triggered by a source change almost never changes the
    multi-GB SIF. Comparing checksums (tens of seconds over ssh) and copying on
    the cluster filesystem beats re-uploading gigabytes over scp, which on
    Windows has no resume. Returns True when the staged SIF is ready.
    """
    ssh = shutil.which("ssh")
    if not ssh:
        return False
    try:
        present = subprocess.run(
            [
                ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                alias, f"test -s {shlex.quote(existing_sif)}",
            ],
            check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if present.returncode != 0:
        return False
    local_digest = _file_sha256(local_image)
    remote_digest = _remote_sha256(alias, existing_sif)
    if not remote_digest or remote_digest != local_digest:
        return False
    print(
        f"{local_image.name} on {alias} matches the local image (sha256); "
        "copying it into the staged repo instead of re-uploading."
    )
    _run_provision_ssh(
        alias,
        f"cp {shlex.quote(existing_sif)} {shlex.quote(staged_sif)}",
        timeout=30 * 60,
    )
    return True


def _upload_provision_file(source: Path, alias: str, remote_path: str) -> None:
    """Upload one file with rsync when available, otherwise OpenSSH scp."""
    if shutil.which("rsync"):
        command = [
            "rsync",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=20 "
            "-o ServerAliveInterval=30 -o ServerAliveCountMax=6",
            "--partial",
            "--progress",
            str(source),
            f"{alias}:{remote_path}",
        ]
    else:
        scp = shutil.which("scp")
        if not scp:
            raise RuntimeError("neither rsync nor scp was found")
        command = [
            scp,
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
            str(source),
            f"{alias}:{remote_path}",
        ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"failed to upload {source.name} to {alias}:{remote_path}")


def _remote_container_build_command(
    *,
    repo_dir: str,
    definition_name: str,
    image_name: str,
) -> str:
    quoted_repo = shlex.quote(repo_dir)
    quoted_definition = shlex.quote(definition_name)
    quoted_image = shlex.quote(image_name)
    return (
        f"cd {quoted_repo} && "
        "runtime=''; "
        "if command -v apptainer >/dev/null 2>&1; then runtime=apptainer; "
        "elif command -v singularity >/dev/null 2>&1; then runtime=singularity; "
        "elif command -v module >/dev/null 2>&1; then "
        "module load singularity >/dev/null 2>&1 || true; "
        "command -v singularity >/dev/null 2>&1 && runtime=singularity; fi; "
        "test -n \"$runtime\" || { echo 'Singularity/Apptainer is unavailable' >&2; exit 1; }; "
        # Some HPC sites force a bind path (e.g. /app1) into EVERY Apptainer
        # invocation via apptainer.conf or APPTAINER_BIND, including `build`.
        # During build the %post runs under --writable, where a bind to a
        # destination absent from the base image ("destination /app1 doesn't
        # exist in container") is fatal and can't be auto-created. Our build
        # only does apt/pip/ollama installs against the internet, so it needs
        # NO site bind paths -- suppress them for this one command only.
        "APPTAINER_NO_MOUNT=bind SINGULARITY_NO_MOUNT=bind "
        f"\"$runtime\" build --fakeroot {quoted_image} {quoted_definition}"
    )


def _remote_manifest(alias: str, target_repo: str) -> str:
    """Fetch the deployed ``.rag_manifest`` for ``target_repo`` on ``alias``.

    Returns "" when the remote repo or its manifest does not exist (i.e. first
    deploy or a deploy from an older build that predates manifests). Errors are
    swallowed into "" so the caller falls back to a full provision.
    """
    remote_manifest = posixpath.join(target_repo, _REPO_MANIFEST_NAME)
    ssh = shutil.which("ssh")
    if not ssh:
        return ""
    try:
        result = subprocess.run(
            [
                ssh,
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=20",
                alias,
                f"cat {shlex.quote(remote_manifest)} 2>/dev/null || true",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def provision_hpc_cluster(
    *,
    alias: str,
    user: str,
    storage_root: str,
    relative_repo: str,
    archive_path: Path,
    image_name: str,
    definition_name: str,
    local_image: Path | None,
    force: bool = False,
) -> None:
    """Stage and activate one complete repository + SIF under its cluster root.

    When ``force`` is false (the default), compares the local source manifest
    against the manifest at the already-deployed ``target_repo``. If they match
    AND the SIF is already present there, the upload/extract/swap cycle is
    skipped -- the remote is current. Set ``force=True`` to redeploy
    unconditionally.
    """
    if not _SSH_ALIAS_RE.fullmatch(alias):
        raise RuntimeError(f"invalid SSH alias for provisioning: {alias!r}")
    if not _SSH_USER_RE.fullmatch(user):
        raise RuntimeError(f"invalid SSH username for provisioning: {user!r}")
    _validate_relative_repo(relative_repo, alias)
    scratch_root = PurePosixPath(storage_root)
    if not scratch_root.is_absolute() or ".." in scratch_root.parts:
        raise RuntimeError(f"invalid storage root for {alias}: {storage_root!r}")
    scratch_root = str(scratch_root)
    storage_parent = str(PurePosixPath(scratch_root).parent)
    target_repo = posixpath.join(scratch_root, relative_repo)
    target_sif = posixpath.join(target_repo, image_name)
    login_link = f"$HOME/{relative_repo}"

    # --- Freshness check -------------------------------------------------
    # Compare the local source fingerprint against the deployed remote one.
    # The comparison ignores config.toml (rewritten by every setup run), so a
    # config-only change degrades to a one-file sync instead of a full
    # re-upload of the source archive and the multi-GB SIF.
    local_manifest = _build_repo_manifest(ROOT)
    local_core_fingerprint = _manifest_fingerprint(
        _build_repo_manifest(ROOT, exclude=_CONFIG_SYNC_EXCLUDE)
    )
    local_fingerprint = _manifest_fingerprint(local_manifest)
    if not force and local_core_fingerprint:
        remote_manifest = _remote_manifest(alias, target_repo)
        remote_core_fingerprint = _manifest_fingerprint(
            _manifest_without(remote_manifest, _CONFIG_SYNC_EXCLUDE)
        )
        if remote_core_fingerprint and remote_core_fingerprint == local_core_fingerprint:
            # Source is current; only confirm the SIF survived.
            ssh = shutil.which("ssh")
            if ssh:
                check = subprocess.run(
                    [
                        ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        alias, f"test -s {shlex.quote(target_sif)}",
                    ],
                    check=False, timeout=30,
                )
                if check.returncode == 0:
                    if _manifest_fingerprint(remote_manifest) == local_fingerprint:
                        print(f"\n{alias}: already up to date ({target_repo}); skipping upload.")
                        return
                    _sync_config_only(
                        alias, target_repo, local_manifest, archive_path.parent
                    )
                    print(
                        f"\n{alias}: source unchanged; synced config.toml only "
                        f"({target_repo}); skipped archive/SIF upload."
                    )
                    return
            # SIF missing -> fall through to re-provision (rebuilds SIF only).

    token = secrets.token_hex(6)
    stage_root = posixpath.join(scratch_root, f".rag_setup_{token}")
    staged_repo = posixpath.join(stage_root, "repo")
    remote_archive = posixpath.join(stage_root, "repository.tar.gz")

    print(f"\nProvisioning {alias} -> {target_repo}")
    prepare = (
        f"test -d {shlex.quote(storage_parent)} && mkdir -p {shlex.quote(scratch_root)} "
        f"{shlex.quote(staged_repo)}"
    )
    _run_provision_ssh(alias, prepare)
    try:
        print(f"Uploading repository source to {alias}...")
        _upload_provision_file(archive_path, alias, remote_archive)
        extract = (
            f"tar -xzf {shlex.quote(remote_archive)} "
            f"-C {shlex.quote(staged_repo)} && rm -f {shlex.quote(remote_archive)}"
        )
        _run_provision_ssh(alias, extract, timeout=600)

        if local_image is not None and local_image.is_file():
            staged_sif = posixpath.join(staged_repo, image_name)
            reused = _reuse_remote_sif_if_unchanged(
                alias, target_sif, staged_sif, local_image
            )
            if not reused:
                print(f"Uploading {image_name} to {alias} (large transfer)...")
                _upload_provision_file(local_image, alias, staged_sif)
        else:
            print(f"Building {image_name} on {alias}...")
            _run_provision_ssh(
                alias,
                _remote_container_build_command(
                    repo_dir=staged_repo,
                    definition_name=definition_name,
                    image_name=image_name,
                ),
                timeout=4 * 60 * 60,
            )

        verify = (
            f"test -s {shlex.quote(posixpath.join(staged_repo, image_name))} "
            f"&& test -f {shlex.quote(posixpath.join(staged_repo, 'main.py'))}"
        )
        _run_provision_ssh(alias, verify)

        target_parent = posixpath.dirname(target_repo)
        link_parent = posixpath.dirname(login_link)
        previous_repo = target_repo + ".rag-setup-previous"
        activate = (
            f"mkdir -p {shlex.quote(target_parent)} \"{link_parent}\"; "
            f"if [ -e \"{login_link}\" ] && [ ! -L \"{login_link}\" ]; then "
            f"echo 'Refusing to replace non-symlink {login_link}' >&2; exit 1; fi; "
            f"rm -rf {shlex.quote(previous_repo)}; "
            f"if [ -e {shlex.quote(target_repo)} ]; then "
            f"mv {shlex.quote(target_repo)} {shlex.quote(previous_repo)}; fi; "
            f"mv {shlex.quote(staged_repo)} {shlex.quote(target_repo)}; "
            f"ln -sfn {shlex.quote(target_repo)} \"{login_link}\"; "
            f"test -s {shlex.quote(posixpath.join(target_repo, image_name))}"
        )
        _run_provision_ssh(alias, activate, timeout=600)
        print(f"Activated {alias}:{relative_repo} ({target_repo})")
    finally:
        try:
            _run_provision_ssh(
                alias,
                f"rm -rf {shlex.quote(stage_root)}",
                timeout=120,
            )
        except RuntimeError:
            pass


def provision_hpc_servers(values: SetupValues, *, force: bool = False) -> None:
    """Package once, then provision the CPU cluster.

    With ``force=False`` (the default) the cluster is only re-uploaded/rebuilt
    when its deployed manifest is missing or differs from the local source.
    """
    if values.mode != "hpc":
        return
    with tempfile.TemporaryDirectory(prefix="rag-hpc-setup-") as temp_dir:
        archive_path = create_repository_archive(Path(temp_dir) / "repository.tar.gz")
        provision_hpc_cluster(
            alias=values.cpu_host,
            user=values.cpu_user,
            storage_root=_storage_root(
                values.cpu_storage_root,
                values.cpu_user,
                "/hpctmp",
            ),
            relative_repo=values.cpu_repo,
            archive_path=archive_path,
            image_name="rag_pipeline_cpu.sif",
            definition_name="Singularity.cpu.def",
            local_image=ROOT / "rag_pipeline_cpu.sif",
            force=force,
        )


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    return value if isinstance(value, dict) else {}


def _nested(mapping: dict[str, Any], *keys: str, default: Any = "") -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _configured_ollama_port(config: dict[str, Any]) -> int:
    raw = str(_nested(config, "ollama", "host", default="http://127.0.0.1:11434"))
    try:
        return int(urllib.parse.urlparse(raw).port or 11434)
    except (TypeError, ValueError):
        return 11434


def _embeddings_backend_view(config: dict[str, Any], chat_backend: str) -> str:
    """Effective embeddings backend after the same env override the app applies.

    Mirrors ``src.embeddings.resolve_embeddings_backend``: ``EMBEDDINGS_BACKEND``
    wins, then ``[embeddings].backend`` (absent = the "ollama" default --
    locally hosted all-minilm), and ``""`` inherits the chat backend.
    An invalid configured value falls back to the default, mirroring the
    app's warn-and-ignore behaviour.
    """
    env = os.environ.get("EMBEDDINGS_BACKEND", "").strip().lower()
    if env:
        return env
    value = _nested(config, "embeddings", "backend", default=None)
    resolved = "ollama" if value is None else str(value).strip().lower()
    if resolved not in ("", "soclaas", "ollama"):
        resolved = "ollama"
    return resolved or chat_backend


_DEFAULT_LLM_BASE_URL = "https://soclaas-api.comp.nus.edu.sg"


def _llm_api_view(config: dict[str, Any]) -> dict[str, str]:
    """Effective [llm_api] settings after the same env overrides the app applies.

    Mirrors ``src/llm_api.resolve_api_key`` precedence: the env var named in
    ``key_env`` (default ``SOCLAAS_API_KEY``) wins, then ``LLM_API_KEY``, then
    the value stored in ``config.toml``. ``LLM_BACKEND`` overrides the backend.
    """
    backend = str(_nested(config, "llm_api", "backend", default="soclaas"))
    env_backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    if env_backend:
        backend = env_backend
    key_env = str(_nested(config, "llm_api", "key_env", default="SOCLAAS_API_KEY"))
    api_key = str(_nested(config, "llm_api", "api_key", default=""))
    key_source = "config.toml" if api_key else ""
    for name in dict.fromkeys((key_env, "LLM_API_KEY")):
        value = os.environ.get(name, "").strip()
        if value:
            api_key = value
            key_source = f"environment ({name})"
            break
    return {
        "backend": backend,
        "base_url": str(_nested(config, "llm_api", "base_url", default=_DEFAULT_LLM_BASE_URL)),
        "models_path": str(_nested(config, "llm_api", "models_path", default="/v1/models")),
        "api_key": api_key,
        "key_source": key_source,
    }


def _verify_llm_api_key(base_url: str, models_path: str, api_key: str) -> tuple[bool, str]:
    """Best-effort live check of the key against the backend's models endpoint."""
    url = base_url.rstrip("/") + models_path
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = getattr(response, "status", 200)
            response.read()
    except urllib.error.HTTPError as exc:
        detail = f"HTTP {exc.code} from {url}"
        if exc.code in (401, 403):
            detail += " -- key rejected; double-check the key value"
        return False, detail
    except (OSError, urllib.error.URLError) as exc:
        return False, f"could not reach {url}: {exc}"
    if 200 <= status < 300:
        return True, f"accepted by {url}"
    return False, f"HTTP {status} from {url}"


def _login_relative_repo_default(raw: str, user: str) -> str:
    """Convert a legacy absolute home path into a login-relative default."""
    value = str(raw or "").strip().replace("\\", "/")
    if value and not value.startswith("/"):
        return value
    for prefix in (f"/home/{user}/", f"/users/{user}/"):
        if user and value.startswith(prefix):
            return value[len(prefix):]
    if value:
        return PurePosixPath(value).name
    return "RAG-Pipeline-Windows"


def _storage_root(raw: str, user: str, default_base: str) -> str:
    value = str(raw or "").strip()
    if not user:
        return value or f"{default_base}/${{USER}}"
    if not value:
        return f"{default_base}/{user}"
    return (
        value.replace("${USER}", user)
        .replace("$USER", user)
        .replace("{username}", user)
    )


def _validate_relative_repo(value: str, label: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or value.startswith("~")
        or ".." in path.parts
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            f"{label} remote repo must be a safe path relative to the SSH login "
            "directory (for example, RAG-Pipeline-Windows)"
        )


def _prompt(label: str, default: str = "", *, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip() or default
        if value or not required:
            return value
        print("  A value is required.")


def _prompt_yes_no(label: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    value = input(f"{label} [{hint}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def _prompt_int(label: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = _prompt(label, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("  Enter a whole number.")
            continue
        if lo <= value <= hi:
            return value
        print(f"  Enter a value between {lo} and {hi}.")


def _validate(values: SetupValues) -> None:
    if values.mode not in {"local", "hpc"}:
        raise ValueError("mode must be 'local' or 'hpc'")
    if not (1 <= values.server_port <= 65535):
        raise ValueError("server port must be between 1 and 65535")
    if not (1 <= values.local_ollama_port <= 65535):
        raise ValueError("local Ollama port must be between 1 and 65535")
    if values.mode == "hpc":
        if not values.cpu_host:
            raise ValueError("CPU SSH alias/host is required")
        if not _SSH_ALIAS_RE.fullmatch(values.cpu_host):
            raise ValueError("CPU SSH alias contains unsupported characters")
        if values.manage_ssh_aliases and not values.cpu_hostname:
            raise ValueError("CPU SSH hostname is required")
        if values.manage_ssh_aliases and not values.cpu_user:
            raise ValueError("CPU SSH username is required")
        if values.manage_ssh_aliases and not values.cpu_identity_file:
            raise ValueError("CPU SSH private-key path is required")
        _validate_relative_repo(values.cpu_repo, "CPU")
        if values.manage_ssh_aliases and values.cpu_storage_root != f"/hpctmp/{values.cpu_user}":
            raise ValueError("Atlas9 CPU storage root must be /hpctmp/<username>")


def _collect_interactive(config: dict[str, Any], args: argparse.Namespace) -> SetupValues:
    existing_hpc = bool(_nested(config, "hpc", "enabled", default=False))
    default_mode = args.mode or ("hpc" if existing_hpc else "local")
    print("\nRAG Pipeline guided setup")
    print("-------------------------")
    mode = _prompt("Mode (local/hpc)", default_mode, required=True).lower()
    server_host = _prompt(
        "Web server bind host",
        args.server_host or str(_nested(config, "server", "host", default="127.0.0.1")),
        required=True,
    )
    server_port = _prompt_int(
        "Web server port",
        int(args.server_port or _nested(config, "server", "port", default=8000)),
        1,
        65535,
    )
    values = SetupValues(mode=mode, server_host=server_host, server_port=server_port)
    values.llm_api_key = args.set_api_key or ""
    if not values.llm_api_key:
        llm = _llm_api_view(config)
        embeddings_backend = _embeddings_backend_view(config, llm["backend"])
        if llm["backend"] == "soclaas" and not llm["api_key"]:
            needs = (
                "chat, vision, and embeddings"
                if embeddings_backend == "soclaas"
                else "chat and vision (embeddings run on local Ollama)"
            )
            print(
                f"\nThe default LLM backend is the hosted SoCLAaS API; {needs}\n"
                "need an API key."
            )
            values.llm_api_key = _prompt(
                "SoCLAaS API key (blank to skip; set later with --set-api-key)",
                "",
            )
    values.local_ollama_port = args.ollama_port or _configured_ollama_port(config)
    if mode == "hpc":
        values.manage_ssh_aliases = True
        values.cpu_host = _prompt(
            "CPU cluster SSH alias",
            args.cpu_host or str(_nested(config, "hpc", "cpu", "ssh_host")),
            required=True,
        )
        existing_cpu_alias = read_ssh_alias(args.ssh_config, values.cpu_host)
        values.cpu_hostname = _prompt(
            "CPU cluster login hostname",
            args.cpu_hostname or existing_cpu_alias.get("hostname", ""),
            required=True,
        )
        values.cpu_user = _prompt(
            "CPU cluster SSH username",
            args.cpu_user or existing_cpu_alias.get("user", ""),
            required=True,
        )
        values.cpu_storage_root = f"/hpctmp/{values.cpu_user}"
        values.cpu_identity_file = _prompt(
            "CPU SSH private key (created automatically if missing)",
            args.cpu_key
            or existing_cpu_alias.get("identityfile", "")
            or f"~/.ssh/rag_{values.cpu_host}_ed25519",
            required=True,
        )
        values.cpu_repo = _prompt(
            "CPU repo path relative to SSH login directory",
            args.cpu_repo
            or _login_relative_repo_default(
                str(_nested(config, "hpc", "cpu", "remote_repo_dir")),
                values.cpu_user,
            ),
            required=True,
        )
    return values


def _collect_non_interactive(config: dict[str, Any], args: argparse.Namespace) -> SetupValues:
    mode = args.mode or ("hpc" if _nested(config, "hpc", "enabled", default=False) else "local")
    manage_ssh = bool(args.setup_ssh or args.cpu_hostname)
    cpu_alias = args.cpu_host or str(_nested(config, "hpc", "cpu", "ssh_host"))
    cpu_user = args.cpu_user or ""
    return SetupValues(
        mode=mode,
        server_host=args.server_host or str(_nested(config, "server", "host", default="127.0.0.1")),
        server_port=args.server_port or int(_nested(config, "server", "port", default=8000)),
        manage_ssh_aliases=manage_ssh,
        cpu_host=cpu_alias,
        cpu_hostname=args.cpu_hostname or "",
        cpu_user=cpu_user,
        cpu_identity_file=(
            args.cpu_key or (f"~/.ssh/rag_{cpu_alias}_ed25519" if manage_ssh and cpu_alias else "")
        ),
        cpu_repo=args.cpu_repo or str(_nested(config, "hpc", "cpu", "remote_repo_dir")),
        cpu_storage_root=_storage_root(
            str(_nested(config, "hpc", "cpu", "storage_root")),
            cpu_user,
            "/hpctmp",
        ),
        local_ollama_port=args.ollama_port or _configured_ollama_port(config),
        llm_api_key=args.set_api_key or "",
    )


def _cluster_corpus_dirs(values: SetupValues) -> dict[str, str] | None:
    """Absolute cluster corpus dirs under ``/hpctmp/<user>/rag-corpus``.

    Keeping data/processed/db OUTSIDE ``remote_repo_dir`` is a correctness
    requirement, not a preference: provisioning atomically replaces the repo
    directory (old -> ``.rag-setup-previous`` -> deleted on the next run), so
    anything stored inside it -- an uploaded corpus, a finished parse -- is
    destroyed by a later re-provision. Returns None when the username is
    unknown (legacy repo-relative dirs are kept then; provisioning will fail
    loudly asking for the username anyway).
    """
    if values.mode != "hpc" or not values.cpu_user:
        return None
    corpus_root = _storage_root(values.cpu_storage_root, values.cpu_user, "/hpctmp")
    corpus_root = f"{corpus_root.rstrip('/')}/rag-corpus"
    return {
        "remote_data_dir": f"{corpus_root}/data",
        "remote_processed_dir": f"{corpus_root}/processed_docs",
        "remote_db_dir": f"{corpus_root}/db",
    }


def configure(path: Path, values: SetupValues) -> None:
    hpc_updates: dict[str, Any] = {
        "enabled": values.mode == "hpc",
        "poll_interval_seconds": 15.0,
    }
    corpus_dirs = _cluster_corpus_dirs(values)
    if corpus_dirs is not None:
        hpc_updates.update(corpus_dirs)
    else:
        hpc_updates.update({
            "remote_data_dir": "data",
            "remote_processed_dir": "processed_docs",
            "remote_db_dir": "db",
        })
    updates: dict[str, dict[str, Any]] = {
        "server": {
            "host": values.server_host,
            "port": values.server_port,
            "bind_all": values.server_host in {"0.0.0.0", "::"},
        },
        "ollama": {"host": f"http://127.0.0.1:{values.local_ollama_port}"},
        "hpc": hpc_updates,
    }
    if values.mode == "hpc":
        updates.update({
            "hpc.cpu": {
                "ssh_host": values.cpu_host,
                "remote_repo_dir": values.cpu_repo,
                "container_sif": "rag_pipeline_cpu.sif",
                "storage_root": values.cpu_storage_root or "/hpctmp/${USER}",
            },
        })
    if values.llm_api_key:
        updates["llm_api"] = {"api_key": values.llm_api_key}
    update_toml_sections(path, updates)


def _command_check(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    return (bool(path), path or "not found")


def _runtime_python() -> Path:
    candidate = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return candidate if candidate.exists() else Path(sys.executable)


def _runtime_dependencies_ready(python: Path | None = None) -> bool:
    python = python or _runtime_python()
    command = [
        str(python),
        "-c",
        "import fastapi, uvicorn, lancedb, pydantic",
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False).returncode == 0


def install_dependencies() -> None:
    """Create a project-local venv and install the pinned requirements."""
    venv_dir = ROOT / ".venv"
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print("Creating .venv...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    print("Installing runtime dependencies (this can take several minutes)...")
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        cwd=ROOT,
        check=True,
    )


def _http_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def _ollama_model_installed(ollama_port: int, model: str) -> bool:
    """Best-effort check that ``model`` is pulled in the local Ollama."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{ollama_port}/api/tags", timeout=3.0
        ) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (OSError, ValueError, urllib.error.URLError):
        return True  # endpoint unreachable/unparseable: don't pile on the port check
    names = {
        str(item.get("name", "")).split(":")[0]
        for item in (payload.get("models") or [])
        if isinstance(item, dict)
    }
    return not names or model in names


def run_checks(
    path: Path,
    values: SetupValues,
    *,
    cluster_required: bool = False,
) -> bool:
    """Preflight checks. WARN never blocks; FAIL does.

    ``cluster_required`` marks a run that will talk to the HPC cluster
    (provisioning or an --initial-corpus parse). Cluster connectivity is only
    a hard dependency then -- daily serving needs just Ollama + the LLM API,
    so an unreachable campus cluster must not block the web server from
    starting.
    """
    print("\nPreflight checks")
    print("----------------")
    # Each check is (label, status, detail) with status OK / WARN / FAIL.
    # WARN does not fail the run; FAIL does.
    checks: list[tuple[str, str, str]] = []
    checks.append((
        "Python 3.11+",
        "OK" if sys.version_info >= (3, 11) else "FAIL",
        sys.version.split()[0],
    ))
    runtime = _runtime_python()
    checks.append((
        "Python dependencies",
        "OK" if _runtime_dependencies_ready(runtime) else "FAIL",
        str(runtime),
    ))
    try:
        with socket.socket() as probe:
            probe.bind((values.server_host, values.server_port))
        checks.append(("Web port", "OK", f"{values.server_host}:{values.server_port} available"))
    except OSError as exc:
        already_running = _http_ready(f"http://127.0.0.1:{values.server_port}/api/health")
        checks.append((
            "Web port",
            "OK" if already_running else "FAIL",
            "RAG web server already running" if already_running else str(exc),
        ))

    config_payload = _load(path)
    llm = _llm_api_view(config_payload)
    embeddings_backend = _embeddings_backend_view(config_payload, llm["backend"])
    if llm["backend"] == "soclaas":
        if llm["api_key"]:
            checks.append(("SoCLAaS API key", "OK", f"set via {llm['key_source']}"))
        else:
            affected = (
                "chat/vision/embeddings"
                if embeddings_backend == "soclaas"
                else "chat/vision"
            )
            checks.append((
                "SoCLAaS API key",
                "WARN",
                f"not set -- {affected} will fail; rerun setup with "
                "--set-api-key <key> or export SOCLAAS_API_KEY",
            ))

    # Local Ollama, in EVERY mode: embeddings default to a locally hosted
    # all-minilm, so index builds and query-time embedding fail without it
    # regardless of where parsing happens. (Historically this checked only
    # local mode; with HPC parse-only deployments the dependency inverted.)
    ready = _http_ready(f"http://127.0.0.1:{values.local_ollama_port}/api/version")
    ollama_required = llm["backend"] == "ollama" or embeddings_backend == "ollama"
    if ready:
        detail = "ready"
        embedding_model = str(_nested(
            config_payload, "models", "embedding_model", default="all-minilm"
        ))
        if embeddings_backend == "ollama" and not _ollama_model_installed(
            values.local_ollama_port, embedding_model
        ):
            checks.append((
                "Local Ollama",
                "WARN",
                f"reachable but '{embedding_model}' is not pulled; run: "
                f"ollama pull {embedding_model}",
            ))
        else:
            checks.append(("Local Ollama", "OK", detail))
    else:
        if ollama_required:
            checks.append((
                "Local Ollama",
                "FAIL",
                'not reachable -- required for [embeddings].backend = "ollama" '
                "(start Ollama and pull the embedding model)",
            ))
        else:
            checks.append((
                "Local Ollama",
                "WARN",
                "not reachable -- optional unless an ollama backend is active",
            ))

    if values.mode != "local":
        # Cluster tooling/connectivity is only a hard requirement on runs that
        # will actually use the cluster; serving runs degrade to WARN.
        severity = "FAIL" if cluster_required else "WARN"
        ok, detail = _command_check("ssh")
        checks.append(("ssh", "OK" if ok else severity, detail))
        if values.cpu_identity_file:
            key_path = Path(os.path.expandvars(values.cpu_identity_file)).expanduser()
            checks.append((
                "CPU SSH key",
                "OK" if key_path.is_file() else severity,
                str(key_path),
            ))
        rsync_ok, rsync_detail = _command_check("rsync")
        scp_ok, scp_detail = _command_check("scp")
        checks.append((
            "file transfer",
            "OK" if (rsync_ok or scp_ok) else severity,
            rsync_detail if rsync_ok else scp_detail,
        ))
        # Import only after configuration is written and ROOT is importable.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        old_config = os.environ.get("RAG_PIPELINE_CONFIG")
        os.environ["RAG_PIPELINE_CONFIG"] = str(path)
        try:
            from src.config import load_config
            from src.hpc_backend import HpcBackend

            remote = HpcBackend(load_config(path).hpc).check_connections()
            outcome = remote["cpu"]
            checks.append((
                "CPU cluster",
                "OK" if bool(outcome["ok"]) else severity,
                str(outcome["detail"]),
            ))
        finally:
            if old_config is None:
                os.environ.pop("RAG_PIPELINE_CONFIG", None)
            else:
                os.environ["RAG_PIPELINE_CONFIG"] = old_config

    for label, status, detail in checks:
        print(f"  {status:4}  {label}: {detail}")
    return all(status != "FAIL" for _, status, _ in checks)


# Ctrl+C reaches the web server directly (it shares this console), and its own
# graceful shutdown -- finalizing queued/running jobs into the durable ledger,
# then terminating its subprocesses -- can legitimately take ~30s. Wait that
# long before escalating to terminate()/kill(), which on Windows is
# TerminateProcess and kills the server mid-shutdown; startup recovery then
# mistakes the stop for a crash and re-enqueues every interrupted job.
GRACEFUL_SHUTDOWN_WAIT_SECONDS = 45.0


def _exited_within(child: subprocess.Popen, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout))
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    return child.poll() is not None


def start_instance(path: Path, values: SetupValues) -> int:
    children: list[subprocess.Popen[str]] = []
    environment = dict(os.environ)
    environment["RAG_PIPELINE_CONFIG"] = str(path)
    try:
        print(f"Starting web server: http://{values.server_host}:{values.server_port}")
        children.append(subprocess.Popen(
            [str(_runtime_python()), "-m", "src.web_app"],
            cwd=ROOT,
            env=environment,
            text=True,
        ))
        return children[-1].wait()
    except KeyboardInterrupt:
        if children:
            print(
                f"Ctrl+C received; giving the web server up to "
                f"{GRACEFUL_SHUTDOWN_WAIT_SECONDS:.0f}s to finish shutting down..."
            )
            if not _exited_within(children[-1], GRACEFUL_SHUTDOWN_WAIT_SECONDS):
                print("Web server did not exit in time; forcing it to stop.")
        return 130
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            if child.poll() is None:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure, verify, and launch one RAG web + HPC instance."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("local", "hpc"))
    parser.add_argument("--server-host")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--ssh-config", type=Path, default=DEFAULT_SSH_CONFIG)
    parser.add_argument("--setup-ssh", action="store_true")
    parser.add_argument("--cpu-host", help="CPU cluster SSH alias")
    parser.add_argument("--cpu-hostname", help="CPU cluster login hostname")
    parser.add_argument("--cpu-user", help="CPU cluster SSH username")
    parser.add_argument("--cpu-key", help="CPU cluster SSH private-key path")
    parser.add_argument("--cpu-repo")
    parser.add_argument(
        "--skip-key-install",
        action="store_true",
        help="Generate/configure keys but do not add public keys remotely.",
    )
    provision_group = parser.add_mutually_exclusive_group()
    provision_group.add_argument(
        "--provision-hpc",
        action="store_true",
        help="Upload the repository and provision the CPU SIF on the cluster "
             "(forces a full redeploy even if the remote is current).",
    )
    provision_group.add_argument(
        "--provision-if-needed",
        action="store_true",
        help="Provision the cluster only if its deployed source is stale or the "
             "SIF is missing (the default behavior of the start launchers).",
    )
    provision_group.add_argument(
        "--skip-hpc-provision",
        action="store_true",
        help="Configure connections only; do not upload/build remote artifacts.",
    )
    parser.add_argument("--ollama-port", type=int)
    parser.add_argument(
        "--initial-corpus",
        type=Path,
        default=None,
        metavar="ZIP",
        help="Deploy an initial PDF corpus: parse it on the HPC cluster, fetch "
             "the Markdown home, and build the local index (requires hpc mode). "
             "Nested directories inside the zip are supported.",
    )
    parser.add_argument(
        "--allow-degraded-vision",
        action="store_true",
        help="With --initial-corpus: proceed without a SoCLAaS key even though "
             "vision enrichment is enabled (figure descriptions will degrade).",
    )
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        help="With --initial-corpus: stop after fetching processed_docs/; build "
             "the index later with main.py --mode index.",
    )
    parser.add_argument(
        "--set-api-key",
        default="",
        metavar="KEY",
        help="Persist a SoCLAaS API key into [llm_api].api_key in config.toml "
             "(the SOCLAAS_API_KEY / LLM_API_KEY environment variables still "
             "override it at runtime).",
    )
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Create .venv and install requirements.txt before checking/starting.",
    )
    parser.add_argument(
        "--no-install-deps",
        action="store_true",
        help="Do not offer to install missing dependencies in interactive mode.",
    )
    return parser


def _decide_dependency_install(
    *,
    dependencies_ready: bool,
    install_deps: bool,
    no_install_deps: bool,
    check_only: bool,
    non_interactive: bool,
    prompt_yes_no=_prompt_yes_no,
) -> bool:
    """Decide whether to create .venv and install requirements.

    A non-interactive run with missing dependencies installs them instead of
    failing preflight -- that is what makes the one-click launchers work on a
    fresh checkout. ``--no-install-deps`` opts out; ``--check-only`` never
    installs.
    """
    if dependencies_ready or no_install_deps or check_only:
        return install_deps
    if install_deps:
        return True
    if non_interactive:
        return True
    return prompt_yes_no(
        "Runtime dependencies are missing. Create .venv and install them?",
        True,
    )


def main(argv: list[str] | None = None) -> int:
    """User-facing entry point: converts failures into concise messages."""
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("\nSetup interrupted.", file=sys.stderr)
        return 130
    except EOFError:
        print(
            "\nNo interactive input is available. Rerun from a terminal, or use "
            "--non-interactive with explicit flags (e.g. --mode local "
            "--server-host 127.0.0.1 --server-port 8000).",
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - one-click UX: no raw tracebacks
        print(f"\nUnexpected setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.ssh_config = args.ssh_config.expanduser().resolve()
    path = args.config.resolve()
    config_preexisted = path.exists()
    existing = _load(path)
    values = (
        _collect_non_interactive(existing, args)
        if args.non_interactive
        else _collect_interactive(existing, args)
    )
    try:
        _validate(values)
    except ValueError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 2

    # Decide whether (and how) to provision the HPC clusters. The three flags
    # are mutually exclusive; none set means "only provision when the
    # interactive SSH-alias setup flow just ran" (legacy behavior).
    provision_mode = "none"
    if args.provision_hpc:
        provision_mode = "force"
    elif args.provision_if_needed:
        provision_mode = "if-needed"
    elif values.mode == "hpc" and not args.skip_hpc_provision and values.manage_ssh_aliases:
        # Interactive first-run flow: SSH aliases were just written, so the
        # remote has never been provisioned -- deploy unconditionally.
        provision_mode = "force"

    if not args.check_only:
        # Resolve the CPU SSH username early: configure() needs it to place the
        # cluster corpus directories under /hpctmp/<user>/rag-corpus (outside
        # the provision-swapped repo), and provisioning needs it below.
        if values.mode == "hpc" and not values.cpu_user:
            values.cpu_user = read_ssh_alias(args.ssh_config, values.cpu_host).get("user", "")
        try:
            setup_ssh_credentials(
                values,
                install_public_keys=not args.skip_key_install,
            )
            configure_ssh_aliases(args.ssh_config, values)
            configure(path, values)
            if values.llm_api_key:
                llm = _llm_api_view(_load(path))
                ok, detail = _verify_llm_api_key(
                    llm["base_url"], llm["models_path"], llm["api_key"]
                )
                if ok:
                    print(f"\nSoCLAaS API key verified ({detail}).")
                else:
                    print(f"\nWARNING: SoCLAaS API key not verified ({detail}).")
                    print("The key was saved; check it if chat/embeddings fail.")
            should_provision = provision_mode != "none" and values.mode == "hpc"
            if should_provision:
                if not values.cpu_user:
                    raise RuntimeError(
                        "CPU SSH username is required for remote provisioning"
                    )
                values.cpu_storage_root = f"/hpctmp/{values.cpu_user}"
                provision_hpc_servers(values, force=provision_mode == "force")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Configuration failed: {exc}", file=sys.stderr)
            return 1
        if values.manage_ssh_aliases:
            print(f"\nSSH aliases saved to {args.ssh_config}")
            ssh_backup = args.ssh_config.with_name(args.ssh_config.name + ".rag-setup.bak")
            if ssh_backup.exists():
                print(f"Previous SSH configuration backed up to {ssh_backup}")
        print(f"\nConfiguration saved to {path}")
        if path.with_suffix(path.suffix + ".bak").exists():
            print(f"Previous configuration backed up to {path}.bak")
        if args.non_interactive and not config_preexisted and values.mode == "local":
            print(
                "\nNOTE: No config.toml existed before this run, so one was created "
                "in LOCAL mode -- PDF parsing runs on this machine.\n"
                "      For cluster-backed parsing, run the guided setup once "
                "(setup.cmd / ./setup.sh) and choose mode 'hpc'."
            )

    dependencies_ready = _runtime_dependencies_ready()
    should_install = _decide_dependency_install(
        dependencies_ready=dependencies_ready,
        install_deps=args.install_deps,
        no_install_deps=args.no_install_deps,
        check_only=args.check_only,
        non_interactive=args.non_interactive,
    )
    if should_install:
        try:
            install_dependencies()
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"Dependency installation failed: {exc}", file=sys.stderr)
            return 1

    # Initial corpus deployment (parse on HPC, fetch, index locally). Runs
    # after dependency installation (the index build needs the venv) and
    # before preflight checks (which then validate the freshly built state).
    if args.initial_corpus and not (args.check_only or args.configure_only):
        if values.mode != "hpc":
            print(
                "\n--initial-corpus requires hpc mode; this instance is "
                "configured for local parsing. Run the guided setup (setup.cmd) "
                "and choose mode 'hpc'.",
                file=sys.stderr,
            )
            return 1
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.hpc_corpus import CorpusError, run_initial_corpus

        try:
            corpus_exit = run_initial_corpus(
                args.initial_corpus,
                config_path=path,
                allow_degraded_vision=args.allow_degraded_vision,
                skip_index_build=args.skip_index_build,
            )
        except CorpusError as exc:
            print(f"\nInitial corpus deployment failed: {exc}", file=sys.stderr)
            return 1
        if corpus_exit != 0:
            return corpus_exit

    checks_ok = True
    if not args.skip_checks:
        checks_ok = run_checks(
            path,
            values,
            cluster_required=(
                provision_mode != "none" and values.mode == "hpc"
            ) or bool(args.initial_corpus),
        )
    if not checks_ok:
        print("\nPreflight failed; fix the items above and rerun setup.", file=sys.stderr)
        return 1
    if args.configure_only or args.check_only:
        return 0

    should_start = args.start
    if not args.non_interactive and not args.start:
        should_start = _prompt_yes_no("Start the instance now?", True)
    if not should_start:
        print(
            "Ready. Start later with setup.cmd --non-interactive --start "
            "(Windows) or ./setup.sh --non-interactive --start."
        )
        return 0
    return start_instance(path, values)


if __name__ == "__main__":
    raise SystemExit(main())
