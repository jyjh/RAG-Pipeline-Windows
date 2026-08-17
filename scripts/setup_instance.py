"""Guided, one-command setup for the web server and HPC connections.

Run interactively with ``python scripts/setup_instance.py``.  The command uses
only the Python standard library, so it can configure and diagnose a fresh
checkout before the project's optional runtime dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
    gpu_host: str = ""
    gpu_hostname: str = ""
    gpu_user: str = ""
    gpu_identity_file: str = ""
    gpu_repo: str = ""
    gpu_storage_root: str = ""
    local_ollama_port: int = 11434


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
    upsert_ssh_alias(
        path,
        alias=values.gpu_host,
        hostname=values.gpu_hostname,
        user=values.gpu_user,
        identity_file=values.gpu_identity_file,
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
    """Generate and optionally authorize the dedicated keys for both clusters."""
    if values.mode != "hpc" or not values.manage_ssh_aliases:
        return
    clusters = (
        (
            values.cpu_host,
            values.cpu_hostname,
            values.cpu_user,
            values.cpu_identity_file,
        ),
        (
            values.gpu_host,
            values.gpu_hostname,
            values.gpu_user,
            values.gpu_identity_file,
        ),
    )
    for alias, hostname, user, raw_key in clusters:
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


def _build_repo_manifest(source_root: Path = ROOT) -> str:
    """Return a deterministic content fingerprint of the archived source tree.

    Walks ``source_root`` with the SAME exclusions as the archive filter so the
    manifest exactly describes the bytes that get shipped. Each regular file is
    one line "<posix_rel_path>\\0<sha256>", sorted by path -- stable across runs
    and independent of filesystem walk order or mtimes. The whole buffer is then
    SHA-256'd into a header line so a single comparison suffices to detect any
    change, but the per-file lines are retained for diagnosing what differed.
    """
    entries: list[tuple[str, str]] = []
    excluded = _REPO_ARCHIVE_EXCLUDED_NAMES

    def _excluded(rel: PurePosixPath) -> bool:
        parts = rel.parts
        if any(p in excluded for p in parts):
            return True
        if any(p.startswith((".index_build_", ".tmp_test_", ".pytest_")) for p in parts):
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
    # If they agree AND the SIF is already in place, there is nothing to do.
    local_manifest = _build_repo_manifest(ROOT)
    local_fingerprint = _manifest_fingerprint(local_manifest)
    if not force and local_fingerprint:
        remote_manifest = _remote_manifest(alias, target_repo)
        remote_fingerprint = _manifest_fingerprint(remote_manifest)
        if remote_fingerprint and remote_fingerprint == local_fingerprint:
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
                    print(f"\n{alias}: already up to date ({target_repo}); skipping upload.")
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
            print(f"Uploading {image_name} to {alias} (large transfer)...")
            _upload_provision_file(
                local_image,
                alias,
                posixpath.join(staged_repo, image_name),
            )
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
    """Package once, then provision the CPU and GPU clusters.

    With ``force=False`` (the default) each cluster is only re-uploaded/rebuilt
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
        provision_hpc_cluster(
            alias=values.gpu_host,
            user=values.gpu_user,
            storage_root=_storage_root(
                values.gpu_storage_root,
                values.gpu_user,
                "/scratch",
            ),
            relative_repo=values.gpu_repo,
            archive_path=archive_path,
            image_name="rag_pipeline.sif",
            definition_name="Singularity.def",
            local_image=(
                ROOT / "rag_pipeline.sif"
                if (ROOT / "rag_pipeline.sif").is_file()
                else None
            ),
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


def _validate(values: SetupValues) -> None:
    if values.mode not in {"local", "hpc"}:
        raise ValueError("mode must be 'local' or 'hpc'")
    if not (1 <= values.server_port <= 65535):
        raise ValueError("server port must be between 1 and 65535")
    if not (1 <= values.local_ollama_port <= 65535):
        raise ValueError("local Ollama tunnel port must be between 1 and 65535")
    if values.mode == "hpc":
        for label, host, hostname, user, repo in (
            ("CPU", values.cpu_host, values.cpu_hostname, values.cpu_user, values.cpu_repo),
            ("GPU", values.gpu_host, values.gpu_hostname, values.gpu_user, values.gpu_repo),
        ):
            if not host:
                raise ValueError(f"{label} SSH alias/host is required")
            if not _SSH_ALIAS_RE.fullmatch(host):
                raise ValueError(f"{label} SSH alias contains unsupported characters")
            if values.manage_ssh_aliases and not hostname:
                raise ValueError(f"{label} SSH hostname is required")
            if values.manage_ssh_aliases and not user:
                raise ValueError(f"{label} SSH username is required")
            identity = (
                values.cpu_identity_file if label == "CPU" else values.gpu_identity_file
            )
            if values.manage_ssh_aliases and not identity:
                raise ValueError(f"{label} SSH private-key path is required")
            _validate_relative_repo(repo, label)
        if values.cpu_host == values.gpu_host:
            raise ValueError("CPU and GPU clusters must use different SSH aliases")
        if values.manage_ssh_aliases:
            if values.cpu_storage_root != f"/hpctmp/{values.cpu_user}":
                raise ValueError("Atlas9 CPU storage root must be /hpctmp/<username>")
            if values.gpu_storage_root != f"/scratch/{values.gpu_user}":
                raise ValueError("Vanda GPU storage root must be /scratch/<username>")


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
    server_port = int(_prompt(
        "Web server port",
        str(args.server_port or _nested(config, "server", "port", default=8000)),
        required=True,
    ))
    values = SetupValues(mode=mode, server_host=server_host, server_port=server_port)
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
        values.gpu_host = _prompt(
            "GPU cluster SSH alias",
            args.gpu_host or str(_nested(config, "hpc", "gpu", "ssh_host")),
            required=True,
        )
        existing_gpu_alias = read_ssh_alias(args.ssh_config, values.gpu_host)
        values.gpu_hostname = _prompt(
            "GPU cluster login hostname",
            args.gpu_hostname or existing_gpu_alias.get("hostname", ""),
            required=True,
        )
        values.gpu_user = _prompt(
            "GPU cluster SSH username",
            args.gpu_user or existing_gpu_alias.get("user", values.cpu_user),
            required=True,
        )
        values.gpu_storage_root = f"/scratch/{values.gpu_user}"
        values.gpu_identity_file = _prompt(
            "GPU SSH private key (created automatically if missing)",
            args.gpu_key
            or existing_gpu_alias.get("identityfile", "")
            or f"~/.ssh/rag_{values.gpu_host}_ed25519",
            required=True,
        )
        values.gpu_repo = _prompt(
            "GPU repo path relative to SSH login directory",
            args.gpu_repo
            or _login_relative_repo_default(
                str(_nested(config, "hpc", "gpu", "remote_repo_dir")),
                values.gpu_user,
            ),
            required=True,
        )
        values.local_ollama_port = int(_prompt(
            "Local Ollama tunnel port",
            str(args.ollama_port or _configured_ollama_port(config)),
            required=True,
        ))
    return values


def _collect_non_interactive(config: dict[str, Any], args: argparse.Namespace) -> SetupValues:
    mode = args.mode or ("hpc" if _nested(config, "hpc", "enabled", default=False) else "local")
    manage_ssh = bool(args.setup_ssh or args.cpu_hostname or args.gpu_hostname)
    cpu_alias = args.cpu_host or str(_nested(config, "hpc", "cpu", "ssh_host"))
    gpu_alias = args.gpu_host or str(_nested(config, "hpc", "gpu", "ssh_host"))
    cpu_user = args.cpu_user or ""
    gpu_user = args.gpu_user or ""
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
        gpu_host=gpu_alias,
        gpu_hostname=args.gpu_hostname or "",
        gpu_user=gpu_user,
        gpu_identity_file=(
            args.gpu_key or (f"~/.ssh/rag_{gpu_alias}_ed25519" if manage_ssh and gpu_alias else "")
        ),
        gpu_repo=args.gpu_repo or str(_nested(config, "hpc", "gpu", "remote_repo_dir")),
        gpu_storage_root=_storage_root(
            str(_nested(config, "hpc", "gpu", "storage_root")),
            gpu_user,
            "/scratch",
        ),
        local_ollama_port=args.ollama_port or _configured_ollama_port(config),
    )


def configure(path: Path, values: SetupValues) -> None:
    updates = {
        "server": {
            "host": values.server_host,
            "port": values.server_port,
            "bind_all": values.server_host in {"0.0.0.0", "::"},
        },
        "ollama": {"host": f"http://127.0.0.1:{values.local_ollama_port}"},
        "hpc": {
            "enabled": values.mode == "hpc",
            "remote_data_dir": "data",
            "remote_db_dir": "db",
            "poll_interval_seconds": 15.0,
        },
    }
    if values.mode == "hpc":
        updates.update({
            "hpc.cpu": {
                "ssh_host": values.cpu_host,
                "remote_repo_dir": values.cpu_repo,
                "container_sif": "rag_pipeline_cpu.sif",
                "storage_root": values.cpu_storage_root or "/hpctmp/${USER}",
            },
            "hpc.gpu": {
                "ssh_host": values.gpu_host,
                "remote_repo_dir": values.gpu_repo,
                "container_sif": "rag_pipeline.sif",
                "storage_root": values.gpu_storage_root or "/scratch/${USER}",
            },
        })
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


def run_checks(path: Path, values: SetupValues) -> bool:
    print("\nPreflight checks")
    print("----------------")
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python 3.11+", sys.version_info >= (3, 11), sys.version.split()[0]))
    runtime = _runtime_python()
    checks.append((
        "Python dependencies",
        _runtime_dependencies_ready(runtime),
        str(runtime),
    ))
    try:
        with socket.socket() as probe:
            probe.bind((values.server_host, values.server_port))
        checks.append(("Web port", True, f"{values.server_host}:{values.server_port} available"))
    except OSError as exc:
        already_running = _http_ready(f"http://127.0.0.1:{values.server_port}/api/health")
        checks.append((
            "Web port",
            already_running,
            "RAG web server already running" if already_running else str(exc),
        ))

    if values.mode == "local":
        ready = _http_ready(f"http://127.0.0.1:{values.local_ollama_port}/api/version")
        checks.append(("Local Ollama", ready, "ready" if ready else "not reachable"))
    else:
        ok, detail = _command_check("ssh")
        checks.append(("ssh", ok, detail))
        for label, raw_key in (
            ("CPU SSH key", values.cpu_identity_file),
            ("GPU SSH key", values.gpu_identity_file),
        ):
            if raw_key:
                key_path = Path(os.path.expandvars(raw_key)).expanduser()
                checks.append((label, key_path.is_file(), str(key_path)))
        rsync_ok, rsync_detail = _command_check("rsync")
        scp_ok, scp_detail = _command_check("scp")
        checks.append((
            "file transfer",
            rsync_ok or scp_ok,
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
            for name in ("cpu", "gpu"):
                outcome = remote[name]
                checks.append((
                    f"{name.upper()} cluster",
                    bool(outcome["ok"]),
                    str(outcome["detail"]),
                ))
        finally:
            if old_config is None:
                os.environ.pop("RAG_PIPELINE_CONFIG", None)
            else:
                os.environ["RAG_PIPELINE_CONFIG"] = old_config

    for label, ok, detail in checks:
        print(f"  {'OK' if ok else 'FAIL':4}  {label}: {detail}")
    return all(ok for _, ok, _ in checks)


def _start_tunnel(values: SetupValues) -> subprocess.Popen[str]:
    # DEPRECATED: LLM serving now runs on the SoCLAaS API; this SSH tunnel for
    # the GPU Ollama serving job is obsolete. Retained for reference; no longer
    # called by start_instance().
    if os.name == "nt":
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            raise RuntimeError("PowerShell was not found; cannot start the tunnel")
        command = [
            powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(ROOT / "scripts" / "tunnel_daemon.ps1"),
            "-JumpHost", values.gpu_host,
            "-HostFile", "~/.rag_ollama_serving_host",
            "-LocalPort", str(values.local_ollama_port),
        ]
    else:
        command = [
            "bash", str(ROOT / "scripts" / "tunnel_daemon.sh"),
            "--jump-host", values.gpu_host,
            "--host-file", "~/.rag_ollama_serving_host",
            "--local-port", str(values.local_ollama_port),
        ]
    return subprocess.Popen(command, cwd=ROOT, text=True)


def _submit_gpu_job(path: Path) -> str:
    # DEPRECATED: the GPU Ollama serving job is obsolete (SoCLAaS API now serves
    # LLM/vision/embeddings). Retained for reference; no longer called by
    # start_instance().
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config import load_config
    from src.hpc_backend import HpcBackend

    return HpcBackend(load_config(path).hpc).submit_serve_job()


def start_instance(path: Path, values: SetupValues, *, submit_gpu: bool) -> int:
    children: list[subprocess.Popen[str]] = []
    environment = dict(os.environ)
    environment["RAG_PIPELINE_CONFIG"] = str(path)
    try:
        if submit_gpu:
            # DEPRECATED: kept for backward CLI compatibility, but it is a no-op.
            # LLM/vision/embedding serving moved to the hosted SoCLAaS API
            # ([llm_api] in config.toml); the GPU Ollama serving job + SSH tunnel
            # are obsolete. The CPU ingest/index HPC path is unaffected.
            print(
                "NOTE: --submit-gpu-job is deprecated and ignored. LLM serving now "
                "runs on the SoCLAaS API; the GPU Ollama serving job + SSH tunnel "
                "are no longer started."
            )

        print(f"Starting web server: http://{values.server_host}:{values.server_port}")
        children.append(subprocess.Popen(
            [str(_runtime_python()), "-m", "src.web_app"],
            cwd=ROOT,
            env=environment,
            text=True,
        ))
        return children[-1].wait()
    except KeyboardInterrupt:
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
    parser.add_argument("--gpu-host", help="GPU cluster SSH alias")
    parser.add_argument("--gpu-hostname", help="GPU cluster login hostname")
    parser.add_argument("--gpu-user", help="GPU cluster SSH username")
    parser.add_argument("--gpu-key", help="GPU cluster SSH private-key path")
    parser.add_argument("--gpu-repo")
    parser.add_argument(
        "--skip-key-install",
        action="store_true",
        help="Generate/configure keys but do not add public keys remotely.",
    )
    provision_group = parser.add_mutually_exclusive_group()
    provision_group.add_argument(
        "--provision-hpc",
        action="store_true",
        help="Upload the repository and provision CPU/GPU SIFs on both clusters "
             "(forces a full redeploy even if the remote is current).",
    )
    provision_group.add_argument(
        "--provision-if-needed",
        action="store_true",
        help="Provision each cluster only if its deployed source is stale or the "
             "SIF is missing (the default behavior of the start launchers).",
    )
    provision_group.add_argument(
        "--skip-hpc-provision",
        action="store_true",
        help="Configure connections only; do not upload/build remote artifacts.",
    )
    parser.add_argument("--ollama-port", type=int)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument(
        "--submit-gpu-job",
        action="store_true",
        help="DEPRECATED no-op: LLM serving now runs on the SoCLAaS API; the GPU Ollama job is not submitted.",
    )
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.ssh_config = args.ssh_config.expanduser().resolve()
    path = args.config.resolve()
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

    if not args.check_only:
        try:
            setup_ssh_credentials(
                values,
                install_public_keys=not args.skip_key_install,
            )
            configure_ssh_aliases(args.ssh_config, values)
            configure(path, values)
            # Decide whether (and how) to provision the HPC clusters. The three
            # flags are mutually exclusive; none set means "only provision when
            # the interactive SSH-alias setup flow just ran" (legacy behavior).
            provision_mode = "none"
            if args.provision_hpc:
                provision_mode = "force"
            elif args.provision_if_needed:
                provision_mode = "if-needed"
            elif values.mode == "hpc" and not args.skip_hpc_provision and values.manage_ssh_aliases:
                # Interactive first-run flow: SSH aliases were just written, so
                # the remote has never been provisioned -- deploy unconditionally.
                provision_mode = "force"
            should_provision = provision_mode != "none" and values.mode == "hpc"
            if should_provision:
                if not values.cpu_user:
                    values.cpu_user = read_ssh_alias(
                        args.ssh_config, values.cpu_host
                    ).get("user", "")
                if not values.gpu_user:
                    values.gpu_user = read_ssh_alias(
                        args.ssh_config, values.gpu_host
                    ).get("user", "")
                if not values.cpu_user or not values.gpu_user:
                    raise RuntimeError(
                        "CPU/GPU SSH usernames are required for remote provisioning"
                    )
                values.cpu_storage_root = f"/hpctmp/{values.cpu_user}"
                values.gpu_storage_root = f"/scratch/{values.gpu_user}"
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

    dependencies_ready = _runtime_dependencies_ready()
    should_install = args.install_deps
    if (
        not dependencies_ready
        and not args.non_interactive
        and not args.check_only
        and not args.no_install_deps
        and not should_install
    ):
        should_install = _prompt_yes_no(
            "Runtime dependencies are missing. Create .venv and install them?",
            True,
        )
    if should_install:
        try:
            install_dependencies()
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"Dependency installation failed: {exc}", file=sys.stderr)
            return 1

    checks_ok = True
    if not args.skip_checks:
        checks_ok = run_checks(path, values)
    if not checks_ok:
        print("\nPreflight failed; fix the items above and rerun setup.", file=sys.stderr)
        return 1
    if args.configure_only or args.check_only:
        return 0

    should_start = args.start
    submit_gpu = args.submit_gpu_job
    if submit_gpu:
        # Deprecated no-op flag; surface it once so the operator knows.
        print(
            "NOTE: --submit-gpu-job is deprecated and ignored. LLM serving now runs "
            "on the SoCLAaS API; the GPU Ollama serving job is no longer submitted."
        )
    if not args.non_interactive and not args.start:
        should_start = _prompt_yes_no("Start the instance now?", True)
        # The GPU Ollama serving job is DEPRECATED (SoCLAaS API now serves LLMs),
        # so we no longer prompt to submit it.
    if not should_start:
        print("Ready. Start later with setup.cmd --non-interactive --start")
        return 0
    return start_instance(path, values, submit_gpu=submit_gpu)


if __name__ == "__main__":
    raise SystemExit(main())
