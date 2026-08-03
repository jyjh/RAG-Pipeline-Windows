from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from scripts.setup_instance import (
    SetupValues,
    _validate,
    configure,
    configure_ssh_aliases,
    create_repository_archive,
    ensure_ssh_private_key,
    install_ssh_public_key,
    _login_relative_repo_default,
    _remote_container_build_command,
    provision_hpc_cluster,
    read_ssh_alias,
    update_toml_sections,
)


def test_update_toml_sections_preserves_unrelated_values(safe_tmp_path):
    config = safe_tmp_path / "config.toml"
    config.write_text(
        '[models]\nllm_model = "keep-me"\n\n[server]\nhost = "old"\nport = 1\n',
        encoding="utf-8",
    )
    update_toml_sections(config, {"server": {"host": "127.0.0.1", "port": 8000}})
    text = config.read_text(encoding="utf-8")
    assert 'llm_model = "keep-me"' in text
    assert 'host = "127.0.0.1"' in text
    assert "port = 8000" in text
    assert config.with_suffix(".toml.bak").exists()


def test_configure_writes_two_cluster_connections(safe_tmp_path):
    config = safe_tmp_path / "config.toml"
    config.write_text("[server]\nhost = \"old\"\n", encoding="utf-8")
    values = SetupValues(
        mode="hpc",
        server_host="0.0.0.0",
        server_port=8080,
        cpu_host="cpu-login",
        cpu_repo="rag-cpu",
        gpu_host="gpu-login",
        gpu_repo="rag-gpu",
    )
    configure(config, values)
    text = config.read_text(encoding="utf-8")
    assert "enabled = true" in text
    assert 'ssh_host = "cpu-login"' in text
    assert 'ssh_host = "gpu-login"' in text
    assert "bind_all = true" in text


def test_configure_ssh_aliases_creates_and_updates_both_clusters(safe_tmp_path):
    ssh_config = safe_tmp_path / ".ssh" / "config"
    ssh_config.parent.mkdir()
    original = (
        "Host unrelated\n"
        "    HostName elsewhere.example\n\n"
        "Host cpu-login\n"
        "    HostName old-cpu.example\n"
        "    User old-user\n"
        "    Compression yes\n"
    )
    ssh_config.write_text(original, encoding="utf-8")
    values = SetupValues(
        mode="hpc",
        server_host="127.0.0.1",
        server_port=8000,
        manage_ssh_aliases=True,
        cpu_host="cpu-login",
        cpu_hostname="cpu.example.edu",
        cpu_user="student",
        cpu_identity_file="~/.ssh/id_cpu",
        cpu_repo="rag-cpu",
        gpu_host="gpu-login",
        gpu_hostname="gpu.example.edu",
        gpu_user="student",
        gpu_identity_file="~/.ssh/id_gpu",
        gpu_repo="rag-gpu",
    )
    configure_ssh_aliases(ssh_config, values)

    cpu = read_ssh_alias(ssh_config, "cpu-login")
    gpu = read_ssh_alias(ssh_config, "gpu-login")
    text = ssh_config.read_text(encoding="utf-8")
    assert cpu["hostname"] == "cpu.example.edu"
    assert cpu["user"] == "student"
    assert cpu["identityfile"] == "~/.ssh/id_cpu"
    assert gpu["hostname"] == "gpu.example.edu"
    assert gpu["identityfile"] == "~/.ssh/id_gpu"
    assert "Compression yes" in text
    assert "Host unrelated" in text
    assert ssh_config.with_name("config.rag-setup.bak").read_text(encoding="utf-8") == original


def test_validate_rejects_absolute_hpc_repo():
    values = SetupValues(
        mode="hpc",
        server_host="127.0.0.1",
        server_port=8000,
        cpu_host="cpu",
        cpu_repo="/home/me/rag",
        gpu_host="gpu",
        gpu_repo="/remote/gpu",
    )
    try:
        _validate(values)
    except ValueError as exc:
        assert "relative to the SSH login" in str(exc)
    else:
        raise AssertionError("absolute HPC repo path should fail validation")


def test_legacy_home_repo_default_becomes_login_relative():
    assert (
        _login_relative_repo_default(
            "/home/student/projects/RAG-Pipeline-Windows",
            "student",
        )
        == "projects/RAG-Pipeline-Windows"
    )


def test_ensure_ssh_private_key_generates_ed25519_pair(safe_tmp_path, monkeypatch):
    private_key = safe_tmp_path / "keys" / "cpu_ed25519"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("-f") + 1])
        output.write_text("private", encoding="utf-8")
        Path(str(output) + ".pub").write_text(
            "ssh-ed25519 AAAATEST rag-pipeline-cpu\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.setup_instance.shutil.which", lambda name: "ssh-keygen")
    monkeypatch.setattr("scripts.setup_instance.subprocess.run", fake_run)
    private, public = ensure_ssh_private_key(str(private_key), "cpu-login")
    assert private == private_key
    assert public == Path(str(private_key) + ".pub")
    assert "-t" in calls[0] and "ed25519" in calls[0]
    assert calls[0][calls[0].index("-N") + 1] == ""


def test_install_ssh_public_key_installs_and_verifies(safe_tmp_path, monkeypatch):
    private_key = safe_tmp_path / "id_ed25519"
    public_key = safe_tmp_path / "id_ed25519.pub"
    private_key.write_text("private", encoding="utf-8")
    public_key.write_text("ssh-ed25519 AAAATEST setup-test\n", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "BatchMode=yes" in command:
            return subprocess.CompletedProcess(command, 0, stdout="RAG_SSH_KEY_OK", stderr="")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.setup_instance.shutil.which", lambda name: "ssh")
    monkeypatch.setattr("scripts.setup_instance.subprocess.run", fake_run)
    install_ssh_public_key(
        hostname="cpu.example.edu",
        user="student",
        private_key=private_key,
        public_key=public_key,
    )
    assert len(calls) == 2
    assert "student@cpu.example.edu" in calls[0]
    assert "authorized_keys" in calls[0][-1]
    assert "grep -qxF" in calls[0][-1]
    assert "BatchMode=yes" in calls[1]


def test_repository_archive_excludes_local_data_indexes_and_sifs(safe_tmp_path):
    source = safe_tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("print('ok')", encoding="utf-8")
    (source / "config.toml").write_text("[server]\n", encoding="utf-8")
    (source / "data").mkdir()
    (source / "data" / "private.pdf").write_bytes(b"pdf")
    (source / "db").mkdir()
    (source / "db" / "index.bin").write_bytes(b"index")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("secret", encoding="utf-8")
    (source / "rag_pipeline_cpu.sif").write_bytes(b"large")
    destination = safe_tmp_path / "repository.tar.gz"

    create_repository_archive(destination, source)
    with tarfile.open(destination, "r:gz") as archive:
        names = set(archive.getnames())
    assert "./main.py" in names
    assert "./config.toml" in names
    assert not any(name.startswith("./data") for name in names)
    assert not any(name.startswith("./db") for name in names)
    assert not any(name.startswith("./.git") for name in names)
    assert not any(name.endswith(".sif") for name in names)


def test_remote_container_build_uses_fakeroot_and_selected_definition():
    command = _remote_container_build_command(
        repo_dir="/scratch/student/rag-gpu",
        definition_name="Singularity.def",
        image_name="rag_pipeline.sif",
    )
    assert "apptainer" in command
    assert "singularity" in command
    assert "build --fakeroot rag_pipeline.sif Singularity.def" in command


def test_provision_cluster_stages_under_hpctmp_and_activates_login_link(
    safe_tmp_path,
    monkeypatch,
):
    archive = safe_tmp_path / "repository.tar.gz"
    archive.write_bytes(b"archive")
    image = safe_tmp_path / "rag_pipeline_cpu.sif"
    image.write_bytes(b"image")
    remote_commands = []
    uploads = []
    monkeypatch.setattr(
        "scripts.setup_instance._run_provision_ssh",
        lambda alias, command, **kwargs: remote_commands.append((alias, command)),
    )
    monkeypatch.setattr(
        "scripts.setup_instance._upload_provision_file",
        lambda source, alias, remote: uploads.append((Path(source), alias, remote)),
    )
    monkeypatch.setattr("scripts.setup_instance.secrets.token_hex", lambda size: "abc123")

    provision_hpc_cluster(
        alias="cpu-login",
        user="student",
        storage_root="/hpctmp/student",
        relative_repo="rag-cpu",
        archive_path=archive,
        image_name="rag_pipeline_cpu.sif",
        definition_name="Singularity.cpu.def",
        local_image=image,
    )

    assert uploads[0] == (
        archive,
        "cpu-login",
        "/hpctmp/student/.rag_setup_abc123/repository.tar.gz",
    )
    assert uploads[1][0] == image
    assert uploads[1][2].endswith("/repo/rag_pipeline_cpu.sif")
    joined = "\n".join(command for _, command in remote_commands)
    assert "/hpctmp/student/rag-cpu" in joined
    assert "$HOME/rag-cpu" in joined
    assert "ln -sfn" in joined


def test_provision_gpu_cluster_uses_vanda_scratch_root(safe_tmp_path, monkeypatch):
    archive = safe_tmp_path / "repository.tar.gz"
    archive.write_bytes(b"archive")
    remote_commands = []
    monkeypatch.setattr(
        "scripts.setup_instance._run_provision_ssh",
        lambda alias, command, **kwargs: remote_commands.append(command),
    )
    monkeypatch.setattr(
        "scripts.setup_instance._upload_provision_file",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("scripts.setup_instance.secrets.token_hex", lambda size: "gpu123")

    provision_hpc_cluster(
        alias="gpu-login",
        user="student",
        storage_root="/scratch/student",
        relative_repo="rag-gpu",
        archive_path=archive,
        image_name="rag_pipeline.sif",
        definition_name="Singularity.def",
        local_image=None,
    )

    joined = "\n".join(remote_commands)
    assert "test -d /scratch" in joined
    assert "/scratch/student/rag-gpu" in joined
    assert "build --fakeroot rag_pipeline.sif Singularity.def" in joined
    assert "/hpctmp/student" not in joined


def test_setup_cli_help_runs():
    result = subprocess.run(
        ["python", "scripts/setup_instance.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--submit-gpu-job" in result.stdout
    assert "--cpu-hostname" in result.stdout
    assert "--gpu-hostname" in result.stdout
    assert "--skip-key-install" in result.stdout
    assert "--provision-hpc" in result.stdout
    assert "--skip-hpc-provision" in result.stdout


def test_repo_manifest_is_deterministic_and_excludes_artifacts(safe_tmp_path):
    """The manifest fingerprint is stable for unchanged source and ignores the
    same paths the archive filter drops (.git, data, *.sif, etc.)."""
    from scripts.setup_instance import _build_repo_manifest, _manifest_fingerprint

    src = safe_tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (src / "README.md").write_text("docs", encoding="utf-8")
    # Artifacts that MUST be excluded from both archive and manifest.
    (src / ".git").mkdir()
    (src / ".git" / "config").write_text("git", encoding="utf-8")
    (src / "data").mkdir()
    (src / "data" / "x.bin").write_bytes(b"\0" * 100)
    (src / "rag_pipeline.sif").write_bytes(b"\x9d" * 50)

    fp1 = _manifest_fingerprint(_build_repo_manifest(src))
    fp2 = _manifest_fingerprint(_build_repo_manifest(src))
    assert fp1 and fp1 == fp2  # deterministic

    # Changing an artifact that is excluded must NOT change the fingerprint.
    (src / "data" / "x.bin").write_bytes(b"\1" * 200)
    (src / ".git" / "config").write_text("changed", encoding="utf-8")
    assert _manifest_fingerprint(_build_repo_manifest(src)) == fp1

    # Changing a shipped source file MUST change the fingerprint.
    (src / "pkg" / "main.py").write_text("print('bye')\n", encoding="utf-8")
    assert _manifest_fingerprint(_build_repo_manifest(src)) != fp1


def test_create_repository_archive_embeds_manifest(safe_tmp_path):
    """The shipped tarball contains a .rag_manifest entry whose fingerprint
    matches a freshly computed one for the same source."""
    import io
    import tarfile
    from scripts.setup_instance import (
        _build_repo_manifest,
        _manifest_fingerprint,
        create_repository_archive,
    )

    src = safe_tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print(1)\n", encoding="utf-8")
    archive = create_repository_archive(safe_tmp_path / "repo.tgz", source_root=src)

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        assert ".rag_manifest" in names
        member = tar.extractfile(".rag_manifest")
        embedded = member.read().decode("utf-8") if member else ""

    expected = _manifest_fingerprint(_build_repo_manifest(src))
    assert _manifest_fingerprint(embedded) == expected


def test_provision_skips_upload_when_remote_manifest_matches(safe_tmp_path, monkeypatch):
    """When the deployed manifest matches the local source AND the SIF exists,
    provision_hpc_cluster must skip the upload/extract/build/activate cycle."""
    from scripts.setup_instance import (
        _build_repo_manifest,
        _manifest_fingerprint,
        provision_hpc_cluster,
    )

    # Point ROOT at a throwaway source tree so the manifest is computed against
    # deterministic, controlled content.
    src = safe_tmp_path / "repo"
    src.mkdir()
    (src / "main.py").write_text("print('deployed')\n", encoding="utf-8")
    monkeypatch.setattr("scripts.setup_instance.ROOT", src)

    local_manifest = _build_repo_manifest(src)
    local_fp = _manifest_fingerprint(local_manifest)
    assert local_fp

    # Stub ssh so _remote_manifest returns a matching manifest, and the SIF
    # existence check succeeds.
    def fake_ssh_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = local_manifest
            stderr = ""
        return R()

    # The freshness path calls subprocess.run with a list arg whose first
    # element is the ssh binary. Detect the manifest fetch vs the SIF test by
    # inspecting the remote command string.
    def fake_subprocess_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        # cmd is [ssh, -o, ..., alias, remote_command]
        remote = cmd[-1] if isinstance(cmd, list) and len(cmd) > 1 else ""
        if remote.startswith("cat ") and ".rag_manifest" in remote:
            class RM:
                returncode = 0
                stdout = local_manifest
                stderr = ""
            return RM()
        # test -s <sif> -> success (SIF exists)
        return R()

    monkeypatch.setattr("scripts.setup_instance.subprocess.run", fake_subprocess_run)

    uploads = []
    remote_commands = []
    monkeypatch.setattr(
        "scripts.setup_instance._upload_provision_file",
        lambda *a, **k: uploads.append(a),
    )
    monkeypatch.setattr(
        "scripts.setup_instance._run_provision_ssh",
        lambda alias, command, **k: remote_commands.append(command),
    )

    provision_hpc_cluster(
        alias="gpu-login",
        user="student",
        storage_root="/scratch/student",
        relative_repo="rag-gpu",
        archive_path=safe_tmp_path / "repository.tar.gz",
        image_name="rag_pipeline.sif",
        definition_name="Singularity.def",
        local_image=None,
        force=False,
    )

    assert uploads == []          # no upload happened
    assert remote_commands == []  # no staging/extract/build/activate happened


def test_provision_redeploys_when_remote_manifest_differs(safe_tmp_path, monkeypatch):
    """A stale or absent remote manifest triggers a full provision."""
    from scripts.setup_instance import provision_hpc_cluster

    src = safe_tmp_path / "repo"
    src.mkdir()
    (src / "main.py").write_text("print('new')\n", encoding="utf-8")
    monkeypatch.setattr("scripts.setup_instance.ROOT", src)

    # _remote_manifest returns "" (no deployed manifest -> first deploy).
    def fake_subprocess_run(cmd, **kwargs):
        class R:
            returncode = 1
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr("scripts.setup_instance.subprocess.run", fake_subprocess_run)

    uploads = []
    remote_commands = []
    monkeypatch.setattr(
        "scripts.setup_instance._upload_provision_file",
        lambda *a, **k: uploads.append(a),
    )
    monkeypatch.setattr(
        "scripts.setup_instance._run_provision_ssh",
        lambda alias, command, **k: remote_commands.append(command),
    )

    provision_hpc_cluster(
        alias="cpu-login",
        user="student",
        storage_root="/hpctmp/student",
        relative_repo="rag-cpu",
        archive_path=safe_tmp_path / "repository.tar.gz",
        image_name="rag_pipeline_cpu.sif",
        definition_name="Singularity.cpu.def",
        local_image=None,
        force=False,
    )

    assert uploads != []          # source was uploaded
    joined = "\n".join(remote_commands)
    assert "tar -xzf" in joined   # extracted
    assert "ln -sfn" in joined    # activated


def test_provision_force_ignores_matching_manifest(safe_tmp_path, monkeypatch):
    """force=True must redeploy even when the remote manifest would match."""
    from scripts.setup_instance import (
        _build_repo_manifest,
        _manifest_fingerprint,
        provision_hpc_cluster,
    )

    src = safe_tmp_path / "repo"
    src.mkdir()
    (src / "main.py").write_text("print('x')\n", encoding="utf-8")
    monkeypatch.setattr("scripts.setup_instance.ROOT", src)
    local_manifest = _build_repo_manifest(src)

    ssh_calls = []
    def fake_subprocess_run(cmd, **kwargs):
        ssh_calls.append(cmd)
        class R:
            returncode = 0
            stdout = local_manifest
            stderr = ""
        return R()
    monkeypatch.setattr("scripts.setup_instance.subprocess.run", fake_subprocess_run)

    uploads = []
    remote_commands = []
    monkeypatch.setattr(
        "scripts.setup_instance._upload_provision_file",
        lambda *a, **k: uploads.append(a),
    )
    monkeypatch.setattr(
        "scripts.setup_instance._run_provision_ssh",
        lambda alias, command, **k: remote_commands.append(command),
    )

    provision_hpc_cluster(
        alias="gpu-login",
        user="student",
        storage_root="/scratch/student",
        relative_repo="rag-gpu",
        archive_path=safe_tmp_path / "repository.tar.gz",
        image_name="rag_pipeline.sif",
        definition_name="Singularity.def",
        local_image=None,
        force=True,
    )

    # Even with a matching manifest, force=True provisions.
    assert uploads != []
    assert ssh_calls == []  # force short-circuits the manifest fetch entirely
    joined = "\n".join(remote_commands)
    assert "tar -xzf" in joined


def test_start_cli_passes_provision_if_needed():
    """The start launchers invoke setup_instance with --provision-if-needed,
    --non-interactive, and --start. Verify the flags are accepted together."""
    result = subprocess.run(
        ["python", "scripts/setup_instance.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--provision-if-needed" in result.stdout

