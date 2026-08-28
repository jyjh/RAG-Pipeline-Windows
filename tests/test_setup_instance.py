from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from scripts.setup_instance import (
    SetupValues,
    _decide_dependency_install,
    _embeddings_backend_view,
    _llm_api_view,
    _validate,
    configure,
    configure_ssh_aliases,
    create_repository_archive,
    ensure_ssh_private_key,
    install_ssh_public_key,
    main,
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


def test_configure_writes_cpu_cluster_connection(safe_tmp_path):
    config = safe_tmp_path / "config.toml"
    config.write_text("[server]\nhost = \"old\"\n", encoding="utf-8")
    values = SetupValues(
        mode="hpc",
        server_host="0.0.0.0",
        server_port=8080,
        cpu_host="cpu-login",
        cpu_repo="rag-cpu",
    )
    configure(config, values)
    text = config.read_text(encoding="utf-8")
    assert "enabled = true" in text
    assert 'ssh_host = "cpu-login"' in text
    assert "bind_all = true" in text


def test_configure_writes_corpus_dirs_outside_repo(safe_tmp_path):
    """With a known username the corpus dirs must live OUTSIDE remote_repo_dir.

    Provisioning atomically replaces the repo directory, so data/processed/db
    inside it would be wiped by any later re-provision (e.g. after a config
    edit changed the source manifest).
    """
    config = safe_tmp_path / "config.toml"
    config.write_text("[server]\nhost = \"old\"\n", encoding="utf-8")
    values = SetupValues(
        mode="hpc",
        server_host="0.0.0.0",
        server_port=8080,
        cpu_host="cpu-login",
        cpu_user="student",
        cpu_storage_root="/hpctmp/student",
        cpu_repo="rag-cpu",
    )
    configure(config, values)
    text = config.read_text(encoding="utf-8")
    assert 'remote_data_dir = "/hpctmp/student/rag-corpus/data"' in text
    assert 'remote_processed_dir = "/hpctmp/student/rag-corpus/processed_docs"' in text
    assert 'remote_db_dir = "/hpctmp/student/rag-corpus/db"' in text


def test_configure_keeps_legacy_dirs_without_username(safe_tmp_path):
    config = safe_tmp_path / "config.toml"
    values = SetupValues(
        mode="hpc",
        server_host="127.0.0.1",
        server_port=8000,
        cpu_host="cpu-login",
        cpu_repo="rag-cpu",
    )
    configure(config, values)
    text = config.read_text(encoding="utf-8")
    assert 'remote_data_dir = "data"' in text
    assert 'remote_processed_dir = "processed_docs"' in text


def test_configure_ssh_aliases_creates_and_updates(safe_tmp_path):
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
    )
    configure_ssh_aliases(ssh_config, values)

    cpu = read_ssh_alias(ssh_config, "cpu-login")
    text = ssh_config.read_text(encoding="utf-8")
    assert cpu["hostname"] == "cpu.example.edu"
    assert cpu["user"] == "student"
    assert cpu["identityfile"] == "~/.ssh/id_cpu"
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
        repo_dir="/hpctmp/student/rag-cpu",
        definition_name="Singularity.cpu.def",
        image_name="rag_pipeline_cpu.sif",
    )
    assert "apptainer" in command
    assert "singularity" in command
    assert "build --fakeroot rag_pipeline_cpu.sif Singularity.cpu.def" in command


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


def test_provision_cluster_verifies_storage_parent_before_use(safe_tmp_path, monkeypatch):
    """provision_hpc_cluster derives the storage parent from the absolute root
    and confirms it exists before creating the staging directory."""
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
    monkeypatch.setattr("scripts.setup_instance.secrets.token_hex", lambda size: "cpu123")

    provision_hpc_cluster(
        alias="cpu-login",
        user="student",
        storage_root="/scratch/student",
        relative_repo="rag-cpu",
        archive_path=archive,
        image_name="rag_pipeline_cpu.sif",
        definition_name="Singularity.cpu.def",
        local_image=None,
    )

    joined = "\n".join(remote_commands)
    assert "test -d /scratch" in joined
    assert "/scratch/student/rag-cpu" in joined
    assert "build --fakeroot rag_pipeline_cpu.sif Singularity.cpu.def" in joined


def test_setup_cli_help_runs():
    result = subprocess.run(
        ["python", "scripts/setup_instance.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--cpu-hostname" in result.stdout
    assert "--skip-key-install" in result.stdout
    assert "--provision-hpc" in result.stdout
    assert "--skip-hpc-provision" in result.stdout
    assert "--set-api-key" in result.stdout


def test_configure_writes_api_key_only_when_provided(safe_tmp_path):
    config = safe_tmp_path / "config.toml"
    config.write_text('[llm_api]\nbackend = "soclaas"\n', encoding="utf-8")

    values = SetupValues(mode="local", server_host="127.0.0.1", server_port=8000)
    configure(config, values)
    assert "api_key" not in config.read_text(encoding="utf-8")

    values = SetupValues(
        mode="local", server_host="127.0.0.1", server_port=8000, llm_api_key="sk-test"
    )
    configure(config, values)
    assert 'api_key = "sk-test"' in config.read_text(encoding="utf-8")
    # The unrelated backend line survives the update.
    assert 'backend = "soclaas"' in config.read_text(encoding="utf-8")


def test_llm_api_view_matches_app_precedence(monkeypatch):
    monkeypatch.delenv("SOCLAAS_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)

    view = _llm_api_view({"llm_api": {"api_key": "config-key"}})
    assert view["api_key"] == "config-key"
    assert view["key_source"] == "config.toml"
    assert view["backend"] == "soclaas"

    monkeypatch.setenv("LLM_API_KEY", "llm-env-key")
    view = _llm_api_view({"llm_api": {"api_key": "config-key"}})
    assert view["api_key"] == "llm-env-key"

    monkeypatch.setenv("SOCLAAS_API_KEY", "soclaas-env-key")
    view = _llm_api_view({"llm_api": {"api_key": "config-key"}})
    assert view["api_key"] == "soclaas-env-key"  # key_env wins over LLM_API_KEY

    monkeypatch.setenv("LLM_BACKEND", "OLLAMA")
    view = _llm_api_view({})
    assert view["backend"] == "ollama"
    assert view["base_url"]  # a default base URL is always available


def test_embeddings_backend_view_matches_app_precedence(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_BACKEND", raising=False)

    # Absent [embeddings].backend = the app default: local Ollama, even when
    # chat runs on soclaas (the split deployment).
    assert _embeddings_backend_view({}, "soclaas") == "ollama"
    # Explicit "" inherits the chat backend.
    assert _embeddings_backend_view({"embeddings": {"backend": ""}}, "ollama") == "ollama"
    assert _embeddings_backend_view({"embeddings": {"backend": ""}}, "soclaas") == "soclaas"
    # Explicit value wins over both the default and the chat backend.
    assert _embeddings_backend_view({"embeddings": {"backend": "soclaas"}}, "ollama") == "soclaas"
    # Invalid configured value falls back to the default (warn-and-ignore).
    assert _embeddings_backend_view({"embeddings": {"backend": "gpu"}}, "soclaas") == "ollama"
    # Env override wins over the configured value.
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "soclaas")
    assert _embeddings_backend_view({"embeddings": {"backend": "ollama"}}, "ollama") == "soclaas"


def test_dependency_install_decision():
    plan = _decide_dependency_install
    # Deps present -> never install (unless explicitly asked).
    assert plan(dependencies_ready=True, install_deps=False, no_install_deps=False,
                check_only=False, non_interactive=False) is False
    assert plan(dependencies_ready=True, install_deps=True, no_install_deps=False,
                check_only=False, non_interactive=False) is True
    # Missing deps + non-interactive -> auto-install (one-click behavior).
    assert plan(dependencies_ready=False, install_deps=False, no_install_deps=False,
                check_only=False, non_interactive=True) is True
    # Opt-outs are honored.
    assert plan(dependencies_ready=False, install_deps=False, no_install_deps=True,
                check_only=False, non_interactive=True) is False
    assert plan(dependencies_ready=False, install_deps=False, no_install_deps=False,
                check_only=True, non_interactive=True) is False
    # Interactive -> asks.
    asked = []
    assert plan(dependencies_ready=False, install_deps=False, no_install_deps=False,
                check_only=False, non_interactive=False,
                prompt_yes_no=lambda label, default: asked.append(label) or True) is True
    assert asked


def test_main_wraps_keyboard_interrupt_and_eof(monkeypatch):
    from scripts import setup_instance

    def interrupt(argv=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(setup_instance, "_main", interrupt)
    assert main([]) == 130

    def no_stdin(argv=None):
        raise EOFError()

    monkeypatch.setattr(setup_instance, "_main", no_stdin)
    assert main([]) == 1


def test_main_wraps_unexpected_errors(monkeypatch):
    from scripts import setup_instance

    def boom(argv=None):
        raise RuntimeError("wizard exploded")

    monkeypatch.setattr(setup_instance, "_main", boom)
    assert main([]) == 1


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

    assert uploads == []          # no upload happened
    assert remote_commands == []  # no staging/extract/build/activate happened


def test_provision_config_only_change_syncs_config_alone(safe_tmp_path, monkeypatch):
    """A config.toml-only change must not re-upload the archive or the SIF.

    config.toml is rewritten by every setup run (API key, ports). Without the
    sans-config freshness comparison, `--set-api-key` followed by start.cmd
    would re-upload the full source archive AND the multi-GB SIF over
    non-resumable Windows scp -- and wipe any corpus still stored inside the
    repo directory via the repo replacement.
    """
    from scripts.setup_instance import _build_repo_manifest, provision_hpc_cluster

    src = safe_tmp_path / "repo"
    src.mkdir()
    (src / "main.py").write_text("print('same')\n", encoding="utf-8")
    (src / "config.toml").write_text("old = true\n", encoding="utf-8")
    deployed_manifest = _build_repo_manifest(src)

    # Now change ONLY config.toml locally.
    (src / "config.toml").write_text("api_key = \"new\"\n", encoding="utf-8")
    monkeypatch.setattr("scripts.setup_instance.ROOT", src)

    def fake_subprocess_run(cmd, **kwargs):
        remote = cmd[-1] if isinstance(cmd, list) and len(cmd) > 1 else ""
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        if remote.startswith("cat ") and ".rag_manifest" in remote:
            class RM:
                returncode = 0
                stdout = deployed_manifest
                stderr = ""
            return RM()
        return R()  # test -s <sif> -> SIF exists

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

    # Exactly two one-file syncs: config.toml and the fresh manifest.
    uploaded_names = sorted(path.name for path, _, _ in uploads)
    assert uploaded_names == [".rag_manifest", "config.toml"]
    # No staging/extract/activate cycle ran.
    assert remote_commands == []


def _provision_sif_test_harness(safe_tmp_path, monkeypatch, *, remote_sha):
    """Shared scaffolding for the SIF dedup tests; returns (uploads, commands)."""
    from scripts.setup_instance import provision_hpc_cluster

    src = safe_tmp_path / "repo"
    src.mkdir()
    (src / "main.py").write_text("print('v2')\n", encoding="utf-8")
    monkeypatch.setattr("scripts.setup_instance.ROOT", src)

    # Remote manifest fetch fails -> source considered stale -> full provision.
    # The SIF existence probe (`test -s`) inside the dedup helper must succeed.
    def fake_subprocess_run(cmd, **kwargs):
        remote = cmd[-1] if isinstance(cmd, list) and len(cmd) > 1 else ""
        class R:
            def __init__(self, rc):
                self.returncode = rc
            stdout = ""
            stderr = ""
        return R(0 if remote.startswith("test -s ") else 1)

    monkeypatch.setattr("scripts.setup_instance.subprocess.run", fake_subprocess_run)

    local_sif = safe_tmp_path / "rag_pipeline_cpu.sif"
    local_sif.write_bytes(b"fake-sif-bytes")
    from scripts.setup_instance import _file_sha256
    digest = _file_sha256(local_sif)
    monkeypatch.setattr(
        "scripts.setup_instance._remote_sha256",
        lambda alias, path: digest if remote_sha == "match" else "0" * 64,
    )

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
        local_image=local_sif,
        force=False,
    )
    return uploads, remote_commands


def test_provision_reuses_matching_remote_sif(safe_tmp_path, monkeypatch):
    """Checksum-identical remote SIF is cp'd from the old repo, not re-uploaded."""
    uploads, remote_commands = _provision_sif_test_harness(
        safe_tmp_path, monkeypatch, remote_sha="match"
    )
    sif_uploads = [u for u in uploads if u[0].name == "rag_pipeline_cpu.sif"]
    assert sif_uploads == [], "unchanged SIF must not be re-uploaded"
    assert any("cp " in command and "rag_pipeline_cpu.sif" in command
               for command in remote_commands)


def test_provision_reuploads_sif_when_checksum_differs(safe_tmp_path, monkeypatch):
    uploads, remote_commands = _provision_sif_test_harness(
        safe_tmp_path, monkeypatch, remote_sha="differ"
    )
    sif_uploads = [u for u in uploads if u[0].name == "rag_pipeline_cpu.sif"]
    assert sif_uploads, "a changed SIF must be uploaded"
    assert not any("cp " in command for command in remote_commands)


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
        alias="cpu-login",
        user="student",
        storage_root="/hpctmp/student",
        relative_repo="rag-cpu",
        archive_path=safe_tmp_path / "repository.tar.gz",
        image_name="rag_pipeline_cpu.sif",
        definition_name="Singularity.cpu.def",
        local_image=None,
        force=True,
    )

    # Even with a matching manifest, force=True provisions.
    assert uploads != []
    assert ssh_calls == []  # force short-circuits the manifest fetch entirely
    joined = "\n".join(remote_commands)
    assert "tar -xzf" in joined


def _hpc_values(**overrides) -> "SetupValues":
    base = dict(
        mode="hpc",
        server_host="127.0.0.1",
        server_port=8000,
        cpu_host="cpu-login",
        cpu_repo="rag-cpu",
    )
    base.update(overrides)
    return SetupValues(**base)


def test_run_checks_ollama_fails_in_hpc_mode(safe_tmp_path, monkeypatch):
    """Embeddings are workstation-local in EVERY mode: a missing Ollama must
    fail preflight even when parsing is delegated to the cluster."""
    import scripts.setup_instance as si

    config = safe_tmp_path / "config.toml"
    config.write_text(
        "[server]\nhost = \"127.0.0.1\"\nport = 8000\n\n[hpc]\nenabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(si, "_http_ready", lambda url, timeout=2.0: False)
    ok = si.run_checks(config, _hpc_values(), cluster_required=False)
    assert ok is False  # Local Ollama FAIL (embeddings default to ollama)


def test_run_checks_cluster_down_warns_when_not_required(safe_tmp_path, monkeypatch):
    """Daily serving must not be blocked by a campus-cluster outage."""
    import scripts.setup_instance as si

    config = safe_tmp_path / "config.toml"
    config.write_text(
        "[server]\nhost = \"127.0.0.1\"\nport = 8000\n\n[hpc]\nenabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(si, "_http_ready", lambda url, timeout=2.0: True)
    monkeypatch.setattr(si, "_ollama_model_installed", lambda port, model: True)
    monkeypatch.setattr(si, "_command_check", lambda name: (False, "not found"))
    monkeypatch.setattr(si, "_runtime_dependencies_ready", lambda python=None: True)

    class Unreachable:
        def check_connections(self):
            return {"cpu": {"ok": False, "configured": True, "detail": "SSH refused"}}

    monkeypatch.setattr("src.hpc_backend.HpcBackend", lambda cfg: Unreachable())
    # Not provisioning/parsing: cluster problems degrade to WARN -> start OK.
    assert si.run_checks(config, _hpc_values(), cluster_required=False) is True
    # Provisioning or --initial-corpus will need the cluster: FAIL -> blocked.
    assert si.run_checks(config, _hpc_values(), cluster_required=True) is False


def test_run_checks_ollama_warns_when_model_missing(safe_tmp_path, monkeypatch, capsys):
    import scripts.setup_instance as si

    config = safe_tmp_path / "config.toml"
    config.write_text(
        "[server]\nhost = \"127.0.0.1\"\nport = 8000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(si, "_http_ready", lambda url, timeout=2.0: True)
    monkeypatch.setattr(si, "_ollama_model_installed", lambda port, model: False)
    monkeypatch.setattr(si, "_command_check", lambda name: (True, "found"))
    monkeypatch.setattr(si, "_runtime_dependencies_ready", lambda python=None: True)
    values = SetupValues(mode="local", server_host="127.0.0.1", server_port=8000)
    assert si.run_checks(config, values) is True  # WARN, not FAIL
    assert "ollama pull all-minilm" in capsys.readouterr().out


def test_fresh_config_local_mode_notice(safe_tmp_path, capsys, monkeypatch):
    """A non-interactive run on a fresh checkout must say it defaulted to
    LOCAL mode instead of silently parsing everything on the workstation."""
    monkeypatch.chdir(safe_tmp_path)
    config = safe_tmp_path / "config.toml"
    rc = main([
        "--non-interactive", "--configure-only", "--skip-checks",
        "--no-install-deps", "--mode", "local",
        "--config", str(config),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LOCAL mode" in out
    assert "setup.cmd" in out


def test_initial_corpus_rejected_in_local_mode(safe_tmp_path, capsys, monkeypatch):
    """--initial-corpus with a local-mode config fails fast with guidance."""
    monkeypatch.chdir(safe_tmp_path)
    config = safe_tmp_path / "config.toml"
    config.write_text("[server]\nhost = \"127.0.0.1\"\nport = 8000\n", encoding="utf-8")
    corpus_zip = safe_tmp_path / "corpus.zip"
    corpus_zip.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    rc = main([
        "--non-interactive", "--skip-checks", "--no-install-deps",
        "--config", str(config), "--initial-corpus", str(corpus_zip),
    ])
    assert rc == 1
    assert "hpc" in capsys.readouterr().err.lower()


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

