from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.setup_instance import (
    SetupValues,
    _validate,
    configure,
    configure_ssh_aliases,
    ensure_ssh_private_key,
    install_ssh_public_key,
    _login_relative_repo_default,
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
