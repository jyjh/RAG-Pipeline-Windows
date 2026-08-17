import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.config import load_config, OllamaConfig, PipelineConfig
import src.local_rag as local_rag
import src.web_app as web_app


def test_ollama_config_defaults_and_toml_loading(tmp_path):
    cfg = PipelineConfig()
    assert isinstance(cfg.ollama, OllamaConfig)
    assert cfg.ollama.host == "http://127.0.0.1:11434"
    assert cfg.ollama.hosts == []
    assert cfg.ollama.fallback_enabled is True
    assert cfg.ollama.chat_health_check_interval_seconds == 5.0
    assert cfg.ollama.chat_max_lost_health_checks == 5

    toml_content = """
[ollama]
host = "http://remote-gpu:11434"
hosts = ["http://backup-1:11434", "http://backup-2:11434"]
fallback_enabled = false
chat_health_check_interval_seconds = 2.0
chat_max_lost_health_checks = 3
"""
    config_file = tmp_path / "config.toml"
    config_file.write_text(toml_content, encoding="utf-8")

    loaded = load_config(config_file)
    assert loaded.ollama.host == "http://remote-gpu:11434"
    assert loaded.ollama.hosts == ["http://backup-1:11434", "http://backup-2:11434"]
    assert loaded.ollama.fallback_enabled is False
    assert loaded.ollama.chat_health_check_interval_seconds == 2.0
    assert loaded.ollama.chat_max_lost_health_checks == 3


def test_ollama_host_precedence(monkeypatch, tmp_path):
    local_rag._ACTIVE_OLLAMA_HOST = None

    # Case 4: Default fallback when no env, no active, no config file
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("RAG_PIPELINE_CONFIG", raising=False)
    assert local_rag._ollama_host() == "http://127.0.0.1:11434"

    # Case 3: Config file host
    toml_content = '[ollama]\nhost = "http://toml-gpu:11434"\n'
    cfg_path = tmp_path / "test_config.toml"
    cfg_path.write_text(toml_content, encoding="utf-8")
    monkeypatch.setenv("RAG_PIPELINE_CONFIG", str(cfg_path))
    assert local_rag._ollama_host() == "http://toml-gpu:11434"

    # Case 2: Active dynamic host overrides config file host
    local_rag._ACTIVE_OLLAMA_HOST = "http://dynamic-failover:11434"
    assert local_rag._ollama_host() == "http://dynamic-failover:11434"

    # Case 1: OLLAMA_HOST env var overrides active dynamic host & config file
    monkeypatch.setenv("OLLAMA_HOST", "http://env-override:11434")
    assert local_rag._ollama_host() == "http://env-override:11434"

    # Clean up state
    local_rag._ACTIVE_OLLAMA_HOST = None


def test_probe_ollama_endpoints(monkeypatch):
    def fake_healthy(host, timeout=3.0):
        return host in {"http://healthy-1:11434", "http://healthy-2:11434"}

    monkeypatch.setattr(local_rag, "_ollama_server_healthy", fake_healthy)

    hosts = ["http://unhealthy:11434", "http://healthy-1:11434", "http://healthy-2:11434"]
    result = local_rag.probe_ollama_endpoints(hosts)
    assert result == "http://healthy-1:11434"

    all_unhealthy = ["http://down-1:11434", "http://down-2:11434"]
    assert local_rag.probe_ollama_endpoints(all_unhealthy) is None
    assert local_rag.probe_ollama_endpoints([]) is None


def test_wait_for_ollama_recovery_failover(monkeypatch):
    local_rag._ACTIVE_OLLAMA_HOST = "http://dead-primary:11434"

    def fake_healthy(host, timeout=3.0):
        return host == "http://backup-gpu:11434"

    monkeypatch.setattr(local_rag, "_ollama_server_healthy", fake_healthy)

    candidate_hosts = ["http://dead-primary:11434", "http://backup-gpu:11434"]
    recovered = local_rag._wait_for_ollama_recovery(
        health_check_interval=0.01,
        max_lost_health_checks=1,
        candidate_hosts=candidate_hosts,
    )
    assert recovered is True
    assert local_rag._ACTIVE_OLLAMA_HOST == "http://backup-gpu:11434"

    # Clean up state
    local_rag._ACTIVE_OLLAMA_HOST = None


def test_wait_for_ollama_recovery_updates_env_var(monkeypatch):
    local_rag._ACTIVE_OLLAMA_HOST = "http://dead-primary:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://dead-primary:11434")

    def fake_healthy(host, timeout=3.0):
        return host == "http://backup-gpu:11434"

    monkeypatch.setattr(local_rag, "_ollama_server_healthy", fake_healthy)

    candidate_hosts = ["http://dead-primary:11434", "http://backup-gpu:11434"]
    recovered = local_rag._wait_for_ollama_recovery(
        health_check_interval=0.01,
        max_lost_health_checks=1,
        candidate_hosts=candidate_hosts,
    )
    assert recovered is True
    assert local_rag._ACTIVE_OLLAMA_HOST == "http://backup-gpu:11434"
    assert os.environ.get("OLLAMA_HOST") == "http://backup-gpu:11434"
    assert local_rag._ollama_host() == "http://backup-gpu:11434"

    # Clean up state
    local_rag._ACTIVE_OLLAMA_HOST = None



def test_web_app_load_chat_config(tmp_path):
    toml_content = """
[chat]
context_window = 4096
llm_num_predict = 2048

[ollama]
host = "http://chat-gpu:11434"
hosts = ["http://chat-backup:11434"]
fallback_enabled = true
chat_health_check_interval_seconds = 3.0
chat_max_lost_health_checks = 4
"""
    cfg_file = tmp_path / "chat_config.toml"
    cfg_file.write_text(toml_content, encoding="utf-8")

    chat_cfg = web_app._load_chat_config(cfg_file)
    assert chat_cfg["ollama_host"] == "http://chat-gpu:11434"
    assert chat_cfg["ollama_hosts"] == ["http://chat-backup:11434"]
    assert chat_cfg["ollama_fallback_enabled"] is True
    assert chat_cfg["ollama_health_check_interval"] == 3.0
    assert chat_cfg["ollama_max_lost_health_checks"] == 4


def test_health_and_metrics_endpoints_ollama_status(monkeypatch):
    # Exercise the dormant Ollama branch of the status snapshot.
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setattr(
        local_rag,
        "_ollama_host",
        lambda: "http://127.0.0.1:11434",
    )
    monkeypatch.setattr(
        local_rag,
        "_get_ollama_candidate_hosts",
        lambda: ["http://127.0.0.1:11434", "http://remote-gpu:11434"],
    )
    monkeypatch.setattr(
        local_rag,
        "_ollama_server_healthy",
        lambda host=None, timeout=1.5: host == "http://127.0.0.1:11434",
    )

    client = TestClient(web_app.app)

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["ollama_active_host"] == "http://127.0.0.1:11434"
    assert data_health["ollama_candidate_hosts"] == ["http://127.0.0.1:11434", "http://remote-gpu:11434"]
    assert data_health["ollama_reachability"] == {
        "http://127.0.0.1:11434": True,
        "http://remote-gpu:11434": False,
    }

    res_metrics = client.get("/api/metrics")
    assert res_metrics.status_code == 200
    data_metrics = res_metrics.json()
    assert data_metrics["ollama_active_host"] == "http://127.0.0.1:11434"
    assert data_metrics["ollama_candidate_hosts"] == ["http://127.0.0.1:11434", "http://remote-gpu:11434"]
    assert data_metrics["ollama_reachability"] == {
        "http://127.0.0.1:11434": True,
        "http://remote-gpu:11434": False,
    }
