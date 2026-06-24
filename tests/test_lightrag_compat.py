import asyncio
import sys
import types

from src.lightrag_compat import LightRAGApi, local_ollama_model_complete, register_vector_storage


def test_register_vector_storage_marks_storage_as_vector_compatible():
    api = LightRAGApi(
        LightRAG=object,
        QueryParam=object,
        EmbeddingFunc=object,
        STORAGES={},
        STORAGE_IMPLEMENTATIONS={
            "VECTOR_STORAGE": {
                "implementations": ["NanoVectorDBStorage"],
                "required_methods": ["query", "upsert"],
            }
        },
        STORAGE_ENV_REQUIREMENTS={},
        chunking_by_token_size=object(),
        compute_mdhash_id=object(),
        ollama_model_complete=object(),
    )

    register_vector_storage(api, "LanceDBStorage", "src.lancedb_storage")

    assert api.STORAGES["LanceDBStorage"] == "src.lancedb_storage"
    assert api.STORAGE_ENV_REQUIREMENTS["LanceDBStorage"] == []
    assert "LanceDBStorage" in api.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]


def test_local_ollama_adapter_maps_lightrag_kwargs(monkeypatch):
    calls = {}

    class FakeHashingKV:
        global_config = {"llm_model_name": "gemma4"}

    class FakeAsyncClient:
        def __init__(self, host=None, **kwargs):
            calls["host"] = host
            calls["client_kwargs"] = kwargs

        async def chat(self, **kwargs):
            calls["chat_kwargs"] = kwargs
            return {"message": {"content": "ok"}}

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        types.SimpleNamespace(AsyncClient=FakeAsyncClient),
    )

    response = asyncio.run(
        local_ollama_model_complete(
            "prompt",
            system_prompt="system",
            history_messages=[{"role": "assistant", "content": "history"}],
            hashing_kv=FakeHashingKV(),
            entity_extraction=True,
            temperature=0.2,
            max_tokens=64,
            timeout=0,
        )
    )

    assert response == "ok"
    assert calls["host"] is None
    assert calls["client_kwargs"] == {}
    assert calls["chat_kwargs"]["model"] == "gemma4"
    assert calls["chat_kwargs"]["format"] == "json"
    assert calls["chat_kwargs"]["options"] == {
        "temperature": 0.2,
        "num_predict": 64,
    }
    assert calls["chat_kwargs"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "history"},
        {"role": "user", "content": "prompt"},
    ]
