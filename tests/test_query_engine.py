import asyncio

import src.query as query


class FakeQueryParam:
    def __init__(self, mode):
        self.mode = mode


class FakeRag:
    def __init__(self):
        self.initialized = False

    async def initialize_storages(self):
        self.initialized = True

    async def aquery(self, question, param):
        assert self.initialized
        return f"{question}:{param.mode}"


def test_query_engine_uses_lazy_query_param(monkeypatch):
    monkeypatch.setattr(query, "make_query_param", FakeQueryParam)

    engine = query.QueryEngine.__new__(query.QueryEngine)
    engine.rag = FakeRag()

    result = asyncio.run(engine._async_ask("question", "hybrid"))

    assert result == "question:hybrid"
