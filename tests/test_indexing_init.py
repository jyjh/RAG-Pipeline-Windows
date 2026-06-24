import asyncio

from src.indexing import Indexer


class FakeRag:
    def __init__(self):
        self.calls = 0

    async def initialize_storages(self):
        self.calls += 1


def test_indexer_storage_initialization_is_idempotent():
    indexer = Indexer.__new__(Indexer)
    indexer.rag = FakeRag()
    indexer._storages_initialized = False

    asyncio.run(indexer._ensure_storages_initialized())
    asyncio.run(indexer._ensure_storages_initialized())

    assert indexer.rag.calls == 1
    assert indexer._storages_initialized
