from src.chunking import build_chunks_for_blocks
from src.config import ChunkingConfig
from src.schema import BlockRecord, Citation, QueryResponse, RetrievedBlock


def test_duplicate_chunks_are_grouped_not_removed():
    blocks = [
        BlockRecord(f"block_{i}", "doc_a", 1, "text", i,
                    markdown="Identical setup note about damper shims and preload.")
        for i in range(1, 4)
    ]
    result = build_chunks_for_blocks("doc_a", blocks, ChunkingConfig(max_tokens=8, overlap_tokens=0))
    assert len(result.chunks) == 3
    assert len(set(chunk.duplicate_group_id for chunk in result.chunks)) == 1


def test_query_response_shape_contains_public_api_fields():
    citation = Citation("doc", "Setup Notes", 2, "block", "table", 0.9, "Spring rate table.", "/assets/asset")
    retrieved = RetrievedBlock("block", "chunk", "doc", "Setup Notes", 2, "table", 0.9, "Spring rate table.", "/assets/asset")
    payload = QueryResponse("Use the cited spring rate table. [C1]", [citation], [retrieved], {"total_seconds": 0.1}).to_dict()
    assert set(payload) == {"answer", "citations", "retrieved_blocks", "timings"}
    assert payload["citations"][0]["block_id"] == "block"
    assert payload["citations"][0]["asset_url"] == "/assets/asset"

