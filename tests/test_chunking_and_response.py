from src.chunking import build_chunks_for_blocks
from src.config import ChunkingConfig
from src.schema import BlockRecord, Citation, QueryResponse, RetrievedBlock


def test_duplicate_chunks_are_grouped_not_removed():
    blocks = [
        BlockRecord(
            block_id=f"block_{i}",
            doc_id="doc_a",
            page_num=1,
            modality="text",
            reading_order=i,
            markdown="Identical setup note about damper shims and preload.",
        )
        for i in range(1, 4)
    ]
    result = build_chunks_for_blocks(
        "doc_a",
        blocks,
        ChunkingConfig(max_tokens=8, overlap_tokens=0),
    )

    assert len(result.chunks) == 3
    assert len(set(chunk.duplicate_group_id for chunk in result.chunks)) == 1


def test_query_response_shape_contains_public_api_fields():
    citation = Citation(
        doc_id="doc",
        document_title="Setup Notes",
        page=2,
        block_id="block",
        modality="table",
        score=0.9,
        snippet="Spring rate table.",
        asset_url="/assets/asset",
    )
    retrieved = RetrievedBlock(
        block_id="block",
        chunk_id="chunk",
        doc_id="doc",
        document_title="Setup Notes",
        page=2,
        modality="table",
        score=0.9,
        text="Spring rate table.",
        asset_url="/assets/asset",
    )

    payload = QueryResponse(
        answer="Use the cited spring rate table. [C1]",
        citations=[citation],
        retrieved_blocks=[retrieved],
        timings={"total_seconds": 0.1},
    ).to_dict()

    assert set(payload) == {"answer", "citations", "retrieved_blocks", "timings"}
    assert payload["citations"][0]["block_id"] == "block"
    assert payload["citations"][0]["asset_url"] == "/assets/asset"

