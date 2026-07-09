from src.schema import AssetRecord, BlockRecord, DocumentRecord, PageRecord
from src.store import SQLiteBlockStore


def test_store_persists_documents_pages_blocks_and_assets(tmp_path):
    store = SQLiteBlockStore(tmp_path)
    doc = DocumentRecord("doc_test", "Brake System Notes", "brakes.pdf", "abc", page_count=1)
    page = PageRecord(doc.doc_id, 1, "digital", text_chars=120, image_count=1)
    asset = AssetRecord("asset_plot", doc.doc_id, "block_plot", 1, "figure_crop", str(tmp_path / "plot.png"))
    block = BlockRecord("block_plot", doc.doc_id, 1, "figure", 1,
                        markdown="Brake pressure trace with threshold at 45 bar.", asset_id=asset.asset_id)
    store.upsert_document(doc)
    store.upsert_page(page)
    store.upsert_asset(asset)
    store.upsert_block(block)
    assert store.get_document(doc.doc_id).title == "Brake System Notes"
    assert store.list_blocks(doc.doc_id)[0].content_for_index() == block.markdown
    assert store.get_asset(asset.asset_id).path.endswith("plot.png")

