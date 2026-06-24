import shutil
import uuid
from pathlib import Path

from src.indexing import run_indexing


def test_run_indexing_uses_local_vector_indexer(monkeypatch):
    tmp_path = Path.cwd() / f".tmp_test_indexing_{uuid.uuid4().hex}"
    md_dir = tmp_path / "md"
    md_dir.mkdir(parents=True)
    db_dir = tmp_path / "db"
    try:
        md_dir.joinpath("doc.md").write_text("content", encoding="utf-8")
        calls = {}

        class FakeLocalVectorIndexer:
            def __init__(self, **kwargs):
                calls["init"] = kwargs

            def index_markdown(self, markdown_dir):
                calls["markdown_dir"] = markdown_dir

        monkeypatch.setattr("src.local_rag.LocalVectorIndexer", FakeLocalVectorIndexer)

        run_indexing(
            str(md_dir),
            str(db_dir),
            embedding_model="nomic-embed-text",
            embedding_batch_size=2,
            embedding_timeout=15.0,
            progress_enabled=False,
        )

        assert calls == {
            "init": {
                "working_dir": str(db_dir),
                "embedding_model": "nomic-embed-text",
                "embedding_batch_size": 2,
                "embedding_timeout": 15.0,
                "progress_enabled": False,
            },
            "markdown_dir": str(md_dir),
        }
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
