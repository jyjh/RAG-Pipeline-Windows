"""DEPRECATED: legacy SQLite-backed v1 pipeline module.

Not imported by the active web app or the LanceDB-based ingestion/indexing
pipeline. Retained for legacy test coverage. See src/store.py for the full
deprecation note. New work belongs in the LanceDB-backed modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import PipelineConfig, load_config
from src.generation import QueryService
from src.indexing_v1 import StructuredIndexer
from src.ingestion_v1 import StructuredIngestor
from src.store import SQLiteBlockStore


def create_app(config: PipelineConfig | None = None):
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field

    cfg = config or load_config()
    store = SQLiteBlockStore(cfg.paths.db_dir)
    query_service = QueryService(cfg, store)
    app = FastAPI(title="Local FSAE RAG Pipeline", version="1.0.0")

    class QueryRequest(BaseModel):
        question: str = Field(..., min_length=1)
        mode: str = "hybrid"
        top_k: int | None = None
        document_ids: list[str] | None = None

    class IngestRequest(BaseModel):
        path: str | None = None

    def require_token(x_api_token: str | None = Header(default=None)) -> None:
        if cfg.server.lan and cfg.server.api_token and x_api_token != cfg.server.api_token:
            raise HTTPException(status_code=401, detail="Invalid API token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "db": str(Path(cfg.paths.db_dir) / "rag.sqlite"),
                "documents": len(store.list_documents())}

    @app.get("/documents")
    def documents(_: None = Depends(require_token)) -> list[dict[str, Any]]:
        return [doc.to_dict() for doc in store.list_documents()]

    @app.get("/documents/{doc_id}/blocks")
    def blocks(doc_id: str, _: None = Depends(require_token)) -> list[dict[str, Any]]:
        return [block.to_dict() for block in store.list_blocks(doc_id)]

    @app.get("/assets/{asset_id}")
    def asset(asset_id: str, _: None = Depends(require_token)):
        record = store.get_asset(asset_id)
        if record is None or not Path(record.path).exists():
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(record.path, media_type=record.mime_type)

    @app.post("/ingest")
    def ingest(request: IngestRequest, _: None = Depends(require_token)) -> dict[str, Any]:
        return {"document_ids": StructuredIngestor(cfg, store).ingest_path(request.path or cfg.paths.data_dir)}

    @app.post("/index")
    def index(_: None = Depends(require_token)) -> dict[str, Any]:
        return StructuredIndexer(cfg, store).rebuild()

    @app.post("/query")
    def query(request: QueryRequest, _: None = Depends(require_token)) -> dict[str, Any]:
        return query_service.ask(request.question, request.mode, request.top_k, request.document_ids).to_dict()

    return app


def run_app(config_path: str | None = None, host: str | None = None, port: int | None = None,
            config: PipelineConfig | None = None) -> None:
    import uvicorn
    cfg = config or load_config(config_path)
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port)

