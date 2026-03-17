# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local RAG (Retrieval-Augmented Generation) pipeline for querying STEM documents (textbooks, PDFs, research papers). Entirely local with no cloud dependencies. Optimized for: Ryzen 9 9950X3D + RTX 4000 Ada 20GB + 128GB RAM.

## Setup

```bash
# Pull required Ollama models
ollama pull deepseek-r1:32b
ollama pull qwen2.5vl:7b
ollama pull nomic-embed-text

# Activate conda environment
conda activate RAG_env

pip install -r requirements.txt
```

## Running the Pipeline

```bash
# Phase 1: Convert PDFs to enriched markdown
python main.py --mode ingest --data_dir data

# Phase 2: Build knowledge graph + vector index from markdown
python main.py --mode index --md_dir processed_docs --db_dir db

# Phase 3: Query
python main.py --mode query --question "Your question here" --query_mode hybrid

# Run all phases end-to-end
python main.py --mode all
```

Query modes: `local` (vector ANN), `global` (knowledge graph), `hybrid` (both).

## Architecture

The pipeline has three sequential phases managed by `main.py`:

**Ingestion** (`src/ingestion.py`): Docling (CUDA) parses PDFs into markdown. Figures/charts are detected, base64-encoded, sent to Qwen2.5-VL for STEM descriptions, then injected as blockquotes at their original layout position.

**Indexing** (`src/indexing.py` + `src/utils.py`): Full markdown documents are inserted into LightRAG (not pre-chunked) to preserve document-level hierarchy. A custom `chunking_func` uses the DeepSeek Qwen2 tokenizer for accurate token counts and performs LanceDB ANN dedup (cosine > 0.90) to filter near-duplicate chunks before NER dispatch. DeepSeek-R1 extracts entities on GPU while Nomic-Embed-Text-v1.5 generates 768d vectors on CPU concurrently. MRL bridge edges (256d) are dual-injected into both `.graphml` and `relationships_vdb` for full hybrid query coverage.

**Querying** (`src/query.py`): A query is embedded (768d, CPU, with `"search_query: "` prefix via `contextvars`), used for ANN search in LanceDB (local mode), traversal in the knowledge graph (global mode), or both merged into a large context window for DeepSeek-R1 Chain-of-Thought reasoning (hybrid mode).

## Key Design Decisions

**VRAM management**: The RTX 4000 Ada has 20GB. Models are swapped sequentially via `manage_vram()` in `src/utils.py`, which unloads inactive models via `ollama.generate(..., keep_alive="0")` before loading the next one.

**CPU-bound embedding**: Nomic is intentionally kept on CPU to free VRAM for LLMs. It uses `asyncio.to_thread()` to run concurrently with GPU-based DeepSeek-R1 entity extraction during indexing.

**Full 768d vectors**: The pipeline retains all 768 dimensions from Nomic (no truncation for indexing/querying). Bridge edges and quotient graph use 256d MRL truncation. L2 normalization is always applied after truncation.

**Asymmetric prefixing (contextvars)**: Nomic requires `"search_document: "` for indexed content and `"search_query: "` for queries. A `contextvars.ContextVar` flag (`embedding_mode`) deterministically switches the prefix — no heuristics. Set to `"document"` during indexing, `"query"` during retrieval.

**Tokenizer alignment**: LightRAG's default tiktoken `o200k_base` (200K vocab) is replaced with a DeepSeek Qwen2 tokenizer shim (152K vocab) so chunk token counts match the executing LLM's actual token budget.

**LanceDB storage adapter**: LightRAG-HKU 1.4.10 has no built-in LanceDB backend. `src/lancedb_storage.py` implements `BaseVectorStorage` and is registered via monkey-patching `lightrag.kg.STORAGES` before instance construction.

**Embedding cache**: `EmbeddingEngine._cache` stores vectors keyed on `(text, dim, prefix)`. The custom `chunking_func` embeds chunks for dedup; LightRAG's subsequent embedding call is a cache hit — zero redundant computation.

**Chunk size**: 1200 tokens with 200-token overlap (measured by DeepSeek Qwen2 tokenizer).

**Async patterns**: `asyncio.run()` wraps LightRAG's async API for CLI usage. If integrating into FastAPI/Streamlit in the future, refactor to native `await` — `asyncio.run()` will conflict with existing event loops.

## Directory Structure

- `data/` — Input PDFs
- `processed_docs/` — Enriched markdown output from ingestion
- `db/` — LightRAG working directory (vector index, knowledge graph, metadata)
- `db/lancedb/` — LanceDB tables (entities, relationships, chunks, bridge entities)
- `src/` — Pipeline modules:
  - `ingestion.py` — PDF parsing + vision enrichment (Docling + Qwen2.5-VL)
  - `embeddings.py` — `EmbeddingEngine` (Nomic MRL, CPU, cached) + `embedding_mode` contextvar
  - `indexing.py` — `Indexer` with custom chunking_func, bridge edge dual injection, quotient graph
  - `query.py` — `QueryEngine` with contextvars prefix switching
  - `utils.py` — VRAM management, LightRAG factory, DeepSeek tokenizer shim, LanceDB registration
  - `lancedb_storage.py` — Custom `BaseVectorStorage` adapter for LanceDB
