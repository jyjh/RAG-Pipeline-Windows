# Local FSAE RAG Pipeline

Local retrieval-augmented generation pipeline for NUS FSAE knowledge transfer. The project ingests technical PDFs and notes, extracts text/tables/equations/figures, builds a local retrieval index, and answers questions without cloud APIs.

## Goals And Constraints

- Keep all document processing, embeddings, retrieval, and generation local.
- Support STEM and engineering documents: textbooks, lecture notes, research papers, reports, scanned PDFs, tables, figures, charts, and equations.
- Preserve enough source context for users to inspect where an answer came from.
- Target workstation: Ryzen 9 9950X3D, RTX 4000 Ada 20GB VRAM, 128GB RAM, Windows 11 or WSL2.
- Optimize for reliable technical knowledge transfer between FSAE batches.

## Current Capabilities

The current source tree contains the LightRAG-based local STEM pipeline:

- `src/ingestion.py`: parses PDFs with Docling and exports enriched Markdown.
- `src/indexing.py`: inserts Markdown into LightRAG, builds vector/graph stores, bridge edges, and quotient graph artifacts.
- `src/query.py`: queries the local LightRAG index.
- `src/embeddings.py`: loads Nomic embeddings on CPU with document/query prefix handling.
- `src/utils.py`: manages Ollama model loading/unloading, DeepSeek tokenizer alignment, and LightRAG construction.
- `src/lancedb_storage.py`: local LanceDB vector-storage adapter for LightRAG.

Implemented behavior described by the old planning docs:

- Docling PDF layout parsing with CUDA acceleration where available.
- Lazy `qwen2.5vl:7b` vision enrichment for figures/charts/diagrams.
- STEM-focused vision prompt requesting LaTeX equations, chart axes/values, and component labels.
- Inline vision-description injection so figures remain near surrounding explanatory text.
- Local Ollama inference using `deepseek-r1:32b`.
- CPU-bound `nomic-ai/nomic-embed-text-v1.5` embeddings.
- 768-dimensional normalized vectors for retrieval.
- DeepSeek/Qwen tokenizer alignment for chunk sizing.
- LanceDB-backed vector storage for LightRAG.
- MRL bridge-edge and quotient-graph experiments for future cross-document synthesis.

The `processed_docs/` Markdown files are corpus data generated from sample PDFs, not project documentation.

## Important Working Tree Note

At the time this README was consolidated, the `D:` workspace showed several tracked non-document files as deleted, including `main.py`, `requirements.txt`, `config.example.toml`, and newer structured-v1 modules/tests. This README documents the files that are currently visible in `D:\Documents\GitHub\RAG-Pipeline-Windows`. Restore the deleted tracked files before relying on commands that reference them.

## Expected Repository Layout

```text
data/                 Input PDFs
processed_docs/       Generated/enriched Markdown corpus files
db/                   Generated local database/index artifacts
src/                  Pipeline source modules
tests/                Tests, if restored in the working tree
README.md             Canonical project documentation
```

Generated databases, assets, caches, and model artifacts should not be committed.

## Setup

Create an environment:

```bash
conda create -n ragpipeline python=3.11 -y
conda activate ragpipeline
```

Install PyTorch with a CUDA wheel appropriate for the workstation. Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install required packages. If `requirements.txt` has been restored, use:

```bash
pip install -r requirements.txt
```

Otherwise install the core runtime packages manually:

```bash
pip install docling lightrag lancedb ollama sentence-transformers transformers numpy pydantic networkx scikit-learn python-dotenv tqdm
```

Install and verify Ollama:

```bash
ollama --version
ollama pull deepseek-r1:32b
ollama pull qwen2.5vl:7b
```

The embedding model is loaded through SentenceTransformers:

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")
print(m.encode(["search_document: test"]).shape)
```

## Usage

If `main.py` is restored, the intended CLI flow is:

```bash
python main.py --mode ingest --data_dir data --md_dir processed_docs
python main.py --mode index --md_dir processed_docs --db_dir db
python main.py --mode query --db_dir db --question "Explain the bias-variance tradeoff" --query_mode hybrid
```

The module entrypoints can also be used for the default directories:

```bash
python -m src.ingestion
python -m src.indexing
```

Query modes:

- `local`: vector search over chunks.
- `global`: graph/relationship search.
- `hybrid`: combined retrieval path; this is the default intended mode.

## Architecture

1. **Ingestion**
   - PDFs are parsed by Docling.
   - Tables and structured items are exported as Markdown where supported.
   - Figures/charts are cropped in memory and sent to local Qwen2.5-VL through Ollama.
   - Vision descriptions are inserted inline at the original document position.

2. **Embeddings**
   - Nomic runs on CPU to preserve GPU VRAM for DeepSeek and Qwen.
   - Document chunks use the `search_document:` prefix.
   - Queries use the `search_query:` prefix.
   - Embeddings are L2-normalized.

3. **Indexing**
   - Full Markdown documents are inserted into LightRAG.
   - LightRAG chunks text using a tokenizer aligned to DeepSeek-R1-Distill-Qwen-32B.
   - LanceDB stores vectors locally.
   - Optional bridge edges and quotient graph artifacts support future cross-domain exploration.

4. **Querying**
   - Questions are embedded locally.
   - LightRAG retrieves vector and graph context.
   - DeepSeek-R1 synthesizes an answer from retrieved context.

## Hardware And Runtime Notes

- DeepSeek-R1 32B at Q4 quantization is close to the RTX 4000 Ada 20GB VRAM limit.
- Qwen2.5-VL is loaded lazily during ingestion only when figures are detected.
- Ollama `keep_alive="0"` can be used to evict inactive models between phases.
- Nomic embeddings run on CPU to avoid competing with inference models for VRAM.
- If indexing causes VRAM pressure, reduce LightRAG entity extraction gleaning or lower context/chunk sizes.

## Validation

Recommended validation once the working tree is restored:

```bash
python -m compileall src
python -m src.ingestion
python -m src.indexing
python main.py --mode query --question "What sources discuss regularization?"
```

Validation scenarios to maintain:

- Born-digital PDF text extraction.
- Scanned PDF OCR.
- Mixed text/image PDF.
- Table-heavy pages.
- Equation-heavy pages.
- Figure/chart-heavy pages.
- Duplicate or near-duplicate content across documents.
- Query answers with inspectable source evidence.
- Offline execution after model weights and Python packages are cached locally.

## Roadmap

Near term:

- Restore or regenerate missing tracked runtime files in the `D:` workspace if needed.
- Validate the current LightRAG path end to end on the target workstation.
- Pin package versions after successful validation.
- Add real golden PDF fixtures for scanned, mixed, table, equation, and figure-heavy documents.
- Benchmark pages/minute, indexing time, query latency, VRAM/RAM usage, and retrieval quality.

Medium term:

- Add a citation-safe structured block store as the durable source of truth.
- Add a local FastAPI service and browser UI for team members.
- Add local reranking and retrieval evaluation.
- Preserve all source block/page references through chunking and deduplication.

Graph/advanced retrieval:

- Keep LightRAG graph retrieval as an optional synthesis layer.
- Ensure any bridge edges are injected into retrieval paths actually used at query time.
- Use quotient graph artifacts only through a deliberate query preprocessor, not by polluting entity namespaces.

## Development Notes

- Do not use cloud APIs or hosted models.
- Do not commit generated `db/`, `processed_docs/` test outputs, model caches, or Python cache files unless intentionally curating a fixture.
- Keep the README as the only root-level project documentation file.
- Treat `processed_docs/*.md` as corpus data, not documentation.

