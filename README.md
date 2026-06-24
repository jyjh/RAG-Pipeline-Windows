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
- `src/embeddings.py`: uses local Ollama embeddings by default, with hash and SentenceTransformers alternatives.
- `src/utils.py`: manages Ollama model loading/unloading, DeepSeek tokenizer alignment, and LightRAG construction.
- `src/lancedb_storage.py`: local LanceDB vector-storage adapter for LightRAG.

Implemented behavior described by the old planning docs:

- Docling PDF layout parsing with CUDA acceleration where available.
- Lazy `qwen2.5vl:7b` vision enrichment for figures/charts/diagrams.
- STEM-focused vision prompt requesting LaTeX equations, chart axes/values, and component labels.
- Inline vision-description injection so figures remain near surrounding explanatory text.
- Local Ollama inference using `gemma4`.
- Local Ollama embeddings with `nomic-embed-text` by default; hash fallback and optional CPU-bound SentenceTransformers embeddings.
- 768-dimensional normalized vectors for retrieval.
- DeepSeek/Qwen tokenizer alignment for chunk sizing.
- LanceDB-backed vector storage for LightRAG.
- MRL bridge-edge and quotient-graph experiments for future cross-document synthesis.

The `processed_docs/` Markdown files are corpus data generated from sample PDFs, not project documentation.

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

Install required packages. 
```bash
pip install -r requirements.txt
```

Otherwise install the core runtime packages manually:

```bash
pip install docling lightrag lancedb ollama sentence-transformers transformers numpy pydantic networkx scikit-learn python-dotenv tqdm
```

Install and verify Ollama on Windows:

```powershell
# Use the full path when `ollama` is not on PATH.
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
& $ollama --version
```

Start the local Ollama server. Either launch the Windows tray app:

```powershell
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
```

Or run the server in a dedicated terminal:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_NO_CLOUD = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_KEEP_ALIVE = "5m"
& $ollama serve
```

In a second terminal, verify that the local server responds:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

Download the required local models:

```powershell
& $ollama pull gemma4
& $ollama pull nomic-embed-text
& $ollama pull qwen2.5vl:7b
& $ollama list
```

Verify the embedding endpoint before indexing:

```powershell
$body = @{ model = "nomic-embed-text"; input = "embedding health check" } | ConvertTo-Json
$response = Invoke-RestMethod `
  -Uri "http://127.0.0.1:11434/api/embed" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  -TimeoutSec 30
$response.embeddings[0].Count
```

The final command should print an embedding dimension, normally `768`. If `/api/tags` works but `/api/embed` times out, Ollama is running but the embedding model runner is wedged or stalled; use the recovery steps in [Troubleshooting](#troubleshooting).

The pipeline sends single query embeddings to `/api/embed` with `input` as a string, matching this health check. Multi-chunk indexing batches use an array input only when a batch contains more than one text.

For local embeddings without Hugging Face/SentenceTransformers, use Ollama:

```powershell
python main.py --mode index --md_dir processed_docs --db_dir db --rag_backend local --embedding_batch_size 1 --embedding_timeout 30
```

The SentenceTransformers backend remains available for a cached Hugging Face model:

```bash
python main.py --mode index --md_dir processed_docs --db_dir db --embedding_backend sentence-transformers
```

## Usage

Intended CLI flow is:

```bash
python main.py --mode ingest --data_dir data --md_dir processed_docs
python main.py --mode index --md_dir processed_docs --db_dir db
python main.py --mode query --db_dir db --question "Explain the bias-variance tradeoff" --query_mode hybrid
```

For born-digital PDFs, ingest defaults to pypdf text extraction and does not run Docling asset enrichment unless requested. Use `--asset_triggers images` to enrich pages with embedded images, or `--asset_triggers all` to also enrich table/equation heuristic pages. Ingestion shows per-document and per-page progress bars by default; pass `--no_progress` to disable them.

Indexing defaults to `--rag_backend auto`, `--embedding_backend ollama`, `--embedding_model nomic-embed-text`, and `--tokenizer_backend byte`. Use `--rag_backend local` to bypass LightRAG and use the local vector fallback directly. Use `--embedding_batch_size 1` if Ollama struggles with larger embedding batches, and `--embedding_timeout 30` to fail clearly instead of waiting indefinitely. Use `--embedding_backend hash` as a no-model fallback. Use `--embedding_backend sentence-transformers --embedding_model nomic-ai/nomic-embed-text-v1.5` for the Hugging Face backend; add `--tokenizer_backend deepseek` to use the DeepSeek/Qwen tokenizer; add `--allow_embedding_download` only when network downloads are acceptable.

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
   - Born-digital PDF text is extracted with pypdf by default.
   - Docling is used for scanned/image-only PDFs or targeted asset enrichment.
   - Tables and structured items are exported as Markdown where supported.
   - Figures/charts can be sent to local Qwen2.5-VL through Ollama.

2. **Embeddings**
   - Ollama embeddings run locally by default with `nomic-embed-text`.
   - Hash embeddings are available as a no-model fallback.
   - Optional Nomic embeddings run on CPU to preserve GPU VRAM for inference models.
   - Document chunks use the `search_document:` prefix.
   - Queries use the `search_query:` prefix.
   - Embeddings are L2-normalized.

3. **Indexing**
   - Full Markdown documents are inserted into LightRAG.
   - LightRAG chunks text with a byte tokenizer by default; the DeepSeek/Qwen tokenizer is opt-in.
   - LanceDB stores vectors locally.
   - Optional bridge edges and quotient graph artifacts support future cross-domain exploration.

4. **Querying**
   - Questions are embedded locally.
   - LightRAG retrieves vector and graph context.
   - `gemma4` synthesizes an answer from retrieved context.

## Hardware And Runtime Notes

- DeepSeek-R1 32B at Q4 quantization is close to the RTX 4000 Ada 20GB VRAM limit.
- Qwen2.5-VL is loaded lazily during ingestion only when figures are detected.
- Ollama `keep_alive="0"` can be used to evict inactive models between phases.
- Hash embeddings avoid model-loading stalls; optional Nomic embeddings run on CPU to avoid competing with inference models for VRAM.
- If indexing causes VRAM pressure, reduce LightRAG entity extraction gleaning or lower context/chunk sizes.

## Troubleshooting

### Ollama Embedding Timeout

Symptom:

```text
Ollama embedding preflight failed
Ollama request timed out after ... at http://127.0.0.1:11434/api/embed
```

This means the pipeline is able to reach the Ollama server, but the local embedding endpoint did not return. Fix it in this order:

1. Stop the current indexing command with `Ctrl+C`.
2. Stop stale Python pipeline processes from prior runs if they are still active.
3. Restart Ollama:

```powershell
Get-Process "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
```

4. Verify the server and embedding model:

```powershell
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
& $ollama list
& $ollama ps

$body = @{ model = "nomic-embed-text"; input = "embedding health check" } | ConvertTo-Json
$response = Invoke-RestMethod `
  -Uri "http://127.0.0.1:11434/api/embed" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  -TimeoutSec 30
$response.embeddings[0].Count
```

The pipeline uses the same string-input request shape for single query embeddings. If this health check is fast but query embeddings are slow, verify that the code includes the singleton string-input fix in `src/embeddings.py`.

5. Retry indexing with conservative embedding settings:

```powershell
python main.py --mode index --md_dir processed_docs --db_dir db --rag_backend local --embedding_batch_size 1 --embedding_timeout 30
```

If the embedding health check still times out after restarting Ollama, use the hash backend to keep working without Ollama embeddings:

```powershell
python main.py --mode index --md_dir processed_docs --db_dir db --rag_backend local --embedding_backend hash
```

Hash embeddings are deterministic and fully local, but retrieval quality is lower than `nomic-embed-text`.

### Ollama Not Found On PATH

If `ollama` is not recognized, use the full Windows install path:

```powershell
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
& $ollama --version
```

To add it to the current PowerShell session:

```powershell
$env:Path = "$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
```

### Port Or Server Conflicts

Ollama should listen on `127.0.0.1:11434` by default. Check the port:

```powershell
netstat -ano | findstr 11434
```

If another stale process owns the port, stop Ollama processes and start the app again:

```powershell
Get-Process "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
```

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
