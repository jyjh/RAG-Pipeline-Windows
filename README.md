# Local FSAE RAG Pipeline

Local retrieval-augmented generation pipeline for NUS FSAE knowledge transfer. The project ingests technical PDFs and notes, extracts text/tables/equations/figures, builds a local vector index, and answers questions through Ollama without cloud APIs.

## Goals And Constraints

- Keep all document processing, embeddings, retrieval, and generation local.
- Use Ollama for both text generation and embeddings.
- Support STEM and engineering documents: textbooks, lecture notes, research papers, reports, scanned PDFs, tables, figures, charts, and equations.
- Preserve enough source context for users to inspect where an answer came from.
- Target workstation: Ryzen 9 9950X3D, RTX 4000 Ada 20GB VRAM, 128GB RAM, Windows 11 or WSL2.

## Current Capabilities

- `src/ingestion.py`: parses PDFs with pypdf/Docling and exports enriched Markdown.
- `src/indexing.py`: chunks Markdown and writes `db/local_vector_index.json`.
- `src/local_rag.py`: performs vector retrieval over the local JSON index and asks Ollama to answer from retrieved context.
- `src/query.py`: thin query wrapper around the local Ollama RAG path.
- `src/web_app.py`: local FastAPI browser UI for uploads, queued indexing, index edits, and chat.
- `src/embeddings.py`: calls Ollama `/api/embed` with `nomic-embed-text` by default.
- `src/utils.py`: optional Ollama model load/unload helpers.

Implemented behavior:

- Born-digital PDF text extraction with pypdf.
- Docling PDF layout parsing with CUDA acceleration where available.
- Lazy `qwen2.5vl:7b` vision enrichment for figures/charts/diagrams.
- Inline vision-description injection so figures remain near surrounding explanatory text.
- Local Ollama inference using `gemma4` by default.
- Local Ollama embeddings using `nomic-embed-text` by default.
- 768-dimensional normalized vectors for retrieval.
- Document chunks use the `search_document:` prefix; queries use `search_query:`.

The `processed_docs/` Markdown files are corpus data generated from sample PDFs, not project documentation.

## Expected Repository Layout

```text
data/                 Input PDFs
processed_docs/       Generated/enriched Markdown corpus files
db/                   Generated local vector index artifacts
web/                  Static browser UI for the local FastAPI app
src/                  Pipeline source modules
tests/                Unit tests
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

Install required packages:

```bash
pip install -r requirements.txt
```

Install and verify Ollama on Windows:

```powershell
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

## Usage

Intended CLI flow:

```bash
python main.py --mode ingest --data_dir data --md_dir processed_docs
python main.py --mode index --md_dir processed_docs --db_dir db
python main.py --mode query --db_dir db --question "Explain the bias-variance tradeoff"
```

For born-digital PDFs, ingest defaults to pypdf text extraction and does not run Docling asset enrichment unless requested. Use `--asset_triggers images` to enrich pages with embedded images, or `--asset_triggers all` to also enrich table/equation heuristic pages. Ingestion shows per-document and per-page progress bars by default; pass `--no_progress` to disable them.

Indexing and query use Ollama embeddings only. Use `--embedding_model` to select a different Ollama embedding model, `--embedding_batch_size 1` if Ollama struggles with larger embedding batches, and `--embedding_timeout 30` to fail clearly instead of waiting indefinitely. Query mode requests `--llm_num_predict 2048` by default; increase it if your local model needs more room for long answers.

The module entrypoints can also be used for the default directories:

```bash
python -m src.ingestion
python -m src.indexing
```

Run the local browser UI:

```bash
uvicorn src.web_app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The web app accepts PDF uploads, queues ingestion/indexing work in the background, lets users inspect/edit/delete index records, and streams chat answers as Ollama produces them. The chat panel exposes sampler controls for temperature, top-k (`Max K`), context window, and maximum output tokens. When Ollama returns model thinking, the chat UI shows it in a collapsible block above the answer and reports clearly if the model stops before producing final answer text. Chat output is rendered as Markdown with local LaTeX-to-MathML formatting. Queued ingestion/indexing waits before expensive phases while chat queries are active; a phase already running is not forcibly interrupted. Query generation defaults to temperature `0.9`, top-k `40`, context window `8192`, and max output `4096`.

## Architecture

1. **Ingestion**
   - Born-digital PDF text is extracted with pypdf by default.
   - Docling is used for scanned/image-only PDFs or targeted asset enrichment.
   - Tables and structured items are exported as Markdown where supported.
   - Figures/charts can be sent to local Qwen2.5-VL through Ollama.

2. **Embeddings**
   - Ollama embeddings run locally with `nomic-embed-text` by default.
   - Document chunks use the `search_document:` prefix.
   - Queries use the `search_query:` prefix.
   - Embeddings are L2-normalized.

3. **Indexing**
   - Markdown documents are split into overlapping chunks.
   - Chunks are embedded through Ollama.
   - The local index is written to `db/local_vector_index.json`.

4. **Querying**
   - Questions are embedded locally through Ollama.
   - The nearest chunks are retrieved from the local JSON vector index.
   - `gemma4` synthesizes an answer from retrieved context and cites source numbers.

## Hardware And Runtime Notes

- Qwen2.5-VL is loaded lazily during ingestion only when figures are detected.
- Ollama `keep_alive="0"` can be used to evict inactive models between phases.
- If indexing causes VRAM pressure, reduce Ollama parallelism or lower the embedding batch size.

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

5. Retry indexing with conservative embedding settings:

```powershell
python main.py --mode index --md_dir processed_docs --db_dir db --embedding_batch_size 1 --embedding_timeout 30
```

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

Recommended validation:

```bash
python -m compileall src
pytest
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
- Offline execution after Ollama models and Python packages are cached locally.

## Roadmap

Near term:

- Pin package versions after successful validation.
- Add real golden PDF fixtures for scanned, mixed, table, equation, and figure-heavy documents.
- Benchmark pages/minute, indexing time, query latency, VRAM/RAM usage, and retrieval quality.

Medium term:

- Add a citation-safe structured block store as the durable source of truth.
- Add a local FastAPI service and browser UI for team members.
- Add local reranking and retrieval evaluation.
- Preserve all source block/page references through chunking and deduplication.

## Development Notes

- Do not use cloud APIs or hosted models.
- Do not commit generated `db/`, `processed_docs/` test outputs, model caches, or Python cache files unless intentionally curating a fixture.
- Keep the README as the only root-level project documentation file.
- Treat `processed_docs/*.md` as corpus data, not documentation.
