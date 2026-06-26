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
- `src/indexing.py`: builds section-aware summary/chunk records and writes the local vector store.
- `src/local_rag.py`: performs two-tier retrieval over the local LanceDB index and asks Ollama to answer from retrieved context.
- `src/query.py`: thin query wrapper around the local Ollama RAG path.
- `src/web_app.py`: local FastAPI browser UI for uploads, queued indexing, index edits, and chat.
- `src/embeddings.py`: calls Ollama `/api/embed` with `nomic-embed-text` by default.
- `src/utils.py`: optional Ollama model load/unload helpers.

Implemented behavior:

- Born-digital PDF text extraction with pypdf.
- Scanned/image-only PDF OCR through Docling with RapidOCR ONNX by default.
- Docling PDF layout parsing with CUDA acceleration where available.
- Lazy `qwen2.5vl:7b` vision enrichment for figures/charts/diagrams.
- Inline vision-description injection so figures remain near surrounding explanatory text.
- Local Ollama inference using `gemma4` by default.
- Local Ollama embeddings using `nomic-embed-text` by default.
- 768-dimensional normalized vectors for retrieval.
- Document chunks use the `search_document:` prefix; queries use `search_query:`.
- Section-aware chunking from PDF outlines/bookmarks, table-of-contents parsing, or heading fallback.
- LanceDB-backed vector storage in `db/lancedb`.

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

For born-digital PDFs, ingest defaults to pypdf text extraction plus targeted Docling enrichment on pages with embedded images. Use `--asset_triggers none` for text-only extraction, or `--asset_triggers all` to also enrich table/equation heuristic pages. Scanned/image-only PDFs fall back to Docling OCR with RapidOCR/ONNX Runtime by default. If OCR fails or returns no content, ingestion analyzes page images with the configured vision model and writes searchable `[Page Image Analysis]` Markdown instead of silently producing an empty file. Ingestion shows per-document and per-page progress bars by default; pass `--no_progress` to disable them.

OCR defaults live under `[ingestion]` in `config.toml`:

```toml
ocr_backend = "rapidocr"
ocr_langs = ["english"]
ocr_force_full_page = true
rapidocr_backend = "onnxruntime"
```

Optional backends are `auto`, `tesseract_cli`, `tesseract`, and `easyocr`. Tesseract backends require a system Tesseract install and language data; set `tesseract_cmd`, `tesseract_data_path`, and `tesseract_psm` when using them. Diagram support is currently search-oriented: diagrams are represented as inline vision descriptions embedded with nearby OCR/page text, not as retrieved image thumbnails.

Indexing and query use Ollama embeddings only. Use `--embedding_model` to select a different Ollama embedding model, `--embedding_batch_size 1` if Ollama struggles with larger embedding batches, and `--embedding_timeout 30` to fail clearly instead of waiting indefinitely. Indexing writes chunks to LanceDB, with `--summary_mode hybrid`, `--chunk_target_tokens 900`, and `--chunk_overlap_tokens 120` by default; the overlap is only used when a detected section is too large. Reindexing reuses existing vectors when a record ID, content hash, embedding model, and embedding dimension are unchanged, and writes `db/index_manifest.json` with per-document chunk/quality counts for the browser UI. Query mode defaults for `context_window`, `llm_num_predict`, and `min_relevance_score` are set in `config.toml`; local retrieval also reranks vector candidates with a bounded lexical score, controlled by `LOCAL_RAG_RETRIEVAL_LEXICAL_WEIGHT` when needed. Ollama chat generation does not use a request timeout; after a connection loss it cancels only after `--ollama_max_lost_health_checks` failed health checks spaced by `--ollama_health_check_interval`.

The module entrypoints can also be used for the default directories:

```bash
python -m src.ingestion
python -m src.indexing
```

Run the local browser UI:

```bash
uvicorn src.web_app:app --host 127.0.0.1 --port 8000
```

The browser UI reads server host, port, polling intervals, and update target from `config.toml` under `[server]`.
By default, `/api/health` and `/api/jobs` are polled once per minute:

```toml
[server]
host = "127.0.0.1"
port = 8000
update_remote = "origin"
update_branch = "main"
health_poll_interval_ms = 60000
jobs_poll_interval_ms = 60000
```

Open `http://127.0.0.1:8000`. The web app accepts PDF uploads, queues ingestion/indexing work in the background, lets users inspect/edit/delete index records, and streams chat answers as Ollama produces them. Uploaded and indexed PDFs are searchable by title/hash and listed with quality badges, chunk counts, trust status, visible trust notes, approve/stale review actions, targeted re-run actions, and download links for source verification. Trust metadata is stored in `data/.document_trust.json`; unreviewed, rejected, stale, expired, missing, or poorly extracted sources are flagged before users rely on them. The top-right update button checks the configured remote branch, pulls fast-forward updates when available, and restarts the uvicorn server. It blocks updates if tracked files are dirty, the server is not running the configured branch, Git history diverged, or chat/indexing work is active. The Index page defaults to 20 rows, can show 50 or 100 rows per page, and can load all matching rows in 100-row HTTP batches. The chat panel saves conversations in the browser's `localStorage` and includes a collapsible saved-chat sidebar with rename/delete controls. Uploads are tracked by PDF SHA-256 hash across queued and ingested files; duplicate uploads are rejected unless the browser confirmation prompt is accepted for a forced re-upload. Forced re-upload cleanup waits until the job reaches ingestion, then removes existing indexed records for that parent PDF before fresh ingestion/indexing. The PDF table's `Re-run` action stages only the selected source PDF, re-runs ingestion for that source, then reindexes the corpus. The chat panel exposes sampler controls for temperature, top-k (`Max K`), context window, relevance floor, maximum output tokens, and web-search enablement. When Ollama returns model thinking, the chat UI shows it in a collapsible block above the answer and reports clearly if the model stops before producing final answer text. Chat output is rendered as Markdown with local LaTeX-to-MathML formatting and includes a Sources panel populated from retrieved chunks and web results. Local sources include an `Open page` link to the cited PDF page when page metadata is available, plus a download link for the full PDF. Queued ingestion/indexing waits before expensive phases while chat queries are active; a phase already running is not forcibly interrupted.

## Architecture

1. **Ingestion**
   - Born-digital PDF text is extracted with pypdf by default.
   - Docling OCR is used for scanned/image-only PDFs or targeted asset enrichment.
   - If Docling cannot parse a scan, page images are described by the local vision model so retrieval still has searchable context.
   - Tables and structured items are exported as Markdown where supported.
   - Figures/charts can be sent to local Qwen2.5-VL through Ollama.

2. **Embeddings**
   - Ollama embeddings run locally with `nomic-embed-text` by default.
   - Document chunks use the `search_document:` prefix.
   - Queries use the `search_query:` prefix.
   - Embeddings are L2-normalized.

3. **Indexing**
   - PDFs are partitioned by outline/bookmark, contents-page entries, or Markdown heading fallback.
   - Title/cover and contents pages are excluded from retrieval records.
   - Document and section summary rows plus leaf chunk rows are embedded through Ollama.
   - Reindexing reuses unchanged vectors from the existing LanceDB table and writes `db/index_manifest.json` with per-document record, chunk, page, and extraction-quality counts.
   - The local index is written to LanceDB under `db/lancedb` by default.

4. **Querying**
   - Questions are embedded locally through Ollama.
   - Summary hits are expanded to child chunks, while direct chunk hits are used as answer context.
   - Context selection uses a stricter relevance floor plus an input-prompt budget capped at 60% of the model context window instead of a fixed chunk count.
   - Candidate order is a hybrid of vector score and lexical query-term support, while the initial relevance gate still uses the vector score.
   - Ollama native tool calls let the model pull additional local context and optional keyless web-search results before final answer streaming; web search is skipped once the current prompt already exceeds the input-prompt budget.
   - The default chat system prompt can be set in `config.toml` under `[chat] system_prompt` or overridden with `--system_prompt`.
   - `gemma4` synthesizes an answer from tool-returned sources and cites `[S#]` local chunks or `[W#]` web results shown in the Sources panel.
   - Final answers are checked for unknown source IDs and weak lexical support against cited tool results; warnings are shown in the chat notice area when citations look suspect.

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
- Add broader retrieval evaluation with golden question sets.
- Preserve all source block/page references through chunking and deduplication.

## Development Notes

- Do not use cloud APIs or hosted models.
- Do not commit generated `db/`, `processed_docs/` test outputs, model caches, or Python cache files unless intentionally curating a fixture.
- Keep the README as the only root-level project documentation file.
- Treat `processed_docs/*.md` as corpus data, not documentation.
