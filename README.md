# Local FSAE RAG Pipeline

Retrieval-augmented generation pipeline for NUS FSAE knowledge transfer. The project ingests technical PDFs and notes, extracts text/tables/equations/figures, builds a local vector index, and answers questions through the hosted SoCLAaS LLM API. Bulk PDF parsing/OCR runs on the HPC CPU cluster (its only role) or locally; embedding and index construction run on the workstation against a locally hosted all-minilm (Ollama); chat and vision inference are served by the SoCLAaS API. A local Ollama chat backend remains supported as a dormant fallback for fully-offline operation.

## Goals And Constraints

- Serve chat (gemma4:26b) and vision (qwen3-vl:32b) through the hosted OpenAI-compatible SoCLAaS API; serve embeddings from a locally hosted `all-minilm` (384-d) via Ollama.
- Keep the compute-heavy PDF parsing/OCR (Docling) on the free HPC CPU cluster — its only usage — and the chunking, LanceDB, ANN build, and embedding work local.
- Support a dormant local Ollama chat/vision backend for offline operation (`[llm_api].backend = "ollama"`).
- Support STEM and engineering documents: textbooks, lecture notes, research papers, reports, scanned PDFs, tables, figures, charts, and equations.
- Preserve enough source context for users to inspect where an answer came from.
- Target workstation: Ryzen 9 9950X3D, RTX 4000 Ada 20GB VRAM, 128GB RAM, Windows 11 or WSL2.

## Current Capabilities

- `src/ingestion.py`: parses PDFs with pypdf/Docling and exports enriched Markdown.
- `src/indexing.py`: builds section-aware summary/chunk records and writes the local vector store.
- `src/local_rag.py`: performs two-tier retrieval over the local LanceDB index and asks the SoCLAaS LLM to answer from retrieved context.
- `src/query.py`: thin query wrapper around the local RAG path.
- `src/web_app.py`: local FastAPI browser UI for uploads, queued indexing, index edits, and chat.
- `src/auto_tag.py`: LLM source-group auto-tagger plugin (gemma4) for the document-trust registry.
- `src/llm_api.py`: OpenAI-compatible client for the hosted SoCLAaS API (chat, vision, embeddings) and the `soclaas`/`ollama` backend selector.
- `src/embeddings.py`: calls the embeddings transport selected by `[embeddings].backend` (default: local Ollama `all-minilm`).

Implemented behavior:

- Born-digital PDF text extraction with pypdf.
- Scanned/image-only PDF OCR through Docling with RapidOCR ONNX by default.
- Docling PDF layout parsing with CUDA acceleration where available.
- `qwen3-vl:32b` vision enrichment (via SoCLAaS chat-completions image parts) for figures/charts/diagrams.
- Inline vision-description injection so figures remain near surrounding explanatory text.
- Chat inference using `gemma4:26b` via the SoCLAaS API by default.
- Embeddings using a locally hosted `all-minilm` (Ollama) by default; `[embeddings].backend = "soclaas"` switches them to the hosted API's `bge-m3` (1024-d) instead.
- 384-dimensional normalized vectors for retrieval (all-MiniLM-L6-v2 geometry).
- all-minilm and bge-m3 are instruction-free (no prefixes); switching to `nomic-embed-text` or an e5 model applies `search_document:`/`search_query:` prefixes automatically.
- Section-aware chunking from PDF outlines/bookmarks, table-of-contents parsing, or heading fallback.
- LanceDB-backed vector storage in `db/lancedb`.
- Categories ("split databases"): user-created partitions of the corpus, each an independent LanceDB index under `db/categories/<key>`. Create them in Admin (or `POST /api/categories`), upload straight into one (upload form's Category selector), move documents between them from the Library (select rows → Move to — vectors are reused, so a move never re-embeds), and toggle which ones a chat searches with the Ask tab's category chips (default: all). Membership is per source and lives in `data/.categories.json.sqlite3`; the default index `db/` is the implicit "General" category, so existing deployments need no migration. Deleting a category requires moving its documents out first. Backups, rebuild, compaction, and HPC delegation currently operate on the General index only; a full rebuild automatically drops categorized sources from General so they are not duplicated.
- LLM auto-tagging: PDFs uploaded without a source group are classified into Official / Student Research / Unofficial by `gemma4:26b` (filename + first-page excerpt, batched, confidence-floored) and written to the trust registry like manual tags, but flagged `auto_tagged` with model/confidence/reason for review; a manual tag always overrides. Configure under `[auto_tag]`; trigger a sweep with the web UI's "Auto-tag ungrouped" button or `POST /api/pdfs/trust/auto-tag`.

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

### Quick start (one click)

Double-click `setup.cmd` on Windows (or run `./setup.sh` on Linux/macOS). The wizard creates a project `.venv`, installs `requirements.txt`, writes `config.toml` (backing up any existing file), prompts for the server bind address/port and the SoCLAaS API key, verifies the key against the API, runs preflight checks, and starts the web server. Missing dependencies install automatically — including in non-interactive runs (`--no-install-deps` opts out).

Common options (all pass through to `scripts/setup_instance.py`):

```powershell
.\setup.cmd --set-api-key <key>             # persist the SoCLAaS key into config.toml
.\setup.cmd --non-interactive --check-only  # re-verify an existing setup
.\setup.cmd --non-interactive --start       # configure-if-needed + start (what start.cmd does)
```

`start.cmd` / `./start.sh` skip the prompts entirely and only provision the HPC cluster when its deployed source is stale. In local mode the wizard checks local Ollama, which is required for the default local embedding backend (`all-minilm`) and the dormant offline chat backend; the preflight fails only when Ollama is unreachable.

The rest of this section describes the manual, conda-based setup for development.

### Manual setup (development)

Create an environment:

```bash
conda create -n ragpipeline python=3.14 -y
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

Configure the SoCLAaS API key. The pipeline reads it from the `SOCLAAS_API_KEY` environment variable (or `LLM_API_KEY`, or `[llm_api].api_key` in `config.toml`). On PowerShell:

```powershell
$env:SOCLAAS_API_KEY = "<your-key>"
```

The endpoint and backend are configured under `[llm_api]` in `config.toml` (defaults: `backend = "soclaas"`, `base_url = "https://soclaas-api.comp.nus.edu.sg"`). Available models: `gemma4:26b` (chat + planner), `qwen3-vl:32b` (vision).

Embeddings run locally by default: `[embeddings].backend = "ollama"` (env `EMBEDDINGS_BACKEND` overrides) serves them from a local Ollama host while chat/vision stay on SoCLAaS. Install Ollama, start it, and pull the model:

```powershell
ollama pull all-minilm
```

Verify the embedding endpoint before indexing:

```powershell
$body = @{ model = "all-minilm"; input = "embedding health check" } | ConvertTo-Json
$response = Invoke-RestMethod `
  -Uri "http://127.0.0.1:11434/api/embed" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  -TimeoutSec 30
$response.embeddings[0].Count
```

The final command should print `384` (all-minilm dimension). To embed through the SoCLAaS API instead, set `[embeddings].backend = "soclaas"` (or `$env:EMBEDDINGS_BACKEND = "soclaas"`) plus `[models] embedding_model = "bge-m3"`, `embedding_dim = 1024`, and verify the key can reach `soclaas-api.comp.nus.edu.sg/v1/embeddings`.

> **Re-index required.** Switching the embedding model (all-minilm 384-d, nomic-embed-text 768-d, bge-m3 1024-d) invalidates any existing index. Delete `db/` (or the staged build dir) and rebuild — the indexer's model+dimension reuse guard enforces this automatically and the query engine raises a clear re-index error on a dimension mismatch.

#### Dormant local Ollama chat fallback (optional)

For fully-offline chat/vision operation, set `[llm_api].backend = "ollama"` (or `LLM_BACKEND=ollama`) and point `[ollama].host` at a local Ollama server. Install Ollama, start `ollama serve`, and pull the models you want to use. The default local chat/planner substitute is `qwen3:4b-instruct` (`ollama pull qwen3:4b-instruct`; ~2.5 GB at Q4_K_M with native tool-calling, sized for a 4 GB GPU) — without it, the cloud tag `gemma4:26b` base-name-matches to `gemma4:latest` (gemma4 e4b, 9.6 GB), which no consumer GPU of this class can serve. For vision, `local_vision_model = "qwen2.5vl:3b"` (`ollama pull qwen2.5vl:3b`); `all-minilm` is already required for the default local embeddings. Automatic fallbacks refuse any installed variant larger than `LOCAL_MODEL_MAX_BYTES` (default 4 GiB) instead of silently loading a model the GPU cannot serve. Chat/vision transports then route to that local server instead of the SoCLAaS API. This path is not used by default.

#### Scaling embeddings across multiple Ollama replicas

When embedding through local Ollama (the default), single-host embedding is the dominant indexing bottleneck at 100GB-scale. The pipeline can round-robin embedding batches across any number of Ollama replicas in parallel. Set `OLLAMA_EMBED_HOSTS` to a comma-separated list of replica base URLs (this lever does not apply when `[embeddings].backend = "soclaas"`, which is already horizontally scaled server-side):

```powershell
# Two Ollama servers (e.g. one per GPU). Batches are round-robined and
# dispatched concurrently across the replicas.
$env:OLLAMA_EMBED_HOSTS = "http://127.0.0.1:11434,http://127.0.0.1:11435"
```

To run multiple replicas on one host, start separate `ollama serve` processes on different ports (`OLLAMA_HOST=127.0.0.1:11435 ollama serve`). Fine-tune in-flight batches per host with `OLLAMA_EMBED_CONCURRENCY` (default `1` = one batch per host at a time; raise it when the Ollama server is configured with `OLLAMA_NUM_PARALLEL>1`). With a single replica and the default concurrency, embedding behavior is unchanged (serial, one batch at a time). The embedding batch size itself defaults to 128 texts per request (override via `OLLAMA_EMBED_BATCH_SIZE` or `--embedding_batch_size`).

## Usage

Intended CLI flow:

```bash
python main.py --mode ingest --data_dir data --md_dir processed_docs
python main.py --mode index --md_dir processed_docs --db_dir db
python main.py --mode query --db_dir db --question "Explain the bias-variance tradeoff"
```

For born-digital PDFs, ingest defaults to pypdf text extraction plus targeted Docling enrichment on pages that appear to contain embedded images, code, or formulas. Pictures use the vision-description path (qwen3-vl:32b via the SoCLAaS API), while detected code/formula pages use Docling code/formula enrichment before Markdown is chunked. Use `--asset_triggers none` for text-only extraction, `--asset_triggers images` for legacy picture-only enrichment, or `--asset_triggers all` to also enrich table heuristic pages. Scanned/image-only PDFs fall back to Docling OCR with RapidOCR/ONNX Runtime by default. If OCR fails or returns no content, ingestion analyzes page images with the configured vision model and writes searchable `[Page Image Analysis]` Markdown instead of silently producing an empty file. Ingestion shows per-document and per-page progress bars by default; pass `--no_progress` to disable them.

OCR defaults live under `[ingestion]` in `config.toml`:

```toml
ocr_backend = "rapidocr"
ocr_langs = ["english"]
ocr_force_full_page = true
rapidocr_backend = "onnxruntime"
```

Optional backends are `auto`, `tesseract_cli`, `tesseract`, and `easyocr`. Tesseract backends require a system Tesseract install and language data; set `tesseract_cmd`, `tesseract_data_path`, and `tesseract_psm` when using them. Successfully described Docling figures/charts are stored as PNG assets under `[paths] asset_dir` (`db/assets` by default), marked in Markdown with stable image-asset IDs, and shown as thumbnails/links in local source panels alongside their searchable vision descriptions.

The scanned-PDF OCR engine itself is chosen with `[ingestion] scanned_ocr_engine`: `docling` (default) or `unlimited_ocr` (Baidu's Unlimited-OCR VLM via the local Ollama host, `unlimited_ocr_model`/`unlimited_ocr_dpi`). The Unlimited-OCR route is not recommended as the primary engine: measured on real corpus scans (`eval/ocr/RESULTS.md`, rerunnable via `scripts/eval_ocr.py`), the Ollama build ties Docling on clean text and wins on tables but hallucinates or truncates on a third of the sample pages because it resizes pages to ~1024 px and lacks the vendor anti-repetition guard.

Indexing and query embed through the local `all-minilm` (Ollama) by default and chat through the SoCLAaS API (`gemma4:26b`). Use `--embedding_model` to select a different embedding model, `--embedding_batch_size` to tune batch size, and `--embedding_timeout 30` to fail clearly instead of waiting indefinitely. Indexing writes chunks to LanceDB, with `--summary_mode hybrid`, `--chunk_target_tokens 900`, and `--chunk_overlap_tokens 120` by default; the overlap is only used when a detected section is too large. Reindexing reuses existing vectors when a record ID, content hash, embedding model, and embedding dimension are unchanged, and writes `db/index_manifest.json` with per-document chunk/quality counts for the browser UI. Query mode defaults for `context_window`, `llm_num_predict`, and `min_relevance_score` are set in `config.toml`; local retrieval also reranks vector candidates with a bounded lexical score, controlled by `LOCAL_RAG_RETRIEVAL_LEXICAL_WEIGHT` when needed. The SoCLAaS chat path retries transient errors with bounded exponential backoff (configurable via `[llm_api].retries`); the dormant Ollama fallback cancels after `--ollama_max_lost_health_checks` failed health checks spaced by `--ollama_health_check_interval`.

The module entrypoints can also be used for the default directories:

```bash
python -m src.ingestion
python -m src.indexing
```

### Initial corpus build: one command (`--initial-corpus`)

The CPU cluster's only role is the one-time bulk parse of the initial PDF corpus. Embeddings are workstation-local, so the cluster cannot build the index — it produces `processed_docs/` Markdown and the index is built locally afterwards. From a configured setup (`setup.cmd` provisions the repo + SIF under `/hpctmp/<user>/<repo>`; the corpus itself lives under `/hpctmp/<user>/rag-corpus/`, deliberately outside the repo directory so a later re-provision cannot wipe it):

```powershell
start.cmd --initial-corpus corpus.zip
```

That single command (implemented by `scripts/hpc_corpus.py`) extracts the zip locally under `data/corpus/<name>/` — **nested directories are preserved**; the pipeline discovers PDFs recursively and disambiguates duplicate names safely — uploads the corpus to the cluster, submits and monitors the ingest-only PBS job (qsub/qstat polling, exit-code verification, live progress), fetches the processed Markdown home, verifies the fetched corpus against the zip, and builds the local index with `main.py --mode index`. Local Ollama must be running with the configured embedding model (`all-minilm` by default) pulled; if vision enrichment is enabled (`[ingestion] vision_enabled = true`, the default), a usable SoCLAaS key must be reachable (env `SOCLAAS_API_KEY`, `[llm_api].api_key`, or `~/rag_soclaas_key` on the login node) — otherwise the command fails fast instead of silently producing `[Image description failed]` markers (override with `--allow-degraded-vision`).

Re-runs are cheap: PDFs whose content hash already maps to processed Markdown are skipped cluster-side, so resuming after a walltime kill or an interrupted fetch costs nothing. Useful flags: `--skip-index-build` (parse + fetch only), `--allow-degraded-vision`.

Afterwards, start the server (`start.cmd`). Subsequent web-UI uploads ingest and index locally; the cluster is not used again.

<details>
<summary>Manual equivalent (appendix — prefer <code>--initial-corpus</code>)</summary>

1. Extract the zip locally (structure preserved) and upload it to the corpus dir:

   ```powershell
   scp corpus.zip <cpu-alias>:/hpctmp/<user>/rag-corpus/data/
   ssh <cpu-alias> "cd /hpctmp/<user>/rag-corpus/data && unzip -oq corpus.zip && rm corpus.zip"
   ```

   Do **not** flatten with `unzip -j`: same-named PDFs in different directories would silently overwrite each other, while the pipeline itself handles nested duplicates safely. On Windows, `rsync` is not needed — the programmatic path below falls back to `scp` automatically.

2. Generate and submit the ingest-only job (`--skip-index` stops after Docling/pypdf parsing, before any embeddings):

   ```powershell
   python -m src.hpc --cpu --skip-index --input-data-dir /hpctmp/<user>/rag-corpus/data --processed-dir /hpctmp/<user>/rag-corpus/processed_docs -o ingest_only.pbs
   scp ingest_only.pbs <cpu-alias>:<remote_repo_dir>/
   ssh <cpu-alias> "cd <remote_repo_dir> && qsub ingest_only.pbs"
   ```

   Monitor with `qstat -u $USER` and the job output file `rag_ingest_index.o<jobid>`, and **check the exit status before fetching** — a walltime kill leaves a partial `processed_docs/` that would otherwise silently index an incomplete corpus. Raise `[hpc.cpu.pbs_overrides] walltime` for large corpora.

3. Bring the processed Markdown home and build the index locally:

   ```powershell
   rsync -P <cpu-alias>:/hpctmp/<user>/rag-corpus/processed_docs/ ./processed_docs/
   python main.py --mode index --md_dir processed_docs --db_dir db
   ```

</details>

Programmatically, the flow maps to `HpcBackend.push_corpus_dir()` → `submit_ingest_index(skip_index=True)` → `fetch_processed_docs()`. To instead build the entire index on the cluster (embeddings via the SoCLAaS API), generate the job without `--skip-index`, export `EMBEDDINGS_BACKEND=soclaas` (with `[models]` set to bge-m3/1024) for the job, and rsync `db/` home with `HpcBackend.fetch_index()`.

Run the local browser UI:

```bash
python -m src.web_app
```

### Guided one-command server + HPC setup

On Windows, double-click `setup.cmd` (or run it from PowerShell). On Linux or
macOS, run `./setup.sh`. The wizard:

- preserves the rest of `config.toml` and writes `config.toml.bak`;
- creates or updates the exact CPU alias in `~/.ssh/config`, preserving
  unrelated hosts and writing `config.rag-setup.bak`;
- creates a dedicated Ed25519 key when missing, installs the public key without
  duplicating `authorized_keys` entries, and verifies key-only login;
- packages the current working-tree source, stages Atlas9 CPU under
  `/hpctmp/<username>/<repo-path>`, and creates the login-relative symlink;
- uploads an existing matching SIF or builds it remotely with
  Singularity/Apptainer `--fakeroot`, then verifies it before activation;
- configures the web bind address and local Ollama fallback endpoint;
- collects the cluster's login hostname, username, private key, and repository
  path relative to the directory entered immediately after SSH;
- verifies Python dependencies, the web port, SSH, file transfer, `qsub`, the
  remote repository, and the Singularity image;
- optionally creates `.venv` and installs `requirements.txt`;
- starts the web server.

Useful non-interactive forms:

```powershell
# Re-check an existing configuration without changing it.
.\setup.cmd --non-interactive --check-only

# Configure a service instance in automation.
.\setup.cmd --non-interactive --mode hpc --server-host 0.0.0.0 `
  --setup-ssh `
  --cpu-host nus_hpc_cpu --cpu-hostname cpu-login.example.edu `
  --cpu-user me --cpu-repo rag-cpu `
  --configure-only

# Start (provisioning the CPU cluster only if its deployed source is stale).
.\setup.cmd --non-interactive --start
```

With `--setup-ssh`, a missing key defaults to
`~/.ssh/rag_<alias>_ed25519` and is generated without a passphrase so the web
service can reconnect unattended. Initial public-key installation may request
the cluster password or MFA once. Use `--skip-key-install` when an administrator
must install the generated `.pub` file instead.

Interactive HPC setup provisions the CPU cluster automatically. Atlas9 uses
`/hpctmp/<username>`. The CPU image is uploaded from `rag_pipeline_cpu.sif`
when present, otherwise it is built remotely from `Singularity.cpu.def`.
Repository activation retains the prior deployment as
`<repo>.rag-setup-previous`. Use `--skip-hpc-provision` for connection-only
configuration, or `--provision-hpc` to reprovision an existing non-interactive
configuration.

The interactive wizard creates the SSH alias for you. Existing exact alias
blocks are updated in place; wildcard/group blocks and unrelated hosts are left
untouched. It also creates and authorizes the private key. Windows OpenSSH
`scp` is supported when `rsync` is unavailable. Press Ctrl+C to stop the local
server.

The browser UI reads server host, port, polling intervals, and update target from `config.toml` under `[server]`.
Use `bind_all = true` to listen on all IPv4 interfaces (`0.0.0.0`) instead of only loopback. If you start
with `uvicorn` directly, pass the same bind explicitly, for example `uvicorn src.web_app:app --host 0.0.0.0 --port 8000`.
By default, `/api/health` and `/api/jobs` are polled once per minute:

```toml
[server]
host = "127.0.0.1"
bind_all = false
port = 8000
update_remote = "origin"
update_branch = "main"
health_poll_interval_ms = 60000
jobs_poll_interval_ms = 60000
```

Open `http://127.0.0.1:8000`. The UI is a dependency-free ES-module app served from `web/` (`web/app.js` entry + `web/js/*` modules; no build step). It is organized as a collapsible sidebar with six views:

- **Documents** — drag-and-drop PDF/zip upload with per-file source-group and category staging, plus the jobs table with live progress bars, rate/ETA, cancellation, and completion toasts; a global banner shows the active job from any tab.
- **Library** — the PDF trust/review table: server-side search plus group/trust/category facet filters, sortable columns, bulk tag/move/re-run/delete (destructive bulk uses typed confirmation), quality and trust badges with a legend, per-row actions (approve, flag stale with an inline note, re-run, re-index, delete), and an in-app preview that shows the original PDF or the extracted Markdown text.
- **Review** — index browsing with a Chunks view (search, pagination, vector-search diagnostics, in-place record editing with Markdown preview) and a Documents view (one row per source document, expandable to that document's records).
- **Ask** — multi-turn chat: the last exchanges are sent with each question and the retrieval planner resolves follow-up references against them; answer modes (Precise/Balanced/Deep) map to sampler presets; answers stream with Markdown/LaTeX rendering, a Sources panel (group badges, scores, page links, extracted-image lightbox, stale-source warnings, per-answer settings/duration meta), copy/regenerate/edit-and-resend actions, and a saved-chat sidebar with search, pin, rename-in-place, and Markdown export. Chats persist in the browser's localStorage.
- **Admin** — ops dashboard (index/queue/LLM/embedding/disk/category cards with 24 h trend sparklines), security-posture and startup-repair banners, API key management (issue, role, disable, delete; plaintext shown once), update manager (branch, current vs. target commit, blocking reasons), and guarded maintenance (backup/restore, compact, rebuild vector index, full re-ingest/rebuild with typed confirmation).
- **Guide** — new-user walkthrough (replayable from Settings).

Keyboard shortcuts (`?` shows the cheat sheet): `g`+`d/l/r/a/m/g` switches views, `/` focuses the current view's search, `n` starts a new chat, `Esc` closes dialogs. Light/dark themes follow the system with a manual toggle; the layout is usable down to phone widths. Uploaded and indexed PDFs are tracked by SHA-256 hash; duplicates are rejected unless a forced re-upload is confirmed. Downloads and previews are path-contained: sources must live under the configured roots (`[paths] corpus_roots` covers bulk corpora on other volumes). The UI respects `[server]` host/port/poll intervals/update target as before, and only one web instance may run per data directory unless `RAG_ALLOW_MULTIPLE_WEB_INSTANCES=1` is set.

## Architecture

1. **Ingestion**
   - Born-digital PDF text is extracted with pypdf by default.
   - Docling OCR is used for scanned/image-only PDFs or targeted asset enrichment.
   - If Docling cannot parse a scan, page images are described by the local vision model so retrieval still has searchable context.
   - Tables and structured items are exported as Markdown where supported.
   - Figures/charts can be sent to `qwen3-vl:32b` through the SoCLAaS API.

2. **Embeddings**
   - Embeddings run on a locally hosted `all-minilm` (Ollama, 384-d) by default, selected by `[embeddings].backend` independently of the chat/vision backend.
   - all-minilm and bge-m3 need no instruction prefixes; nomic/e5 models get `search_document:`/`search_query:` applied automatically.
   - Embeddings are L2-normalized.
   - `[embeddings].backend = "soclaas"` switches embeddings to the hosted API's `bge-m3` (1024-d); a model/dim switch requires a full re-index.

3. **Indexing**
   - PDFs are partitioned by outline/bookmark, contents-page entries, or Markdown heading fallback.
   - Title/cover and contents pages are excluded from retrieval records.
   - Document and section summary rows plus leaf chunk rows are embedded through the active embeddings backend (local Ollama by default).
   - Reindexing reuses unchanged vectors from the existing LanceDB table and writes `db/index_manifest.json` with per-document record, chunk, page, and extraction-quality counts.
   - The local index is written to LanceDB under `db/lancedb` by default.

4. **Querying**
   - Questions are embedded through the active embeddings backend (local Ollama by default).
   - Summary hits are expanded to child chunks, while direct chunk hits are used as answer context.
   - Context selection uses a stricter relevance floor plus an input-prompt budget capped at 60% of the model context window instead of a fixed chunk count.
   - Candidate order is a hybrid of vector score and lexical query-term support, while the initial relevance gate still uses the vector score.
   - OpenAI-style tool calls let the model pull additional local context and optional keyless web-search results before final answer streaming; web search is skipped once the current prompt already exceeds the input-prompt budget.
   - The default chat system prompt can be set in `config.toml` under `[chat] system_prompt` or overridden with `--system_prompt`.
   - `gemma4:26b` synthesizes an answer from tool-returned sources and cites `[S#]` local chunks or `[W#]` web results shown in the Sources panel.
   - Final answers are checked for unknown source IDs and weak lexical support against cited tool results; warnings are shown in the chat notice area when citations look suspect.

## Hardware And Runtime Notes

- Chat and vision inference are served by the SoCLAaS API; those calls only need outbound HTTPS to `soclaas-api.comp.nus.edu.sg` plus the API key. The workstation also runs local Ollama for embeddings (`all-minilm`).
- The CPU HPC job does the non-LLM parse work (Docling OCR/parsing) and runs ingest-only by default (`--skip-index`); the workstation embeds with local all-minilm and builds LanceDB/ANN. A full ingest+index cluster job remains available for `EMBEDDINGS_BACKEND=soclaas` deployments.
- The dormant Ollama fallback's `keep_alive="0"` can evict inactive models between phases, and VRAM pressure can be relieved by reducing Ollama parallelism or the embedding batch size.

## Troubleshooting

The Ollama-specific sections below apply to the local Ollama embedding backend (the default) and to the **dormant offline chat fallback** (`[llm_api].backend = "ollama"`). On the SoCLAaS path (chat/vision by default, and embeddings when `[embeddings].backend = "soclaas"`), errors surface as `SoCLAaS ... HTTP <code>` messages — verify `SOCLAAS_API_KEY`, the base URL, and that the host can reach `soclaas-api.comp.nus.edu.sg`.

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

$body = @{ model = "all-minilm"; input = "embedding health check" } | ConvertTo-Json
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
- Benchmark pages/minute, indexing time, query latency, RAM usage, and retrieval quality.

Medium term:

- Add broader retrieval evaluation with golden question sets.
- Preserve all source block/page references through chunking and deduplication.
- Wire `HpcBackend.submit_ingest_index`/`fetch_index` into the web job queue so HPC
  builds can be triggered from the UI (the CLI path is automated via
  `start.cmd --initial-corpus`; the upload-driven web UI still ingests locally).

## Development Notes

- The primary LLM backend is the hosted SoCLAaS API; the local Ollama transport is
  a dormant offline fallback. Keep both code paths working.
- Do not commit generated `db/`, `processed_docs/` test outputs, model caches, or Python cache files unless intentionally curating a fixture.
- Keep the README as the only root-level project documentation file; `docs/` holds
  historical runbooks and design notes.
- Treat `processed_docs/*.md` as corpus data, not documentation.
