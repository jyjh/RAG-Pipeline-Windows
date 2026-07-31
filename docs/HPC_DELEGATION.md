# HPC Delegation — web app on a server, work on the cluster

How the FastAPI web app (running on a lightweight server) delegates to the NUS
HPC cluster: **bulk indexing on the free CPU cluster, chat LLM on the paid GPU
cluster.** This doc is the architecture; `docs/HPC_RUNBOOK.md` is the operator
recipe for running it.

## The two halves (and why one is already done)

The web app does two heavy things, and they want opposite architectures:

| Half | Where it runs | Cost model | Status |
|------|---------------|------------|--------|
| **Build** (ingest PDFs → OCR → embed → LanceDB index) | Free CPU cluster | Free, parallelizable | `src/hpc_backend.py` (this round) |
| **Chat** (7B LLM answers queries against the index) | Paid GPU cluster | Billed per hour, only while chatting | Already done — no code |

The chat half needs **no new code**: the web app already treats Ollama as
`http://host:port` via `OLLAMA_HOST` / `[ollama].host`. You run the serving job
(`scripts/nus_hpc_serve.pbs`) on the GPU cluster, start the SSH tunnel
(`scripts/tunnel_daemon.{sh,ps1}`), and point `OLLAMA_HOST` at `127.0.0.1:11434`.
To the web app, the cluster GPU is just another local Ollama.

All real engineering is the **build half** — getting the web app to submit
ingest/index PBS jobs and pull the result back.

## Data flow

```
                         BUILD HALF (free CPU cluster)
  user ──▶ web app ──ssh/qsub──▶ login node ──▶ PBS job on CPU node
   │         (server)                            │  (Singularity + bulk_ingest.py)
   │                                            ▼
   │                                       db/ on HPC
   │                                            │
   │              ◀────rsync db/ back───────────┘
   │              │
   │         cache invalidate  ◀── known gap: needs /api/hpc/reload (next step)
   ▼
  query
   │
   ▼
 CHAT HALF (paid GPU cluster, on demand)
  user ──▶ web app ──▶ OLLAMA_HOST=127.0.0.1:11434 ──tunnel──▶ GPU node (serve job)
                                                              (ollama serve, gemma4)
```

The build half is asynchronous (minutes to hours); the chat half is synchronous
per request.

## The build half — how `HpcBackend` plugs in

The web app already has a clean seam for "run the build and report progress":
`RagJobQueue._run_pipeline_subprocess` (`src/web_app_classes/rag_job_queue.py:834`)
→ `_run_job_subprocess` (`src/web_app.py:787`). Today that shells out to a local
`python main.py --mode ingest|index` and parses `__RAG_PROGRESS__` lines
(`src/progress_protocol.py`) into the job's `progress` dict, which the UI polls.

`src/hpc_backend.py::HpcBackend` mirrors that contract but runs the work on HPC:

1. **Submit:** generate a PBS script (reuses `src.hpc.generate_pbs_script` with
   `--cpu` overrides), `ssh ssh_host "cat > /repo/.hpc_ingest_*.pbs"` it over,
   `cd /repo && qsub ...`, capture the JOBID.
2. **Poll + relay:** loop `qstat -f <JOBID>` until terminal; each poll also
   `tail -c +N` the job's combined stdout file and parse `__RAG_PROGRESS__` lines
   into the **same** `progress_callback` the local runner uses — so the existing
   UI progress bar works unchanged.
3. **Cancel:** `cancel_event` triggers `qdel <JOBID>`.
4. **Fetch:** `fetch_index()` rsyncs the remote `db/` back into the local `db/`.

Because the contract matches, the eventual wiring (next step) is small: in
`_run_job_subprocess`, branch on `cfg.hpc.enabled` — local subprocess vs
`HpcBackend`.

## Config

`[hpc]` in `config.toml` (dataclass in `src/config.py::HpcConfig`, default
`enabled = false` so existing setups are untouched). **Two separate clusters**
are configured independently because the free CPU cluster and the paid GPU
cluster are different machines with different login nodes — build work routes
to `[hpc.cpu]`, the serving job routes to `[hpc.gpu]`:

```toml
[hpc]
enabled = true
remote_data_dir = "data"             # pre-staged corpus on the CPU cluster (--input-dir)
remote_db_dir = "db"                 # where bulk_ingest.py writes the index (CPU cluster)
poll_interval_seconds = 15.0

[hpc.cpu]                            # FREE cluster — runs ingest/index
ssh_host = "nus_hpc_cpu"             # ~/.ssh/config alias for the CPU login node
remote_repo_dir = "rag"              # relative to the directory entered by SSH
container_sif = "rag_pipeline_cpu.sif"

[hpc.cpu.pbs_overrides]              # merged into generate_pbs_script() (CPU defaults: ngpus=0, queue=cpu)
ncpus = 16
walltime = "08:00:00"

[hpc.gpu]                            # PAID cluster — runs the Ollama serving job for chat
ssh_host = "nus_hpc_gpu"             # ~/.ssh/config alias for the GPU login node (DIFFERENT host)
remote_repo_dir = "rag"              # relative to the directory entered by SSH
container_sif = "rag_pipeline.sif"   # the CUDA image, not the CPU one

[hpc.gpu.pbs_overrides]              # merged into generate_serve_pbs_script() (GPU defaults: ngpus=1, queue=gpu)
walltime = "08:00:00"
```

`HpcBackend` validates each cluster lazily (only the one a method needs), so a
build-only deployment with no GPU config still works. Chat half config (separate
— the Ollama client points at the tunnel, no `[hpc]` involvement):

```toml
[ollama]
host = "http://127.0.0.1:11434"      # the tunnel lands here
hosts = []                           # add a local fallback if you have one
```

## Two assumptions (and their fallbacks)

1. **The web-app server can SSH to both cluster login nodes** (key-based,
   configured in `~/.ssh/config` as `[hpc.cpu].ssh_host` and `[hpc.gpu].ssh_host`
   — two different hosts). This is required — "the web app calls back to HPC"
   means the server must reach each cluster's login node. **Fallback if not:**
   run a small broker on a host that *can* reach HPC (e.g. a login node itself),
   and have the web app POST to it; or trigger jobs manually and just use the
   `fetch_index` half to pull results.

2. **The corpus is pre-staged on HPC** (`remote_data_dir` points at PDFs already
   there). Simplest path. **Fallback (not built yet):** before `qsub`, rsync the
   user's uploaded PDFs out — `HpcBackend` would gain a `push_uploads()` step
   (`rsync local data/.upload_queue/ → remote_data_dir/`). This is additive and
   documented as a next step.

## Known gaps (explicitly NOT done this round)

These are the next steps; flagged so the scope is honest:

- **`_run_job_subprocess` selection logic** — branch on `cfg.hpc.enabled` to use
  `HpcBackend` instead of the local subprocess. Small change once the backend
  exists; deliberately deferred so the live job queue isn't touched until you've
  reviewed `HpcBackend`.
- **Cache invalidation after `fetch_index`** — the server opens `db/` lazily and
  caches it for the process lifetime (`_index_store`, `web_app.py:718`). An
  *externally*-produced index won't be seen until `_invalidate_index_caches(db)`
  (`web_app.py:1110`) is called. Today only in-process jobs call it. After
  `fetch_index`, the caller must trigger invalidation — either a new
  `/api/hpc/reload` endpoint or auto-invalidate at the end of `fetch_index`.
- **Upload-rsync-out** — pushing web-app-uploaded PDFs to HPC (assumption 2's
  fallback).
- **Reconciliation of Ollama paths** — embeddings (build) use the in-job Ollama
  on HPC; chat uses the tunnel. No conflict, but `[embeddings].hosts` and
  `[ollama].host` serve different halves and must not be confused.

## Manual smoke test (once you point it at real creds)

`HpcBackend` is runnable standalone from a REPL before any web-app wiring:

```python
from src.config import load_config
from src.hpc_backend import HpcBackend
cfg = load_config().hpc            # with [hpc] enabled in your config.toml
b = HpcBackend(cfg)
result = b.submit_ingest_index("data")   # blocks, relays progress to stdout
print(result.remote_db_dir)
b.fetch_index(local_db_dir="db")         # rsync the index back
```

Then `python -m src.web_app` and query — if you also started the serve job +
tunnel, chat hits the GPU cluster.

## Files

| File | Role |
|------|------|
| `src/hpc_backend.py` | `HpcBackend` — the build-half orchestrator (SSH qsub/qstat/rsync + progress relay) |
| `src/config.py` | `HpcConfig` dataclass wired as `cfg.hpc` (default off) |
| `src/hpc.py` | `generate_pbs_script` / `generate_serve_pbs_script` (reused, unchanged) |
| `src/progress_protocol.py` | `__RAG_PROGRESS__` parsing (reused, unchanged) |
| `scripts/nus_hpc_serve.pbs` | The chat-half serving job (unchanged) |
| `scripts/tunnel_daemon.{sh,ps1}` | The chat-half tunnel (unchanged) |
| `docs/HPC_RUNBOOK.md` | Operator recipe for the cluster side |
