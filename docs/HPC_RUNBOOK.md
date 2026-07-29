# NUS HPC Runbook

End-to-end recipe for running the FSAE RAG pipeline on the NUS HPC compute
cluster: build the LanceDB index on a GPU node, then chat through the web app
on your laptop while Ollama runs on a cluster GPU.

This is the topology this runbook targets:

```
 ┌─────────────┐    SSH (2-hop, -J)     ┌──────────┐                ┌──────────────────┐
 │  Laptop     │ ─────────────────────▶ │ login    │ ─── private ─▶ │ GPU compute node │
 │ web_app.py  │   127.0.0.1:11434 ───▶ │ node     │    network     │ ollama serve     │
 │ reads db/   │                        │ (jump)   │                │ (PBS serve job)  │
 │ locally     │ ◀───────────────────── │          │ ◀───────────── │                  │
 └─────────────┘                        └──────────┘                └──────────────────┘
        ▲
        │ db/ copied home via rsync after the ingest job
        │
  built on HPC by nus_hpc_ingest_index.pbs
```

Two PBS jobs do the work:

| Job | Script | Lifetime | Purpose |
|-----|--------|----------|---------|
| Ingest+index | `scripts/nus_hpc_ingest_index.pbs` | Runs to completion, exits | OCR/embed/index PDFs → writes `db/` |
| Ollama serving | `scripts/nus_hpc_serve.pbs` | Long-lived (until walltime) | Keeps `ollama serve` alive on a GPU; publishes its hostname |

> **Fallback (simpler) topology.** If you discover NUS lets you run a persistent
> `ollama serve` on a **login or dev node** (no PBS needed), skip the serving
> job entirely and run the tunnel daemon in single-hop mode straight at that
> host. Today's `tunnel_daemon` default already does this. This runbook assumes
> the general case (GPUs only on compute nodes allocated via PBS).

---

## Prerequisites (one-time, on your laptop)

1. **SSH access** to NUS HPC, key-based. Add an alias to `~/.ssh/config`:

   ```sshconfig
   Host nus_hpc
       HostName <nus-login-hostname>     # e.g. aspsus.nus.edu.sg — ask your cluster admin
       User <your-nus-username>
       IdentityFile ~/.ssh/id_ed25519
       ServerAliveInterval 15
       ServerAliveCountMax 3
   ```

2. **A Linux box with root/fakeroot** to build the Singularity image (a Linux
   workstation, a VM, or WSL2). You cannot build SIFs on the NUS login node
   (no root) and not on Windows directly.

3. **Project checked out** on the cluster (in your home dir or project storage)
   so the PBS scripts and `scripts/bulk_ingest.py` are available there. The
   `.pbs` scripts bind-mount `${PWD}:/app`, so `qsub` them from the repo root.

4. **Ollama models known to your config.** Open `config.toml` and confirm the
   `[models]` tags match what the PBS jobs will `ollama pull`:
   `nomic-embed-text`, `qwen2.5vl:7b`, `gemma4`, `qwen2.5:1.5b`. If `gemma4` is
   a custom alias you made with `ollama create`, create it in the model store
   once (Phase 2) before the first ingest — otherwise the pull step will fail.

---

## CPU-only clusters — read this if you have no GPU allocation

The phases below assume GPU nodes (the CUDA container, the `gpu` queue, Ollama on
a GPU). If your allocation is **CPU-only**, the same artifacts work — you build a
different image, pass `--cpu` to the generator, and shift your scale-out strategy
from "one fast GPU" to "many CPU nodes sharding embeddings." The honest caveat:
**interactive chat against a cluster-hosted 7B LLM on CPU is too slow to be worth
it**, so on CPU the cluster is for *building the index*, not for serving chat.

### What changes

| | GPU (phases below) | CPU-only |
|---|---|---|
| Image | `Singularity.def` → `rag_pipeline.sif` | `Singularity.cpu.def` → `rag_pipeline_cpu.sif` (ubuntu base, CPU torch; ~3–5 GB smaller, no CUDA) |
| Job script | `qsub scripts/nus_hpc_ingest_index.pbs` (hardcoded GPU) | `python -m src.hpc --cpu -o myjob.pbs && qsub myjob.pbs` (0 GPUs, `cpu` queue, more cores) |
| Config | `config.toml` (`accelerator = "auto"`) | `config.cpu.toml` (`accelerator = "cpu"`, `vision_enabled = false`) — see below |
| Chat | Cluster GPU via serving job + tunnel | **Run the LLM locally** after copying `db/` home |
| Embedding throughput | One GPU (fast per-host) | **Multi-node sharding** — N Ollama replicas across N nodes (the real CPU lever) |

### 1. Build the CPU image

```bash
sudo singularity build rag_pipeline_cpu.sif Singularity.cpu.def
#  -- or --
singularity build --fakeroot rag_pipeline_cpu.sif Singularity.cpu.def
rsync -P rag_pipeline_cpu.sif nus_hpc:~/rag_pipeline_cpu.sif
```

### 2. CPU config (`config.cpu.toml`)

The PBS script can't set `accelerator` or `vision_enabled` — those are read from
`config.toml` by the pipeline, not the job script. So keep a CPU-specific config
and point the pipeline at it:

```bash
cp config.example.toml config.cpu.toml
```

Then edit `config.cpu.toml`:

```toml
[ingestion]
accelerator = "cpu"          # avoid CUDA-detection churn; you have no GPU
vision_enabled = false        # qwen2.5-vl crawls on CPU; disable for batch ingest
ingestion_workers = 4         # CPU nodes are core-rich; raise to use them

[embeddings]
# THE CPU scale-out lever: list one Ollama replica per node you're sharding
# across. See "Multi-node embedding sharding" below.
hosts = ["http://node-a:11434", "http://node-b:11434", "http://node-c:11434"]
concurrency = 2               # in-flight batches per host; tune to core count
```

Run the ingest job pointed at this config:

```bash
RAG_PIPELINE_CONFIG=config.cpu.toml qsub myjob.pbs
# (or export RAG_PIPELINE_CONFIG in your shell before qsub -v)
```

### 3. Multi-node embedding sharding (the real CPU win)

A single CPU Ollama replica is slow. The config comment in `config.example.toml`
calls `[embeddings] hosts` "the PRIMARY scale-out lever — N replicas ≈ N×
throughput." On a CPU cluster the way to realize this is:

1. Submit N serving jobs (`python -m src.hpc --cpu --serve ...`), one per node,
   each listening on `127.0.0.1:11434` on its own node.
2. Either run one ingest job whose `config.cpu.toml` lists all N replicas in
   `[embeddings] hosts` (requires the nodes to be networked so the indexer can
   reach each replica — typical within a job array / co-scheduled allocation), **or**
3. Split your corpus into N shards and run N independent ingest jobs, one per
   node, each building a `db/` fragment you merge afterward.

> **Native embeddings vs Ollama replicas:** `config.example.toml` notes that
> `[models].native_embeddings = true` (SentenceTransformers on the local CPU) is
> 2–5× faster than the Ollama HTTP path *per host* — but it shards less cleanly
> across nodes. For a single-node CPU build, prefer native embeddings; for a
> multi-node build, prefer Ollama replicas via `[embeddings] hosts`.

### 4. After the build: copy `db/` home and chat locally

```bash
rsync -P --delete nus_hpc:~/<path-to-repo>/db/ ./db/
python -m src.web_app      # local Ollama (CPU or whatever you have) for chat
```

On CPU you generally **skip Phases 5–6** (the serving job + 2-hop tunnel) — that
topology exists to put a *fast GPU* behind the chat path, which isn't the case
on CPU. Build on the cluster, copy the index home, chat locally.

---

## Phase 1 — Build the Singularity image (once)

On your Linux build box:

```bash
# fakeroot avoids needing real root; run once to set up the namespace mapping:
singularity shell --fakeroot /dev/null  # priming step on some systems

# Build (this takes 15-40 min and pulls several GB):
sudo singularity build rag_pipeline.sif Singularity.def
#  -- or, with fakeroot already configured --
singularity build --fakeroot rag_pipeline.sif Singularity.def
```

Copy the resulting image to NUS shared storage (home is visible to all compute
nodes):

```bash
rsync -P rag_pipeline.sif nus_hpc:~/rag_pipeline.sif
```

> The `.pbs` scripts find the SIF at `${CONTAINER_SIF:-rag_pipeline.sif}` in the
> working directory, falling back to `/hpctmp2/$USER/rag_pipeline.sif`. Putting
> it in your home/repo root satisfies the first lookup.

---

## Phase 2 — Seed the model store (once)

Pick a **persistent, shared** location for the Ollama model registry. The
default is `${HOME}/ollama_models` — home is shared across compute nodes, which
is what you want. Only override this if home has a tight quota (then point it at
shared `/hpctmp2/$USER/ollama_models`).

The first ingest job (Phase 3) pre-pulls these models automatically, so you can
skip this phase and let Phase 3 do it. But running it standalone first is a good
way to fail fast on a bad model tag before committing to a long ingest:

```bash
# Quick one-off: a tiny PBS job that just pulls, then exits.
cat > /tmp/seed_models.pbs <<'EOF'
#PBS -N rag_seed_models
#PBS -l select=1:ncpus=2:mem=8gb:ngpus=1
#PBS -l walltime=01:00:00
#PBS -q gpu
#PBS -j oe
set -e
module load singularity
HOME="${HOME:-$(eval echo ~${USER:-$(whoami)})}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-${HOME}/ollama_models}"
mkdir -p "${OLLAMA_MODELS_DIR}"
CONTAINER_SIF="${CONTAINER_SIF:-rag_pipeline.sif}"
BIND="-B ${HOME}:/srv/home -B ${PWD}:/app -B ${OLLAMA_MODELS_DIR}:/srv/ollama_models"
export OLLAMA_MODELS=/srv/ollama_models
singularity exec --nv ${BIND} "${CONTAINER_SIF}" ollama serve > /tmp/ollama.log 2>&1 &
PID=$!
for i in $(seq 1 30); do
  singularity exec ${BIND} "${CONTAINER_SIF}" curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
  sleep 2
done
for m in nomic-embed-text qwen2.5vl:7b gemma4 qwen2.5:1.5b; do
  singularity exec ${BIND} "${CONTAINER_SIF}" ollama pull "$m"
done
kill $PID
EOF
qsub -v CONTAINER_SIF=rag_pipeline.sif /tmp/seed_models.pbs
```

Watch `qstat -f <jobid>`; if a pull fails, fix the tag in `config.toml` and
re-submit.

---

## Phase 3 — Ingest + index (the build)

From the repo root on the login node (the working dir becomes `/app` inside the
container):

```bash
# Option A: submit the checked-in script directly.
qsub scripts/nus_hpc_ingest_index.pbs

# Option B: generate a tuned variant (more CPUs/GPUs, longer walltime, ...).
python -m src.hpc --ngpus 1 --ncpus 8 --mem 32gb --walltime 12:00:00 -o myjob.pbs
qsub myjob.pbs
```

Useful `qsub` overrides (pass with `-v`):

| Variable | Meaning |
|----------|---------|
| `INPUT_DATA_DIR` | Where PDFs live (default `data`, i.e. `${PWD}/data`) |
| `OLLAMA_MODELS_DIR` | Persistent model store (default `${HOME}/ollama_models`) |
| `OLLAMA_MODELS_TO_PULL` | Space-separated tags (defaults to the 4 above) |
| `CONTAINER_SIF` | Path to the image (default `rag_pipeline.sif`) |

Monitor:

```bash
qstat -u $USER                 # status
cat rag_ingest_index.o<jobid>  # stdout/stderr (PBS -j oe)
```

The job writes `processed_docs/` and `db/` (per `config.toml [paths]`) into the
working directory on shared storage. The streaming indexer checkpoints every
250 files / 15 min, so a killed job can be cheaply resumed by re-`qsub`-ing
(chunking is idempotent per file).

> **Embedding throughput is the bottleneck.** For a large cold corpus, consider
> the multi-host/native-GPU scale-out levers in `[embeddings]` and
> `[models].native_embeddings` (documented in `config.example.toml`). That work
> is out of scope for this runbook but is where the real time savings are.

---

## Phase 4 — Copy the built index home

The web app reads `db/` locally, so bring it to your laptop:

```bash
# From your laptop:
rsync -P --delete nus_hpc:~/<path-to-repo>/db/ ./db/
```

(`--delete` keeps the local copy in sync if you re-run Phase 3 later.)

---

## Phase 5 — Serve Ollama on a GPU (long-lived)

Submit the serving job; it stays alive until its walltime, publishing its
compute-node hostname to `~/.rag_ollama_serving_host`:

```bash
# From the login node, repo root:
qsub scripts/nus_hpc_serve.pbs

# Or a tuned variant:
python -m src.hpc --serve --walltime 08:00:00 -o serve.pbs && qsub serve.pbs
```

Wait until the discovery file is populated (a few seconds after the job starts
running):

```bash
# On the login node:
watch -n 5 'cat ~/.rag_ollama_serving_host 2>/dev/null && echo'
```

When it prints a hostname, Ollama is reachable on that node at port 11434. Check
job status with `qstat -u $USER`. To stop serving early: `qdel <jobid>` — the
job's cleanup trap removes the discovery file.

---

## Phase 6 — Start the 2-hop tunnel (from your laptop)

The tunnel forwards your local `127.0.0.1:11434` to Ollama on the compute node,
hopping through the login host. It re-reads the discovery file on every
reconnect, so a rescheduled serving job is picked up automatically.

**Bash (Linux/macOS/WSL):**

```bash
./scripts/tunnel_daemon.sh \
  --jump-host nus_hpc \
  --host-file '~/.rag_ollama_serving_host'
```

**PowerShell (Windows):**

```powershell
.\scripts\tunnel_daemon.ps1 `
  -JumpHost nus_hpc `
  -HostFile "$HOME\.rag_ollama_serving_host"
```

Leave this running in a terminal. Verify the forward from another shell:

```bash
curl -s http://127.0.0.1:11434/api/version    # should return {"version":"..."}
```

> The `--host-file` path is interpreted **on the jump host**, so the `~` is the
> cluster user's home. The daemon `ssh nus_hpc "cat <file>"` to read it.

---

## Phase 7 — Run the web app (laptop)

```bash
python -m src.web_app
# serves on 127.0.0.1:8000 per config.toml [server]
```

It reads the local `db/` and sends embedding/chat traffic to
`http://127.0.0.1:11434` — which is now your tunnel to the cluster GPU.

Open the UI in a browser, then verify the wiring end-to-end:

```bash
curl -s http://127.0.0.1:8000/api/health | python -m json.tool
```

Expect (the M4 failover surface in `web_app.py`):

```json
{
  "ollama_active_host": "http://127.0.0.1:11434",
  "ollama_candidate_hosts": ["http://127.0.0.1:11434"],
  "ollama_reachability": { "http://127.0.0.1:11434": true }
}
```

`ollama_reachability` being `true` confirms the tunnel is live. If it's `false`,
check the tunnel daemon's output and the serving job (`qstat -f`).

---

## Day-2 operations

- **Resume a killed ingest**: just `qsub` it again. Per-file idempotency in the
  ingestion path and the indexer checkpoint make this cheap.
- **Update models**: edit `OLLAMA_MODELS_TO_PULL` (or `config.toml [models]`)
  and re-submit either job; the pull step will fetch the new tags into the
  persistent store.
- **Serve for longer**: regenerate with a longer walltime, or chain a second
  serving job before the first expires.
- **Run ingest and serve concurrently**: fine — they use independent scratch
  dirs and share only the (read-mostly) model store.

## Troubleshooting

| Symptom | Likely cause / fix |
|--------|--------------------|
| `qsub` job dies within seconds | Walltime too short, or SIF path wrong. Check `rag_ingest_index.o<jobid>`. |
| `ollama pull failed for 'gemma4'` | `gemma4` is an alias, not a published tag. `ollama create` it in the model store once (Phase 2). |
| Models re-download every job | `OLLAMA_MODELS_DIR` is resolving to per-job scratch. Ensure it points at shared storage (home or `/hpctmp2/$USER`). |
| Tunnel: `discovery file is empty or missing` | Serving job isn't running yet, or the path differs from `OLLAMA_HOST_FILE`. `cat ~/.rag_ollama_serving_host` on the login node to confirm. |
| `/api/health` shows `ollama_reachability: false` | Tunnel is down. Check `tunnel_daemon` output; confirm `qstat` shows the serve job is R(un). |
| `singularity: FATAL: could not open image` | SIF not at expected path. `ls -l rag_pipeline.sif` from the `qsub` working dir, or pass `-v CONTAINER_SIF=/full/path`. |
| CRLF errors (`/bin/bash^M: bad interpreter`) | Someone edited a `.pbs`/`.sh` on Windows without `.gitattributes`. `git add --renormalize .` and re-commit. |
