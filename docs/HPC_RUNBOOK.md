# NUS HPC Runbook

Build the FSAE RAG index on NUS HPC, then chat through the web app on your
laptop.

> **Path A (GPU serving) is historical.** LLM serving moved to the hosted
> SoCLAaS API, and the GPU serving job, SSH tunnel, and `Singularity.def` were
> removed from the repository. Path A is retained below for reference only;
> use Path B — the CPU cluster runs the ingest-only bulk parse, embeddings run
> on the workstation (localhosted nomic-embed-text), and chat/vision come from
> the SoCLAaS API.

- **CPU-only cluster (free, unlimited):** Path B (recommended).

For a web-server instance, the recommended entry point is the guided setup from
the repository root:

```powershell
.\setup.cmd
```

On Linux/macOS use `./setup.sh`. It writes the `[hpc]`
configuration, creates/updates the CPU alias in `~/.ssh/config`, records the
repo path relative to the SSH login directory, checks SSH/PBS/images, and can
start the web server. The manual steps below remain useful for
cluster preparation and troubleshooting.

The wizard generates a dedicated Ed25519 key for the alias and installs the
public key remotely. Expect one password/MFA prompt on the first
run. If the site disables password-based key bootstrap, use
`--skip-key-install` and ask the cluster administrator to install the generated
`.pub` file.

It also provisions the current repository and its matching container beneath
`/hpctmp/<username>/<repo-path>` on Atlas9 CPU. A login-directory symlink keeps
the configured repo path short (for example `rag-cpu`). An existing local SIF is
uploaded; a missing SIF is built remotely with Singularity/Apptainer
`--fakeroot`. Use `--skip-hpc-provision` only when managing those artifacts
separately.

## Prerequisites (one-time)

1. **SSH access** to NUS HPC. The guided setup creates the key and alias.
   For manual setup, add an alias to `~/.ssh/config`:

   ```sshconfig
   Host nus_hpc
       HostName <nus-login-hostname>
       User <your-nus-username>
       IdentityFile ~/.ssh/id_ed25519
       ServerAliveInterval 15
       ServerAliveCountMax 3
   ```

2. **A Linux box with root/fakeroot** to build the Singularity image (a Linux
   workstation, VM, or WSL2). You can't build SIFs on the login node or on Windows.

3. **Project checked out** on the cluster. The PBS scripts bind-mount `${PWD}:/app`,
   so `qsub` them from the repo root.

4. **A SoCLAaS API key** provisioned for the job: `~/rag_soclaas_key`
   (chmod 600) on the login node, the `SOCLAAS_API_KEY` qsub environment
   variable, or `[llm_api].api_key` in the staged `config.toml`.

---

## Path A — GPU cluster (paid) — HISTORICAL

> The commands in this path reference `Singularity.def`, `scripts/nus_hpc_serve.pbs`,
> and `scripts/tunnel_daemon.*`, which were removed from the repository when LLM
> serving moved to the SoCLAaS API. Kept for archaeology only.

### A1. Build the image (once)

```bash
sudo singularity build rag_pipeline.sif Singularity.def
#  -- or, with fakeroot configured --
singularity build --fakeroot rag_pipeline.sif Singularity.def
rsync -P rag_pipeline.sif nus_hpc:~/rag_pipeline.sif
```

The scripts find the SIF at `${CONTAINER_SIF:-rag_pipeline.sif}` in the working
directory, falling back to `/scratch/$USER/rag_pipeline.sif` on Vanda.

### A2. Seed the model store (once)

Models must live on **persistent, shared** storage (default `${HOME}/ollama_models`).
The first ingest job pre-pulls them; there is no separate seed job, so the
easiest check is to let A3 pull on its first run (it fails fast on a bad model
tag).

```bash
qsub scripts/nus_hpc_ingest_index.pbs   # first run pre-pulls the models
```

### A3. Ingest + index

From the repo root:

```bash
# Option A: the checked-in script.
qsub scripts/nus_hpc_ingest_index.pbs

# Option B: a tuned variant (more GPUs/CPU, longer walltime, ...).
python -m src.hpc --ngpus 1 --mem 32gb --walltime 12:00:00 -o myjob.pbs
qsub myjob.pbs
```

Useful `qsub -v` overrides: `INPUT_DATA_DIR`, `OLLAMA_MODELS_DIR`,
`OLLAMA_MODELS_TO_PULL`, `CONTAINER_SIF`. Monitor with `qstat -u $USER` and
`cat rag_ingest_index.o<jobid>`. Output lands in `db/` per `config.toml [paths]`.

### A4. Serve Ollama on a GPU + tunnel from your laptop

Chat uses the cluster GPU. Submit the long-lived serving job (it publishes its
compute-node hostname so the tunnel can find it):

```bash
qsub scripts/nus_hpc_serve.pbs
# wait until the discovery file is populated:
watch -n 5 'cat ~/.rag_ollama_serving_host 2>/dev/null && echo'
```

Then from your laptop, start the 2-hop tunnel (laptop → login → compute node):

```bash
./scripts/tunnel_daemon.sh --jump-host nus_hpc --host-file '~/.rag_ollama_serving_host'
# verify:
curl -s http://127.0.0.1:11434/api/version
```

Stop serving early with `qdel <jobid>`.

### A5. Run the web app

See [Copy the index home & run the web app](#copy-the-index-home--run-the-web-app).

---

## Path B — CPU-only cluster (free)

The cluster's only job is the bulk PDF parse (Docling OCR/pypdf extraction into
`processed_docs/` Markdown). Embeddings are workstation-local (nomic-embed-text
via Ollama), so the index cannot be built on the cluster and is built at home;
chat runs on the SoCLAaS API.

### B1. Build the image (once)

```bash
sudo singularity build rag_pipeline_cpu.sif Singularity.cpu.def
#  -- or --
singularity build --fakeroot rag_pipeline_cpu.sif Singularity.cpu.def
rsync -P rag_pipeline_cpu.sif nus_hpc:~/rag_pipeline_cpu.sif
```

### B2. CPU config

The PBS script can't set `accelerator` or `vision_enabled` — those are read from
`config.toml`. Keep a CPU-specific config:

```bash
cp config.example.toml config.cpu.toml
```

Edit `config.cpu.toml`:

```toml
[ingestion]
accelerator = "cpu"          # no CUDA on this cluster
ingestion_workers = 4        # CPU nodes are core-rich
# vision enrichment (qwen3-vl) runs through the SoCLAaS API, not local
# compute, so vision_enabled can stay true -- provision the API key at
# ~/rag_soclaas_key (chmod 600) on the login node.
```

Embeddings never run on the cluster in the default (ingest-only) flow, so no
embedding settings matter there.

### B3. Ingest (the cluster's only job)

Preferred — run everything from the workstation with one command (extract,
upload, submit, monitor with exit-code verification, fetch, index):

```bash
./start.sh --initial-corpus corpus.zip      # or start.cmd on Windows
```

Manual equivalent — stage the corpus first. An initial zip of PDFs is fine;
unpack it on the login node into the corpus data dir, KEEPING the directory
structure (the pipeline discovers PDFs recursively and handles duplicate
names, whereas `unzip -j` flattening silently overwrites same-named PDFs from
different directories):

```bash
mkdir -p /hpctmp/$USER/rag-corpus/data && cd /hpctmp/$USER/rag-corpus/data
unzip -oq corpus.zip && rm corpus.zip
```

The corpus lives under `/hpctmp/$USER/rag-corpus/` — outside the deployed
repo directory — so a re-provision (which atomically replaces the repo)
cannot wipe it.

Submit the ingest-only job (`--skip-index` makes it run
`bulk_ingest.py --skip-index`, stopping after parsing):

```bash
python -m src.hpc --cpu --skip-index \
  --input-data-dir /hpctmp/$USER/rag-corpus/data \
  --processed-dir /hpctmp/$USER/rag-corpus/processed_docs \
  -o ingest_only.pbs
RAG_PIPELINE_CONFIG=config.cpu.toml qsub ingest_only.pbs
```

Bring the Markdown home and build the index locally (local Ollama with
`nomic-embed-text` pulled). Check the PBS exit status first — a walltime
kill leaves a partial `processed_docs/` that would otherwise silently index
an incomplete corpus:

```bash
rsync -P nus_hpc:/hpctmp/$USER/rag-corpus/processed_docs/ ./processed_docs/
python main.py --mode index --md_dir processed_docs --db_dir db
```

Programmatic equivalent (what `--initial-corpus` runs, plus exit-code
verification and a fetched-corpus completeness check):
`HpcBackend.push_corpus_dir()` → `submit_ingest_index(skip_index=True)` →
`fetch_processed_docs()`.

Variant — build the whole index on the cluster instead (embeddings via the
SoCLAaS API; also set `[models] embedding_model = "bge-m3"`,
`embedding_dim = 1024` in the config the job reads, and expect a full re-index
when switching back):

```bash
python -m src.hpc --cpu -o myjob.pbs
RAG_PIPELINE_CONFIG=config.cpu.toml EMBEDDINGS_BACKEND=soclaas qsub myjob.pbs
```

### B4. Copy the corpus home & run the web app

See [Copy the corpus home & run the web app](#copy-the-corpus-home--run-the-web-app).

### B5. Do you ever need GPU? Measure it.

The config's "weeks of embedding work" warning is a *single-host* number. Native
embeddings are 2–5× faster, and the corpus shards across N free nodes. Measure
the real rate on your corpus:

```bash
# size the job first (no models/network):
python scripts/estimate_embed_throughput.py --count --sample-dir processed_docs/
# measure native-CPU rate on one node, project across node counts:
python scripts/estimate_embed_throughput.py --native --sample-dir processed_docs/ --num-nodes 1,4,8,16
# scale a small sample up to your real corpus:
python scripts/estimate_embed_throughput.py --native --sample-dir processed_docs/sample/ --scale 50 --num-nodes 16
```

If the free-CPU projection is acceptable (hours), you never need the paid GPU
cluster for bulk work. If your corpus is heavily scanned and vision dominates, a
*vision-only* GPU pass could be worth building — ask.

---

## Copy the corpus home & run the web app

The default (ingest-only) path brings `processed_docs/` home and builds the
index locally — see B3. The cluster-built-index variant instead syncs the
finished `db/`:

```bash
rsync -P --delete nus_hpc:~/<path-to-repo>/db/ ./db/
```

Run the web app locally (it reads local `db/`, embeds queries through local
Ollama `nomic-embed-text`, and chats via the SoCLAaS API):

```bash
python -m src.web_app    # serves on 127.0.0.1:8000
```

Verify the wiring:

```bash
curl -s http://127.0.0.1:8000/api/health | python -m json.tool
# expect: "llm_backend": "soclaas" (reachable) for chat/vision, and
# "embeddings_backend": "ollama" with "embeddings_reachable": true
```

---

## Day-2 operations

- **Resume a killed ingest:** just `qsub` it again — per-file idempotency plus the
  indexer checkpoint make this cheap.
- **Update models:** edit `OLLAMA_MODELS_TO_PULL` (or `config.toml [models]`) and
  re-submit either job.
- **Serve longer:** regenerate with a longer walltime, or chain a second serving
  job before the first expires.

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| Job dies within seconds | Walltime too short, or SIF path wrong. Check `rag_ingest_index.o<jobid>`. |
| `ollama pull failed for 'gemma4'` | `gemma4` is an alias, not a published tag. `ollama create` it in the model store once. |
| Models re-download every job | `OLLAMA_MODELS_DIR` resolves to per-job scratch. Point it at shared storage (home, Atlas9 `/hpctmp/$USER`, or Vanda `/scratch/$USER`). |
| Tunnel: `discovery file is empty or missing` | Serving job isn't running yet. `cat ~/.rag_ollama_serving_host` on the login node. |
| `/api/health` shows `ollama_reachability: false` | Tunnel down, or serve job not `R(un)`. Check `tunnel_daemon` output and `qstat`. |
| `singularity: FATAL: could not open image` | SIF not at expected path. `ls -l rag_pipeline.sif` from the `qsub` dir, or pass `-v CONTAINER_SIF=/full/path`. |
| `/bin/bash^M: bad interpreter` | Someone edited a `.pbs`/`.sh` on Windows. `git add --renormalize .` and re-commit. |
| `install.sh` fails with `requires zstd`, or `ollama-linux-amd64.tar.gz` returns 404 | The Ollama installer now ships a `.tar.zst` (zstd-compressed), and the old gzip `ollama.com/download/...tar.gz` URL is gone. The `.def` files apt-install `zstd` and the fallback uses the current asset `github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst` extracted with `tar --zstd -xf`. If it still fails, `sudo apt-get install -y zstd` on the build host. |
| `pip ... from versions: ... max X` for a package you know is newer (e.g. onnxruntime caps at 1.23.2) | The build host is hitting a **stale PyPI mirror/proxy**, not the real PyPI. The `.def` files force pypi.org as primary (`--index-url https://pypi.org/simple`), so a clean rebuild usually fixes it. If it persists, build with `PIP_INDEX_URL=https://pypi.org/simple singularity build ...`. Don't lower the pin — the version exists on real PyPI with a matching wheel. |
| `requires-python >=3.11` / `max 1.23.2` specifically for **onnxruntime or numpy** during the build | The container was building on Python 3.10 (ubuntu:22.04 / cuda base) — those two pins need ≥3.11 and have no cp310 wheel. The `.def` files now build on **Python 3.14** (CPU image: `python:3.14-slim` base; GPU image: deadsnakes `python3.14` on the CUDA base, all `pip` routed through `python3.14 -m pip`). This also matches the local dev Python, so what you test locally is what the container runs. |
