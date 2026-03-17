# Setup Guide: Local STEM RAG Pipeline

Optimized for **Ryzen 9 9950X3D · RTX 4000 Ada 20GB · 128GB DDR5 · Windows 11 / WSL2**.

---

## 1. Prerequisites

### Hardware Requirements
| Component | Minimum | This Build |
| :--- | :--- | :--- |
| GPU | NVIDIA (8GB+ VRAM, CUDA 12.x) | RTX 4000 Ada 20GB |
| CPU | 8-core with AVX2 | Ryzen 9 9950X3D (AVX-512, 128MB L3) |
| RAM | 32GB | 128GB DDR5 |
| Storage | 100GB NVMe free | 500GB+ recommended |

### Why these specs matter
- **20GB VRAM** is the practical minimum to run DeepSeek-R1-32B at Q4_K_M quantization (~19–22GB) without CPU layer offloading, which would reduce inference speed by 10–20x.
- **AVX-512 + large L3 cache** on the Ryzen 9 accelerates the dense matrix multiplications inside SentenceTransformer (Nomic embedding). The 128MB 3D V-Cache keeps embedding batch data resident across calls, reducing DRAM bandwidth pressure.
- **128GB RAM** allows LanceDB's memory-mapped vector index to be kept fully resident, eliminating disk I/O latency during ANN search entirely.

---

## 2. Software Prerequisites

### 2.1 CUDA Toolkit
The pipeline requires CUDA 12.1 or higher for:
- **Docling** — runs Heron (layout analysis) and TableFormer (table reconstruction) on the GPU via PyTorch's CUDA backend.
- **PyTorch** — all GPU tensor operations require a matching CUDA runtime.
- **Ollama** — uses CUDA for quantized model inference.

Install from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). Select **Windows → x86_64 → exe (local)**.

Verify after installation:
```bash
nvcc --version
# Expected: Cuda compilation tools, release 12.x
nvidia-smi
# Expected: Driver Version >= 525.xx, CUDA Version 12.x
```

### 2.2 Ollama
Ollama is a local model server that handles:
- **Quantized model loading** — converts model weights to GGUF format and loads them into VRAM using llama.cpp under the hood.
- **VRAM lifecycle management** — the `keep_alive` parameter controls how long a model stays in VRAM after inference. The pipeline uses this to swap models between phases.
- **REST API** — exposes a local HTTP endpoint that the Python `ollama` library calls; each `ollama.generate()` in the pipeline is an HTTP POST to `localhost:11434`.

Download from [ollama.com](https://ollama.com) and install. Verify:
```bash
ollama --version
```

Pull the three required models:
```bash
# Primary reasoning LLM (~19GB download, Q4_K_M quantized)
# Used for: entity extraction during indexing, CoT answer generation during querying
ollama pull deepseek-r1:32b

# Vision-Language Model (~5GB download, Q4 quantized)
# Used for: describing figures, charts, and diagrams found in PDFs
ollama pull qwen2.5vl:7b

# Embedding model (reference only — the pipeline uses the HuggingFace version directly via SentenceTransformer, NOT this Ollama copy)
# Pull for reference/testing, but it is not called by the pipeline
ollama pull nomic-embed-text
```

> **Why SentenceTransformer instead of Ollama for embeddings?**
> The pipeline loads `nomic-embed-text-v1.5` directly via `SentenceTransformer` with `device="cpu"`. This is intentional: it forces embedding onto the CPU, preserving ~2GB of VRAM for the LLMs. The Ollama-served version would load into VRAM and compete with DeepSeek-R1.

### 2.3 Conda (Miniconda)
Conda is used instead of `venv` for three reasons:
1. **Binary dependency management** — `venv` is Python-only; CUDA-linked packages (PyTorch, torchvision) require exact CUDA/cuDNN runtime versions that conda can resolve and install atomically.
2. **Environment reproducibility** — `conda env export` captures both Python packages and native library versions, making the environment fully reproducible across machines.
3. **Isolation from system Python** — avoids conflicts with system-level packages on Windows/WSL2 where the system Python may have conflicting CUDA or numpy builds.

Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install. Verify:
```bash
conda --version
```

---

## 3. Environment Setup

### Step 1 — Create the conda environment
```bash
conda create -n ragpipeline python=3.11 -y
conda activate ragpipeline
```
Python 3.11 is recommended over 3.12 for compatibility with `docling` and `lightrag` which have dependencies that may not yet have 3.12 wheels.

### Step 2 — Install PyTorch with CUDA
Install PyTorch using the official pip wheel URL rather than the conda pytorch channel. The pip wheels are released faster and always match the latest CUDA versions:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **What this does:** Downloads PyTorch wheels pre-compiled against CUDA 12.1. These wheels include the CUDA runtime libraries, so PyTorch does not depend on the system CUDA installation at runtime — it ships its own `libcudart`. Replace `cu121` with `cu124` if you are running CUDA 12.4+.

Verify GPU is visible to PyTorch:
```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA RTX 4000 Ada Generation
```

### Step 3 — Install pipeline dependencies
```bash
pip install -r requirements.txt
```

> **What each package provides:**
> - `docling` — PDF parsing engine; uses PyTorch CUDA backend for layout analysis and OCR acceleration.
> - `lightrag` — GraphRAG framework; manages chunking, NER via LLM, and the NetworkX knowledge graph.
> - `lancedb` — embedded columnar vector database; stores 768d Nomic embeddings with memory-mapped ANN index.
> - `ollama` — Python client for the Ollama REST API; used for model swapping and LLM/VLM inference calls.
> - `sentence-transformers` — loads `nomic-embed-text-v1.5` on CPU; handles tokenization, forward pass, and pooling.
> - `scikit-learn` — KMeans for pre-indexing chunk deduplication and quotient graph partitioning.
> - `networkx` — reads LightRAG's `.graphml` entity graph for MRL bridge edge computation and quotient graph construction.
> - `pydantic` — data validation; Ollama's `GenerateResponse` object is a Pydantic model.
> - `numpy` — vector truncation and L2 normalization of MRL embeddings.
> - `python-dotenv`, `tqdm` — used internally by LightRAG and Docling.

### Step 4 — Verify the Nomic model downloads correctly
On first run, SentenceTransformer will download `nomic-ai/nomic-embed-text-v1.5` from HuggingFace (~550MB) and cache it in `~/.cache/huggingface/`. Verify manually:
```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")
print(m.encode(["search_document: test"]).shape)
# Expected: (1, 768)
```
`trust_remote_code=True` is required because Nomic's model uses a custom pooling architecture not included in the base transformers library — the model repo ships its own `modeling_nomic_bert.py` that gets executed locally.

---

## 4. Directory Structure

```
RAGprojectWindows/
├── data/                   # Place input PDFs here
├── processed_docs/         # Enriched markdown output from ingestion phase
├── db/                     # LightRAG working directory (vector index + knowledge graph)
├── src/
│   ├── ingestion.py        # DocumentProcessor (Docling + Qwen VLM), EmbeddingEngine (Nomic)
│   ├── indexing.py         # Indexer — feeds markdown into LightRAG's graph + vector pipeline
│   ├── query.py            # QueryEngine — hybrid local/global/graph retrieval
│   └── utils.py            # VRAM management, LightRAG instance factory
├── main.py                 # CLI orchestrator
└── requirements.txt
```

The `processed_docs/` and `db/` directories are created automatically by the pipeline if they do not exist.

---

## 5. Running the Pipeline

### Full pipeline (ingest → index, then query separately)
```bash
# Phase 1: Convert PDFs to enriched markdown
# Docling parses layout on GPU; Qwen2.5-VL describes figures (loaded lazily)
python main.py --mode ingest --data_dir data --md_dir processed_docs

# Phase 2: Build knowledge graph + vector index
# DeepSeek-R1 extracts entities; Nomic embeds chunks on CPU in parallel
python main.py --mode index --md_dir processed_docs --db_dir db

# Phase 3: Query
python main.py --mode query --db_dir db --question "Explain the relationship between entropy and the Boltzmann distribution" --query_mode hybrid
```

### Run ingest and index together
```bash
python main.py --mode all --data_dir data --md_dir processed_docs --db_dir db
```

### Query modes
| Mode | Mechanism | Best for |
| :--- | :--- | :--- |
| `local` | ANN cosine search on 768d vectors in LanceDB | Specific definitions, direct quotes |
| `global` | Graph traversal via entity nodes and relational edges | Cross-document relationships, theory synthesis |
| `hybrid` | Both combined into one context window | General STEM reasoning (default) |

---

## 6. Technical Notes on Key Mechanisms

### VRAM Swapping
The pipeline calls `ollama.generate(model=name, prompt=" ", keep_alive="0")` to evict a model from VRAM before loading the next one. Under the hood, this tells Ollama's llama.cpp backend to release the GPU memory allocation and unmap the GGUF weight file. The `keep_alive="0"` string form is used (rather than integer `0`) for compatibility across Ollama versions.

### Lazy VLM Loading
`DocumentProcessor._ensure_vision_loaded()` uses a boolean flag (`_vision_model_loaded`) to load Qwen2.5-VL on the first figure detected. If a batch of PDFs contains no images, the VLM is never loaded and ~8GB VRAM remains available to Docling throughout the entire ingestion phase.

### MRL Embedding — 768d Full Fidelity
`nomic-embed-text-v1.5` is trained with Matryoshka Representation Learning, meaning the first N dimensions of a 768d vector form a valid lower-dimensional embedding. The pipeline retains all 768 dimensions (no truncation) to maximize retrieval accuracy on STEM terminology where subtle semantic distinctions matter. After encoding, vectors are L2-normalized so that cosine similarity is equivalent to a dot product, which is what LanceDB's ANN index uses internally.

### Pre-Indexing Chunk Deduplication
Before any LightRAG NER call, `index_markdown()` pre-chunks all markdown files using `EmbeddingEngine.chunk_text()` (1200-token / 300-token overlap, 4 chars≈1 token approximation). All chunks across all documents are then passed to `EmbeddingEngine.deduplicate_chunks()`, which embeds them at full 768d, runs KMeans to narrow pairwise comparison to within-cluster pairs, and removes chunks with cosine similarity above 0.90.

This step eliminates verbatim duplicate content (republished theorems, repeated definitions across textbooks) before paying DeepSeek-R1 NER cost. Threshold 0.90 preserves cross-perspective chunks (same topic, different exposition, scoring 0.65–0.85) while removing near-verbatim copies.

### Single Event Loop for Indexing
`indexing.py` collects all deduplicated chunks, then calls `asyncio.run(_async_index_all(...))` once. Each `rag.insert()` call awaits completion before the next begins, preventing concurrent writes to LightRAG's shared graph and vector store. This replaces the previous pattern of calling `asyncio.run()` inside a `for` loop, which created and destroyed the event loop on every file and risked corrupting LightRAG's internal async state.

### MRL Bridge Edges
After LightRAG indexing, `build_mrl_bridge_edges()` reads the entity graph (`.graphml`), embeds all entity node names at 128d using the same Nomic model, and computes pairwise cosine similarity. Entity pairs that exceed similarity 0.82 but have no existing graph edge are written as cross-domain bridge edges to `db/mrl_bridge_edges.json`. These edges capture semantic kinship between entities that never co-occurred in the same 1200-token chunk — for example, "entropy" in thermodynamics and "entropy" in information theory, which are indexed from different textbooks and never share a chunk context.

128d is chosen because lower MRL prefixes generalize better across domain vocabulary shifts. In the full 768d space, disciplinary synonyms diverge due to different surrounding contexts. In the coarser 128d prefix they converge, correctly linking them.

### Quotient Graph
`build_quotient_graph()` loads the entity graph enriched with bridge edges, embeds all entity names at 256d, and clusters them with KMeans (sqrt heuristic, clamped to [4, 100] clusters). Each cluster becomes a quotient node; inter-cluster edges (from both LightRAG NER and bridge edges) are inherited as quotient edges and written to `db/quotient_graph.json`.

The quotient graph provides a coarse-grained corpus map: instead of navigating thousands of entity nodes, queries can resolve to a thematic cluster (e.g., "probability theory", "linear algebra") and then drill down. 256d is the right resolution — 128d over-merges domain vocabulary; 768d yields one cluster per entity with no compression benefit.

### Inline Vision Injection and Chunk Context
Vision descriptions are injected at the exact document position of each figure during item iteration, not appended at the end. This is essential for RAG quality: LightRAG chunks at 1200 tokens, so a figure description appended after all document text would land in an isolated chunk with no surrounding context. DeepSeek-R1's NER on that chunk would extract entities like "x-axis: Energy (eV)" without knowing the surrounding text discusses "Boltzmann distributions" — the graph edge between them would never be created. With inline injection, the figure description, the preceding paragraph, and the following paragraph all land in the same or adjacent chunks, giving both the NER pass and the embedding an accurate picture of what the figure represents.

Text content from non-figure items is extracted via `item.export_to_markdown()` at the item level (valid for tables and structured elements in Docling 2.x) with a fallback to `item.text` for plain text items.

### Chunk Size and Graph Density
Chunks of 1200 tokens with 300-token overlap are passed to both the embedding function and the NER LLM. Smaller chunks mean individual mathematical definitions, theorems, or proofs map to distinct graph nodes rather than being merged. The 300-token overlap ensures that concepts near chunk boundaries are captured in adjacent nodes and connected via shared entities during deduplication.

---

## 7. Pinning Dependency Versions

Once the pipeline has been validated end-to-end, pin all package versions to prevent silent API breakage:

```bash
conda activate ragpipeline
pip freeze | grep -E "docling|lightrag|lancedb|ollama|sentence.transformers|torch|numpy|pydantic" > versions.txt
```

Then update `requirements.txt` with the pinned versions (e.g., `docling==2.x.x`). See the note in `requirements.txt`.
