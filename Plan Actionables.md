# Plan & Actionables - Local RAG Pipeline (v4.0)

### Purpose of this Documentation
This document defines the end-to-end technical architecture and roadmap for a local-only STEM RAG pipeline. It is optimized for the **Ryzen 9 9950X3D, RTX 4000 Ada 20GB VRAM, 128GB RAM**, prioritizing structural accuracy for textbooks and relational synthesis for complex theories.

---

## 1. End-to-End Pipeline Technical Workflow

### Phase 1: Orchestration & Resource Management
**Scripts:** `main.py`, `src/utils.py`
**Mechanism:**
The pipeline uses a **lazy and sequential model swapping** strategy to maximize the 20GB RTX 4000 Ada GPU.

1. **No Preloading Before Ingestion:** `main.py` starts the ingestion phase immediately without loading any model. The vision model (Qwen2.5-VL) is loaded lazily inside `DocumentProcessor._ensure_vision_loaded()` only when the first figure is detected in a PDF. If a document contains no images, the VLM is never loaded and ~8GB of VRAM remains free.
2. **VRAM Eviction Before Indexing:** When transitioning from ingestion to indexing, `manage_vram()` calls `ollama.generate(model=name, prompt=" ", keep_alive="0")` to evict any resident models (including Qwen if it was loaded). It then pre-loads DeepSeek-R1-32B with a 30-minute `keep_alive` window.
3. **Eviction Protocol:** Ollama's `keep_alive` API controls how long a model stays resident in VRAM after inference. Setting `keep_alive="0"` (string form, more reliable across Ollama versions than integer `0`) causes immediate eviction after the next request.

### Phase 2: Document Ingestion & Vision Enrichment
**Script:** `src/ingestion.py`
**Mechanism:**
1. **Docling Parsing (GPU Accelerated):** `DocumentProcessor` initializes Docling with `AcceleratorOptions(device=AcceleratorDevice.CUDA)`, hooking into PyTorch's CUDA backend to run layout analysis (Heron model) and table reconstruction (TableFormer) on the RTX 4000 GPU.
2. **Inline Markdown Assembly:** The pipeline iterates all document items once via `doc.iterate_items()`. For each item:
   - **Figure items** (`picture`, `figure`, `chart`): vision description is injected at that exact position in the output.
   - **All other items**: text is extracted via `item.export_to_markdown()` (item-level call — valid Docling API for tables and structured items) with a fallback to `item.text`.
   - Label detection uses `str(item.label).lower()` for version-safe string comparison.
   - **Why inline matters for RAG:** A figure must appear in the same 1200-token chunk as its surrounding text. If a figure showing "Boltzmann probability density vs. energy" is appended at the end of the document, LightRAG's NER processes it in an isolated chunk with no surrounding context. The entity graph cannot link the figure to "Boltzmann distribution" or "most probable energy state" unless those terms are in the same chunk. Inline injection ensures the vision description, the text preceding the figure, and the text following it all land in the same or adjacent chunks.
3. **Lazy Vision Trigger:** On first figure detection, `_ensure_vision_loaded()` warms up Qwen2.5-VL via Ollama. Subsequent figures reuse the already-loaded model.
4. **In-Memory Encoding:** The PIL Image returned by `item.get_image(doc)` is buffered to a PNG byte stream (`io.BytesIO`) and base64-encoded in memory — no temporary files on disk.
5. **STEM-Focused VLM Prompt:** The vision prompt explicitly requests LaTeX representations for equations, axis labels and exact values for charts, and component labels for schematics. This produces machine-readable STEM content rather than generic descriptions.

### Phase 3: Matryoshka Representation Learning (MRL) Embedding
**Scripts:** `src/embeddings.py` (Refactored from `src/ingestion.py`), `src/utils.py`
**Mechanism:**
1. **CPU-Bound Model Initialization:** `nomic-embed-text-v1.5` is loaded via `SentenceTransformer` with `device="cpu"`, keeping ~2GB of VRAM available for the LLMs. The Ryzen 9's AVX-512 SIMD instructions and large 3D V-Cache (128MB L3) accelerate the dense matrix multiplications required for embedding.
2. **Asynchronous Threading:** The custom `mrl_embedding_func` wraps `get_mrl_embeddings()` in `asyncio.to_thread()`. This offloads the blocking CPU computation to a background OS thread, freeing the async event loop to issue the next LLM NER call concurrently — CPU and GPU work in parallel on the same chunk.
3. **Asymmetric Prefixing (ContextVars):** Text chunks must be prefixed with `"search_document: "` and queries with `"search_query: "`. Because LightRAG uses a single `EmbeddingFunc` hook, the pipeline uses Python's `contextvars` to set an explicit `embedding_mode` flag before insertion or querying. The `mrl_embedding_func` reads this thread-safe flag to deterministically apply the correct prefix, resolving Nomic's asymmetric retrieval requirement without fragile heuristics.
4. **Full-Fidelity Vectors:** The model generates 768-dimensional dense vectors. The pipeline retains all 768 dimensions — no truncation. Given 128GB of system RAM, the storage overhead (~3GB per 1M vectors) is negligible while retaining 100% of Nomic's semantic spatial information.
5. **L2 Normalization:** Vectors are L2-normalized (`/ np.linalg.norm(...)`) to unit length, ensuring cosine similarity equals dot product — required for LanceDB's ANN cosine index.

### Phase 4: Knowledge Graph Construction
**Scripts:** `src/indexing.py`, `src/utils.py`
**Mechanism:**

#### Step 1 — Tokenizer Alignment & Document Insertion
To avoid severe VRAM bloat and context truncation, LightRAG's default OpenAI tokenizer (`o200k_base`) is replaced with a compatibility shim around HuggingFace's `AutoTokenizer` for `DeepSeek-R1-Distill-Qwen-32B`. This ensures chunks sized to 1200 tokens exactly match the token count processed by the local LLM. Full documents (not pre-chunks) are passed to `rag.insert()` to preserve document-level hierarchy and long-range entity co-occurrence.

#### Step 2 — Custom Chunking & Deduplication Hook (`chunking_func`)
Deduplication is injected via LightRAG's `chunking_func` constructor parameter. The custom function chunks the document using the DeepSeek tokenizer and compares chunks against a global LanceDB index to filter out near-duplicates (cosine > 0.90) before NER dispatch. 
**Mechanism:** To prevent the system from embedding chunks twice (once for semantic dedup, once natively by LightRAG), a `_cache` dictionary is added to the shared `EmbeddingEngine` instance. This achieves O(log N) incremental deduplication and massive GPU compute savings without redundant CPU overhead.

#### Step 3 — LightRAG Indexing (Entity Extraction + Vector Storage)
Filtered chunks pass into LightRAG's async pipeline:
- **LanceDB Configuration:** `create_lightrag_instance()` explicitly sets `vector_storage="LanceDBStorage"` to bypass the default in-memory O(N) NanoVectorDB. It utilizes `use_mmap=True` and `max_memory="64GB"` for zero-disk-latency ANN search.
- **Entity Extraction (Hardware Concurrency):** For each chunk, LightRAG dispatches a NER prompt to DeepSeek-R1-32B on the GPU. Simultaneously, Nomic embedding runs on the CPU.
- **Graph Deduplication:** LightRAG uses DeepSeek-R1 to resolve entity variants globally (e.g., "P", "p-value", "p value" → single node).
- **Dual Storage:** 768d chunk embeddings stored in LanceDB. Extracted entity nodes and relational edges stored in LightRAG's NetworkX graph.

#### Step 4 — MRL Bridge Edges (256d ANN Search & Dual Injection)
After indexing, cross-domain semantic connections are computed. Instead of an O(N²) in-memory pairwise matrix, all entity names are embedded at 256d and inserted into a dedicated LanceDB table. An O(N log N) ANN radius search retrieves entity pairs with > 0.82 similarity.
**Mechanism (Dual Injection):** These bridge edges are injected into TWO locations to satisfy LightRAG's hybrid retrieval:
1. **NetworkX Graph:** Injected into the `.graphml` file with required schema attributes (`weight`, `description`, `source_id="mrl_bridge"`, `keywords`) so local graph traversal can rank and describe them.
2. **Vector DB:** The edge description is embedded and injected into LightRAG's `relationships_vdb` (LanceDB) so semantic global queries can retrieve them.

#### Step 5 — Quotient Graph (256d, KMeans)
`build_quotient_graph()` maps coarse-grained corpus clusters. Entity names are embedded at 256d, and KMeans clusters them into semantically coherent groups. 
**Mechanism:** This quotient graph is written to `db/quotient_graph.json`. It is intentionally kept separate from the primary `.graphml` entity graph to avoid namespace pollution. Querying this structure requires a custom two-stage query pre-processor.

### Phase 5: Querying & Hybrid Retrieval
**Script:** `src/query.py`
**Mechanism:**
1. **Query Input:** User provides a natural language question via `--question`.
2. **Query Embedding:** The question is embedded using the 768d CPU-bound MRL function, deterministically prefixed with `"search_query: "` via `contextvars`.
3. **Local / Vector Search:** LanceDB performs ANN cosine similarity search on the memory-mapped 768d chunks and `relationships_vdb`.
4. **Global / Graph Search:** LightRAG traverses the NetworkX knowledge graph for entity nodes and relational edges connected to keywords in the query.
5. **CoT Generation:** Retrieved chunks and graph context (including dual-injected MRL bridge edges) are synthesized by DeepSeek-R1-32B using Chain-of-Thought reasoning.

---

## 2. Hardware Resource & VRAM Management Map

| Phase | Active Models | Approx. VRAM | Strategy |
| :--- | :--- | :--- | :--- |
| **Ingestion (no images)** | Docling (CUDA) only | ~4–6 GB | VLM never loaded if no figures detected |
| **Ingestion (with images)** | Docling + Qwen2.5-VL-7B (Q4) | ~10–14 GB | VLM loaded lazily on first figure |
| **Indexing** | DeepSeek-R1-32B (Q4_K_M) | ~19–22 GB | Qwen evicted before load; Nomic on CPU |
| **Querying** | DeepSeek-R1-32B (Q4_K_M) | ~19–22 GB | Same quantization as indexing |

> **Note on Q8 Querying:** Running DeepSeek-R1-32B at Q8 requires ~32GB of VRAM — exceeding the RTX 4000 Ada's 20GB budget by 60%. Q8 in this configuration would require heavy CPU layer offloading via Ollama, which reduces inference speed to unusable levels for interactive querying. Q4_K_M is the correct target for this hardware.

### CPU/GPU Utilization
- **GPU (RTX 4000 Ada 20GB):** LLM inference (DeepSeek-R1), VLM vision tasks (Qwen2.5-VL), document layout analysis (Docling CUDA).
- **CPU (Ryzen 9 9950X3D):** Nomic MRL embedding (AVX-512 accelerated), LightRAG graph deduplication logic, LanceDB memory-mapped RAM index management.

---

## 3. Technical Stack (Local-Only)

| Layer | Tool | Rationale |
| :--- | :--- | :--- |
| **Layout** | **IBM Docling** | Best-in-class table/math extraction for STEM. CUDA-accelerated via PyTorch backend. |
| **Vision** | **Qwen2.5-VL-7B (via Ollama)** | Strong OCR and diagram reasoning; lazy-loaded only when figures are present. |
| **Graph** | **LightRAG** | Token-efficient GraphRAG; LLM-based entity deduplication is more accurate than vector clustering for STEM notation. |
| **Inference** | **Ollama** | Manages model quantization and CUDA memory; exposes a local REST API for model swapping. |
| **Vector DB** | **LanceDB** | Serverless, zero-copy, memory-mappable to RAM for near-zero-latency ANN search. |
| **Embedding** | **Nomic MRL (CPU, 768d)** | Full 768d vectors retained; CPU execution preserves VRAM for LLMs. ContextVars manages query vs doc prefixing. |

---

## 4. Known Limitations & Bottlenecks

### 1. VRAM OOM During Indexing
- **Risk:** DeepSeek-R1-32B at Q4_K_M requires ~19–22GB. Large prompts at peak can push past 20GB, causing Ollama to offload layers to CPU and dramatically slow inference.
- **Mitigation:** Tokenizer alignment to DeepSeek's Qwen2 vocabulary strictly enforces prompt limits. Nomic runs on CPU (saves ~2GB). Monitor `ollama logs` during indexing. If OOM occurs consistently, reduce `entity_extract_max_gleaning` from 1 to 0 in `utils.py`.

### 2. Async Event Loop Conflicts (Future UI Integration)
- **Risk:** `asyncio.run()` in `indexing.py` and `query.py` cannot be called from within a running event loop. Integrating into Streamlit, FastAPI, or Gradio will raise `RuntimeError: asyncio.run() cannot be called from a running event loop`.
- **Mitigation:** Current CLI usage is unaffected. Future API/UI integration requires refactoring `asyncio.run()` calls to native `await` syntax.

### 3. Docling API Fragility
- **Risk:** Figure detection uses `str(item.label).lower()` string comparison. If Docling changes its label naming convention across versions, image extraction will silently produce no vision descriptions.
- **Mitigation:** Pin the `docling` version in `requirements.txt`. Test after any upgrade with a known PDF containing figures.

### 4. Module Cohesion (`EmbeddingEngine`)
- **Risk:** Code organization currently places core embedding logic (`EmbeddingEngine`) inside `src/ingestion.py`, which should solely be responsible for PDF layout parsing.
- **Mitigation:** Extract `EmbeddingEngine` into a dedicated `src/embeddings.py` to ensure logical module boundaries.

---

## 5. Roadmap

### M1: Ingestion Engine
- [x] Docling CUDA-accelerated PDF parsing with two-pass markdown assembly.
- [x] Lazy VLM loading — Qwen2.5-VL loaded on first figure, skipped entirely for text-only PDFs.
- [x] STEM-focused vision prompt: LaTeX equations, axis values, component labels.

### M2: Knowledge Mapping
- [x] LightRAG configured with DeepSeek-R1-32B, utilizing a custom Qwen2 Tokenizer shim.
- [x] Nomic MRL 768d full-fidelity embeddings on CPU with L2 normalization and explicit `contextvars` intent prefixing.
- [x] Full-document `rag.insert()` with custom `chunking_func` and `_cache` mapping for semantic deduplication.
- [x] LanceDB explicit configuration for ANN memory-mapping.
- [x] MRL bridge edges at 256d — LanceDB ANN search followed by dual-injection into `.graphml` and `relationships_vdb`.
- [x] Quotient graph at 256d — coarse-grained corpus map written to `db/quotient_graph.json` requiring custom query pre-processor.
- [ ] **Validation:** Confirm notation deduplication across documents (e.g., "p-value" / "weights" / "P" → single node).
- [ ] **Validation:** Inspect `mrl_bridge_edges.json` and vector dual-injection logs.

### M3: Hardware Optimisation
- [x] Sequential model swapping with lazy VLM loading to prevent unnecessary VRAM consumption.
- [x] CPU-GPU concurrency via `asyncio.to_thread` (Nomic on CPU overlaps with DeepSeek-R1 on GPU).
- [x] Single event loop for all file insertions in `indexing.py` — no per-file loop destruction.
- [x] LanceDB memory-mapped to 64GB RAM pool for zero-disk-latency ANN search.

### M4: Validation
- [ ] Verify 100% local execution (all model weights pre-downloaded; disconnect from network before test run).
- [ ] Benchmark query-to-answer latency on Ryzen 9 9950X3D + RTX 4000 Ada.
- [ ] Pin all package versions in `requirements.txt` after validation run.