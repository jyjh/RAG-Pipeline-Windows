# Primary backend is the hosted OpenAI-compatible SoCLAaS API
# (https://soclaas-api.comp.nus.edu.sg). Set [llm_api].backend = "ollama"
# (or env LLM_BACKEND=ollama) to fall back to a local Ollama server for
# fully-offline operation -- the Ollama transport stays in the tree for that.
DEFAULT_LLM_BACKEND = "soclaas"

DEFAULT_LLM_MODEL = "gemma4:26b"
DEFAULT_VISION_MODEL = "qwen3-vl:32b"
DEFAULT_VISION_ENABLED = True

# SoCLAaS exposes only three models (bge-m3, gemma4:26b, qwen3-vl:32b), so the
# query planner reuses the main chat model instead of a dedicated small model.
DEFAULT_PLANNER_MODEL = "gemma4:26b"
DEFAULT_PLANNER_MAX_QUERIES = 3

# Chat/planner model served by the LOCAL Ollama fallback. Sized for a 4 GB
# GPU (GTX 1650-class): qwen3:4b-instruct is ~2.5 GB at Q4_K_M with native
# tool-calling, whereas the base-name fallback for the cloud tag would pull in
# gemma4:latest (gemma4 e4b, 9.6 GB) -- far past both VRAM and acceptable cold
# -load time on this class of hardware. SoCLAaS deployments never use this
# name; it is only reached through llm_api.resolve_local_model.
DEFAULT_LOCAL_LLM_MODEL = "qwen3:4b-instruct"

# Ollama keep_alive for local chat/planner requests. Ollama unloads a model
# after 5 minutes by default; on this hardware a cold reload costs minutes,
# which the warm-up thread would then have to redo after every idle gap.
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"

# Embeddings default to a locally hosted all-minilm (Ollama, 384-d) while
# chat/vision stay on SoCLAaS; [embeddings].backend = "soclaas" switches
# embeddings to the API's bge-m3 (1024-d) instead. Changing the model/dim
# invalidates any existing index -- a full re-index is required (the indexer's
# model+dim reuse guard enforces this automatically).
DEFAULT_EMBEDDING_MODEL = "all-minilm"
DEFAULT_EMBEDDING_DIM = 384
# Texts per embedding request; also the timeout-measurement baseline. The
# shipped config.example.toml pins 64 for 4 GB GPUs; 128 remains the
# unconfigured default (all-minilm is small enough to batch larger).
DEFAULT_EMBEDDING_BATCH_SIZE = 128
DEFAULT_EMBEDDING_TIMEOUT = 30.0

# Chat/query defaults shared by main.py, web_app.py, and src/local_rag.py.
# These are the single source of truth; the config dataclasses and the web
# request models reference them, not copies.
DEFAULT_TEMPERATURE = 0.3
DEFAULT_NUM_PREDICT = 4096
DEFAULT_SAMPLER_TOP_K = 40
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_LLM_TIMEOUT = 120.0
DEFAULT_RETRIEVAL_CANDIDATE_K = 80
DEFAULT_RETRIEVAL_MIN_SCORE = 0.50
DEFAULT_RETRIEVAL_RELATIVE_CUTOFF = 0.72
DEFAULT_RETRIEVAL_RRF_K = 60
DEFAULT_CONTEXT_TOKEN_FRACTION = 0.60
DEFAULT_WEB_SEARCH_ENABLED = True
DEFAULT_WEB_SEARCH_TIMEOUT = 8.0
DEFAULT_WEB_SEARCH_MAX_RESULTS = 5

# LLM source-group auto-tagging ([auto_tag]; see src/auto_tag.py). Documents
# per classification request, confidence floor below which a PDF stays
# ungrouped, excerpt budget per PDF, and per-request LLM timeout.
DEFAULT_AUTO_TAG_MODEL = ""
# Small batches keep the JSON reply inside a local model's reliability and
# generation budget (see src/auto_tag.py DEFAULT_BATCH_SIZE rationale).
DEFAULT_AUTO_TAG_BATCH_SIZE = 5
DEFAULT_AUTO_TAG_MIN_CONFIDENCE = 0.6
DEFAULT_AUTO_TAG_EXCERPT_CHARS = 1200
DEFAULT_AUTO_TAG_TIMEOUT_SECONDS = 120.0
DEFAULT_AUTO_TAG_MAX_ITEMS_PER_RUN = 200
# Dormant-Ollama chat failover cadence (health checks after connection loss).
DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL = 5.0
DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS = 5

DEFAULT_PDF_PARSER_MODE = "hybrid"
DEFAULT_DOCLING_ACCELERATOR = "auto"
DEFAULT_ASSET_TRIGGERS = "auto"
DEFAULT_ASSET_DIR = "db/assets"
DEFAULT_CODE_ENRICHMENT = False
DEFAULT_FORMULA_ENRICHMENT = True

DEFAULT_OCR_BACKEND = "rapidocr"
DEFAULT_OCR_LANGS = ("english",)
DEFAULT_OCR_FORCE_FULL_PAGE = True
DEFAULT_OCR_BITMAP_AREA_THRESHOLD = 0.05
DEFAULT_RAPIDOCR_BACKEND = "onnxruntime"
DEFAULT_TESSERACT_CMD = "tesseract"
DEFAULT_TESSERACT_DATA_PATH = ""
DEFAULT_TESSERACT_PSM = None

SUPPORTED_OCR_BACKENDS = ("auto", "rapidocr", "tesseract_cli", "tesseract", "easyocr")
SUPPORTED_RAPIDOCR_BACKENDS = ("onnxruntime", "openvino", "paddle", "torch")

# Scanned-PDF OCR engine. "docling" = the Docling pipeline's OCR plugin
# (see SUPPORTED_OCR_BACKENDS); "unlimited_ocr" = Baidu's Unlimited-OCR VLM
# served by the local Ollama host (render pages -> one vision request each).
DEFAULT_SCANNED_OCR_ENGINE = "docling"
SUPPORTED_SCANNED_OCR_ENGINES = ("docling", "unlimited_ocr", "vision_ocr")
DEFAULT_UNLIMITED_OCR_MODEL = "frob/unlimited-ocr"
# The Ollama/llama.cpp build of Unlimited-OCR resizes every page to ~1024px
# regardless of render dpi, and lacks the vendor's ngram anti-repetition
# guard: at 300 dpi dense text becomes illegible and the model hallucinates
# (eval/ocr/RESULTS.md). 110 dpi kept text legible after the downscale and is
# the highest-fidelity measured setting for the Ollama route.
DEFAULT_UNLIMITED_OCR_DPI = 110
DEFAULT_UNLIMITED_OCR_NUM_CTX = 16384
# vision_ocr engine: a general vision-language model (local Ollama) used as
# the scanned-PDF OCR engine. Unlike the Unlimited-OCR port, qwen2.5-vl tiles
# natively so full 300 dpi renders work. Most accurate local OCR measured
# (eval/ocr/RESULTS.md) at ~3-4 min/page on a 4 GB GPU.
DEFAULT_VISION_OCR_MODEL = "qwen2.5vl:3b"
DEFAULT_VISION_OCR_DPI = 300
DEFAULT_VISION_OCR_NUM_CTX = 8192
