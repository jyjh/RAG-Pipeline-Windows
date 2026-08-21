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

# Embeddings default to a locally hosted nomic-embed-text (Ollama, 768-d) while
# chat/vision stay on SoCLAaS; [embeddings].backend = "soclaas" switches
# embeddings to the API's bge-m3 (1024-d) instead. Changing the model/dim
# invalidates any existing index -- a full re-index is required (the indexer's
# model+dim reuse guard enforces this automatically).
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_EMBEDDING_DIM = 768
# Texts per embedding request; also the timeout-measurement baseline.
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
DEFAULT_CONTEXT_TOKEN_FRACTION = 0.60
DEFAULT_WEB_SEARCH_ENABLED = True
DEFAULT_WEB_SEARCH_TIMEOUT = 8.0
DEFAULT_WEB_SEARCH_MAX_RESULTS = 5
# Dormant-Ollama chat failover cadence (health checks after connection loss).
DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL = 5.0
DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS = 5

DEFAULT_PDF_PARSER_MODE = "hybrid"
DEFAULT_DOCLING_ACCELERATOR = "auto"
DEFAULT_ASSET_TRIGGERS = "auto"
DEFAULT_ASSET_DIR = "db/assets"
DEFAULT_CODE_ENRICHMENT = True
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
