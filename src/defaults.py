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
DEFAULT_PLANNER_TIMEOUT = 30.0

# bge-m3 produces 1024-d dense vectors (vs nomic-embed-text's 768). Changing the
# embedding model/dim invalidates any existing index -- a full re-index is
# required (the indexer's model+dim reuse guard enforces this automatically).
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_EMBEDDING_DIM = 1024

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
