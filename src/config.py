"""Typed, cached config.toml loader -- the app's single config system.

Every consumer (web app, CLI, HPC job, embedding/LLM transports) reads
``load_config()``; the returned :class:`PipelineConfig` is a deep copy, so
callers may mutate it freely. Parsing is cached on the file's
``(path, mtime, size)`` signature, so hot paths (e.g. per-request key lookups
in ``src.llm_api``) do not re-read disk.

Section-by-section dataclasses mirror config.example.toml one-to-one. The
merge warns on unknown keys (old configs keeping removed sections, e.g. the
deleted ``[hpc.gpu]``, are surfaced instead of silently ignored).
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from src.defaults import (
    DEFAULT_AUTO_TAG_BATCH_SIZE,
    DEFAULT_AUTO_TAG_EXCERPT_CHARS,
    DEFAULT_AUTO_TAG_MAX_ITEMS_PER_RUN,
    DEFAULT_AUTO_TAG_MIN_CONFIDENCE,
    DEFAULT_AUTO_TAG_MODEL,
    DEFAULT_AUTO_TAG_TIMEOUT_SECONDS,
    DEFAULT_CODE_ENRICHMENT,
    DEFAULT_CONTEXT_TOKEN_FRACTION,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_DOCLING_ACCELERATOR,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_FORMULA_ENRICHMENT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_LOCAL_LLM_MODEL,
    DEFAULT_NUM_PREDICT,
    DEFAULT_OCR_BACKEND,
    DEFAULT_OCR_BITMAP_AREA_THRESHOLD,
    DEFAULT_OCR_FORCE_FULL_PAGE,
    DEFAULT_OCR_LANGS,
    DEFAULT_SCANNED_OCR_ENGINE,
    DEFAULT_UNLIMITED_OCR_DPI,
    DEFAULT_UNLIMITED_OCR_MODEL,
    DEFAULT_VISION_OCR_DPI,
    DEFAULT_VISION_OCR_MODEL,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_PDF_PARSER_MODE,
    DEFAULT_PLANNER_MAX_QUERIES,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_RAPIDOCR_BACKEND,
    DEFAULT_RETRIEVAL_CANDIDATE_K,
    DEFAULT_RETRIEVAL_MIN_SCORE,
    DEFAULT_RETRIEVAL_RRF_K,
    DEFAULT_RETRIEVAL_RELATIVE_CUTOFF,
    DEFAULT_SAMPLER_TOP_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_TESSERACT_CMD,
    DEFAULT_TESSERACT_DATA_PATH,
    DEFAULT_TESSERACT_PSM,
    DEFAULT_VISION_ENABLED,
    DEFAULT_VISION_MODEL,
    DEFAULT_WEB_SEARCH_ENABLED,
    DEFAULT_WEB_SEARCH_MAX_RESULTS,
    DEFAULT_WEB_SEARCH_TIMEOUT,
    DEFAULT_ASSET_DIR,
    DEFAULT_ASSET_TRIGGERS,
)

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024
DEFAULT_BACKGROUND_WORKER_THREADS = min(4, max(1, (os.cpu_count() or 2) // 2))


@dataclass
class PathsConfig:
    data_dir: str = "data"
    processed_dir: str = "processed_docs"
    db_dir: str = "db"
    asset_dir: str = DEFAULT_ASSET_DIR
    # Extra directories the web UI may serve source PDFs from. Bulk-ingested
    # corpora usually live outside data_dir (e.g. a second drive), and the
    # download/preview path containment rejects anything outside the known
    # roots -- list such roots here, one absolute path per entry.
    corpus_roots: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    llm_model: str = DEFAULT_LLM_MODEL
    # Chat model used when the LOCAL Ollama backend is active (cloud
    # unavailable). Defaults to a 4 GB-GPU-sized substitute (qwen3:4b-instruct)
    # so the local fallback never base-name-matches the cloud tag onto a model
    # this hardware cannot serve (gemma4:26b -> gemma4:latest, 9.6 GB). Set to
    # a different installed tag to override.
    local_llm_model: str = DEFAULT_LOCAL_LLM_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    # Vision model used when the LOCAL Ollama backend is active (cloud
    # unavailable). ``vision_model`` below names the cloud-side model; this
    # one pins the local substitute explicitly (e.g. "qwen2.5vl:3b" on a
    # 4 GB GPU) instead of relying on the registry-hint scan.
    local_vision_model: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = ""
    # all-minilm dense dim (384). Changing model/dim invalidates an existing
    # index (full re-index required); the indexer's reuse guard enforces it.
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    allow_hash_embeddings: bool = True
    native_embeddings: bool = False


@dataclass
class LlmApiConfig:
    """Hosted OpenAI-compatible LLM API (SoCLAaS) used as the primary backend.

    ``backend`` selects the live backend for CHAT and VISION: ``"soclaas"``
    (default, hosted API) or ``"ollama"`` (dormant local fallback for offline
    operation -- the Ollama transport code is retained for this). Embeddings
    have their own selector, ``[embeddings].backend`` (default ``"ollama"``:
    a locally hosted all-minilm), so chat/vision can run on SoCLAaS while
    embeddings stay local; set it to ``"soclaas"`` to embed through the
    hosted API (bge-m3, 1024-d) as well.

    The API key is read from ``api_key`` here but the env var named in
    ``key_env`` (default ``SOCLAAS_API_KEY``; ``LLM_API_KEY`` is also honored)
    overrides it, so secrets need not live in the config file.
    """

    backend: str = "soclaas"
    base_url: str = "https://soclaas-api.comp.nus.edu.sg"
    api_key: str = ""
    chat_path: str = "/v1/chat/completions"
    embeddings_path: str = "/v1/embeddings"
    models_path: str = "/v1/models"
    request_timeout_seconds: float = 120.0
    retries: int = 3
    key_env: str = "SOCLAAS_API_KEY"


@dataclass
class IngestionConfig:
    """PDF-ingestion knobs (parser choice, OCR, vision, parallelism)."""

    parser_mode: str = DEFAULT_PDF_PARSER_MODE
    accelerator: str = DEFAULT_DOCLING_ACCELERATOR
    num_threads: int = 8
    asset_triggers: str = DEFAULT_ASSET_TRIGGERS
    code_enrichment: bool = DEFAULT_CODE_ENRICHMENT
    formula_enrichment: bool = DEFAULT_FORMULA_ENRICHMENT
    vision_enabled: bool = DEFAULT_VISION_ENABLED
    ocr_backend: str = DEFAULT_OCR_BACKEND
    ocr_langs: list[str] = field(default_factory=lambda: list(DEFAULT_OCR_LANGS))
    ocr_force_full_page: bool = DEFAULT_OCR_FORCE_FULL_PAGE
    ocr_bitmap_area_threshold: float = DEFAULT_OCR_BITMAP_AREA_THRESHOLD
    rapidocr_backend: str = DEFAULT_RAPIDOCR_BACKEND
    tesseract_cmd: str = DEFAULT_TESSERACT_CMD
    tesseract_data_path: str = DEFAULT_TESSERACT_DATA_PATH
    tesseract_psm: int | None = DEFAULT_TESSERACT_PSM
    # Which engine OCRs scanned/low-text PDFs: "docling" runs the Docling
    # pipeline's OCR plugin (ocr_backend above); "unlimited_ocr" renders each
    # page and asks Baidu's Unlimited-OCR VLM on the local Ollama host;
    # "vision_ocr" does the same with a general vision model
    # (vision_ocr_model, default qwen2.5vl:3b) and a verbatim-transcription
    # prompt -- the most accurate local option per eval/ocr/RESULTS.md.
    scanned_ocr_engine: str = DEFAULT_SCANNED_OCR_ENGINE
    unlimited_ocr_model: str = DEFAULT_UNLIMITED_OCR_MODEL
    unlimited_ocr_dpi: int = DEFAULT_UNLIMITED_OCR_DPI
    vision_ocr_model: str = DEFAULT_VISION_OCR_MODEL
    vision_ocr_dpi: int = DEFAULT_VISION_OCR_DPI
    ingestion_workers: int = 1
    max_pages_whole_doc: int = 50
    # Estimated on-disk expansion when PDFs become Markdown + .pages.json
    # sidecars, used by the ingest disk-space pre-check. Born-digital PDFs
    # expand ~1.5x; OCR/vision-enriched scanned PDFs can exceed 2x.
    ingest_expansion_factor: float = 2.0


@dataclass
class ChunkingConfig:
    """``[chunking]`` section: section splitting + index-backend selection.

    Consumed by ``main.py --mode index`` so CLI index builds honor the same
    knobs the web UI's reindex form exposes. The web job queue always sends
    explicit values from the request, so these mainly govern direct CLI runs.
    """

    # Target chunk size within one detected section.
    max_tokens: int = 900
    # Overlap used only when splitting an oversized detected section.
    overlap_tokens: int = 120
    # How to derive document/section summary records: hybrid | deterministic | llm.
    summary_mode: str = "hybrid"
    # Vector index storage backend. LanceDB is the only supported backend.
    index_backend: str = "lancedb"


@dataclass
class ChatConfig:
    """``[chat]`` generation/planner settings for query mode."""

    llm_num_predict: int = DEFAULT_NUM_PREDICT
    llm_timeout: float = DEFAULT_LLM_TIMEOUT
    temperature: float = DEFAULT_TEMPERATURE
    max_k: int = DEFAULT_SAMPLER_TOP_K
    context_window: int = DEFAULT_CONTEXT_WINDOW
    # Ollama-only: how long the local server keeps the chat model loaded
    # between requests (Ollama default is 5m; a cold reload on a 4 GB GPU
    # costs minutes). Passed as the /api/chat ``keep_alive`` field; an empty
    # string falls back to the Ollama default. Ignored by the SoCLAaS backend.
    ollama_keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE
    system_prompt: str = ""
    planner_model: str = DEFAULT_PLANNER_MODEL
    planner_enabled: bool = True
    planner_max_queries: int = DEFAULT_PLANNER_MAX_QUERIES


@dataclass
class RetrievalConfig:
    """``[retrieval]`` relevance/candidate tuning for query mode."""

    candidate_top_k: int = DEFAULT_RETRIEVAL_CANDIDATE_K
    min_relevance_score: float = DEFAULT_RETRIEVAL_MIN_SCORE
    relative_relevance_cutoff: float = DEFAULT_RETRIEVAL_RELATIVE_CUTOFF
    context_token_fraction: float = DEFAULT_CONTEXT_TOKEN_FRACTION
    # RRF damping constant for fusing vector + BM25 rankings (60 is the value
    # from the original RRF paper). Higher = lower-ranked lists count more.
    rrf_k: int = DEFAULT_RETRIEVAL_RRF_K


@dataclass
class WebSearchConfig:
    enabled: bool = DEFAULT_WEB_SEARCH_ENABLED
    timeout_seconds: float = DEFAULT_WEB_SEARCH_TIMEOUT
    max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    bind_all: bool = False
    # Legacy alias for bind_all kept so older config files keep loading.
    lan: bool = False
    api_token: str = ""
    health_poll_interval_ms: int = 60_000
    jobs_poll_interval_ms: int = 60_000
    background_worker_threads: int = DEFAULT_BACKGROUND_WORKER_THREADS
    update_remote: str = "origin"
    update_branch: str = "main"
    disk_safety_factor: float = 1.15
    job_workers: int = 1
    query_wait_timeout_seconds: float = 1800.0


@dataclass
class ApiKeysConfig:
    """Per-user API-key auth, rate limiting, and usage tracking.

    This layers on top of the single shared ``[server] api_token`` (which stays
    available as an admin/owner master bypass). When ``enabled`` is true and
    at least one API key exists in the store (or a master token is set), mutating
    ``/api/*`` requests require a valid credential. The store is empty by
    default, so a fresh deployment stays fully open (zero-config).

    Rate limiting is a per-key (or per-master-token) sliding 60s window; the
    global default applies unless a key carries its own override. Usage counters
    are persisted to disk every ``usage_persist_interval`` increments and on
    shutdown to avoid hitting the store on every request.
    """

    enabled: bool = True
    rate_limit_per_minute: int = 60
    usage_persist_interval: int = 50
    key_prefix: str = "rag_"


@dataclass
class UploadsConfig:
    max_upload_bytes: int = 0
    max_corpus_bytes: int = 0
    chunk_bytes: int = DEFAULT_UPLOAD_CHUNK_BYTES


@dataclass
class EmbeddingsConfig:
    """Embedding-engine tuning. These are the dominant cost at corpus scale
    (a 100GB cold index is weeks of embedding work on a single host), so they
    are surfaced in config rather than env-only.

    Precedence at construction time (see ``EmbeddingEngine.__init__``) is:
    explicit constructor arg > env var (``OLLAMA_EMBED_*``) > these config
    values > hardcoded defaults. The env vars remain the escape hatch for
    ad-hoc overrides without editing the config file.
    """

    # Embedding transport, selected INDEPENDENTLY of chat/vision so the common
    # split deployment works: SoCLAaS gemma4/qwen3-vl chat+vision with locally
    # hosted all-minilm embeddings. "ollama" (default) = local Ollama
    # host(s) from [embeddings].hosts / OLLAMA_HOST; "soclaas" = the hosted
    # API's bge-m3 (1024-d; pair it with [models] embedding_model/dim);
    # "" = follow [llm_api].backend. Env EMBEDDINGS_BACKEND overrides.
    # ([models].native_embeddings = true still takes precedence over both.)
    backend: str = "ollama"
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT
    retries: int = 3
    cache_max_entries: int = 50_000
    # Scale the per-request timeout with batch size so a large batch on a slow
    # backend does not silently trip the retry loop. The effective timeout is
    # ``timeout_seconds * max(1.0, batch_size / timeout_batch_baseline)``,
    # capped at ``timeout_max_seconds``. Set ``timeout_batch_baseline`` equal to
    # the batch size at which ``timeout_seconds`` was measured (default 128).
    timeout_batch_baseline: int = DEFAULT_EMBEDDING_BATCH_SIZE
    timeout_max_seconds: float = 600.0
    # List of Ollama replica URLs for multi-host embedding (the primary
    # scale-out lever for the dormant local backend). Empty = single host
    # via ``OLLAMA_HOST``.
    hosts: list[str] = field(default_factory=list)
    # In-flight embedding batches per host. Set >1 when the Ollama server runs
    # with ``OLLAMA_NUM_PARALLEL>1``. 1 = one batch per host at a time.
    concurrency: int = 1
    # Instruction prefixes prepended to document/query texts before embedding.
    # Empty string = use the model's natural behaviour. bge-m3 is
    # instruction-free (empty); nomic-embed-text/e5 want
    # "search_document: " / "search_query: ". Left empty, the runtime
    # auto-resolves per model (see src.llm_api.resolve_embedding_prefix).
    doc_prefix: str = ""
    query_prefix: str = ""


@dataclass
class AutoTagConfig:
    """LLM source-group auto-tagging (see ``src/auto_tag.py``).

    When ``enabled`` (default), PDFs uploaded without a manual source group
    are classified in the background by the chat model and written to the
    trust registry like any manual tag, but flagged ``auto_tagged`` so
    reviewers can audit/override them. ``model`` left empty uses
    ``[models].llm_model`` (gemma4:26b).
    """

    enabled: bool = True
    model: str = DEFAULT_AUTO_TAG_MODEL
    batch_size: int = DEFAULT_AUTO_TAG_BATCH_SIZE
    min_confidence: float = DEFAULT_AUTO_TAG_MIN_CONFIDENCE
    excerpt_chars: int = DEFAULT_AUTO_TAG_EXCERPT_CHARS
    timeout_seconds: float = DEFAULT_AUTO_TAG_TIMEOUT_SECONDS
    max_items_per_run: int = DEFAULT_AUTO_TAG_MAX_ITEMS_PER_RUN


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    hosts: list[str] = field(default_factory=list)
    fallback_enabled: bool = True
    chat_health_check_interval_seconds: float = 5.0
    chat_max_lost_health_checks: int = 5


@dataclass
class HpcClusterConfig:
    """Connection + job settings for the CPU-cluster login node."""

    # SSH alias for the cluster's login node (configured in ~/.ssh/config).
    ssh_host: str = ""
    # Path relative to the SSH login directory where the repo + scripts live.
    # Example: "RAG-Pipeline-Windows" resolves beneath the remote account's
    # default directory without assuming /home, /users, or another site layout.
    remote_repo_dir: str = ""
    # Singularity image filename the PBS job execs on this cluster.
    container_sif: str = "rag_pipeline_cpu.sif"
    # Absolute per-user storage root used for deployment, job scratch and binds.
    # Atlas9 CPU: /hpctmp/<username>.
    storage_root: str = "/hpctmp/${USER}"
    # Resource overrides merged into generate_pbs_script():
    # ncpus/mem/ngpus/queue/walltime/... Empty = generator defaults. Sensible
    # CPU defaults (ngpus=0/queue=cpu) are set on the HpcConfig.cpu factory.
    pbs_overrides: dict = field(default_factory=dict)


def _default_cpu_overrides() -> dict:
    return {"ngpus": 0, "queue": "cpu", "container_sif": "rag_pipeline_cpu.sif"}


@dataclass
class HpcConfig:
    """Delegation of bulk ingestion/indexing to the HPC CPU cluster over SSH.

    When ``enabled`` is False (the default), the web app runs ingestion/indexing
    as local ``main.py`` subprocesses exactly as before -- this section is a
    no-op. When enabled, the job queue submits a PBS ingest/index job to the
    free CPU cluster via ``ssh cpu.ssh_host qsub ...``, relays its progress, and
    rsyncs the built ``db/`` back. See ``docs/HPC_DELEGATION.md`` and
    ``src/hpc_backend.py``. The job calls the SoCLAaS embeddings endpoint
    instead of a local Ollama server; the former GPU serving path was removed.
    """

    # Master switch. Default False = existing local-subprocess behavior untouched.
    enabled: bool = False
    # The CPU cluster (free) runs ingest/index. The pre-staged corpus lives here.
    cpu: HpcClusterConfig = field(default_factory=lambda: HpcClusterConfig(
        container_sif="rag_pipeline_cpu.sif",
        storage_root="/hpctmp/${USER}",
        pbs_overrides=_default_cpu_overrides()))
    # Where the corpus PDFs live on the CPU cluster (PBS --input-dir).
    # Prefer an ABSOLUTE path under /hpctmp/<user>/rag-corpus (what the setup
    # wizard writes): provisioning atomically replaces the repo directory, so
    # anything stored inside it is wiped by a later re-provision. Relative
    # values resolve against remote_repo_dir (legacy deployments).
    remote_data_dir: str = "data"
    # Where an ingest-only PBS job (--skip-index) writes the processed
    # Markdown corpus, relative to remote_repo_dir unless absolute. Fetched
    # back with HpcBackend.fetch_processed_docs for local indexing.
    remote_processed_dir: str = "processed_docs"
    # Where the PBS job writes the index (relative to remote_repo_dir unless
    # absolute). bulk_ingest.py writes db/ to its cwd by default.
    remote_db_dir: str = "db"
    # Seconds between qstat polls while waiting for a PBS job to finish.
    poll_interval_seconds: float = 15.0


@dataclass
class BackupsConfig:
    # 0 disables the automatic backup scheduler; otherwise the web app
    # enqueues an index backup when the newest backup is older than this
    # many hours (checked hourly while the server runs).
    auto_backup_hours: int = 0


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    backups: BackupsConfig = field(default_factory=BackupsConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    uploads: UploadsConfig = field(default_factory=UploadsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    api_keys: ApiKeysConfig = field(default_factory=ApiKeysConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    llm_api: LlmApiConfig = field(default_factory=LlmApiConfig)
    auto_tag: AutoTagConfig = field(default_factory=AutoTagConfig)
    hpc: HpcConfig = field(default_factory=HpcConfig)
    # Raw ``[indexing]`` section (ANN tuning keys); normalization happens in
    # src.vector_store.apply_indexing_config, which tolerates missing keys.
    indexing: dict = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        for value in (self.paths.data_dir, self.paths.processed_dir, self.paths.db_dir, self.paths.asset_dir):
            Path(value).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".toml", ".tml"}:
        import tomllib

        with path.open("rb") as fh:
            return tomllib.load(fh)
    raise ValueError(f"Unsupported config format: {path}")


def _merge_dataclass(target: Any, values: dict[str, Any], *, path: str = "") -> Any:
    for key, value in values.items():
        if not hasattr(target, key):
            logger.warning("Unknown config key [%s] ignored: %s", f"{path}.{key}".lstrip("."), key)
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value, path=f"{path}.{key}".lstrip("."))
        else:
            setattr(target, key, value)
    return target


# Parsed-config cache keyed on (resolved path, mtime, size). ``load_config``
# returns a deep copy so callers can mutate without polluting the cache.
# Only the newest signature per path is retained: keeping every version would
# leak one full PipelineConfig per config-file edit for the process lifetime.
_CONFIG_CACHE: dict[tuple[str, int, int], PipelineConfig] = {}


def _prune_config_cache(current: tuple[str, int, int]) -> None:
    path = current[0]
    for signature in [key for key in _CONFIG_CACHE if key[0] == path and key != current]:
        _CONFIG_CACHE.pop(signature, None)


def default_config_path() -> Path:
    """Repo ``config.toml``, honoring ``RAG_PIPELINE_CONFIG`` (env override).

    Relative env values anchor at the repo root so the resolution is
    independent of the process working directory. Module-level helpers that
    cannot receive an explicit path (``src.llm_api``, ``src.local_rag``) must
    pass this to :func:`load_config` -- a bare ``load_config()`` returns pure
    dataclass defaults and never touches disk, which silently ignores
    config.toml when the env var is unset.
    """
    raw = os.environ.get("RAG_PIPELINE_CONFIG")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else Path(__file__).resolve().parents[1] / raw
    return Path(__file__).resolve().parents[1] / "config.toml"


def load_config(path: str | os.PathLike[str] | None = None) -> PipelineConfig:
    """Load the typed config. No path and no ``RAG_PIPELINE_CONFIG`` = defaults.

    Parsing is cached on the file's ``(path, mtime, size)`` signature; the
    returned dataclass is a deep copy, safe to mutate.
    """
    raw = path or os.environ.get("RAG_PIPELINE_CONFIG") or ""
    if raw:
        chosen = Path(raw)
        try:
            stat = chosen.stat()
            signature = (str(chosen), stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = (str(chosen), 0, 0)
    else:
        chosen = None
        signature = ("<defaults>", 0, 0)
    cached = _CONFIG_CACHE.get(signature)
    if cached is None:
        cfg = PipelineConfig()
        if chosen is not None and chosen.exists():
            _merge_dataclass(cfg, _load_mapping(chosen))
        cfg.ensure_dirs()
        _CONFIG_CACHE[signature] = cfg
        _prune_config_cache(signature)
        cached = cfg
    return copy.deepcopy(cached)
