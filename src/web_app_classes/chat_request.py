from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int | None = Field(DEFAULT_EMBEDDING_BATCH_SIZE, ge=1, le=256)
    embedding_timeout: float | None = Field(DEFAULT_EMBEDDING_TIMEOUT, gt=0)
    temperature: float | None = Field(DEFAULT_TEMPERATURE, ge=0, le=5)
    max_k: int | None = Field(DEFAULT_MAX_K, ge=1, le=500)
    context_window: int | None = Field(CHAT_CONFIG["context_window"], ge=512)
    llm_num_predict: int | None = Field(CHAT_CONFIG["llm_num_predict"], ge=1)
    llm_timeout: float | None = Field(DEFAULT_LLM_TIMEOUT, gt=0)
    web_search_enabled: bool = DEFAULT_WEB_SEARCH_ENABLED
    retrieval_candidate_k: int | None = Field(DEFAULT_RETRIEVAL_CANDIDATE_K, ge=1)
    retrieval_min_score: float | None = Field(CHAT_CONFIG["retrieval_min_score"], ge=0, le=1)
    retrieval_relative_cutoff: float | None = Field(DEFAULT_RETRIEVAL_RELATIVE_CUTOFF, ge=0, le=1)
    context_token_fraction: float | None = Field(DEFAULT_CONTEXT_TOKEN_FRACTION, gt=0, le=1)
    web_search_timeout: float | None = Field(DEFAULT_WEB_SEARCH_TIMEOUT, gt=0)
    web_search_max_results: int | None = Field(DEFAULT_WEB_SEARCH_MAX_RESULTS, ge=1, le=20)
    ollama_health_check_interval: float | None = Field(CHAT_CONFIG["ollama_health_check_interval"], gt=0)
    ollama_max_lost_health_checks: int | None = Field(CHAT_CONFIG["ollama_max_lost_health_checks"], ge=1)
    system_prompt: str | None = CHAT_CONFIG["system_prompt"]
    planner_model: str | None = CHAT_CONFIG["planner_model"]
    planner_enabled: bool = CHAT_CONFIG["planner_enabled"]
    planner_max_queries: int | None = Field(CHAT_CONFIG["planner_max_queries"], ge=0, le=20)

ChatRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, ChatRequest)

