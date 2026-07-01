from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class ChatRequest(BaseModel):
    question: str
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int | None = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_timeout: float | None = DEFAULT_EMBEDDING_TIMEOUT
    temperature: float | None = DEFAULT_TEMPERATURE
    max_k: int | None = DEFAULT_MAX_K
    context_window: int | None = CHAT_CONFIG["context_window"]
    llm_num_predict: int | None = CHAT_CONFIG["llm_num_predict"]
    llm_timeout: float | None = DEFAULT_LLM_TIMEOUT
    web_search_enabled: bool = DEFAULT_WEB_SEARCH_ENABLED
    retrieval_candidate_k: int | None = DEFAULT_RETRIEVAL_CANDIDATE_K
    retrieval_min_score: float | None = CHAT_CONFIG["retrieval_min_score"]
    retrieval_relative_cutoff: float | None = DEFAULT_RETRIEVAL_RELATIVE_CUTOFF
    context_token_fraction: float | None = DEFAULT_CONTEXT_TOKEN_FRACTION
    web_search_timeout: float | None = DEFAULT_WEB_SEARCH_TIMEOUT
    web_search_max_results: int | None = DEFAULT_WEB_SEARCH_MAX_RESULTS
    ollama_health_check_interval: float | None = CHAT_CONFIG["ollama_health_check_interval"]
    ollama_max_lost_health_checks: int | None = CHAT_CONFIG["ollama_max_lost_health_checks"]
    system_prompt: str | None = CHAT_CONFIG["system_prompt"]
    planner_model: str | None = CHAT_CONFIG["planner_model"]
    planner_enabled: bool = CHAT_CONFIG["planner_enabled"]
    planner_max_queries: int | None = CHAT_CONFIG["planner_max_queries"]

ChatRequest.__module__ = _source_module.__name__
finalize_split_class(_source_module, ChatRequest)

