from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.local_rag as _source_module
from src.reliability import (
    SOURCE_GROUP_UNGROUPED,
    load_source_group_map,
    source_group_details,
    source_group_weight,
)

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class LocalQueryEngine:
    def __init__(
        self,
        working_dir: str = "./db",
        *,
        asset_dir: str | Path | None = None,
        trust_path: str | Path | None = None,
        model: str = "gemma4",
        embedding_model: str = "nomic-embed-text",
        embedding_batch_size: int | None = None,
        embedding_timeout: float | None = None,
        progress_enabled: bool = True,
        top_k: int | None = None,
        num_predict: int | None = None,
        llm_timeout: float | None = None,
        temperature: float | None = None,
        sampler_top_k: int | None = None,
        context_window: int | None = None,
        retrieval_candidate_k: int | None = None,
        retrieval_min_score: float | None = None,
        retrieval_relative_cutoff: float | None = None,
        retrieval_lexical_weight: float | None = None,
        context_token_fraction: float | None = None,
        web_search_enabled: bool = True,
        web_search_timeout: float | None = None,
        web_search_max_results: int | None = None,
        tool_max_rounds: int = DEFAULT_TOOL_MAX_ROUNDS,
        tool_max_calls: int = DEFAULT_TOOL_MAX_CALLS,
        ollama_health_check_interval: float | None = None,
        ollama_max_lost_health_checks: int | None = None,
        system_prompt: str | None = None,
        planner_model: str | None = None,
        planner_enabled: bool = True,
        planner_max_queries: int | None = None,
    ):
        from src.embeddings import EmbeddingEngine
        from src.asset_store import ImageAssetStore

        self.working_dir = working_dir
        self.asset_dir = Path(asset_dir) if asset_dir is not None else Path(working_dir) / "assets"
        env_trust_path = os.environ.get("LOCAL_RAG_TRUST_PATH")
        inferred_trust_path = Path(working_dir).resolve().parent / "data" / ".document_trust.json"
        self.trust_path = Path(trust_path or env_trust_path or inferred_trust_path)
        self.source_group_by_hash = load_source_group_map(self.trust_path)
        self.asset_store = ImageAssetStore(self.asset_dir)
        self.model = model
        self.progress_enabled = progress_enabled
        self.top_k = _positive_int(top_k, 5) if top_k is not None else 5
        self.num_predict = _positive_int(
            num_predict or os.environ.get("LOCAL_RAG_NUM_PREDICT"),
            DEFAULT_NUM_PREDICT,
        )
        self.temperature = _positive_float(
            temperature if temperature is not None else os.environ.get("LOCAL_RAG_TEMPERATURE"),
            QUERY_TEMPERATURE,
        )
        self.sampler_top_k = _positive_int(
            sampler_top_k or os.environ.get("LOCAL_RAG_SAMPLER_TOP_K"),
            DEFAULT_SAMPLER_TOP_K,
        )
        self.context_window = _positive_int(
            context_window or os.environ.get("LOCAL_RAG_NUM_CTX"),
            DEFAULT_CONTEXT_WINDOW,
        )
        self.llm_timeout = _positive_float(
            llm_timeout if llm_timeout is not None else os.environ.get("LOCAL_RAG_LLM_TIMEOUT"),
            DEFAULT_LLM_TIMEOUT,
        )
        self.ollama_health_check_interval = _positive_float(
            ollama_health_check_interval
            if ollama_health_check_interval is not None
            else os.environ.get("LOCAL_RAG_OLLAMA_HEALTH_CHECK_INTERVAL"),
            DEFAULT_OLLAMA_HEALTH_CHECK_INTERVAL,
        )
        self.ollama_max_lost_health_checks = _positive_int(
            ollama_max_lost_health_checks or os.environ.get("LOCAL_RAG_OLLAMA_MAX_LOST_HEALTH_CHECKS"),
            DEFAULT_OLLAMA_MAX_LOST_HEALTH_CHECKS,
        )
        self.system_prompt = str(
            system_prompt
            if system_prompt is not None
            else os.environ.get("LOCAL_RAG_SYSTEM_PROMPT") or DEFAULT_QUERY_SYSTEM_PROMPT
        ).strip()
        self.retrieval_candidate_k = _positive_int(
            retrieval_candidate_k or os.environ.get("LOCAL_RAG_RETRIEVAL_CANDIDATE_K"),
            DEFAULT_RETRIEVAL_CANDIDATE_K,
        )
        self.retrieval_min_score = _positive_float(
            retrieval_min_score
            if retrieval_min_score is not None
            else os.environ.get("LOCAL_RAG_RETRIEVAL_MIN_SCORE"),
            DEFAULT_RETRIEVAL_MIN_SCORE,
        )
        self.retrieval_relative_cutoff = _positive_float(
            retrieval_relative_cutoff
            if retrieval_relative_cutoff is not None
            else os.environ.get("LOCAL_RAG_RETRIEVAL_RELATIVE_CUTOFF"),
            DEFAULT_RETRIEVAL_RELATIVE_CUTOFF,
        )
        self.retrieval_lexical_weight = _bounded_float(
            retrieval_lexical_weight
            if retrieval_lexical_weight is not None
            else os.environ.get("LOCAL_RAG_RETRIEVAL_LEXICAL_WEIGHT"),
            DEFAULT_RETRIEVAL_LEXICAL_WEIGHT,
        )
        self.context_token_fraction = _positive_float(
            context_token_fraction
            if context_token_fraction is not None
            else os.environ.get("LOCAL_RAG_CONTEXT_TOKEN_FRACTION"),
            DEFAULT_CONTEXT_TOKEN_FRACTION,
        )
        self.web_search_enabled = bool(web_search_enabled)
        self.web_search_timeout = _positive_float(
            web_search_timeout
            if web_search_timeout is not None
            else os.environ.get("LOCAL_RAG_WEB_SEARCH_TIMEOUT"),
            DEFAULT_WEB_SEARCH_TIMEOUT,
        )
        self.web_search_max_results = _positive_int(
            web_search_max_results or os.environ.get("LOCAL_RAG_WEB_SEARCH_MAX_RESULTS"),
            DEFAULT_WEB_SEARCH_MAX_RESULTS,
        )
        self.tool_max_rounds = _positive_int(tool_max_rounds, DEFAULT_TOOL_MAX_ROUNDS)
        self.tool_max_calls = _positive_int(tool_max_calls, DEFAULT_TOOL_MAX_CALLS)
        self.planner_enabled = bool(planner_enabled)
        self.planner_model = str(
            planner_model
            if planner_model is not None
            else os.environ.get("LOCAL_RAG_PLANNER_MODEL") or DEFAULT_PLANNER_MODEL
        )
        self.planner_max_queries = _positive_int(
            planner_max_queries or os.environ.get("LOCAL_RAG_PLANNER_MAX_QUERIES"),
            DEFAULT_PLANNER_MAX_QUERIES,
        )
        self.engine = EmbeddingEngine(
            model_name=embedding_model,
            ollama_batch_size=embedding_batch_size,
            ollama_timeout=embedding_timeout,
        )
        self.store = default_store(working_dir, prefer_lancedb=True)
        self.record_count = self._load_record_count()

    def _reliability_details(self, record: dict[str, Any]) -> dict[str, Any]:
        explicit_group = str(record.get("source_group") or "").strip()
        if explicit_group:
            return source_group_details(explicit_group)
        source_hash = str(record.get("source_hash") or "").strip()
        details = dict(self.source_group_by_hash.get(source_hash) or {})
        if not details:
            details = {
                "key": SOURCE_GROUP_UNGROUPED,
                "weight": source_group_weight(SOURCE_GROUP_UNGROUPED),
            }
        return details

    def _ollama_options(self) -> dict[str, int | float]:
        return {
            "temperature": self.temperature,
            "top_k": self.sampler_top_k,
            "num_ctx": self.context_window,
            "num_predict": self.num_predict,
        }

    def _tool_messages(self, question: str) -> list[dict[str, Any]]:
        web_instruction = (
            "Use web_search only when local context is insufficient or the user asks for current/external facts, "
            "and only while the prompt remains under the input-context budget."
            if self.web_search_enabled
            else "Do not use web_search; it is disabled for this request."
        )
        system_prompt = self.system_prompt or DEFAULT_QUERY_SYSTEM_PROMPT
        if "{web_instruction}" in system_prompt:
            system_prompt = system_prompt.replace("{web_instruction}", web_instruction)
        else:
            system_prompt = f"{system_prompt}\n\n{web_instruction}"
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": question},
        ]

    def _tool_definitions(self, *, include_web_search: bool | None = None) -> list[dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_local_context",
                    "description": (
                        "Search the local vectorized PDF index for source chunks relevant to the question. "
                        "Call this before answering and again if additional local evidence is needed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The focused retrieval query.",
                            },
                            "exclude_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional local chunk IDs that should not be returned again.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        web_search_available = self.web_search_enabled if include_web_search is None else include_web_search
        if web_search_available:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": (
                            "Search the public web for current or external facts that are not available "
                            "in local PDF context."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The web search query.",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum result count to return.",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            )
        return tools

    def _load_record_count(self) -> int:
        if not self.store.exists():
            _status(
                f"Local query: LanceDB index not found in {self.working_dir}",
                enabled=self.progress_enabled,
            )
            return 0
        count = self.store.count()
        _status(f"Local query: loaded {count} structured record(s).", enabled=self.progress_enabled)
        return count

    def _context_token_budget(self) -> int:
        effective_fraction = min(self.context_token_fraction, DEFAULT_CONTEXT_TOKEN_FRACTION)
        return max(1, int(math.floor(self.context_window * effective_fraction)))

    def _prompt_token_count(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        return estimate_prompt_tokens(messages, tools=tools)

    def _final_answer_instruction(self) -> str:
        return (
            "Write the final answer now using only the tool results above. "
            "Cite every factual claim with the source IDs in square brackets. "
            "If the sources are insufficient, say what information is missing."
        )

    def _final_answer_reserve_tokens(self) -> int:
        return estimate_prompt_tokens([{"role": "user", "content": self._final_answer_instruction()}])

    def _tool_result_token_budget(self, messages: list[dict[str, Any]] | None) -> int:
        if messages is None:
            return self._context_token_budget()
        remaining = (
            self._context_token_budget()
            - self._final_answer_reserve_tokens()
            - self._prompt_token_count(messages)
        )
        return max(0, remaining)

    def _web_search_allowed(self, messages: list[dict[str, Any]]) -> bool:
        if not self.web_search_enabled:
            return False
        tools_with_web = self._tool_definitions(include_web_search=True)
        return self._prompt_token_count(messages, tools=tools_with_web) <= self._context_token_budget()

    def _retrieve(
        self,
        question: str,
        *,
        exclude_ids: set[str] | None = None,
        token_budget: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.record_count:
            return []
        exclude_ids = exclude_ids or set()

        query_vector = self.engine.get_mrl_embeddings(
            [question],
            truncate_dim=768,
            prefix="search_query: ",
        )[0]
        candidates = self.store.search(
            query_vector.tolist(),
            top_k=self.retrieval_candidate_k,
        )
        if not candidates:
            return []

        best_score = max(float(candidate.get("score") or 0.0) for candidate in candidates)
        score_cutoff = max(self.retrieval_min_score, best_score * self.retrieval_relative_cutoff)
        query_terms = _claim_keywords(question)
        token_budget = self._context_token_budget() if token_budget is None else max(0, int(token_budget))
        if token_budget <= 0:
            return []
        used_tokens = 0
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        keyword_cache: dict[str, tuple[set[str], set[str]]] = {}

        def rank_record(record: dict[str, Any], *, inherited_score: float | None = None) -> dict[str, Any]:
            vector_score = float(record.get("score") or inherited_score or 0.0)
            lexical_score = _lexical_relevance(query_terms, record, keyword_cache=keyword_cache)
            hybrid_score = (
                (1.0 - self.retrieval_lexical_weight) * vector_score
                + self.retrieval_lexical_weight * lexical_score
            )
            reliability = self._reliability_details(record)
            reliability_modifier = float(reliability.get("weight") or source_group_weight(SOURCE_GROUP_UNGROUPED))
            final_score = hybrid_score * reliability_modifier
            ranked = dict(record)
            ranked["vector_score"] = round(vector_score, 4)
            ranked["lexical_score"] = round(lexical_score, 4)
            ranked["hybrid_score"] = round(hybrid_score, 4)
            ranked["source_group"] = str(reliability.get("key") or SOURCE_GROUP_UNGROUPED)
            ranked["reliability_modifier"] = round(reliability_modifier, 4)
            ranked["score"] = round(final_score, 4)
            return ranked

        def maybe_add(record: dict[str, Any], *, inherited_score: float | None = None) -> bool:
            nonlocal used_tokens
            record_id = str(record.get("id") or "")
            if not record_id or record_id in seen or record_id in exclude_ids:
                return False
            vector_score = float(record.get("vector_score") or record.get("score") or inherited_score or 0.0)
            if vector_score < score_cutoff:
                return False
            score = float(record.get("score") or record.get("hybrid_score") or vector_score)
            content = str(record.get("content") or "")
            tokens = estimate_context_tokens(content)
            original_tokens = tokens
            content_truncated = False
            remaining_tokens = token_budget - used_tokens
            if remaining_tokens <= 0:
                return False
            if tokens > remaining_tokens:
                content, content_truncated, tokens = _fit_text_to_context_budget(content, remaining_tokens)
                if not content:
                    return False
            item = dict(record)
            item["content"] = content
            item["score"] = score
            item["vector_score"] = round(vector_score, 4)
            item["lexical_score"] = round(float(record.get("lexical_score") or 0.0), 4)
            item["hybrid_score"] = round(float(record.get("hybrid_score") or vector_score), 4)
            item["source_group"] = str(record.get("source_group") or SOURCE_GROUP_UNGROUPED)
            item["reliability_modifier"] = round(
                float(record.get("reliability_modifier") or source_group_weight(SOURCE_GROUP_UNGROUPED)),
                4,
            )
            item["estimated_tokens"] = tokens
            if content_truncated:
                item["content_truncated"] = True
                item["original_estimated_tokens"] = original_tokens
            seen.add(record_id)
            results.append(item)
            used_tokens += tokens
            return True

        ranked_candidates = sorted(
            (rank_record(candidate) for candidate in candidates),
            key=lambda item: (float(item.get("score") or 0.0), float(item.get("vector_score") or 0.0)),
            reverse=True,
        )
        for candidate in ranked_candidates:
            vector_score = float(candidate.get("vector_score") or candidate.get("score") or 0.0)
            if vector_score < score_cutoff:
                continue
            node_type = str(candidate.get("node_type", "chunk"))
            if node_type == "chunk":
                maybe_add(candidate)
            else:
                children = sorted(
                    (
                        rank_record(child, inherited_score=vector_score)
                        for child in self.store.child_chunks(candidate, limit=self.retrieval_candidate_k)
                    ),
                    key=lambda item: (
                        float(item.get("score") or 0.0),
                        float(item.get("vector_score") or 0.0),
                    ),
                    reverse=True,
                )
                for child in children:
                    if not maybe_add(child, inherited_score=vector_score) and results:
                        if used_tokens >= token_budget:
                            break
            if used_tokens >= token_budget:
                break
        return results

    def _local_tool_result(
        self,
        *,
        query: str,
        exclude_ids: set[str],
        citations: CitationRegistry,
        token_budget: int | None = None,
        matches: list[dict[str, Any]] | None = None,
        planner_queries: list[str] | None = None,
    ) -> dict[str, Any]:
        token_budget = self._context_token_budget() if token_budget is None else max(0, int(token_budget))
        result = {
            "tool": "search_local_context",
            "query": query,
            "result_count": 0,
            "context_token_budget": self._context_token_budget(),
            "results": [],
            "instructions": (
                "Use these source IDs for citations. Do not cite local context that is not listed here."
            ),
        }
        if planner_queries:
            result["planner_queries"] = list(planner_queries)
        base_tokens = estimate_json_tokens(result)
        if token_budget <= base_tokens:
            result["error"] = "Local context skipped because the prompt token budget is exhausted."
            return result

        result_budget = token_budget - base_tokens
        if matches is None:
            matches = self._retrieve(query, exclude_ids=exclude_ids, token_budget=result_budget)
        context_blocks = []
        used_tokens = 0
        truncated = False
        for match in matches:
            source = citations.preview_local(match)
            location = " :: ".join(
                part
                for part in [
                    source.get("source_pdf_name", ""),
                    source.get("section_path", ""),
                    source.get("page_label", ""),
                ]
                if part
            )
            block = {
                "source_id": source["id"],
                "citation": source["label"],
                "chunk_id": source["chunk_id"],
                "score": source["score"],
                "vector_score": round(float(match.get("vector_score") or 0.0), 4),
                "lexical_score": round(float(match.get("lexical_score") or 0.0), 4),
                "hybrid_score": round(float(match.get("hybrid_score") or 0.0), 4),
                "reliability_modifier": round(float(match.get("reliability_modifier") or 0.0), 4),
                "source_group": str(match.get("source_group") or SOURCE_GROUP_UNGROUPED),
                "location": location,
                "content": str(match.get("content") or ""),
            }
            remaining_tokens = result_budget - used_tokens
            fitted, block_truncated, block_tokens = _fit_text_field_to_json_budget(
                block,
                "content",
                remaining_tokens,
                truncated_field="content_truncated",
            )
            if fitted is None:
                if context_blocks:
                    break
                continue
            citations.add_local(match)
            if bool(match.get("content_truncated")) and not block_truncated:
                fitted["content_truncated"] = True
            context_blocks.append(fitted)
            used_tokens += block_tokens
            truncated = truncated or block_truncated or bool(match.get("content_truncated"))
            if used_tokens >= result_budget:
                break
        result["results"] = context_blocks
        result["result_count"] = len(context_blocks)
        if truncated:
            result["context_truncated"] = True
        if not context_blocks and matches:
            result["error"] = "Retrieved local context did not fit within the prompt token budget."
        return result

    def search_local_context(
        self,
        *,
        query: str,
        relevance_floor: float | None = None,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        previous_min_score = self.retrieval_min_score
        if relevance_floor is not None:
            try:
                self.retrieval_min_score = max(0.0, float(relevance_floor))
            except (TypeError, ValueError):
                self.retrieval_min_score = previous_min_score
        try:
            return self._local_tool_result(
                query=query,
                exclude_ids=set(),
                citations=CitationRegistry(asset_store=self.asset_store),
                token_budget=token_budget,
            )
        finally:
            self.retrieval_min_score = previous_min_score

    def _web_tool_result(
        self,
        *,
        query: str,
        max_results: int | None,
        citations: CitationRegistry,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        if not self.web_search_enabled:
            return {"tool": "web_search", "query": query, "result_count": 0, "error": "Web search is disabled."}
        token_budget = self._context_token_budget() if token_budget is None else max(0, int(token_budget))
        result = {
            "tool": "web_search",
            "query": query,
            "result_count": 0,
            "provider": "duckduckgo_lite",
            "error": "",
            "results": [],
            "instructions": "Use these web source IDs for citations. Do not cite URLs that are not listed here.",
        }
        base_tokens = estimate_json_tokens(result)
        if token_budget <= base_tokens:
            result["error"] = "Web search skipped because the prompt token budget is exhausted."
            return result

        response = web_search_duckduckgo_lite(
            query,
            max_results=_positive_int(max_results, self.web_search_max_results),
            timeout=self.web_search_timeout,
        )
        results = []
        used_tokens = 0
        result_budget = token_budget - base_tokens
        truncated = False
        for raw_result in response.get("results", []):
            source = citations.preview_web(raw_result)
            block = {
                "source_id": source["id"],
                "citation": source["label"],
                "title": source["title"],
                "url": source["url"],
                "snippet": source["snippet"],
                "provider": source["provider"],
            }
            remaining_tokens = result_budget - used_tokens
            fitted, snippet_truncated, block_tokens = _fit_text_field_to_json_budget(
                block,
                "snippet",
                remaining_tokens,
                truncated_field="snippet_truncated",
            )
            if fitted is None:
                if results:
                    break
                continue
            citation_result = dict(raw_result)
            citation_result["snippet"] = fitted.get("snippet", "")
            citations.add_web(citation_result)
            results.append(fitted)
            used_tokens += block_tokens
            truncated = truncated or snippet_truncated
            if used_tokens >= result_budget:
                break
        result = {
            "tool": "web_search",
            "query": query,
            "result_count": len(results),
            "provider": response.get("provider", "duckduckgo_lite"),
            "error": response.get("error", ""),
            "results": results,
            "instructions": "Use these web source IDs for citations. Do not cite URLs that are not listed here.",
        }
        if truncated:
            result["context_truncated"] = True
        if not results and response.get("results") and not result.get("error"):
            result["error"] = "Web results did not fit within the prompt token budget."
        return result

    def _execute_tool_call(
        self,
        call: dict[str, Any],
        *,
        question: str,
        citations: CitationRegistry,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        if name == "search_local_context":
            query = str(arguments.get("query") or question).strip() or question
            raw_excludes = arguments.get("exclude_ids", [])
            requested_excludes = {str(value) for value in raw_excludes} if isinstance(raw_excludes, list) else set()
            exclude_ids = citations.local_record_ids() | requested_excludes
            result = self._local_tool_result(
                query=query,
                exclude_ids=exclude_ids,
                citations=citations,
                token_budget=self._tool_result_token_budget(messages),
            )
            text = f"Retrieved {result['result_count']} local source chunk(s)."
            if result.get("error"):
                text = str(result["error"])
            return name, result, text

        if name == "web_search":
            query = str(arguments.get("query") or question).strip() or question
            if messages is not None:
                current_tokens = self._prompt_token_count(
                    messages,
                    tools=self._tool_definitions(include_web_search=True),
                )
                context_budget = self._context_token_budget()
                if current_tokens > context_budget:
                    result = {
                        "tool": "web_search",
                        "query": query,
                        "result_count": 0,
                        "error": (
                            "Web search skipped because the current prompt is already "
                            f"estimated at {current_tokens} token(s), above the {context_budget} token input-context budget."
                        ),
                    }
                    return name, result, str(result["error"])
            result = self._web_tool_result(
                query=query,
                max_results=arguments.get("max_results"),
                citations=citations,
                token_budget=self._tool_result_token_budget(messages),
            )
            text = f"Retrieved {result['result_count']} web result(s)."
            if result.get("error"):
                text = str(result["error"])
            return name, result, text

        return name or "unknown", {"error": f"Unknown tool: {name or 'unknown'}"}, "Unknown tool call."

    def _append_tool_result(
        self,
        messages: list[dict[str, Any]],
        *,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    @staticmethod
    def _tool_result_event(tool_name: str, result: dict[str, Any], text: str) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool": tool_name,
            "text": text,
            "result": result,
            "content": json.dumps(result, ensure_ascii=False),
        }

    def _forced_local_tool_call(
        self,
        messages: list[dict[str, Any]],
        *,
        question: str,
        citations: CitationRegistry,
    ) -> tuple[dict[str, Any], str]:
        call = {
            "function": {
                "name": "search_local_context",
                "arguments": {"query": question, "exclude_ids": []},
            }
        }
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        tool_name, result, text = self._execute_tool_call(
            call,
            question=question,
            citations=citations,
            messages=messages,
        )
        self._append_tool_result(messages, tool_name=tool_name, result=result)
        return result, text

    def _eager_local_tool_call(
        self,
        *,
        question: str,
        citations: CitationRegistry,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str] | None:
        """Pre-fetch local context before the main model is invoked.

        Uses a small planner model to expand ``question`` into several diverse
        retrieval queries, runs each through ``_retrieve`` (with an accumulating
        exclude set so a chunk is returned at most once), merges the union, and
        formats the result exactly like a model-issued ``search_local_context``
        tool call. Returns ``None`` when eager retrieval is disabled or the
        index is empty, so the caller falls back to the existing loop.
        """
        if not self.planner_enabled or not self.record_count:
            return None
        token_budget = self._tool_result_token_budget(messages)
        planner_queries = generate_search_queries(
            question,
            model=self.planner_model,
            max_queries=self.planner_max_queries,
            timeout=DEFAULT_PLANNER_TIMEOUT,
        )

        # Shared exclude set across all expanded queries so a chunk is returned
        # at most once even when several queries match it.
        exclude_ids: set[str] = set()
        seen_ids: set[str] = set()
        merged: list[dict[str, Any]] = []
        for query in planner_queries:
            hits = self._retrieve(query, exclude_ids=exclude_ids, token_budget=token_budget)
            if not hits:
                continue
            for hit in hits:
                record_id = str(hit.get("id") or "")
                if record_id and record_id in seen_ids:
                    continue
                if record_id:
                    seen_ids.add(record_id)
                    exclude_ids.add(record_id)
                merged.append(hit)

        if merged:
            merged.sort(
                key=lambda item: (
                    float(item.get("score") or item.get("hybrid_score") or 0.0),
                    float(item.get("vector_score") or 0.0),
                ),
                reverse=True,
            )
        result = self._local_tool_result(
            query=question,
            exclude_ids=set(),
            citations=citations,
            token_budget=token_budget,
            matches=merged,
            planner_queries=planner_queries,
        )
        call = {
            "function": {
                "name": "search_local_context",
                "arguments": {"query": question, "exclude_ids": []},
            }
        }
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        self._append_tool_result(messages, tool_name="search_local_context", result=result)
        text = f"Retrieved {result['result_count']} local source chunk(s) from {len(planner_queries)} query/queries."
        if result.get("error"):
            text = str(result["error"])
        return result, text

    def _run_tool_rounds(self, question: str):
        citations = CitationRegistry(asset_store=self.asset_store)
        messages = self._tool_messages(question)
        local_search_used = False
        tool_calls_used = 0

        # Eager multi-query pre-fetch: retrieve the most relevant local context
        # before the main model is ever invoked, so the model starts its first
        # round already seeing evidence instead of spending a round deciding to
        # call search_local_context. Falls back to the loop's forced search if
        # disabled or the index is empty.
        if self.planner_enabled and self.record_count:
            eager = self._eager_local_tool_call(
                question=question,
                citations=citations,
                messages=messages,
            )
            if eager is not None:
                result, text = eager
                local_search_used = True
                tool_calls_used += 1
                yield {
                    "type": "tool_call",
                    "tool": "search_local_context",
                    "text": "Searching local context...",
                }
                yield self._tool_result_event("search_local_context", result, text)
                yield {"type": "sources", "sources": citations.all_sources()}
                if not result.get("results"):
                    # Eager retrieval found nothing; let the loop attempt web
                    # search or fall through to the empty-context answer path.
                    pass

        for _ in range(self.tool_max_rounds):
            tools = self._tool_definitions(include_web_search=self._web_search_allowed(messages))
            response = _ollama_chat(
                model=self.model,
                messages=messages,
                options=self._ollama_options(),
                stream=False,
                timeout=self.llm_timeout,
                health_check_interval=self.ollama_health_check_interval,
                max_lost_health_checks=self.ollama_max_lost_health_checks,
                tools=tools,
            )
            thinking = _ollama_response_thinking(response)
            if thinking:
                yield {"type": "thinking", "text": thinking}

            tool_calls = _ollama_tool_calls(response)
            if not tool_calls:
                if not local_search_used:
                    yield {
                        "type": "notice",
                        "text": "The model did not call the required local search tool, so local context was retrieved before answering.",
                    }
                    result, text = self._forced_local_tool_call(
                        messages,
                        question=question,
                        citations=citations,
                    )
                    local_search_used = True
                    tool_calls_used += 1
                    yield self._tool_result_event("search_local_context", result, text)
                    yield {"type": "sources", "sources": citations.all_sources()}
                    if not result.get("results"):
                        break
                    continue
                break

            message = _ollama_response_message(response)
            messages.append(
                {
                    "role": "assistant",
                    "content": str(message.get("content") or ""),
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                if tool_calls_used >= self.tool_max_calls:
                    yield {
                        "type": "notice",
                        "text": "Tool-call limit reached; answering from the context already retrieved.",
                    }
                    break
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                tool_name = str(function.get("name") or "unknown")
                yield {"type": "tool_call", "tool": tool_name, "text": f"Running {tool_name}..."}
                tool_name, result, text = self._execute_tool_call(
                    call,
                    question=question,
                    citations=citations,
                    messages=messages,
                )
                tool_calls_used += 1
                if tool_name == "search_local_context":
                    local_search_used = True
                self._append_tool_result(messages, tool_name=tool_name, result=result)
                yield self._tool_result_event(tool_name, result, text)
                yield {"type": "sources", "sources": citations.all_sources()}
            if tool_calls_used >= self.tool_max_calls:
                break

        if not local_search_used:
            result, text = self._forced_local_tool_call(
                messages,
                question=question,
                citations=citations,
            )
            yield self._tool_result_event("search_local_context", result, text)
            yield {"type": "sources", "sources": citations.all_sources()}

        messages.append(
            {
                "role": "user",
                "content": self._final_answer_instruction(),
            }
        )
        yield {
            "type": "_tool_state",
            "messages": messages,
            "citations": citations,
        }

    def ask(self, question: str) -> str:
        return "".join(self.ask_stream(question)).strip()

    def ask_stream(self, question: str):
        for event in self.ask_stream_events(question):
            if event["type"] == "answer":
                yield event["text"]

    def ask_stream_events(self, question: str):
        if self.planner_enabled and self.record_count:
            yield {"type": "notice", "text": "Searching local context..."}
        else:
            yield {"type": "notice", "text": "Planning retrieval tool calls..."}
        try:
            tool_state: dict[str, Any] | None = None
            for event in self._run_tool_rounds(question):
                if event.get("type") == "_tool_state":
                    tool_state = event
                    continue
                yield event
            if tool_state is None:
                yield {"type": "answer", "text": "No local context could be retrieved. Run index mode first."}
                return

            messages = tool_state["messages"]
            citations: CitationRegistry = tool_state["citations"]
            _status(
                f"Local query: streaming answer from Ollama model {self.model} "
                f"(lost_connection_cycles={self.ollama_max_lost_health_checks}, "
                f"health_interval={self.ollama_health_check_interval:g}s)",
                enabled=self.progress_enabled,
            )
            stream = _ollama_chat(
                model=self.model,
                messages=messages,
                options=self._ollama_options(),
                stream=True,
                timeout=self.llm_timeout,
                health_check_interval=self.ollama_health_check_interval,
                max_lost_health_checks=self.ollama_max_lost_health_checks,
            )
            emitted = False
            answer_emitted = False
            thinking_emitted = False
            in_thinking = False
            done_reason = ""
            answer_text = ""
            for chunk in stream:
                done_reason = _ollama_done_reason(chunk) or done_reason
                thinking = _ollama_response_thinking(chunk)
                if thinking:
                    emitted = True
                    thinking_emitted = True
                    yield {"type": "thinking", "text": thinking}

                content = _ollama_response_content(chunk)
                if content:
                    events, in_thinking = _split_think_tag_events(
                        content,
                        in_thinking=in_thinking,
                    )
                    for event in events:
                        emitted = True
                        if event["type"] == "thinking":
                            thinking_emitted = True
                        if event["type"] == "answer":
                            answer_emitted = True
                            answer_text += event["text"]
                        yield event
        except Exception as exc:
            raise RuntimeError(
                f"Local Ollama streaming query failed for model '{self.model}'. "
                f"Run `{_ollama_pull_command(self.model)}` and ensure Ollama is running. "
                f"Original error: {exc}"
            ) from exc

        if not emitted:
            raise RuntimeError(
                f"Ollama returned an empty streamed answer for model '{self.model}'. "
                "Check the model response in Ollama logs or retry with a different --llm_model."
            )
        if thinking_emitted and not answer_emitted:
            if done_reason == "length":
                yield {
                    "type": "notice",
                    "text": (
                        "The model reached its token limit while still producing its thinking. "
                        "The token budget has been raised for future requests; retry the question "
                        "or increase LOCAL_RAG_NUM_PREDICT for longer answers."
                    ),
                }
            else:
                yield {
                    "type": "notice",
                    "text": "The model ended before producing final answer text.",
                }
        elif done_reason == "length":
            yield {
                "type": "notice",
                "text": "The model reached its token limit, so the answer may be incomplete.",
            }
        invalid = sorted(set(re.findall(r"\[[SW]\d+\]", answer_text)) - citations.valid_labels())
        if invalid:
            yield {
                "type": "notice",
                "text": f"The answer cited unknown source ID(s): {', '.join(invalid)}.",
            }
        for warning in citation_support_warnings(answer_text, messages):
            yield {"type": "notice", "text": warning}

LocalQueryEngine.__module__ = _source_module.__name__
finalize_split_class(_source_module, LocalQueryEngine)

