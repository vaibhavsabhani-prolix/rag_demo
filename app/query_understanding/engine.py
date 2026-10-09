import json
import logging
import time
from typing import Any, cast

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from app.config import (
    QUERY_CACHE_SIZE,
    QUERY_LLM_MAX_TOKENS,
    QUERY_LLM_REMOTE_API_KEY,
    QUERY_LLM_REMOTE_BASE_URL,
    QUERY_LLM_REMOTE_MODEL,
    QUERY_LLM_REQUEST_TIMEOUT,
    QUERY_LLM_TEMPERATURE,
)
from app.models.parsed_query import ParsedQuery
from app.query_understanding.cache import QueryCache
from app.query_understanding.exceptions import (
    LLMCommunicationError,
    SchemaValidationError,
)
from app.query_understanding.normalizer import QueryNormalizer
from app.query_understanding.prompt import (
    ANCHOR_ASSISTANT,
    ANCHOR_ASSISTANT_2,
    ANCHOR_USER,
    ANCHOR_USER_2,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def repair_truncated_json(text: str) -> dict[str, Any] | None:
    stack: list[str] = []
    in_string = False
    escape = False
    last_safe_idx: int | None = None
    last_safe_stack: list[str] = []

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            if not in_string:
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                next_ch = text[j] if j < len(text) else ""
                if next_ch == ":":
                    # This string was a KEY, not a value - truncating right
                    # after a dangling key would leave a key with no value.
                    pass
                elif next_ch == "," and not (stack and stack[-1] == "["):
                    pass
                else:
                    last_safe_idx = i + 1
                    last_safe_stack = list(stack)
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_safe_idx = i + 1
            last_safe_stack = list(stack)
        elif ch == "," and stack and stack[-1] == "[":
            last_safe_idx = i
            last_safe_stack = list(stack)

    if last_safe_idx is None:
        return None

    truncated = text[:last_safe_idx].rstrip()
    if truncated.endswith(","):
        truncated = truncated.removesuffix(",")
    closing = "".join("}" if c == "{" else "]" for c in reversed(last_safe_stack))
    candidate = truncated + closing

    try:
        repaired = json.loads(candidate)

        if not isinstance(repaired, dict):
            return None

        return cast(dict[str, Any], repaired)

    except (json.JSONDecodeError, ValueError):
        return None


class QueryUnderstandingEngine:

    def __init__(self) -> None:
        self.base_url = QUERY_LLM_REMOTE_BASE_URL.rstrip("/")
        self.model = QUERY_LLM_REMOTE_MODEL
        self.api_key = QUERY_LLM_REMOTE_API_KEY
        self.timeout = QUERY_LLM_REQUEST_TIMEOUT
        self.temperature = QUERY_LLM_TEMPERATURE
        self.sync_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=1,
        )

        # Normalizer and Thread-Safe Cache
        self.normalizer = QueryNormalizer()
        self.cache = QueryCache(max_size=QUERY_CACHE_SIZE)

    def warm_up(self) -> bool:
        try:
            self.sync_client.chat.completions.create(
                model=self.model,
                messages=self._prepare_request_messages(ANCHOR_USER),
                response_format={"type": "json_object"},
                temperature=self.temperature,
                timeout=self.timeout,
                max_tokens=QUERY_LLM_MAX_TOKENS,
            )
            return True
        except OpenAIError as exc:
            logger.warning("Query understanding LLM warm-up failed: %s", exc)
            return False

    def _prepare_request_messages(self, query: str) -> list[ChatCompletionMessageParam]:

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ANCHOR_USER},
            {"role": "assistant", "content": ANCHOR_ASSISTANT},
            {"role": "user", "content": ANCHOR_USER_2},
            {"role": "assistant", "content": ANCHOR_ASSISTANT_2},
            {"role": "user", "content": query.strip()},
        ]

    # ==============================================================
    # Synchronous Parsing Hot-Path
    # ==============================================================

    def parse(
        self,
        query: str,
        use_cache: bool = True,
    ) -> ParsedQuery:
        print("Phase 1 Started")
        clean_query = query.strip()
        if not clean_query:
            return self.create_fallback(query)

        # 1. Cache Check
        if use_cache:
            cached_result = self.cache.get(clean_query)
            if cached_result is not None:
                return cached_result

        t_start = time.perf_counter()
        try:
            # 2. Remote LLM Request (Structured JSON)
            messages = self._prepare_request_messages(clean_query)
            t_llm_start = time.perf_counter()

            print(
                f"[QueryLLM] Request -> {self.base_url} (model={self.model})"
            )
            completion = self.sync_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                timeout=self.timeout,
                max_tokens=QUERY_LLM_MAX_TOKENS,
            )
            t_llm_end = time.perf_counter()
            usage = completion.usage
            print(
                f"[QueryLLM] Response <- finish_reason={completion.choices[0].finish_reason} "
                f"tokens(prompt={usage.prompt_tokens if usage else '?'},"
                f"completion={usage.completion_tokens if usage else '?'}) "
                f"in {(t_llm_end - t_llm_start) * 1000:.0f}ms"
            )

            raw_text = completion.choices[0].message.content or "{}"
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                repaired = repair_truncated_json(raw_text)
                if repaired is None:
                    raise
                logger.warning(
                    "Query understanding response was truncated (finish_reason=%s); "
                    "recovered a partial parse for query %r",
                    completion.choices[0].finish_reason,
                    clean_query,
                )
                data = repaired
            data["original_query"] = clean_query
            if not data.get("semantic_query"):
                data["semantic_query"] = data.get("query", clean_query)

            raw_parsed = ParsedQuery.model_validate(data)

            # 3. Local Deterministic Normalization
            t_norm_start = time.perf_counter()
            normalized = self.normalizer.normalize(raw_parsed)
            t_norm_end = time.perf_counter()

            # 4. Cache Population
            if use_cache:
                self.cache.put(clean_query, normalized)
            print(
                f"Parsed query in {(t_norm_end - t_start) * 1000:.2f} ms "
                f"(LLM: {(t_llm_end - t_llm_start) * 1000:.2f} ms, "
                f"Norm: {(t_norm_end - t_norm_start) * 1000:.2f} ms): "
            )
            print("Phase 1 Ended")
            return normalized

        except OpenAIError as exc:
            err = LLMCommunicationError(
                f"LLM communication error: {exc}",
                original_query=clean_query,
            )
            logger.error("%s", err)
            return self.create_fallback(clean_query, error=exc)

        except ValidationError as exc:
            err = SchemaValidationError(
                f"Failed to parse query structure: {exc}",
                original_query=clean_query,
            )
            logger.error("%s", err)
            return self.create_fallback(clean_query, error=exc)

    # ==============================================================
    # Fallback Mechanism
    # ==============================================================

    def create_fallback(
        self,
        query: str,
        error: Exception | None = None,
    ) -> ParsedQuery:

        clean_query = query.strip() if query else ""
        if error:
            logger.warning(
                "Using conservative fallback for query %r due to error: %s",
                clean_query,
                error,
            )

        return ParsedQuery(
            original_query=clean_query,
            semantic_query=clean_query,
            structured_query=clean_query,
            evidence_query=clean_query,
            concepts=[],
            relationships=[],
            attributes=[],
            requirements=[],
            constraints=[],
            exclusions=[],
            metadata_filters=[],
            is_metadata_only=False,
        )
