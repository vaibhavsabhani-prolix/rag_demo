"""
Query Understanding Engine

High-performance, domain-generic semantic parser for natural language patent queries.
Orchestrates single-LLM structured output extraction, LRU caching, and deterministic normalization.
"""

import json
import logging
import time
from typing import Any, List, Optional
from openai import AsyncOpenAI, OpenAI, OpenAIError

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
    QueryUnderstandingError,
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


def _repair_truncated_json(text: str) -> Optional[dict]:
    """
    Best-effort recovery for a JSON object cut off mid-generation (a very
    information-dense query can need more structured output than fits in
    the token budget). Walks the text tracking bracket/string nesting,
    remembers the last point where a string, array/object, or top-level
    value cleanly ended, truncates there, and closes whatever brackets
    were still open at that point. This salvages every field the model
    finished generating instead of discarding the whole completion - every
    ParsedQuery field has a safe empty default, so a partial object still
    validates.
    """
    stack: List[str] = []
    in_string = False
    escape = False
    last_safe_idx: Optional[int] = None
    last_safe_stack: Optional[List[str]] = None

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
                    # This string was a value inside an object, with more
                    # keys still to come (next_ch == ","). Truncating here
                    # and closing the object now would produce a
                    # syntactically valid but semantically incomplete
                    # object (e.g. a relationship with "subject" but no
                    # "relation"/"object"), which fails schema validation
                    # even though the JSON itself parses - only a trailing
                    # value (followed by "}"/"]") or an array element
                    # (container is "[") is a genuinely safe cut point.
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
            # Only safe when directly inside an array (separating complete
            # array elements). A comma directly inside an object (stack
            # top "{") separates that object's OWN key-value pairs -
            # truncating there and closing the object early would produce
            # a syntactically valid but semantically incomplete object
            # (e.g. a relationship with "subject" but no "relation"),
            # which fails schema validation even though the JSON parses.
            last_safe_idx = i
            last_safe_stack = list(stack)

    if last_safe_idx is None or last_safe_stack is None:
        return None

    truncated = text[:last_safe_idx].rstrip()
    if truncated.endswith(","):
        truncated = truncated[:-1]
    closing = "".join("}" if c == "{" else "]" for c in reversed(last_safe_stack))
    candidate = truncated + closing

    try:
        repaired = json.loads(candidate)
        return repaired if isinstance(repaired, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


class QueryUnderstandingEngine:
    """
    Production-ready Query Understanding Engine for Patent Semantic Search.
    
    Transforms arbitrary patent queries into strongly-typed ParsedQuery objects with:
      - 1 single remote LLM call per uncached query
      - Deterministic temperature = 0 extraction
      - Native structured JSON output
      - Thread-safe bounded LRU caching
      - Local deterministic metadata and entity normalization
      - Safe error handling with conservative fallback
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
        cache_size: Optional[int] = None,
    ):
        self.base_url = (base_url or QUERY_LLM_REMOTE_BASE_URL).rstrip("/")
        self.model = model or QUERY_LLM_REMOTE_MODEL
        self.api_key = api_key or QUERY_LLM_REMOTE_API_KEY
        self.timeout = timeout if timeout is not None else QUERY_LLM_REQUEST_TIMEOUT
        self.temperature = temperature if temperature is not None else QUERY_LLM_TEMPERATURE

        # Persistent HTTP clients (connection pooling & keep-alive).
        # max_retries=1 caps a timed-out call at 2 attempts instead of the
        # SDK default of 3, so a cold/unresponsive server fails in ~2x the
        # timeout instead of ~3x before falling back.
        self.sync_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=1,
        )
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=1,
        )

        # Normalizer and Thread-Safe Cache
        self.normalizer = QueryNormalizer()
        self.cache = QueryCache(max_size=cache_size or QUERY_CACHE_SIZE)

    def warm_up(self) -> bool:
        """
        Fire a throwaway request at the remote LLM so a cold/unloaded model
        pays its startup latency here instead of on a user's first query.
        Returns True on success, False if the server is unreachable/slow
        (safe to ignore - the engine still works, just slower on first use).
        """
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

    def _prepare_request_messages(self, query: str) -> List[dict[str, Any]]:
        """Assemble the compact anchored message payload for an incoming query."""
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
        raise_on_error: bool = False,
    ) -> ParsedQuery:
        """
        Parse a natural language patent search query into a strongly typed ParsedQuery.

        Args:
            query: Raw user query string.
            use_cache: Whether to check/populate the in-memory cache.
            raise_on_error: If True, raise QueryUnderstandingError on failure;
                           if False, return a safe conservative fallback.

        Returns:
            Validated and normalized ParsedQuery.
        """
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

            print(f"[QueryLLM] Request -> {self.base_url} (model={self.model}, query={clean_query[:80]!r})")
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
                repaired = _repair_truncated_json(raw_text)
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
                f"[QueryLLM] Result: {len(normalized.concepts)} concepts, "
                f"{len(normalized.metadata_filters)} metadata_filters, "
                f"is_metadata_only={normalized.is_metadata_only}"
            )
            logger.debug(
                "Parsed query in %.2f ms (LLM: %.2f ms, Norm: %.2f ms): %r",
                (t_norm_end - t_start) * 1000,
                (t_llm_end - t_llm_start) * 1000,
                (t_norm_end - t_norm_start) * 1000,
                clean_query,
            )
            return normalized

        except OpenAIError as exc:
            err = LLMCommunicationError(
                f"LLM communication error: {exc}",
                original_query=clean_query,
            )
            logger.error("%s", err)
            if raise_on_error:
                raise err from exc
            return self.create_fallback(clean_query, error=exc)

        except Exception as exc:
            err = SchemaValidationError(
                f"Failed to parse query structure: {exc}",
                original_query=clean_query,
            )
            logger.error("%s", err)
            if raise_on_error:
                raise err from exc
            return self.create_fallback(clean_query, error=exc)

    # ==============================================================
    # Asynchronous Parsing Hot-Path
    # ==============================================================

    async def parse_async(
        self,
        query: str,
        use_cache: bool = True,
        raise_on_error: bool = False,
    ) -> ParsedQuery:
        """
        Asynchronously parse a natural language patent search query.
        """
        clean_query = query.strip()
        if not clean_query:
            return self.create_fallback(query)

        if use_cache:
            cached_result = self.cache.get(clean_query)
            if cached_result is not None:
                return cached_result

        t_start = time.perf_counter()
        try:
            messages = self._prepare_request_messages(clean_query)
            t_llm_start = time.perf_counter()

            print(f"[QueryLLM] Request -> {self.base_url} (model={self.model}, query={clean_query[:80]!r})")
            completion = await self.async_client.chat.completions.create(
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
                repaired = _repair_truncated_json(raw_text)
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

            t_norm_start = time.perf_counter()
            normalized = self.normalizer.normalize(raw_parsed)
            t_norm_end = time.perf_counter()

            if use_cache:
                self.cache.put(clean_query, normalized)

            print(
                f"[QueryLLM] Result: {len(normalized.concepts)} concepts, "
                f"{len(normalized.metadata_filters)} metadata_filters, "
                f"is_metadata_only={normalized.is_metadata_only}"
            )

            logger.debug(
                "Async parsed query in %.2f ms (LLM: %.2f ms, Norm: %.2f ms): %r",
                (t_norm_end - t_start) * 1000,
                (t_llm_end - t_llm_start) * 1000,
                (t_norm_end - t_norm_start) * 1000,
                clean_query,
            )
            return normalized

        except OpenAIError as exc:
            err = LLMCommunicationError(
                f"Async LLM communication error: {exc}",
                original_query=clean_query,
            )
            logger.error("%s", err)
            if raise_on_error:
                raise err from exc
            return self.create_fallback(clean_query, error=exc)

        except Exception as exc:
            err = SchemaValidationError(
                f"Async query parse failed: {exc}",
                original_query=clean_query,
            )
            logger.error("%s", err)
            if raise_on_error:
                raise err from exc
            return self.create_fallback(clean_query, error=exc)

    # ==============================================================
    # Batch Support
    # ==============================================================

    def parse_batch(
        self,
        queries: List[str],
        use_cache: bool = True,
        raise_on_error: bool = False,
    ) -> List[ParsedQuery]:
        """Parse multiple queries sequentially."""
        return [self.parse(q, use_cache=use_cache, raise_on_error=raise_on_error) for q in queries]

    async def parse_batch_async(
        self,
        queries: List[str],
        use_cache: bool = True,
        raise_on_error: bool = False,
    ) -> List[ParsedQuery]:
        """Parse multiple queries concurrently using asyncio."""
        import asyncio
        tasks = [
            self.parse_async(q, use_cache=use_cache, raise_on_error=raise_on_error)
            for q in queries
        ]
        return await asyncio.gather(*tasks)

    # ==============================================================
    # Fallback Mechanism
    # ==============================================================

    def create_fallback(
        self,
        query: str,
        error: Optional[Exception] = None,
    ) -> ParsedQuery:
        """
        Generate a safe, conservative ParsedQuery fallback without fabricating false structure.
        """
        clean_query = query.strip() if query else ""
        if error:
            logger.warning("Using conservative fallback for query %r due to error: %s", clean_query, error)

        return ParsedQuery(
            original_query=clean_query,
            semantic_query=clean_query,
            concepts=[],
            relationships=[],
            attributes=[],
            requirements=[],
            constraints=[],
            exclusions=[],
            metadata_filters=[],
            is_metadata_only=False,
        )
