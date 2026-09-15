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
from app.query_understanding.prompt import ANCHOR_ASSISTANT, ANCHOR_USER, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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

        # Persistent HTTP clients (connection pooling & keep-alive)
        self.sync_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        # Normalizer and Thread-Safe Cache
        self.normalizer = QueryNormalizer()
        self.cache = QueryCache(max_size=cache_size or QUERY_CACHE_SIZE)

    def _prepare_request_messages(self, query: str) -> List[dict[str, Any]]:
        """Assemble the compact anchored message payload for an incoming query."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ANCHOR_USER},
            {"role": "assistant", "content": ANCHOR_ASSISTANT},
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

            completion = self.sync_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                timeout=self.timeout,
                max_tokens=400,
            )
            t_llm_end = time.perf_counter()

            raw_text = completion.choices[0].message.content or "{}"
            data = json.loads(raw_text)
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

            completion = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                timeout=self.timeout,
                max_tokens=400,
            )
            t_llm_end = time.perf_counter()

            raw_text = completion.choices[0].message.content or "{}"
            data = json.loads(raw_text)
            data["original_query"] = clean_query
            if not data.get("semantic_query"):
                data["semantic_query"] = data.get("query", clean_query)

            raw_parsed = ParsedQuery.model_validate(data)

            t_norm_start = time.perf_counter()
            normalized = self.normalizer.normalize(raw_parsed)
            t_norm_end = time.perf_counter()

            if use_cache:
                self.cache.put(clean_query, normalized)

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
