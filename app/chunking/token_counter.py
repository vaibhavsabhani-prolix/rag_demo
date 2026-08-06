"""
Token Counter

Wraps the embedding model's tokenizer to provide accurate token counts.
Used throughout the chunking pipeline instead of len(text).

Performance: An LRU cache avoids redundant tokenizer calls for text
that is counted multiple times (e.g. during splitting and re-joining).
"""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer

from app.config import EMBEDDING_MODEL, TOKEN_COUNT_CACHE_SIZE


# ==================================================================
# Cached count function (module-level to avoid `self` in cache key)
# ==================================================================

def _make_cached_counter(tokenizer, maxsize: int):
    """
    Create a cached token-counting function bound to *tokenizer*.

    Using a standalone function (not a method) lets lru_cache key
    on the text string alone, without including `self`.
    """

    @lru_cache(maxsize=maxsize)
    def _count(text: str) -> int:
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))

    return _count


class TokenCounter:
    """
    Token-aware text measurement using the actual embedding model tokenizer.

    Loads the tokenizer once at construction. All downstream components
    share a single TokenCounter instance to avoid redundant loads.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Cached counting function — avoids redundant tokenizer calls
        self._cached_count = _make_cached_counter(
            self._tokenizer,
            maxsize=TOKEN_COUNT_CACHE_SIZE,
        )

    # ==============================================================
    # Core API
    # ==============================================================

    def count(self, text: str) -> int:
        """
        Return the number of tokens in *text*.

        Results are LRU-cached. Repeated calls for the same text
        are O(1) dict lookups instead of O(n) tokenizer passes.
        """

        return self._cached_count(text)

    def tokenize(self, text: str) -> list[int]:
        """
        Return raw token IDs for *text*.
        """

        return self._tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        """

        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

    def truncate(self, text: str, max_tokens: int) -> str:
        """
        Truncate *text* to at most *max_tokens* tokens and decode back.
        """

        ids = self.tokenize(text)

        if len(ids) <= max_tokens:
            return text

        return self.decode(ids[:max_tokens])

    def clear_cache(self) -> None:
        """
        Clear the token count cache.
        Call between documents if memory is a concern.
        """

        self._cached_count.cache_clear()

    def cache_info(self):
        """
        Return cache statistics (hits, misses, size, maxsize).
        """

        return self._cached_count.cache_info()
