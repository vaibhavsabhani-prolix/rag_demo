"""
Token Counter

Provides accurate token counts using the embedding model's tokenizer.
An LRU cache avoids repeated tokenizer calls for the same text.
"""

from functools import lru_cache

from transformers import AutoTokenizer

from app.config import EMBEDDING_MODEL, TOKEN_COUNT_CACHE_SIZE


def _make_cached_counter(tokenizer, maxsize: int):
    """Create a cached token-counting function."""

    @lru_cache(maxsize=maxsize)
    def _count(text: str) -> int:
        if not text:
            return 0

        # verbose=False suppresses the tokenizer's "sequence longer than
        # the maximum for this model" warning. That warning exists to
        # flag input that would index out of bounds when run through the
        # model - but this call only ever counts tokens, and the counts
        # are what the chunker uses to keep every chunk under
        # MAX_CHUNK_TOKENS in the first place. A long patent would
        # otherwise emit it once per oversized unit.
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=False,
                verbose=False,
            )
        )

    return _count


class TokenCounter:
    """Token-aware text measurement using the embedding model tokenizer."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self._cached_count = _make_cached_counter(
            self._tokenizer,
            maxsize=TOKEN_COUNT_CACHE_SIZE,
        )

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""
        return self._cached_count(text)

    def clear_cache(self) -> None:
        """Clear the token count cache."""
        self._cached_count.cache_clear()