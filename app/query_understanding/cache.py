"""
Thread-safe Bounded In-Memory LRU Cache for Query Understanding
"""

from collections import OrderedDict
import threading
from typing import Optional
from app.models.parsed_query import ParsedQuery


class QueryCache:
    """
    Thread-safe, bounded Least-Recently-Used (LRU) cache for ParsedQuery results.
    Prevents repeated expensive LLM calls for identical queries.
    """

    def __init__(self, max_size: int = 1024):
        self.max_size = max(1, max_size)
        self._cache: OrderedDict[str, ParsedQuery] = OrderedDict()
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def _normalize_key(query: str) -> str:
        """Create a deterministic cache key by normalizing whitespace and casing."""
        return " ".join(query.strip().lower().split())

    def get(self, query: str) -> Optional[ParsedQuery]:
        """
        Retrieve a cached ParsedQuery if present.
        Returns a fresh copy to guarantee immutability for caller.
        """
        key = self._normalize_key(query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                cached_obj = self._cache[key]
                # Return a deep copy/model copy to prevent caller mutation of cache
                return cached_obj.model_copy(deep=True)
            self._misses += 1
            return None

    def put(self, query: str, parsed_query: ParsedQuery) -> None:
        """
        Store a validated ParsedQuery in the cache with LRU eviction.
        """
        if not query or not parsed_query:
            return
        key = self._normalize_key(query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = parsed_query.model_copy(deep=True)
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries and reset metrics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(hit_ratio, 4),
            }
