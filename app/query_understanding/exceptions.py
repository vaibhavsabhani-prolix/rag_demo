"""
Query Understanding Exceptions

Specialized error types for query understanding failure modes.
"""

from typing import Optional


class QueryUnderstandingError(Exception):
    """Base exception for all query understanding errors."""

    def __init__(self, message: str, original_query: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.original_query = original_query
        self.details = details or {}

    def __str__(self) -> str:
        base = f"[QueryUnderstandingError] {self.message}"
        if self.original_query:
            base += f" (Query: {self.original_query!r})"
        return base


class LLMCommunicationError(QueryUnderstandingError):
    """Raised when remote LLM service is unreachable, times out, or returns an HTTP error."""
    pass


class SchemaValidationError(QueryUnderstandingError):
    """Raised when the LLM response fails Pydantic schema validation."""
    pass


class NormalizationError(QueryUnderstandingError):
    """Raised when post-processing normalization encounters unrecoverable errors."""
    pass
