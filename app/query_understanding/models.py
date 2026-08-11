"""
Query Understanding Models

Structured representation of a user's natural-language search query.
This is the single contract between QueryUnderstanding (parser.py) and
the search pipeline (app/semantic_search.py) - the pipeline never sees
or interprets natural language, only these typed objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateFilter:
    """
    A candidate filter extracted by LLM query understanding before field resolution.

    meaning: Natural language phrase or relation (e.g. "patented by", "in the US").
    value:   Extracted candidate value (e.g. "Coca Cola", "US").
    """

    meaning: str
    value: object


@dataclass
class MetadataFilter:
    """
    A single structured metadata constraint.

    field:    Canonical snake_case field name. Must be a key in
              FIELD_MAPPING (see field_mapping.py) - the pipeline never
              accepts a field it doesn't recognize.
    operator: One of "equals", "contains", "gt", "gte", "lt", "lte" -
              restricted to what the field's type supports (see
              field_mapping.py's per-field "operators" list).
    value:    Normalized filter value (see normalizer.py).
    """

    field: str
    operator: str
    value: object


@dataclass
class ParsedQuery:
    """
    Output of QueryUnderstanding.parse().

    original_query is preserved verbatim for debugging/logging.
    semantic_query is the portion to embed and search semantically -
    metadata-filter phrases are removed from it (see parser.py).
    candidate_filters holds raw (meaning, value) candidates extracted by the LLM.
    metadata_filters holds the validated MetadataFilter objects after FIELD_MAPPING resolution.
    """

    original_query: str
    semantic_query: str
    candidate_filters: list[CandidateFilter] = field(default_factory=list)
    metadata_filters: list[MetadataFilter] = field(default_factory=list)

    @property
    def has_filters(self) -> bool:
        return len(self.metadata_filters) > 0

    @property
    def is_metadata_only(self) -> bool:
        """
        True when there's no real topic to vector-search - just
        metadata filters (see parser.py Rule 9B). SemanticSearch routes
        these straight to a whole-collection metadata filter instead of
        vector search + rerank (see SemanticSearch._search_by_metadata_only).
        """
        return not self.semantic_query.strip() and self.has_filters
