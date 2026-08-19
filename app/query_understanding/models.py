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
    operator: One of "equals", "contains", "not_equals", "not_contains",
              "gt", "gte", "lt", "lte" - restricted to what the field's
              type supports (see field_mapping.py's per-field
              "operators" list). "not_equals"/"not_contains" express an
              exclusion ("not from China", "excluding Coca Cola") -
              gt/gte/lt/lte need no negated variant since their
              opposite is just another comparison operator
              (e.g. "not after 2018" is "lte 2018").
    value:    Normalized filter value (see normalizer.py).
    """

    field: str
    operator: str
    value: object


@dataclass
class Concept:
    """
    A single concept extracted from the query (object, technology,
    component, material, process, goal, constraint, attribute, etc. -
    see `role`). Generic across domains - the LLM decides what's
    present, nothing here is hardcoded to a technology.
    """

    id: str
    text: str
    role: str
    importance: float = 0.5
    required: bool = False
    semantic_variants: list[str] = field(default_factory=list)


@dataclass
class Goal:
    """
    What the invention should accomplish (as distinct from a Constraint).

    keywords: short literal phrases likely to actually appear in patent
    text (distinct from `text`, which is the natural-language
    description) - mirrors Concept.semantic_variants, since coverage
    checking (see app/reranker.py) is plain substring matching and a
    single natural-language sentence rarely appears verbatim in a patent.
    """

    id: str
    text: str
    importance: float = 0.5
    required: bool = False
    keywords: list[str] = field(default_factory=list)


@dataclass
class Constraint:
    """A condition that must remain satisfied while achieving a Goal."""

    id: str
    text: str
    type: str = ""
    importance: float = 0.5
    required: bool = False
    keywords: list[str] = field(default_factory=list)


@dataclass
class OptimizationTarget:
    """A property the invention should maximize or minimize (e.g. power consumption -> minimize)."""

    id: str
    property: str
    direction: str = "maximize"
    importance: float = 0.5


@dataclass
class Relationship:
    """A directed relation between two concepts (e.g. camera -> used_for -> defect detection)."""

    source: str
    relation: str
    target: str
    importance: float = 0.5


@dataclass
class Requirement:
    """
    A single reranking-time requirement derived from the query's
    concepts/goals/constraints - what a downstream reranker should
    evaluate a candidate chunk against.
    """

    id: str
    description: str
    type: str = ""
    importance: float = 0.5
    required: bool = False
    evaluation_hint: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class RankingWeights:
    """
    Relative weights (should sum to ~1.0) for blending the reranker's
    semantic score with the requirement/relationship/constraint
    coverage signals computed from the fields above. See
    Reranker._blend_scores in app/reranker.py for how these are
    actually applied - weights for signals the query has no data for
    (e.g. no constraints extracted) are redistributed into
    semantic_relevance rather than penalizing every candidate equally.
    """

    semantic_relevance: float = 1.0
    requirement_satisfaction: float = 0.0
    relationship_satisfaction: float = 0.0
    constraint_satisfaction: float = 0.0
    evidence_strength: float = 0.0
    exact_match: float = 0.0


@dataclass
class ParsedQuery:
    """
    Output of QueryUnderstanding.parse().

    original_query is preserved verbatim for debugging/logging.
    semantic_query is the portion to embed and search semantically -
    metadata-filter phrases are removed from it (see parser.py).
    candidate_filters holds raw (meaning, value) candidates extracted by the LLM.
    metadata_filters holds the validated MetadataFilter objects after FIELD_MAPPING resolution.

    The remaining fields (concepts, goals, constraints, optimization,
    exclusions, relationships, requirements, ranking_weights) are the
    dynamic, domain-agnostic requirements structure used by
    Reranker.rerank() to blend a requirement/constraint/relationship
    coverage score in with the semantic cross-encoder score - see
    app/reranker.py. They default to empty/neutral so existing code
    that only cares about semantic_query/metadata_filters is unaffected.
    """

    original_query: str
    semantic_query: str
    candidate_filters: list[CandidateFilter] = field(default_factory=list)
    metadata_filters: list[MetadataFilter] = field(default_factory=list)

    intent: str = ""
    query_type: list[str] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    goals: list[Goal] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    optimization: list[OptimizationTarget] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    ranking_weights: RankingWeights = field(default_factory=RankingWeights)

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

    @property
    def has_requirements_structure(self) -> bool:
        """
        True when query understanding extracted any of the dynamic
        requirements fields - lets Reranker fall back to pure semantic
        ranking when there's nothing to blend in.

        Includes `optimization` explicitly - a pure-optimization query
        (e.g. "reduce power consumption in semiconductor devices") could
        in principle extract an OptimizationTarget without a matching
        Concept, and would otherwise incorrectly fall back to
        pure-semantic-only ranking.
        """
        return bool(
            self.concepts
            or self.goals
            or self.constraints
            or self.optimization
            or self.relationships
            or self.exclusions
            or self.requirements
        )
