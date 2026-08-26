"""
Domain-Agnostic Query Understanding Data Structures

Typed dataclasses representing the parsed query, its metadata filters,
and its dynamic requirements structure for reranking.

The domain-specific aspects live entirely in the LLM prompt and the
allowlists (FIELD_MAPPING, CODE_TO_FIELD). These structures are generic
across any technology domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateFilter:
    """
    A raw (meaning, value) candidate extracted by the LLM before validation.
    Kept on ParsedQuery for debugging and telemetry.
    """

    meaning: str
    value: str | int | float


@dataclass
class MetadataFilter:
    """
    A validated metadata filter ready to apply against Qdrant.

    field:    The canonical Python-side field name (e.g. 'publication_year').
    operator: One of the allowed operators for this field's type
              ('equals', 'gt', 'gte', 'lt', 'lte', 'contains').
    value:    The typed value (int, str, list of str).
    group:    None for a normal, independently-required filter (AND'd with
              every other filter). Set to a shared id when this filter is
              one member of an OR-matched fallback group (e.g. a bare
              country/year value expanded across every field in that
              category by an "ANY_COUNTRY"/"ANY_YEAR" LLM field code - see
              FIELD_GROUPS in field_mapping.py) - a patent needs to satisfy
              only ONE filter per group, but every group (and every
              ungrouped filter) is still required. See FilterEngine.matches().
    """

    field: str
    operator: str
    value: str | int | float | list[str]
    group: str | None = None


# ==================================================================
# Dynamic requirements structure for reranking
# ==================================================================


@dataclass
class Concept:
    """
    A key domain concept identified in the query, with its role and importance.
    Extracted entirely by the LLM - no hardcoded entity types.
    """

    id: str
    text: str
    role: str = "primary_concept"
    importance: float = 1.0
    required: bool = False
    semantic_variants: list[str] = field(default_factory=list)


@dataclass
class Goal:
    """A functional goal or objective the user wants to achieve."""

    id: str
    text: str
    importance: float = 1.0
    required: bool = False
    keywords: list[str] = field(default_factory=list)


@dataclass
class Constraint:
    """
    A domain/engineering constraint mentioned in the query (distinct from
    metadata filters like date or jurisdiction).
    """

    id: str
    text: str
    type: str = ""
    importance: float = 1.0
    required: bool = False
    keywords: list[str] = field(default_factory=list)
    strictness: str = "hard"


@dataclass
class OptimizationTarget:
    """
    A property to maximize or minimize (e.g. minimize latency, maximize yield).

    direction: "maximize" | "minimize"
    property:  The property name (e.g. "latency", "power consumption")
    target_value: Optional target specification
    """

    id: str
    property: str
    direction: str = "maximize"
    target_value: str = ""
    importance: float = 1.0


@dataclass
class Relationship:
    """
    A relationship between two concepts/entities that must hold in relevant patents.
    Generic triple representation: source -[relation]-> target.
    """

    source: str
    relation: str
    target: str
    id: str = ""
    importance: float = 1.0


@dataclass
class Requirement:
    """
    A specific technical requirement that candidate patents must satisfy.
    Derived from concepts, goals, and constraints.
    """

    id: str
    description: str
    type: str = "semantic_match"
    importance: float = 1.0
    required: bool = False
    evaluation_hint: str = ""
    keywords: list[str] = field(default_factory=list)
    priority: str = "medium"
    verification_hint: str = ""


@dataclass
class RankingWeights:
    """
    Adaptive ranking weights suggested by the LLM based on query complexity.
    Values are in [0, 1] and will be normalized by Reranker._compute_weights().
    """

    semantic_relevance: float = 1.0
    requirement_satisfaction: float = 0.0
    relationship_satisfaction: float = 0.0
    constraint_satisfaction: float = 0.0
    evidence_strength: float = 0.0
    exact_match: float = 0.0


@dataclass
class QuestionIntent:
    """
    Dynamic description of requested information when the query is a question.
    Generic across all technology domains without fixed enums or categories.
    """

    is_question: bool = True
    target: str = ""
    expected_answer_type: str = ""
    answer_criteria: str = ""


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

    is_question: bool = False
    question_intent: QuestionIntent | None = None

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
