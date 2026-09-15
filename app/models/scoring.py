"""
Phase 7 Final Patent Scoring & Result Selection Data Models

Structured models for component score breakdowns, final scored patent results,
and complete Phase 7 search results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.models.reranking import RerankedEvidenceChunk
from app.models.verification import RelationshipVerification, RequirementVerification


class ScoreBreakdown(BaseModel):
    """
    Transparent component score breakdown for explainability.
    All component scores are normalized to 0.0 - 1.0.
    """
    model_config = ConfigDict(extra="ignore")

    relationship_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized relationship verification score (0.0 to 1.0)."
    )
    requirement_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized requirement verification score (0.0 to 1.0)."
    )
    reranker_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized BGE cross-encoder relevance score (0.0 to 1.0)."
    )
    retrieval_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized vector retrieval similarity score (0.0 to 1.0)."
    )
    relationship_weighted: float = Field(
        default=0.0,
        description="Weighted contribution from relationship score."
    )
    requirement_weighted: float = Field(
        default=0.0,
        description="Weighted contribution from requirement score."
    )
    reranker_weighted: float = Field(
        default=0.0,
        description="Weighted contribution from reranker score."
    )
    retrieval_weighted: float = Field(
        default=0.0,
        description="Weighted contribution from retrieval score."
    )
    raw_total: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Unrounded weighted total score on 0.0 to 10.0 scale."
    )


class FinalPatentResult(BaseModel):
    """
    A candidate patent with final computed score, score breakdown,
    and all preserved Phase 5 verification and Phase 6 evidence data.
    """
    model_config = ConfigDict(extra="ignore")

    patent_id: str = Field(..., description="Unique patent identifier.")
    final_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Final patent relevance score on 0.0 to 10.0 scale (rounded to 2 decimal places)."
    )
    score_breakdown: ScoreBreakdown = Field(
        default_factory=ScoreBreakdown,
        description="Detailed score component breakdown for explainability."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patent metadata dictionary."
    )
    relationship_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of requested relationships supported by evidence."
    )
    requirement_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of requested requirements supported by evidence."
    )
    best_reranker_score: float = Field(
        default=0.0,
        description="Highest cross-encoder score among this patent's evidence chunks."
    )
    candidate_score: float = Field(
        default=0.0,
        description="Phase 2 / Phase 4 retrieval score."
    )
    supported_count: int = Field(default=0, description="Count of supported relationships.")
    unsupported_count: int = Field(default=0, description="Count of unsupported relationships.")
    contradicted_count: int = Field(default=0, description="Count of contradicted relationships.")
    unknown_count: int = Field(default=0, description="Count of unknown relationships.")
    relationships: List[RelationshipVerification] = Field(
        default_factory=list,
        description="List of verified relationships from Phase 5."
    )
    requirements: List[RequirementVerification] = Field(
        default_factory=list,
        description="List of verified requirements from Phase 5."
    )
    evidence: List[RerankedEvidenceChunk] = Field(
        default_factory=list,
        description="List of bounded evidence chunks scored by BGE."
    )


class FinalSearchResult(BaseModel):
    """
    Complete output produced by Phase 7 Final Patent Scoring & Result Selection.
    """
    model_config = ConfigDict(extra="ignore")

    results: List[FinalPatentResult] = Field(
        default_factory=list,
        description="Ordered list of final qualifying patents (sorted by final_score desc, score >= threshold)."
    )
    total_candidates_evaluated: int = Field(
        default=0,
        description="Total candidate patents evaluated in Phase 7."
    )
    passed_threshold_count: int = Field(
        default=0,
        description="Count of candidates that met or exceeded FINAL_SCORE_THRESHOLD."
    )
    rejected_count: int = Field(
        default=0,
        description="Count of candidates excluded due to score < FINAL_SCORE_THRESHOLD."
    )
    threshold_used: float = Field(
        default=7.0,
        description="FINAL_SCORE_THRESHOLD used for inclusion."
    )
    top_k_limit: int = Field(
        default=10,
        description="FINAL_TOP_K maximum limit applied."
    )
    weights_used: Dict[str, float] = Field(
        default_factory=dict,
        description="Component scoring weights used."
    )
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Latency metrics in milliseconds (scoring_ms, filter_sort_ms, total_ms)."
    )
