"""
Phase 5 Relationship Verification Data Models

Structured models for semantic relationship verification, requirement verification,
evidence citations, and per-patent verification results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RelationshipVerification(BaseModel):
    """
    Verification outcome for a single directed semantic relationship.
    """
    model_config = ConfigDict(extra="ignore")

    relationship_index: int = Field(..., description="Index of the relationship in ParsedQuery.relationships.")
    subject: str = Field(..., description="Subject concept.")
    relation: str = Field(..., description="Predicate / relation connector.")
    object: str = Field(..., description="Object concept.")
    supported: bool = Field(default=False, description="Whether evidence supports this relationship.")
    status: str = Field(
        default="NOT_SUPPORTED",
        description="Status: 'SUPPORTED', 'NOT_SUPPORTED', 'CONTRADICTED', or 'UNKNOWN'."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence in this relationship verification."
    )
    evidence_chunk_ids: List[int] = Field(
        default_factory=list,
        description="List of chunk IDs that provide direct evidence for this relationship."
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Brief explanation of how evidence supports, contradicts, or lacks the relationship."
    )


class RequirementVerification(BaseModel):
    """
    Verification outcome for an explicit query requirement.
    """
    model_config = ConfigDict(extra="ignore")

    requirement_index: int = Field(..., description="Index of the requirement in ParsedQuery.requirements.")
    requirement: str = Field(..., description="Requirement text statement.")
    supported: bool = Field(default=False, description="Whether evidence supports this requirement.")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence in requirement verification."
    )
    evidence_chunk_ids: List[int] = Field(
        default_factory=list,
        description="List of chunk IDs providing evidence for this requirement."
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Brief explanation."
    )


class PatentVerificationResult(BaseModel):
    """
    Semantic relationship & requirement verification result for a single candidate patent.
    """
    model_config = ConfigDict(extra="ignore")

    patent_id: str = Field(..., description="Unique patent identifier.")
    relationships: List[RelationshipVerification] = Field(
        default_factory=list,
        description="List of verified relationship results."
    )
    requirements: List[RequirementVerification] = Field(
        default_factory=list,
        description="List of verified requirement results."
    )
    relationship_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of requested relationships supported by evidence (0.0 - 1.0)."
    )
    requirement_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of requested requirements supported by evidence (0.0 - 1.0)."
    )
    supported_count: int = Field(default=0, description="Count of supported relationships.")
    unsupported_count: int = Field(default=0, description="Count of unsupported relationships.")
    contradicted_count: int = Field(default=0, description="Count of contradicted relationships.")
    unknown_count: int = Field(default=0, description="Count of unknown relationships.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patent metadata dictionary."
    )
    candidate_score: float = Field(
        default=0.0,
        description="Phase 2 / Phase 4 retrieval score."
    )


class VerificationBatchResult(BaseModel):
    """
    Complete output produced by Phase 5 Relationship Verification.
    """
    model_config = ConfigDict(extra="ignore")

    verified_patents: List[PatentVerificationResult] = Field(
        default_factory=list,
        description="List of verification results for all evaluated candidate patents."
    )
    total_evaluated: int = Field(default=0, description="Total candidate patents verified.")
    fully_supported_count: int = Field(default=0, description="Candidates with 100% relationship coverage.")
    partially_supported_count: int = Field(default=0, description="Candidates with >0% and <100% coverage.")
    unsupported_count: int = Field(default=0, description="Candidates with 0% coverage.")
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Latency metrics (llm_verification_ms, total_ms, avg_per_patent_ms)."
    )
