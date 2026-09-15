"""
Phase 6 BGE Reranking Data Models

Structured models for evidence chunk reranking scores, patent-level
intermediate reranker aggregations, and Phase 6 batch results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.models.verification import RelationshipVerification, RequirementVerification


class RerankedEvidenceChunk(BaseModel):
    """
    An evidence chunk scored by the BGE cross-encoder reranker.
    Preserves all retrieval metadata and adds the cross-encoder relevance score.
    """
    model_config = ConfigDict(extra="ignore")

    patent_id: str = Field(..., description="Patent identifier.")
    chunk_id: int = Field(..., description="Chunk sequence index within the patent.")
    text: str = Field(default="", description="Text of the evidence chunk.")
    retrieval_score: float = Field(
        default=0.0,
        description="Vector similarity or retrieval score from Phase 4."
    )
    retrieval_source: str = Field(
        default="initial_candidate",
        description="Origin of evidence chunk: 'initial_candidate', 'evidence_query', or 'neighbor'."
    )
    section: Optional[str] = Field(default=None, description="Patent section heading.")
    document_chunk_index: Optional[int] = Field(default=None, description="Document-wide chunk index.")
    token_count: Optional[int] = Field(default=None, description="Token count of the chunk.")
    reranker_score: float = Field(
        default=0.0,
        description="Relevance score returned by the BGE cross-encoder reranker."
    )


class RerankedPatentResult(BaseModel):
    """
    Candidate patent with Phase 5 verification results and Phase 6 reranker scores.
    Preserves all Phase 5 relationship and requirement verification results.
    """
    model_config = ConfigDict(extra="ignore")

    patent_id: str = Field(..., description="Unique patent identifier.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patent metadata dictionary."
    )
    candidate_score: float = Field(
        default=0.0,
        description="Phase 2 / Phase 4 retrieval score."
    )
    relationship_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of requested relationships supported by evidence (from Phase 5)."
    )
    requirement_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of requested requirements supported by evidence (from Phase 5)."
    )
    relationships: List[RelationshipVerification] = Field(
        default_factory=list,
        description="List of verified relationships from Phase 5."
    )
    requirements: List[RequirementVerification] = Field(
        default_factory=list,
        description="List of verified requirements from Phase 5."
    )
    supported_count: int = Field(default=0, description="Count of supported relationships.")
    unsupported_count: int = Field(default=0, description="Count of unsupported relationships.")
    contradicted_count: int = Field(default=0, description="Count of contradicted relationships.")
    unknown_count: int = Field(default=0, description="Count of unknown relationships.")
    evidence: List[RerankedEvidenceChunk] = Field(
        default_factory=list,
        description="List of bounded evidence chunks scored by BGE."
    )
    best_reranker_score: float = Field(
        default=0.0,
        description="Highest cross-encoder score among this patent's evidence chunks."
    )
    avg_reranker_score: float = Field(
        default=0.0,
        description="Average cross-encoder score across this patent's evidence chunks."
    )


class RerankBatchResult(BaseModel):
    """
    Complete output produced by Phase 6 BGE Reranking.
    """
    model_config = ConfigDict(extra="ignore")

    reranked_patents: List[RerankedPatentResult] = Field(
        default_factory=list,
        description="List of reranked patent candidate results."
    )
    total_candidates: int = Field(default=0, description="Total candidates processed.")
    total_chunks_reranked: int = Field(default=0, description="Total evidence chunks scored by BGE.")
    total_requests: int = Field(default=0, description="Total HTTP requests sent to BGE server.")
    truncated_chunks_count: int = Field(default=0, description="Number of chunks truncated for token budget.")
    reranking_query: str = Field(default="", description="Deterministic reranking query sent to BGE.")
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Latency metrics in milliseconds (reranker_http_ms, total_ms, avg_per_chunk_ms)."
    )
