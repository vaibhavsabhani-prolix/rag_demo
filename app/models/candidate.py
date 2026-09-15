"""
Candidate Retrieval Data Models

Structures for chunks, candidate patents, and aggregated retrieval results
produced by Phase 2 dynamic candidate retrieval.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CandidateChunk(BaseModel):
    """
    An individual patent chunk point retrieved from Qdrant vector search.
    """
    model_config = ConfigDict(extra="ignore")

    point_id: str = Field(..., description="Unique Qdrant point UUID/ID.")
    patent_id: str = Field(..., description="Patent identifier (e.g. 'US1234567A').")
    chunk_id: int = Field(..., description="Chunk index within the patent.")
    score: float = Field(..., description="Vector similarity / retrieval score for this chunk.")
    matched_views: List[str] = Field(
        default_factory=list,
        description="Names of retrieval views that retrieved this chunk point."
    )
    section: Optional[str] = Field(default=None, description="Patent section heading (e.g. 'Claims', 'Description').")
    text: Optional[str] = Field(default=None, description="Chunk text content if loaded.")
    document_chunk_index: Optional[int] = Field(default=None, description="Document-wide chunk index.")
    token_count: Optional[int] = Field(default=None, description="Token count of the chunk.")


class CandidatePatent(BaseModel):
    """
    A candidate patent aggregated from one or more retrieved chunk hits.
    """
    model_config = ConfigDict(extra="ignore")

    patent_id: str = Field(..., description="Unique patent identifier.")
    best_chunk_id: Optional[int] = Field(default=None, description="Chunk ID with the highest retrieval score.")
    retrieval_score: float = Field(
        default=0.0,
        description="Strongest chunk retrieval score representing this patent candidate."
    )
    matched_views: List[str] = Field(
        default_factory=list,
        description="Unique retrieval views that matched any chunk in this patent."
    )
    chunk_count: int = Field(
        default=0,
        description="Number of chunks retrieved for this patent."
    )
    chunks: List[CandidateChunk] = Field(
        default_factory=list,
        description="Retrieved candidate chunk points belonging to this patent."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patent-level metadata (title, assignee, publication date/year, classifications, etc.)."
    )


class CandidateRetrievalResult(BaseModel):
    """
    Complete output produced by Phase 2 Dynamic Candidate Retrieval.
    """
    model_config = ConfigDict(extra="ignore")

    candidates: List[CandidatePatent] = Field(
        default_factory=list,
        description="Bounded, ordered list of patent candidates."
    )
    retrieval_views: Dict[str, str] = Field(
        default_factory=dict,
        description="Dynamic retrieval views constructed for search (view_name -> query text)."
    )
    total_chunk_hits: int = Field(
        default=0,
        description="Total raw chunk points returned across all views before deduplication."
    )
    unique_patents: int = Field(
        default=0,
        description="Total number of unique candidate patents identified."
    )
    is_metadata_only: bool = Field(
        default=False,
        description="Whether this retrieval was exclusively metadata-filtered."
    )
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Latency breakdown in milliseconds (embedding_ms, qdrant_retrieval_ms, merge_ms, total_ms)."
    )


class FilterDiagnostic(BaseModel):
    """
    Diagnostic metrics for an individual metadata filter.
    """
    model_config = ConfigDict(extra="ignore")

    field: str = Field(..., description="Canonical metadata field name.")
    operator: str = Field(..., description="Comparison operator.")
    value: Any = Field(..., description="Target value.")
    passed: int = Field(default=0, description="Number of candidates that passed this filter.")
    failed: int = Field(default=0, description="Number of candidates that failed this filter.")


class FilteredCandidateResult(BaseModel):
    """
    Phase 3 Output: Candidate patents strictly validated against all metadata constraints.
    """
    model_config = ConfigDict(extra="ignore")

    candidates: List[CandidatePatent] = Field(
        default_factory=list,
        description="Filtered list of patent candidates that strictly satisfy all metadata constraints."
    )
    total_before: int = Field(default=0, description="Number of candidates before filtering.")
    total_after: int = Field(default=0, description="Number of candidates surviving all filters.")
    filtered_count: int = Field(default=0, description="Number of candidates rejected by metadata filters.")
    diagnostics: List[FilterDiagnostic] = Field(
        default_factory=list,
        description="Detailed pass/fail breakdown per metadata filter."
    )
    is_metadata_only: bool = Field(
        default=False,
        description="Whether this query was exclusively metadata constraints."
    )
    filter_time_ms: float = Field(
        default=0.0,
        description="Phase 3 execution latency in milliseconds."
    )
