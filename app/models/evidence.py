"""
Phase 4 Evidence Retrieval Data Models

Data structures for evidence chunks and patent evidence collections
produced by Phase 4 Bounded Evidence Retrieval.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EvidenceChunk(BaseModel):
    """
    A single bounded evidence chunk retrieved for relationship verification.
    """
    model_config = ConfigDict(extra="ignore")

    patent_id: str = Field(..., description="Patent identifier.")
    chunk_id: int = Field(..., description="Chunk sequence index within the patent.")
    text: str = Field(default="", description="Full text of the evidence chunk.")
    retrieval_score: float = Field(
        default=0.0,
        description="Vector similarity or relevance retrieval score."
    )
    retrieval_source: str = Field(
        default="initial_candidate",
        description="Origin of evidence chunk: 'initial_candidate', 'evidence_query', or 'neighbor'."
    )
    section: Optional[str] = Field(default=None, description="Patent section heading (e.g. 'Claims', 'Description').")
    document_chunk_index: Optional[int] = Field(default=None, description="Document-wide chunk index.")
    token_count: Optional[int] = Field(default=None, description="Token count of the chunk.")


class PatentEvidence(BaseModel):
    """
    Bounded evidence collection for a single qualified candidate patent.
    """
    model_config = ConfigDict(extra="ignore")

    patent_id: str = Field(..., description="Patent identifier.")
    chunks: List[EvidenceChunk] = Field(
        default_factory=list,
        description="Bounded, ordered list of evidence chunks for this patent."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patent metadata dictionary."
    )
    candidate_score: float = Field(
        default=0.0,
        description="Retrieval score from Phase 2 / Phase 3 candidate ranking."
    )


class EvidenceRetrievalResult(BaseModel):
    """
    Complete output produced by Phase 4 Bounded Evidence Retrieval.
    """
    model_config = ConfigDict(extra="ignore")

    evidence_by_patent: Dict[str, List[EvidenceChunk]] = Field(
        default_factory=dict,
        description="Mapping from patent_id -> List of EvidenceChunk objects."
    )
    patent_evidence_list: List[PatentEvidence] = Field(
        default_factory=list,
        description="Ordered list of PatentEvidence items matching candidate patent order."
    )
    total_candidates: int = Field(default=0, description="Total candidate patents evaluated.")
    total_evidence_chunks: int = Field(default=0, description="Total evidence chunks gathered across all candidates.")
    evidence_query_text: str = Field(default="", description="Constructed dynamic evidence query used for retrieval.")
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Latency breakdown in milliseconds (embedding_ms, qdrant_retrieval_ms, neighbor_ms, deduplication_ms, total_ms)."
    )
